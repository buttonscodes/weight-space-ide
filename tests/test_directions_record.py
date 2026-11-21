"""The reproduction record of a committed direction edit (wsp.directions_record), model-free.

The workspace is the tiny random model of tests/test_steering.py (two layers, d_model 24), opened by a Session;
measurements are made with DR.measure on it or written by save_random. Every expectation is computed here from the
files themselves (hashlib over the bytes on disk, json of the prompt files), not with the module under test.
"""
import hashlib
import json
import os
import platform
import time

import pytest
import torch

from wsp import delta as Dl
from wsp import directions as DR
from wsp import directions_record as REC
from wsp import directions_search as DSR
from wsp.atlas import wiring
from wsp.session import Session
from wsp.wscript import run_script

from test_directions import GOOD, BAD, distributions, save_random
from test_steering import CFG, make_ws

N = CFG["n_layers"]
APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELDS = ["version", "patch", "created", "script", "script_sha256", "rebuildable", "calls", "pieces", "made_on", "base",
          "environment", "origin", "search", "heretic"]
ARA_SMALL = {"lora_rank": 2, "n_optimization_steps": 1, "max_iter": 2}


def ablation_params(name="m", mode="none", **kw):
    return {"direction": name, "direction_index": None, "weight_distributions": distributions(),
            "orthogonalize_direction": True, "row_normalization": mode, "full_normalization_lora_rank": 3, **kw}


def sha_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def mpath(ws, name, ext):
    return os.path.join(ws.dir, "directions", name + ext)


def session(tmp_path):
    make_ws(tmp_path / "rnd")
    sess = Session(str(tmp_path))
    sess.open("rnd")
    return sess, sess.ws


def records(ws):
    return os.path.join(ws.dir, "directions", "records")


def record(ws, pid):
    with open(os.path.join(records(ws), f"{pid}.json")) as f:
        return json.load(f)


def canonical_sha(obj, sort_keys=False):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys)
                          .encode("utf-8")).hexdigest()


def meta_sha(ws, name):
    meta = json.load(open(mpath(ws, name, ".json")))
    return hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def base_path(ws):
    return os.path.join(ws.dir, "base.safetensors")


# ------------------------------------------------------------------ the interpreter lists its calls
def test_the_interpreter_lists_the_measurements_an_edit_used(tmp_path):
    ws = make_ws(tmp_path / "rnd")
    save_random(ws, "m", n_layers=N, d=CFG["d_model"])
    params = ablation_params()
    res = run_script(ws, DR.script_for(params) + '\nv = direction("m", 1)\ng, b = direction_means("m")')
    assert set(res) == {"deltas", "output", "reverted", "calls"}
    assert [c["function"] for c in res["calls"]] == ["ablate_direction", "direction", "direction_means"]
    abl, d, dm = res["calls"]
    assert set(abl) == {"function", "method", "parameters", "measurement", "result"}
    assert abl["method"] == "ablation" and abl["parameters"] == DR.normalize(params)
    meta = json.load(open(mpath(ws, "m", ".json")))
    want = {"name": "m", "meta_sha256": hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "means_sha256": sha_file(mpath(ws, "m", ".safetensors")), "io_sha256": None,
            "prompts_sha256": meta["prompts_sha256"], "created": meta["created"], "counts": meta["counts"]}
    assert abl["measurement"] == want == d["measurement"] == dm["measurement"]
    assert set(abl["result"]) == {"stored", "components", "skipped"}
    assert abl["result"]["stored"] == sum(p.numel() for ps in res["deltas"].values() for p in ps)
    assert (d["method"], d["parameters"], d["result"]) == (None, {"name": "m", "k": 1}, None)
    assert (dm["method"], dm["parameters"], dm["result"]) == (None, {"name": "m"}, None)
    json.dumps(res["calls"])                                        # plain JSON values
    assert run_script(ws, "mlp(1).W_out *= 0.5")["calls"] == []
    # a failing call is not listed: the script stops there
    with pytest.raises(Exception):
        run_script(ws, 'direction("nope", 1)')


# ------------------------------------------------------------------ RC5
def test_new_measurements_record_their_environment(tmp_path):
    import platform
    from wsp.atlas import wiring
    ws = make_ws(tmp_path / "rnd")
    DR.measure(ws, "m", GOOD, BAD)
    meta = json.load(open(mpath(ws, "m", ".json")))
    env = meta["environment"]
    assert env["libraries"]["torch"] == torch.__version__
    assert set(env) == {"libraries", "sampler", "python", "torch_threads", "machine", "system", "runner"}
    assert (env["python"], env["machine"], env["system"]) == (platform.python_version(), platform.machine(),
                                                              platform.system())
    assert env["torch_threads"] == torch.get_num_threads()
    assert env["runner"] == {"class": type(ws._runner).__name__, "dtype": str(getattr(ws._runner, "dtype", None)),
                             "device": str(getattr(ws._runner, "device", None))}
    assert meta["measured_on"] == []                                # variant base: the base weights
    p = ws.commit({"blocks.1.mlp.W_out": [Dl.lowrank((range(0, 2), range(0, 3)), torch.ones(2, 1), torch.ones(1, 3))]}, "x")
    DR.measure(ws, "c", GOOD, BAD, variant="current")
    meta = json.load(open(mpath(ws, "c", ".json")))
    assert meta["measured_on"] == [[p.id, 1.0, float(p.created), int(p.stored)]] == wiring.patch_fingerprint(ws)
    assert DR.load(ws, "c")[0]["environment"]["runner"]["class"] == type(ws._runner).__name__


def test_a_record_of_an_older_measurement_has_null_environment(tmp_path):
    sess, ws = session(tmp_path)
    save_random(ws, "old", n_layers=N, d=CFG["d_model"])          # a meta without environment and measured_on
    assert "environment" not in json.load(open(mpath(ws, "old", ".json")))
    r = sess.apply(DR.script_for(ablation_params("old")))
    m = record(ws, r["patch"])["calls"][0]["measurement"]
    assert m["environment"] is None and m["measured_on"] is None and m["variant"] == "base"


# ------------------------------------------------------------------ RC1
def test_a_committed_ablation_writes_its_record(tmp_path):
    sess, ws = session(tmp_path)
    DR.measure(ws, "m", GOOD, BAD)
    first = sess.apply("mlp(1).W_out *= 0.5")
    assert first["committed"] and "record" not in first and not os.path.exists(records(ws))
    before = wiring.patch_fingerprint(ws)
    params = ablation_params()
    script = DR.script_for(params)
    r = sess.apply(script)
    pid = r["patch"]
    rec = record(ws, pid)
    patch = ws.patch(pid)
    assert list(rec) == FIELDS
    assert rec["version"] == 1 and rec["heretic"] == "64e9991" and rec["origin"] is None and rec["search"] is None
    assert rec["patch"] == {"id": pid, "created": patch.created, "message": patch.message, "author": "human"}
    assert rec["script"] == script and rec["script_sha256"] == hashlib.sha256(script.encode()).hexdigest()
    assert rec["rebuildable"] is True
    assert r["record"] == {"version": 1, "calls": 1, "digest": rec["pieces"]["digest"]} and "record_error" not in r
    (c,) = rec["calls"]
    assert c["function"] == "ablate_direction" and c["method"] == "ablation" and c["parameters"] == DR.normalize(params)
    assert c["seed"] is None and c["heretic"] == {
        "modifier": "Abliteration", "parameters": {"direction_index": None, "weight_distributions": distributions()},
        "settings": {"orthogonalize_direction": True, "row_normalization": "none", "full_normalization_lora_rank": 3}}
    m = c["measurement"]
    meta = json.load(open(mpath(ws, "m", ".json")))
    assert m["name"] == "m" and m["meta_sha256"] == meta_sha(ws, "m")
    assert m["means_sha256"] == sha_file(mpath(ws, "m", ".safetensors")) and m["io_sha256"] is None
    assert (m["prompts_sha256"], m["created"], m["counts"]) == (meta["prompts_sha256"], meta["created"], meta["counts"])
    for k in ("variant", "tokens", "options", "datasets", "module_io", "model", "environment", "measured_on"):
        assert m[k] == meta[k], k
    prompts = json.load(open(mpath(ws, "m", ".prompts.json")))
    assert m["prompts_snapshot"] == canonical_sha(prompts, sort_keys=True)
    snap = os.path.join(records(ws), "prompts", m["prompts_snapshot"] + ".json")
    assert json.load(open(snap)) == DR.prompts(ws, "m") == {"good": GOOD, "bad": BAD}
    assert m["means_snapshot"] == m["means_sha256"]
    means = os.path.join(records(ws), "means", m["means_snapshot"] + ".safetensors")
    assert open(means, "rb").read() == open(mpath(ws, "m", ".safetensors"), "rb").read()
    assert rec["pieces"] == {"digest": DSR.piece_digest(patch.deltas),
                             "stored": sum(p.numel() for ps in patch.deltas.values() for p in ps),
                             "tensors": sorted(patch.deltas)}
    assert rec["made_on"] == before == [[first["patch"], 1.0, ws.patch(first["patch"]).created,
                                         ws.patch(first["patch"]).stored]]
    st = os.stat(base_path(ws))
    assert rec["base"] == {"kind": "wsp", "name": "rnd", "repo": None, "sha": None, "base_sha256": None,
                           "files": [{"name": "base.safetensors", "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
                                      "sha256": sha_file(base_path(ws))}],
                           "hashed": True, "hash_seconds": rec["base"]["hash_seconds"]}
    assert json.load(open(os.path.join(records(ws), "base.json"))) == {
        "files": [["base.safetensors", st.st_size, st.st_mtime_ns, sha_file(base_path(ws))]]}
    env = rec["environment"]
    assert env["libraries"]["torch"] == torch.__version__ and env["platform"] == platform.platform()
    assert env["implementation"] == platform.python_implementation()
    assert env["packages"]["torch"] is not None and set(env["packages"]) == {
        "torch", "transformers", "safetensors", "numpy", "tokenizers", "optuna", "pyarrow"}
    assert env["code_sha256"] == {rel: sha_file(os.path.join(APP, rel)) for rel in (
        "wsp/directions.py", "wsp/directions_ara.py", "wsp/wscript.py", "wsp/delta.py", "wsp/workspace.py")}
    assert env["runner"]["class"] == type(ws._runner).__name__
    # Nothing of the machine beyond the platform values: no absolute path, no host name.
    text = open(os.path.join(records(ws), f"{pid}.json")).read()
    for value in (str(tmp_path), APP, os.path.expanduser("~"), platform.node()):
        assert not value or value not in text, value


# ------------------------------------------------------------------ RC2
def test_dry_runs_refusals_and_plain_scripts_write_nothing(tmp_path):
    sess, ws = session(tmp_path)
    DR.measure(ws, "m", GOOD, BAD)
    script = DR.script_for(ablation_params())
    r = sess.apply(script, dry_run=True)
    assert r["ok"] and not r["committed"] and "record" not in r and "record_error" not in r
    sess.set_task({"scope": ["blocks.0.mlp.neuron.5"]})
    r = sess.apply(script, enforce_scope=True)
    assert r["ok"] is False and "outside the task scope" in r["error"] and ws.patches == []
    sess.set_task({"scope": ["model"]})
    r = sess.apply("mlp(1).W_out *= 0.5")
    assert r["committed"] and "record" not in r
    assert not os.path.exists(records(ws))


# ------------------------------------------------------------------ RC3
def test_an_ara_commit_records_its_seed_and_settings(tmp_path):
    sess, ws = session(tmp_path)
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    params = {"method": "ara", "direction": "m", "start_layer_index": 0, "end_layer_index": None, "seed": 5, **ARA_SMALL}
    r = sess.apply(DR.script_for(params))
    assert r["committed"], r
    rec = record(ws, r["patch"])
    (c,) = rec["calls"]
    assert c["function"] == "ablate_ara" and c["method"] == "ara" and c["seed"] == 5
    from wsp import directions_ara as ARA
    assert c["parameters"] == ARA.normalize(params)
    assert c["heretic"]["modifier"] == "ARA"
    assert c["heretic"]["settings"] == {"preserve_row_magnitudes": True, "lora_rank": 2, "n_optimization_steps": 1,
                                        "learning_rate": 1.0, "max_iter": 2, "history_size": 10}
    assert c["heretic"]["parameters"] == {"start_layer_index": 0, "end_layer_index": N,
                                          "preserve_good_behavior_weight": 1.0, "steer_bad_behavior_weight": 0.01,
                                          "overcorrect_relative_weight": 0.0, "neighbor_count": 1}
    assert c["parameters"]["end_layer_index"] is None                # the recorded call keeps what was asked
    io_json = os.path.join(ws.dir, "directions", "m.io", "io.json")
    assert c["measurement"]["io_sha256"] == sha_file(io_json) and rec["rebuildable"] is True
    assert sorted(os.listdir(records(ws))) == sorted(["base.json", f"{r['patch']}.json", "means", "prompts"])  # no io


# ------------------------------------------------------------------ RC4
def test_the_record_outlives_its_measurement(tmp_path):
    sess, ws = session(tmp_path)
    DR.measure(ws, "m", GOOD, BAD)
    r = sess.apply(DR.script_for(ablation_params()))
    pid = r["patch"]
    path = os.path.join(records(ws), f"{pid}.json")
    saved = open(path, "rb").read()
    old_means = open(mpath(ws, "m", ".safetensors"), "rb").read()
    m = record(ws, pid)["calls"][0]["measurement"]
    ans = REC.answer(ws, pid)
    row = next(x for x in ans["checks"] if x["key"] == "measurement:m")
    assert row["same"] is True and row["message"] is None
    DR.measure(ws, "m", GOOD + ["dog"], BAD, replace=True)
    assert open(path, "rb").read() == saved
    assert json.load(open(os.path.join(records(ws), "prompts", m["prompts_snapshot"] + ".json"))) == {"good": GOOD, "bad": BAD}
    assert open(os.path.join(records(ws), "means", m["means_snapshot"] + ".safetensors"), "rb").read() == old_means
    ans = REC.answer(ws, pid)
    row = next(x for x in ans["checks"] if x["key"] == "measurement:m")
    assert row["same"] is False and row["effect"] == "refuse"
    assert row["then"] == {"meta_sha256": m["meta_sha256"], "means_sha256": m["means_sha256"], "io_sha256": None}
    assert row["now"] == {"meta_sha256": meta_sha(ws, "m"), "means_sha256": sha_file(mpath(ws, "m", ".safetensors")),
                          "io_sha256": None}
    assert row["message"] == (f"measurement 'm' changed since patch {pid} was made (measured again or deleted): measure "
                              f"it again from the record (GET /api/patches/{pid}/record?remeasure=true) and pass the new "
                              f"name as measurement")
    DR.delete(ws, "m")
    row = next(x for x in REC.answer(ws, pid)["checks"] if x["key"] == "measurement:m")
    assert row["same"] is False and row["now"] is None


# ------------------------------------------------------------------ RC6
def test_the_base_hash_is_cached(tmp_path, monkeypatch):
    sess, ws = session(tmp_path)
    DR.measure(ws, "m", GOOD, BAD)
    base = base_path(ws)
    seen = []
    real = REC.sha256_file
    monkeypatch.setattr(REC, "sha256_file", lambda p: (seen.append(p), real(p))[1])
    script = DR.script_for(ablation_params())
    count = lambda: seen.count(base)                                # noqa: E731
    r1 = sess.apply(script)
    assert count() == 1 and record(ws, r1["patch"])["base"]["files"][0]["sha256"] == sha_file(base)
    r2 = sess.apply(script)
    assert count() == 1 and record(ws, r2["patch"])["base"]["hash_seconds"] == 0
    assert record(ws, r2["patch"])["base"]["files"] == record(ws, r1["patch"])["base"]["files"]
    REC.answer(ws, r2["patch"], remeasure_too=True)
    assert count() == 1                                             # GET never hashes
    st = os.stat(base)
    os.utime(base, ns=(st.st_atime_ns, st.st_mtime_ns + 1000))
    ans = REC.answer(ws, r2["patch"])
    assert count() == 1
    row = ans["checks"][0]
    assert row["key"] == "base" and row["same"] is None
    assert row["message"] == "the base's content hash is not cached; a re-run hashes it"
    r3 = sess.apply(script)
    assert count() == 2
    files = record(ws, r3["patch"])["base"]["files"]
    assert files[0]["mtime_ns"] == st.st_mtime_ns + 1000 and files[0]["sha256"] == sha_file(base)


def test_a_base_above_the_limit_is_hashed_later(tmp_path, monkeypatch):
    sess, ws = session(tmp_path)
    DR.measure(ws, "m", GOOD, BAD)
    monkeypatch.setattr(REC, "HASH_AT_COMMIT_MAX_BYTES", 0)
    pid = sess.apply(DR.script_for(ablation_params()))["patch"]
    st = os.stat(base_path(ws))
    b = record(ws, pid)["base"]
    assert b["hashed"] is False and b["hash_seconds"] == 0
    assert b["files"] == [{"name": "base.safetensors", "bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": None}]
    assert not os.path.exists(os.path.join(records(ws), "base.json"))
    ans = REC.answer(ws, pid)
    assert ans["record"]["base"]["files"][0]["sha256"] is None and ans["record"]["base"]["files"][0]["sha256_from"] is None
    assert ans["checks"][0]["same"] is None
    # The re-run's base check (hash=True) hashes it now: passes with the note, and the cache is written.
    row = REC._base_row(ws, record(ws, pid)["base"], pid, hash=True)
    gib = st.st_size / 2 ** 30
    assert row["same"] is True and row["message"] == (
        f"the base was not hashed when the edit was committed ({gib:.2f} GiB); its files have the same size and "
        f"modification time, and their hash now is recorded")
    ans = REC.answer(ws, pid)
    f = ans["record"]["base"]["files"][0]
    assert (f["sha256"], f["sha256_from"]) == (sha_file(base_path(ws)), "later, file unchanged since")
    assert ans["checks"][0]["same"] is True and ans["checks"][0]["message"] is None
    assert record(ws, pid)["base"]["files"][0]["sha256"] is None        # the record itself is never rewritten
    os.utime(base_path(ws), ns=(st.st_atime_ns, st.st_mtime_ns + 1000))
    ans = REC.answer(ws, pid)
    f = ans["record"]["base"]["files"][0]
    assert (f["sha256"], f["sha256_from"]) == (None, None)
    want = ("the base was not hashed when the edit was committed and its file base.safetensors has changed since (size or "
            "modification time): it cannot be shown to be the same")
    assert ans["checks"][0]["same"] is False and ans["checks"][0]["message"] == want
    assert REC._base_row(ws, record(ws, pid)["base"], pid, hash=True)["message"] == want


def test_a_hashed_base_that_changed_is_refused(tmp_path):
    sess, ws = session(tmp_path)
    DR.measure(ws, "m", GOOD, BAD)
    pid = sess.apply(DR.script_for(ablation_params()))["patch"]
    cache = os.path.join(records(ws), "base.json")
    rows = json.load(open(cache))
    rows["files"][0][3] = "ab" * 32                                    # the cache now says another content
    json.dump(rows, open(cache, "w"))
    row = REC.answer(ws, pid)["checks"][0]
    assert row["same"] is False and row["message"] == (
        f"the base weights are not those patch {pid} was made on (base.safetensors differs)")


# ------------------------------------------------------------------ RC7
def test_direction_reads_are_recorded(tmp_path):
    sess, ws = session(tmp_path)
    DR.measure(ws, "m", GOOD, BAD)
    r = sess.apply('v = direction("m", 1)\nmlp(1).W_out *= 0.5')
    rec = record(ws, r["patch"])
    (c,) = rec["calls"]
    assert (c["function"], c["method"], c["parameters"], c["result"]) == ("direction", None, {"name": "m", "k": 1}, None)
    assert c["heretic"] is None and c["seed"] is None and c["measurement"]["meta_sha256"] == meta_sha(ws, "m")
    assert rec["rebuildable"] is False
    r = sess.apply('g, b = direction_means("m")\n' + DR.script_for(ablation_params()))
    rec = record(ws, r["patch"])
    assert [c["function"] for c in rec["calls"]] == ["direction_means", "ablate_direction"]
    assert rec["rebuildable"] is False
    # The same ablation script with a trailing newline is still rebuildable (the script is compared stripped).
    r = sess.apply(DR.script_for(ablation_params()) + "\n")
    assert record(ws, r["patch"])["rebuildable"] is True


# ------------------------------------------------------------------ RC8
def test_a_failed_record_keeps_the_commit(tmp_path, monkeypatch):
    sess, ws = session(tmp_path)
    DR.measure(ws, "m", GOOD, BAD)
    script = DR.script_for(ablation_params())

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(REC, "write", boom)
    r = sess.apply(script)
    assert r["ok"] and r["committed"] and r["record_error"] == "OSError: disk full" and "record" not in r
    assert not os.path.exists(os.path.join(records(ws), f"{r['patch']}.json"))
    assert [p.id for p in ws.patches] == [r["patch"]]
    monkeypatch.undo()
    monkeypatch.setattr(REC, "prepare", lambda *a, **k: 1 / 0)
    r = sess.apply(script)
    assert r["committed"] and r["record_error"] == "ZeroDivisionError: division by zero"
    assert not os.path.exists(os.path.join(records(ws), f"{r['patch']}.json")) and len(ws.patches) == 2


# ------------------------------------------------------------------ RC9
def test_a_reused_patch_id_is_stale(tmp_path, monkeypatch):
    sess, ws = session(tmp_path)
    DR.measure(ws, "m", GOOD, BAD)
    pid = sess.apply(DR.script_for(ablation_params()))["patch"]
    assert REC.answer(ws, pid)["stale"] is False and REC.answer(ws, pid)["record"]["patch"]["id"] == pid
    ws.delete_patch(pid)
    suffix = int(pid.split("-")[1])
    later = (int(time.time()) // 100000 + 1) * 100000 + suffix + 0.25    # the same id, another commit time
    monkeypatch.setattr(time, "time", lambda: later)
    r = sess.apply("mlp(1).W_out *= 0.5")
    monkeypatch.undo()
    assert r["patch"] == pid and ws.patch(pid).created == later
    ans = REC.answer(ws, pid)
    assert ans == {"record": None, "stale": True, "checks": None, "environment": None, "remeasure": None,
                   "remeasure_unavailable": None, "heretic": None}
    # the next direction edit given that id overwrites the stale record
    ws.delete_patch(pid)
    monkeypatch.setattr(time, "time", lambda: later + 100000)
    r = sess.apply(DR.script_for(ablation_params()))
    monkeypatch.undo()
    assert r["patch"] == pid and REC.answer(ws, pid)["stale"] is False
    assert record(ws, pid)["patch"]["created"] == later + 100000


# ------------------------------------------------------------------ RC11
def test_a_search_trial_commit_records_the_search(tmp_path):
    pytest.importorskip("optuna")
    from test_directions_search import make_search, read_json, sdir, session_of, tiny, trial_path
    ws = tiny(tmp_path)
    sess = session_of(tmp_path)
    ws = sess.ws
    make_search(ws, n_trials=2)
    assert DSR.run(ws, "s1")["state"] == "finished"
    r = DSR.apply_trial(sess, "s1", 1)
    assert r["committed"] and r["record"]["calls"] == 1, r
    rec = record(ws, r["patch"])
    s = read_json(os.path.join(sdir(ws, "s1"), "search.json"))
    t = read_json(trial_path(ws, "s1", 1))
    assert rec["origin"]["kind"] == "direction_search" and rec["origin"] == ws.patch(r["patch"]).source["origin"]
    se = rec["search"]
    assert (se["name"], se["id"], se["seed"], se["n_trials"], se["n_startup_trials"]) == (
        "s1", s["id"], s["seed"], s["n_trials"], s["n_startup_trials"])
    assert se["seed_from"] == s["seed_from"] and se["sampler"] == s["sampler"] and se["objectives"] == s["objectives"]
    req = s["scoring"]["request"]
    assert se["max_response_length"] == req["max_response_length"] == 3
    assert se["rendering"] == {k: req[k] for k in ("chat", "system", "prefix", "chat_options")}
    assert [x["prompts_sha256"] for x in se["scorers"]] == [canonical_sha(x["prompts"]) for x in req["scores"]]
    assert [(x["plugin"], x["optimization"]) for x in se["scorers"]] == [(x["plugin"], x["optimization"]) for x in req["scores"]]
    assert se["scorers"][0]["settings"] == {"keyword_markers": ["e", "q"]}
    assert "prompts" not in json.dumps(se["scorers"]).replace("prompts_sha256", "")     # no prompt texts
    assert se["finished_trials"] == 2
    assert se["trial"]["index"] == 1 and se["trial"]["values"] == t["values"]
    assert se["trial"]["displays"] == [x["display"] for x in t["scores"]]
    assert [(x["name"], x["value"], x["display"]) for x in se["trial"]["scores"]] == \
        [(x["name"], x["value"], x["display"]) for x in t["scores"]]
    assert [x["plugin"] for x in se["trial"]["scores"]] == ["KeywordRate", "KLDivergence"]
    assert rec["script"] == t["script"] and rec["rebuildable"] is True


# ------------------------------------------------------------------ RC10
def test_the_record_route(tmp_path):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    sess, ws = session(tmp_path)
    c = TestClient(create_app(sess, "http://127.0.0.1:0"), raise_server_exceptions=False)
    r = c.get("/api/patches/9999-00000/record")
    assert (r.status_code, r.json()) == (404, {"error": "no patch '9999-00000'"})
    plain = sess.apply("mlp(1).W_out *= 0.5")["patch"]
    r = c.get(f"/api/patches/{plain}/record")
    assert r.status_code == 200 and r.json() == {"record": None, "stale": False, "checks": None, "environment": None,
                                                 "remeasure": None, "remeasure_unavailable": None, "heretic": None}
    ws.set_patch(plain, enabled=False)
    DR.measure(ws, "m", GOOD, BAD, winsorization_quantile=0.95)
    pid = sess.apply(DR.script_for(ablation_params()))["patch"]
    rec = record(ws, pid)
    r = c.get(f"/api/patches/{pid}/record")
    out = r.json()
    assert r.status_code == 200 and set(out) == {"record", "stale", "checks", "environment", "remeasure",
                                                 "remeasure_unavailable", "heretic"}
    assert out["stale"] is False and out["remeasure"] is None
    # the Heretic file's rules without a query: typed prompts on a workspace that is not a transformers checkpoint
    assert out["heretic"] == REC.heretic_status(ws, rec, pid) and set(out["heretic"]) == {"runnable", "reasons",
                                                                                          "needs_model"}
    assert out["heretic"]["runnable"] is False and out["heretic"]["needs_model"] is False
    assert "the good prompts include inline texts; Heretic's settings name Hub datasets (a local path would put a " \
        "path of this machine into the file)" in out["heretic"]["reasons"]
    assert {k: v for k, v in out["record"].items() if k != "base"} == {k: v for k, v in rec.items() if k != "base"}
    assert out["record"]["base"]["files"][0]["sha256_from"] == "commit"
    assert [x["key"] for x in out["checks"]] == ["base", "made_on", "measurement:m"]
    assert all(x["effect"] == "refuse" for x in out["checks"])
    assert [x["same"] for x in out["checks"]] == [True, False, True]       # the patch itself is switched on
    made_on = out["checks"][1]
    assert made_on["then"] == [] and made_on["now"] == wiring.patch_fingerprint(ws)
    assert made_on["message"] == (f"the enabled patches are not those patch {pid} was made on (then: none; now: {pid}). "
                                  f"Switch patch {pid} off (and the others back as they were) to re-run it on the weights "
                                  f"it was made on")
    ws.set_patch(pid, enabled=False)
    assert c.get(f"/api/patches/{pid}/record").json()["checks"][1]["same"] is True
    rows = {x["name"]: x for x in out["environment"]}
    assert rows["torch"] == {"name": "torch", "then": torch.__version__, "now": torch.__version__, "severity": "high",
                             "same": True}
    assert rows["torch_threads"]["severity"] == "low" and rows["wsp/wscript.py"]["severity"] == "critical"
    assert rows["runner"]["severity"] == "medium" and rows["platform"]["severity"] == "low"
    assert all(x["same"] for x in out["environment"])
    # remeasure: the measure request that makes the measurement again from the record
    out = c.get(f"/api/patches/{pid}/record", params={"remeasure": "true"}).json()
    assert out["remeasure_unavailable"] == []
    (entry,) = out["remeasure"]
    body = entry["body"]
    assert entry["measurement"] == "m" and body["name"] == f"m-r{pid}" and DR.NAME.match(body["name"])
    assert body == {"name": f"m-r{pid}", "good": GOOD, "bad": BAD, "chat": False, "system": None, "prefix": None,
                    "chat_options": None, "variant": "base", "winsorization_quantile": 0.95, "module_io": False}
    again = DR.measure(ws, **DR.measure_body(body))
    old = json.load(open(mpath(ws, "m", ".json")))
    assert again["prompts_sha256"] == old["prompts_sha256"] and again["options"] == old["options"]
    assert open(mpath(ws, body["name"], ".safetensors"), "rb").read() == open(mpath(ws, "m", ".safetensors"), "rb").read()
    # a long measurement name is cut to a valid one
    assert len(REC.suggested_name("x" * 64, pid)) == 64 and DR.NAME.match(REC.suggested_name("x" * 64, pid))
    # the prompts snapshot gone: no body, and the reason
    snap = rec["calls"][0]["measurement"]["prompts_snapshot"]
    os.replace(os.path.join(records(ws), "prompts", snap + ".json"), os.path.join(str(tmp_path), "moved.json"))
    out = c.get(f"/api/patches/{pid}/record?remeasure=true").json()
    assert out["remeasure"] == [{"measurement": "m", "body": None}]
    assert out["remeasure_unavailable"] == [f"the prompts of measurement 'm' are not in the workspace (snapshot "
                                            f"{snap[:12]} is missing)"]
    # a script that is not the ablate calls alone cannot be rebuilt
    other = sess.apply('v = direction("m", 1)\nmlp(1).W_out *= 0.5')["patch"]
    out = c.get(f"/api/patches/{other}/record?remeasure=true").json()
    assert out["remeasure"] is None and out["remeasure_unavailable"] == [
        f"patch {other}'s script does more than ablate_direction and ablate_ara calls, so it cannot be rebuilt for "
        f"another measurement"]
    assert c.get(f"/api/patches/{other}/record").json()["remeasure"] is None
