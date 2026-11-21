"""Steering in /api/run and /api/chat: the request fields, the report, the variant pushed, and Stop.

No checkpoint is loaded. The routes run a tiny random model of this project (Config + init_params,
as tests/test_steering.py does) with its runner's ``generate`` recorded; expected tokens come from
``wsp.model.forward`` over a copy of the parameters where the push is folded into the MLP output
bias (adding c to neuron N of hook_post adds c * W_out[N] to the layer's output). Stop is checked on
the routes, on the GGUF and transformers runners' fake decoders (hooks removed), and through the
model worker on a socketpair (tests/test_steering_worker.py's harness), where a stopped steered
job leaves the worker usable.
"""
import os
import threading
import time

import pytest
import torch
from fastapi.testclient import TestClient

from wsp import generation_stop
from wsp import steering as S
from wsp.generation_stop import GenerationStopped, Stop
from wsp.model import forward
from wsp.server import create_app

from test_steering import independent_p50, make_ws
from test_steering_worker import (CFG, WorkerRunner, WspHarness, gguf_runner, greedy, hf_runner, hooks_on, loaded,
                                  local_ws, remote)

URL = "http://127.0.0.1:41883"
CHAT = {"messages": [{"role": "user", "content": "abc"}], "force_template": True,
        "template": {"user": "", "assistant": "", "sep": ""}}
ROUTES = [("/api/run", {"prompt": "abc"}), ("/api/chat", CHAT)]


@pytest.fixture()
def sess(tmp_path):
    from wsp.session import Session
    make_ws(tmp_path / "rnd")
    s = Session(str(tmp_path))
    s.open("rnd")
    return s


def client(sess):
    return TestClient(create_app(sess, URL), base_url=URL, raise_server_exceptions=False)


def record(ws, steer_kw=True):
    """Record every generate call. Without ``steer_kw`` the signature has no steer keyword at all,
    as the runners in the stop and runtime tests do not: a request without steer must still work."""
    real, calls = ws.runner.generate, []
    if steer_kw:
        def generate(ids, variant="current", extra=None, **kw):
            calls.append((variant, dict(kw)))
            return real(ids, variant, extra, **kw)
    else:
        def generate(ids, variant="current", extra=None, max_new_tokens=40, temperature=0.0, top_k=0, seed=None,
                     stop_ids=(), on_token=None):
            kw = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, seed=seed)
            if on_token is not None:
                kw["on_token"] = on_token
            calls.append((variant, kw))
            return real(ids, variant, extra, **kw)
    ws.runner.generate = generate
    return calls


def pushed_params(ws, layer, neuron, c, variant="current"):
    """The variant's parameters with ``c`` added to hook_post[..., neuron] of ``layer``, folded into b_out."""
    p = {k: v.clone() for k, v in ws.runner.params(variant).items()}
    p[f"blocks.{layer}.mlp.b_out"] += c * p[f"blocks.{layer}.mlp.W_out"][neuron]
    return p


def greedy_tokens(params, cfg, ids, n):
    ids = list(ids)
    for _ in range(n):
        logits = forward(params, cfg, torch.tensor(ids))
        ids.append(int(logits.reshape(-1, cfg.d_vocab)[-1].argmax()))
    return ids[-n:]


def last_logits(params, cfg, ids):
    return forward(params, cfg, torch.tensor(ids)).reshape(-1, cfg.d_vocab)[-1]


def text_of(route, data, variant):
    return data[variant]["text"] if route == "/api/run" else data[variant]


def status_of(route, data, variant):
    return data[variant]["status"] if route == "/api/run" else data["variants"][variant]


# ------------------------------------------------------------------ the request fields
@pytest.mark.parametrize("route,body", ROUTES)
def test_without_steer_the_runner_is_called_exactly_as_before(sess, route, body):
    calls = record(sess.ws, steer_kw=False)                 # generate has no steer keyword: any steer= would fail
    c = client(sess)
    for extra in ({}, {"steer": None}, {"steer": []}):
        calls.clear()
        r = c.post(route, json={**body, "max_new_tokens": 4, "temperature": 0, **extra})
        assert r.status_code == 200, r.text
        data = r.json()
        assert [v for v, _ in calls] == ["current", "base"] and all("steer" not in kw for _, kw in calls)
        assert "steering" not in data
        if route == "/api/run":
            assert "steered" not in data["current"] and "steered" not in data["base"]


@pytest.mark.parametrize("route,body", ROUTES)
def test_p50_and_recalibrate_without_steering_are_ignored(sess, route, body):
    """They only shape a push: without one (no steer, or an empty one) they are not checked and change nothing."""
    calls = record(sess.ws, steer_kw=False)
    c = client(sess)
    for extra in ({"p50": {"1": 3.0}}, {"p50": "not a map", "recalibrate": "yes"}, {"steer": [], "p50": {"1": -1}, "recalibrate": True}):
        calls.clear()
        r = c.post(route, json={**body, "max_new_tokens": 3, "temperature": 0, **extra})
        assert r.status_code == 200, (extra, r.text)
        assert [v for v, _ in calls] == ["current", "base"] and all("steer" not in kw for _, kw in calls)
        assert "steering" not in r.json()
    # with steering they are still checked
    r = c.post(route, json={**body, "max_new_tokens": 3, "steer": [{"addr": "L1N3", "mode": "add", "dose": 1}], "recalibrate": "yes"})
    assert r.status_code == 400 and "recalibrate" in r.text


@pytest.mark.parametrize("route,body", ROUTES)
def test_steer_reaches_the_current_variant_only_and_the_report_says_what_was_applied(sess, route, body):
    calls = record(sess.ws)
    c = client(sess)
    spec = [{"addr": "L1N3", "mode": "add", "dose": 0.75}, {"addr": "L0N2", "mode": "mul", "dose": 2}]
    r = c.post(route, json={**body, "max_new_tokens": 4, "temperature": 0, "steer": spec})
    assert r.status_code == 200, r.text
    data = r.json()
    assert [v for v, _ in calls] == ["current", "base"]
    (_, current), (_, base) = calls
    assert "steer" not in base
    wire = current["steer"]
    assert wire["version"] == 1 and set(wire["layers"]) == {"0", "1"}
    # The report: one forward pass of the base model measured layer 1; the constant is dose * p50 / ||column||.
    rep = data["steering"]
    assert rep["variant"] == "current" and rep["resolved"] and rep["calibration"]["ran"]
    assert rep["neurons"] == 2 and rep["layers"] == 2 and rep["heads"] == 0 and rep["mlps"] == 0
    p50 = independent_p50(sess.ws)["1"]
    norm = float(sess.ws.runner.params("current")["blocks.1.mlp.W_out"][3].norm())
    e0, e1 = rep["entries"]
    assert e0["addr"] == "L1N3" and e0["source"] == "measured" and e0["unit"] == "resid"
    assert e0["p50"]["1"] == pytest.approx(p50, rel=1e-5) and e0["norm"] == pytest.approx(norm, rel=1e-5)
    assert e0["value"] == pytest.approx(0.75 * p50 / norm, rel=1e-5)
    assert e1["addr"] == "L0N2" and e1["mode"] == "mul" and e1["value"] == 2
    assert wire["p50"]["1"] == pytest.approx(p50, rel=1e-5)
    if route == "/api/run":
        assert data["current"]["steered"] is True and "steered" not in data["base"]
    # Once measured, the next steered request uses the cache and runs no calibration.
    again = c.post(route, json={**body, "max_new_tokens": 4, "temperature": 0, "steer": spec}).json()
    assert again["steering"]["calibration"]["cached"]
    # A given p50 is used as is, and the report says so.
    given = c.post(route, json={**body, "max_new_tokens": 4, "steer": spec, "p50": {"1": 3.5}}).json()
    assert given["steering"]["entries"][0]["source"] == "given"
    assert given["steering"]["entries"][0]["value"] == pytest.approx(0.75 * 3.5 / norm, rel=1e-5)
    assert sess.ws.patches == []                             # steering writes no weight and records no patch


def test_a_steered_run_generates_what_the_pushed_model_generates(sess):
    """End to end on the real FunctionalRunner: tokens and the next-token panel against an independent push."""
    ws, c = sess.ws, client(sess)
    ids, n, dose = ws.encode("abc"), 8, 6.0
    spec = [{"addr": "L1N3", "mode": "add", "dose": dose, "unit": "raw"}]
    plain = greedy_tokens(ws.runner.params("current"), ws.cfg, ids, n)
    expect = greedy_tokens(pushed_params(ws, 1, 3, dose), ws.cfg, ids, n)
    assert expect != plain                                   # the push matters on this model
    data = c.post("/api/run", json={"prompt": "abc", "max_new_tokens": n, "temperature": 0, "steer": spec,
                                    "top_k_show": 5}).json()
    assert data["current"]["text"] == ws.tok.decode(expect)
    assert data["base"]["text"] == ws.tok.decode(greedy_tokens(ws.runner.params("base"), ws.cfg, ids, n))
    # The edited column's next-token bars are the steered distribution; the base column's are unsteered.
    p = last_logits(pushed_params(ws, 1, 3, dose), ws.cfg, ids).softmax(-1)
    top = p.topk(5)
    assert [row[0] for row in data["current"]["next"]] == [ws.tok.token_str(int(i)) for i in top.indices]
    assert [row[1] for row in data["current"]["next"]] == pytest.approx([float(x) for x in top.values], abs=1e-4)
    q = last_logits(ws.runner.params("base"), ws.cfg, ids).softmax(-1).topk(5)
    assert [row[0] for row in data["base"]["next"]] == [ws.tok.token_str(int(i)) for i in q.indices]
    # A resid dose gives the constant from the independent p50 and column norm.
    p50, norm = independent_p50(ws)["1"], float(ws.runner.params("current")["blocks.1.mlp.W_out"][3].norm())
    resid = c.post("/api/run", json={"prompt": "abc", "max_new_tokens": n, "temperature": 0,
                                     "steer": [{"addr": "L1N3", "mode": "add", "dose": 2.0}]}).json()
    assert resid["current"]["text"] == ws.tok.decode(greedy_tokens(pushed_params(ws, 1, 3, 2.0 * p50 / norm), ws.cfg, ids, n))


def test_a_steered_chat_reply_is_the_pushed_model_reply(sess):
    ws, c = sess.ws, client(sess)
    ids, n, dose = ws.encode("abc"), 6, 6.0
    data = c.post("/api/chat", json={**CHAT, "max_new_tokens": n, "temperature": 0,
                                     "steer": [{"addr": "L1N3", "mode": "add", "dose": dose, "unit": "raw"}]}).json()
    assert data["current"] == ws.tok.decode(greedy_tokens(pushed_params(ws, 1, 3, dose), ws.cfg, ids, n)).strip("\n")
    assert data["base"] == ws.tok.decode(greedy_tokens(ws.runner.params("base"), ws.cfg, ids, n)).strip("\n")
    e = data["steering"]["entries"][0]
    assert e["unit"] == "raw" and e["value"] == dose and "source" not in e     # a raw dose is the constant itself
    assert data["steering"]["calibration"] == {"needed": False}


@pytest.mark.parametrize("route,body", ROUTES)
def test_a_bad_steer_entry_is_refused_before_the_model_is_loaded(sess, route, body, monkeypatch):
    import wsp.runner
    built = []

    def refuse(w):
        built.append(w.name)
        raise AssertionError("the runner was built for a request that should have been refused first")
    monkeypatch.setattr(wsp.runner, "make_runner", refuse)
    assert sess.ws._runner is None
    c = client(sess)
    ok = [{"addr": "L1N3", "mode": "add", "dose": 0.75}]
    for extra, message in [
        ({"steer": [{"addr": "L0H1", "mode": "add", "dose": 1}]}, "steer entry 1 (L0H1): add pushes neurons only"),
        ({"steer": [ok[0], {"addr": "L9N1", "mode": "add", "dose": 1}]}, "steer entry 2 (L9N1)"),
        ({"steer": [{"addr": "L1N40", "mode": "mul", "dose": 2}]}, "does not exist"),
        ({"steer": [{"addr": "L1N3", "mode": "push", "dose": 2}]}, "steer entry 1"),
        ({"steer": [{"addr": "L1N3", "mode": "add", "dose": "big"}]}, "steer entry 1"),
        ({"steer": [{"addr": "blocks.0.mlp", "mode": "add", "dose": 1}]}, "add pushes neurons only"),
        ({"steer": "L1N3 +0.75"}, "steer"),
        ({"steer": ok, "p50": {"1": -1}}, "above 0"),
        ({"steer": ok, "recalibrate": "yes"}, "recalibrate must be"),
    ]:
        r = c.post(route, json={**body, **extra})
        assert r.status_code == 400, (extra, r.text)
        assert message in r.json()["error"], (extra, r.json()["error"])
    assert built == [] and sess.ws._runner is None


# ------------------------------------------------------------------ events and Stop
@pytest.mark.parametrize("route,body", ROUTES)
def test_steering_events_carry_the_request_stream_id(sess, route, body, monkeypatch):
    monkeypatch.setattr(S, "PROGRESS_AFTER", 0.0)
    monkeypatch.setattr(S, "PROGRESS_EVERY", 0.0)
    monkeypatch.setattr(S, "READ_GAP", 0)
    seen, publish = [], sess.publish
    sess.publish = lambda e: (seen.append(e), publish(e))[1]
    c = client(sess)
    spec = [{"addr": "L1N3", "mode": "add", "dose": 0.75}, {"addr": "L1N5", "mode": "add", "dose": -0.5}]
    assert c.post(route, json={**body, "max_new_tokens": 2, "steer": spec, "stream": "s-1"}).status_code == 200
    steering = [e for e in seen if e.get("type") == "steering"]
    states = [e["state"] for e in steering]
    assert states[:2] == ["calibrating", "calibrated"] and "reading" in states and states[-1] == "read"
    assert all(e["stream"] == "s-1" and e["workspace"] == "rnd" for e in steering)
    # Without a stream id the events still arrive, untagged.
    seen.clear()
    c.post(route, json={**body, "max_new_tokens": 2, "steer": spec, "recalibrate": True})
    assert [e for e in seen if e.get("type") == "steering"] and all("stream" not in e for e in seen if e.get("type") == "steering")


@pytest.mark.parametrize("route,body", ROUTES)
def test_a_stop_during_calibration_ends_the_request_before_any_generation(sess, route, body):
    calls = record(sess.ws)
    c = client(sess)
    stops, publish = c.app.state.generation_stops, sess.publish

    def stop_while_calibrating(e):
        publish(e)
        if e.get("type") == "steering" and e.get("state") == "calibrating":
            assert stops.stop("cal-1")
    sess.publish = stop_while_calibrating
    spec = [{"addr": "L1N3", "mode": "add", "dose": 0.75}]
    data = c.post(route, json={**body, "max_new_tokens": 6, "steer": spec, "stream": "cal-1"}).json()
    assert data["stopped"] is True and calls == []
    assert status_of(route, data, "current") == status_of(route, data, "base") == "not_run"
    assert data["steering"]["resolved"] is False and data["steering"]["calibration"]["ran"]
    # The measurement finished and was kept; the next request generates, steered, from the cache.
    sess.publish = publish
    again = c.post(route, json={**body, "max_new_tokens": 6, "steer": spec, "stream": "cal-2"}).json()
    assert "stopped" not in again and again["steering"]["calibration"]["cached"]
    assert [v for v, _ in calls] == ["current", "base"] and "steer" in calls[0][1] and "steer" not in calls[1][1]


@pytest.mark.parametrize("route,body", ROUTES)
def test_a_stop_during_column_reads_ends_the_request_before_any_generation(sess, route, body, monkeypatch):
    monkeypatch.setattr(S, "PROGRESS_AFTER", 0.0)
    monkeypatch.setattr(S, "PROGRESS_EVERY", 0.0)
    monkeypatch.setattr(S, "READ_GAP", 0)
    calls = record(sess.ws)
    c = client(sess)
    stops, publish = c.app.state.generation_stops, sess.publish

    def stop_while_reading(e):
        publish(e)
        if e.get("type") == "steering" and e.get("state") == "reading":
            stops.stop("read-1")
    sess.publish = stop_while_reading
    spec = [{"addr": "L1N3", "mode": "add", "dose": 0.75}, {"addr": "L1N5", "mode": "add", "dose": 0.5}]
    data = c.post(route, json={**body, "max_new_tokens": 6, "steer": spec, "p50": {"1": 3.0}, "stream": "read-1"}).json()
    assert data["stopped"] is True and calls == []
    assert status_of(route, data, "current") == status_of(route, data, "base") == "not_run"
    assert data["steering"]["resolved"] is False                # stopped during the reads: nothing was applied


@pytest.mark.parametrize("route,body", ROUTES)
def test_a_stop_during_a_steered_reply_keeps_its_steered_prefix_and_the_model_runs_unsteered_after(sess, route, body, monkeypatch):
    monkeypatch.setattr("wsp.token_stream.INTERVAL", 0)
    ws, c = sess.ws, client(sess)
    stops, publish = c.app.state.generation_stops, sess.publish
    ids, n, dose = ws.encode("abc"), 10, 6.0
    spec = [{"addr": "L1N3", "mode": "add", "dose": dose, "unit": "raw"}]

    def stop_on_the_second_token(e):
        publish(e)
        if e.get("type") == "generation" and e.get("variant") == "current" and e.get("tokens") == 2:
            stops.stop("st-1")
    sess.publish = stop_on_the_second_token
    data = c.post(route, json={**body, "max_new_tokens": n, "temperature": 0, "steer": spec, "stream": "st-1"}).json()
    assert data["stopped"] is True
    assert status_of(route, data, "current") == "stopped" and status_of(route, data, "base") == "not_run"
    steered = ws.tok.decode(greedy_tokens(pushed_params(ws, 1, 3, dose), ws.cfg, ids, n))
    got = text_of(route, data, "current")
    assert got and len(got) < len(steered) and steered.startswith(got)
    # Nothing stays applied: the next request without steer is the plain model, in full.
    sess.publish = publish
    plain = c.post(route, json={**body, "max_new_tokens": n, "temperature": 0, "stream": "st-2"}).json()
    assert "stopped" not in plain and "steering" not in plain
    expect = ws.tok.decode(greedy_tokens(ws.runner.params("current"), ws.cfg, ids, n))
    assert text_of(route, plain, "current") == (expect if route == "/api/run" else expect.strip("\n"))


# ------------------------------------------------------------------ hooks removed on every exit path
def test_gguf_steering_hooks_are_removed_when_the_reply_is_stopped(tmp_path, monkeypatch):
    r, model, G = gguf_runner(tmp_path, monkeypatch)
    wire, _ = S.prepare(r.ws, [{"addr": "L1N4", "mode": "add", "dose": 3.0, "unit": "raw"}, {"addr": "L0H0", "mode": "mul", "dose": 2}])
    full = r.generate([1, 2, 3], max_new_tokens=8, steer=wire)
    streamed = []

    def on_token(t):
        streamed.append(t)
        if len(streamed) == 2:
            raise GenerationStopped()
    with pytest.raises(GenerationStopped):
        r.generate([1, 2, 3], max_new_tokens=8, steer=wire, on_token=on_token)
    assert streamed == full[:2]
    assert all(hooks_on(m) == 0 for m in model.modules())
    assert r.generate([1, 2, 3], max_new_tokens=8, steer=wire) == full
    plain = r.generate([1, 2, 3], max_new_tokens=8)
    assert plain != full and all(hooks_on(m) == 0 for m in model.modules())


def test_transformers_steering_hooks_are_removed_when_the_reply_is_stopped(tmp_path):
    r, model, tensors = hf_runner(tmp_path)
    spec = [{"addr": "L1N4", "mode": "add", "dose": 4.0, "unit": "raw"}]
    prompt, n = [1, 5, 2, 7], 8
    full = r.generate(prompt, "current", max_new_tokens=n, steer=spec)
    streamed = []

    def on_token(t):
        streamed.append(t)
        if len(streamed) == 2:
            raise GenerationStopped()
    got = r.generate(prompt, "current", max_new_tokens=n, steer=spec, on_token=on_token)
    assert got == full[:2] and streamed == full[:2]
    assert all(hooks_on(m) == 0 for m in model.modules())
    assert r.generate(prompt, "current", max_new_tokens=n) == greedy(loaded(tensors), prompt, n)


# ------------------------------------------------------------------ through the model worker
class SlowHarness(WspHarness):
    """The worker's runner waits a little after each token, so a stop lands mid-reply."""

    def _runner(self, ws):
        runner = super()._runner(ws)
        real = runner.generate

        def generate(ids, variant="current", extra=None, on_token=None, **kw):
            def slow(t):
                on_token(t)
                time.sleep(0.05)
            return real(ids, variant, extra, on_token=slow if on_token else None, **kw)
        runner.generate = generate
        return runner


class Live:
    """A live-text callback with its Stop, as RemoteRunner reads it (token_stream.TextStream)."""

    def __init__(self, after):
        self.stop, self.ids, self.after = Stop(), [], after

    def __call__(self, token):
        self.ids.append(token)
        if len(self.ids) >= self.after:
            self.stop.set()
            raise GenerationStopped()


def test_a_stopped_steered_worker_job_ends_early_and_the_worker_stays_usable(tmp_path, monkeypatch):
    h = SlowHarness(tmp_path)
    h.load()
    here = local_ws(tmp_path)
    r, sent, _ = remote(h, monkeypatch)
    h.ws._runtime_client.directory = h.runtime              # where the worker looks for stop.json
    ids, n = here.encode("abc"), 12
    spec = [{"addr": "L1N3", "mode": "add", "dose": 6.0, "unit": "raw"}, {"addr": "L0H1", "mode": "mul", "dose": 0}]
    wire, _ = S.prepare(here, spec)
    expect = here.runner.generate(ids, max_new_tokens=n, steer=wire)
    live = Live(after=2)
    got = r.generate(ids, max_new_tokens=n, steer=spec, on_token=live)
    assert sent[-1][0] == "generate" and sent[-1][1]["steer"]["layers"] == wire["layers"]
    assert 2 <= len(got) < n and got == expect[:len(got)]
    assert not (h.runtime / generation_stop.FILE).exists()  # the request was cleared once answered
    # The same worker answers in full: steered, then plain, with nothing left over from the stopped job.
    assert r.generate(ids, max_new_tokens=n, steer=spec) == expect
    assert r.generate(ids, max_new_tokens=n) == here.runner.generate(ids, max_new_tokens=n)
    assert len(h.runners) == 1                               # no reload
    h.ws._runtime_client = None
    h.shutdown()
