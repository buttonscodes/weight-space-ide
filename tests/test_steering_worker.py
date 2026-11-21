"""Steering and probes across the model-worker boundary, and on the GGUF runner's cached decode.

Part 1 runs `serve_worker` on a socketpair (as tests/test_worker_protocol.py does) over a tiny
random model of this project (Config + init_params): the worker's runner is a FunctionalRunner,
the server side is a real Workspace and a RemoteRunner on the same channel.

Part 2 drives `GGUFRunner.generate` / `next_logits`, and `HFRunner.generate` (its fake
`model.generate` decodes token by token), over a small fake decoder module with a KV cache,
written here (no transformers, no ggml, no checkpoint is run): the steering hooks are real
forward pre-hooks on its `down_proj` / `o_proj` modules.

Part 3 reads output-column norms from synthetic quantized bytes (the gguf package's own
dequantize; no GGUF file).
"""
import json
import math
import os
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from wsp import delta as Dl
from wsp import probes as P
from wsp import steering as S
from wsp import worker_jobs
from wsp.remote_runner import ERRORS, RemoteRunner
from wsp.runner import FunctionalRunner, HFRunner
from wsp.runtime_events import RuntimeRefusal, event
from wsp.worker_jobs import JOBS, encode_job, strict_json
from wsp.tokenizer import VocabTokenizer
from wsp.workspace import Ref, Workspace

from test_steering import CFG, make_ws
from test_worker_protocol import Harness


# ------------------------------------------------------------------ part 1: the worker boundary
class WorkerRunner(FunctionalRunner):
    """This project's runner plus GGUFRunner's ``next_logits`` signature, so next-token jobs run."""

    def next_logits(self, ids, variant="current", extra=None, steer=None):
        hooks = S.hooks_for(self.ws, steer, variant, extra) if steer is not None else None
        return self.run(list(ids), variant, extra, hooks=hooks or None)[0][0, -1]


class WspHarness(Harness):
    def __init__(self, tmp_path):
        super().__init__(tmp_path, make_runner=self._runner)
        make_ws(self.dir)                                   # config, base weights and tokenizer in the worker's directory
        self.ws = self.open_workspace(str(self.dir))        # the server's view

    def _runner(self, ws):
        runner = WorkerRunner(ws)
        self.runners.append(runner)
        return runner

    def open_workspace(self, directory):
        ws = Workspace.load(directory)
        ws.store.path = str(self.source)                    # the identity the load job checks
        ws.meta["base_sha256"] = "fixture"
        return ws


class FakeClient:
    """The server-side runtime client surface RemoteRunner and job_scope use."""

    def __init__(self, binding):
        self.binding, self.operation_id, self.last_event, self.names, self.ops = binding, None, None, [], []

    @contextmanager
    def operation(self, ws, name, **kwargs):
        outer = self.operation_id is None
        if outer:
            self.operation_id = "op"
        self.names.append(name)
        self.ops.append((name, kwargs))
        try:
            yield
        finally:
            if outer:
                self.operation_id = None


def remote(h, monkeypatch):
    """A RemoteRunner on the harness channel, with the server workspace in worker mode."""
    r = object.__new__(RemoteRunner)
    r.ws, r.lock, r.channel = h.ws, threading.RLock(), h.channel
    r.lost = r.lost_in = r.pid = None
    r.max_ctx, r.hooks, r._tok = CFG["n_ctx"], frozenset(), None
    h.ws._runtime_client = FakeClient(h.lease.binding)
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
        received.append((frame.type, dict(frame.header), {n: tuple(t.shape) for n, t in frame.tensors.items()}))
        return frame
    monkeypatch.setattr(worker_jobs, "encode_job", encode)
    monkeypatch.setattr(h.channel, "recv", recv)
    return r, sent, received


def local_ws(tmp_path):
    """An in-process copy of the same model, for the expected values."""
    return make_ws(tmp_path / "local")


SPEC = [{"addr": "L1N3", "mode": "add", "dose": 2.5}, {"addr": "L0N2", "mode": "mul", "dose": 3},
        {"addr": "blocks.1.mlp", "mode": "mul", "dose": 0.5}, {"addr": "L0H1", "mode": "mul", "dose": 0}]


def test_calibration_and_probe_jobs_run_in_the_worker_and_return_numbers_only(tmp_path):
    h = WspHarness(tmp_path)
    h.load()
    here = local_ws(tmp_path)
    ref = S.reference_ids(here)
    frame, _, _ = h.call("steer_calibrate", {"ids": ref})
    assert frame.type == "result" and not frame.tensors, frame.header
    assert frame["value"] == strict_json(S.calibrate_job(here, ref))
    assert not os.path.exists(os.path.join(h.dir, S.CALIBRATION_FILE))          # the worker writes nothing
    wire, _ = S.prepare(here, SPEC)
    ids = here.encode("abcabd")
    cases = [
        ("probe_reads", {"ids": ids, "neurons": [[1, 3], [0, 2]], "variants": ["base", "current"], "steer": wire},
         lambda: P.reads(here, ids, [[1, 3], [0, 2]], ["base", "current"], wire)),
        ("probe_fires", {"ids": ids, "position": -1, "k": 7, "sign": "both", "variant": "current"},
         lambda: P.fires(here, ids, -1, 7, "both", "current")),
        ("probe_contrast", {"a": [ids, ids[:3]], "b": [here.encode("zz")], "k": 5, "variant": "current"},
         lambda: P.contrast(here, [ids, ids[:3]], [here.encode("zz")], 5, "current")),
        ("probe_next", {"ids": ids, "variant": "current", "steer": wire, "top": 5},
         lambda: P.next_token(here, ids, "current", wire, 5)),
    ]
    for name, args, expected in cases:
        frame, _, _ = h.call(name, args)
        assert frame.type == "result" and frame.tensors == {}, (name, frame.header)
        assert frame["value"] == strict_json(expected()), name
        assert len(json.dumps(frame["value"])) < 8000, name                     # reduced next to the model
    assert h.call("probe_next", {"ids": ids, "variant": "current", "steer": wire, "top": 5})[0]["value"]["kl"] > 0
    # A C3 list is never prepared inside the worker (calibration belongs to the server).
    frame, _, _ = h.call("probe_next", {"ids": ids, "variant": "current", "steer": SPEC, "top": 5})
    assert frame.type == "error" and frame["exception"] == "SteeringError" and "prepared by the server" in frame["message"]
    frame, _, _ = h.call("probe_reads", {"ids": ids, "neurons": [[0, 99]], "variants": ["current"], "steer": None})
    assert frame.type == "error" and "neuron 99 does not exist" in frame["message"]
    frame, _, _ = h.call("generate", {"ids": ids, "variant": "current", "max_new_tokens": 2, "temperature": 0.0, "top_k": 0,
                                      "seed": None, "stop_ids": [], "steer": {"version": 1, "layers": {"1": {"add_resid": [[3], [1.0]]}}}})
    assert frame.type == "error" and "resid push but no p50" in frame["message"]
    h.shutdown()


def test_generate_carries_the_wire_form_and_streams_steered_tokens(tmp_path):
    h = WspHarness(tmp_path)
    h.load()
    here = local_ws(tmp_path)
    wire, _ = S.prepare(here, SPEC)
    ids = here.encode("abc")
    args = {"ids": ids, "variant": "current", "max_new_tokens": 8, "temperature": 0.0, "top_k": 0, "seed": None, "stop_ids": []}
    plain, plain_tokens, _ = h.call("generate", args)
    steered, steered_tokens, _ = h.call("generate", {**args, "steer": wire})
    assert plain["value"] == plain_tokens == here.runner.generate(ids, max_new_tokens=8)
    assert steered["value"] == steered_tokens == here.runner.generate(ids, max_new_tokens=8, steer=wire)
    assert steered["value"] != plain["value"]
    frame, _, _ = h.call("next_logits", {"ids": ids, "variant": "current", "steer": wire})
    assert torch.equal(frame.tensors["value"], WorkerRunner(here).next_logits(ids, steer=wire))
    h.shutdown()


def test_remote_runner_prepares_on_the_server_and_ships_only_the_wire_form(tmp_path, monkeypatch):
    h = WspHarness(tmp_path)
    h.load()
    here = local_ws(tmp_path)
    r, sent, received = remote(h, monkeypatch)
    ids = here.encode("abc")
    expected_wire, _ = S.prepare(here, SPEC)
    out = r.generate(ids, max_new_tokens=6, steer=SPEC)
    assert out == here.runner.generate(ids, max_new_tokens=6, steer=SPEC)
    # Calibration ran once in the worker; the server wrote the cache; generate carried the resolved wire form.
    assert [name for name, _ in sent] == ["steer_calibrate", "generate"]
    shipped = sent[1][1]["steer"]
    assert shipped["version"] == 1 and shipped["layers"] == expected_wire["layers"]
    assert shipped["p50"] == pytest.approx(expected_wire["p50"])
    assert os.path.exists(os.path.join(h.dir, S.CALIBRATION_FILE))
    sent.clear()
    assert r.generate(ids, max_new_tokens=6, steer=SPEC) == out and [n for n, _ in sent] == ["generate"]   # cached
    # Without steer the job arguments are exactly as before.
    sent.clear()
    r.generate(ids, max_new_tokens=2)
    assert set(sent[0][1]) == set(JOBS["generate"].args) - {"steer"}
    sent.clear()
    assert torch.equal(r.next_logits(ids, steer=SPEC), WorkerRunner(here).next_logits(ids, steer=expected_wire))
    assert sent[0][0] == "next_logits" and sent[0][1]["steer"]["layers"] == expected_wire["layers"]
    # Whole probes forward as named jobs; every result frame is JSON without tensors.
    sent.clear()
    received.clear()
    wire, _ = S.prepare(h.ws, SPEC)
    assert P.reads(h.ws, ids, [[1, 3]], ["base", "current"], wire) == strict_json(P.reads(here, ids, [[1, 3]], ["base", "current"], wire))
    assert P.fires(h.ws, ids, k=4) == strict_json(P.fires(here, ids, k=4))
    assert P.contrast(h.ws, [ids], [here.encode("zz")], 3) == strict_json(P.contrast(here, [ids], [here.encode("zz")], 3))
    assert P.next_token(h.ws, ids, "current", wire, 4) == strict_json(P.next_token(here, ids, "current", wire, 4))
    assert [n for n, _ in sent] == ["probe_reads", "probe_fires", "probe_contrast", "probe_next"]
    results = [(kind, tensors) for kind, header, tensors in received if kind == "result"]
    assert len(results) == 4 and all(tensors == {} for _, tensors in results)
    for name, arguments in sent:
        encode_job(name, arguments)                         # every forwarded argument is one the worker row accepts
    # Each job opens its runtime operation with the tokens it runs (the longest text), not 0.
    client = h.ws._runtime_client
    P.next_token(h.ws, ids, "current", None, 4)
    P.contrast(h.ws, [ids[:2]], [here.encode("zzzzz")], 3)
    scoped = [(n, kw["tokens"], kw["retained"]) for n, kw in client.ops       # the outer job_scope of each public job
              if n in ("steer_calibrate",) + tuple(n for n in JOBS if n.startswith("probe_")) and "extra_stored" not in kw and kw]
    assert scoped == [("steer_calibrate", len(S.reference_ids(here)), True),
                      ("probe_reads", 3, True), ("probe_fires", 3, True), ("probe_contrast", 3, True), ("probe_next", 3, True),
                      ("probe_next", 3, False), ("probe_contrast", 5, True)]
    # A refusal in the worker keeps its type across the boundary (so the server answers 400 with its own words).
    assert ERRORS["SteeringError"] is S.SteeringError and ERRORS["ProbeError"] is P.ProbeError
    with pytest.raises(P.ProbeError, match="k must be an integer from 1 to 200"):
        P.fires(h.ws, ids, k=0)
    with pytest.raises(S.SteeringError, match="resid push but no p50"):
        P.next_token(h.ws, ids, "current", {"version": 1, "layers": {"1": {"add_resid": [[3], [1.0]]}}}, 4)
    # Arbitrary hooks still cannot cross.
    with pytest.raises(RuntimeError, match="Forward hooks cannot cross the model-worker boundary"):
        r.run(ids, hooks=S.hooks_for(here, wire))
    h.ws._runtime_client = None
    h.shutdown()


def test_the_new_jobs_are_rows_of_the_job_table():
    assert {"steer_calibrate", "probe_reads", "probe_fires", "probe_contrast", "probe_next"} <= set(JOBS)
    assert "steer" in JOBS["generate"].args and "steer" in JOBS["next_logits"].args
    assert all(JOBS[n].result == "json" and not JOBS[n].stream and not JOBS[n].extra
               for n in ("steer_calibrate", "probe_reads", "probe_fires", "probe_contrast", "probe_next"))


# ------------------------------------------------------------------ part 2: the GGUF runner's cached decode
V, D, F, H, LAYERS = 12, 8, 10, 2, 2


def checkpoint(directory):
    """A tiny llama-named safetensors checkpoint and config, so arch.detect gives the usual module map."""
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(3)
    w = lambda *shape: torch.randn(*shape, generator=g) * 0.5  # noqa: E731
    t = {"model.embed_tokens.weight": w(V, D), "model.norm.weight": torch.ones(D), "lm_head.weight": w(V, D)}
    for l in range(LAYERS):
        p = f"model.layers.{l}"
        t.update({f"{p}.input_layernorm.weight": torch.ones(D), f"{p}.post_attention_layernorm.weight": torch.ones(D),
                  f"{p}.self_attn.q_proj.weight": w(D, D), f"{p}.self_attn.k_proj.weight": w(D, D),
                  f"{p}.self_attn.v_proj.weight": w(D, D), f"{p}.self_attn.o_proj.weight": w(D, D),
                  f"{p}.mlp.gate_proj.weight": w(F, D), f"{p}.mlp.up_proj.weight": w(F, D), f"{p}.mlp.down_proj.weight": w(D, F)})
    os.makedirs(directory, exist_ok=True)
    save_file(t, os.path.join(directory, "model.safetensors"))
    json.dump({"model_type": "llama", "architectures": ["LlamaForCausalLM"], "vocab_size": V, "hidden_size": D,
               "intermediate_size": F, "num_hidden_layers": LAYERS, "num_attention_heads": H, "num_key_value_heads": H,
               "max_position_embeddings": 64, "rms_norm_eps": 1e-6}, open(os.path.join(directory, "config.json"), "w"))
    return t


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(D, D, bias=False) for _ in range(4))

    def forward(self, x, past):
        b, t, _ = x.shape
        q, k, v = (m(x).view(b, t, H, D // H).transpose(1, 2) for m in (self.q_proj, self.k_proj, self.v_proj))
        if past is not None:
            k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
        n = k.shape[2]
        mask = torch.ones(t, n, dtype=torch.bool).tril(n - t)
        a = (q @ k.transpose(-1, -2) / math.sqrt(D // H)).masked_fill(~mask, float("-inf")).softmax(-1)
        z = (a @ v).transpose(1, 2).reshape(b, t, D)
        return self.o_proj(z), (k, v)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj, self.up_proj = nn.Linear(D, F, bias=False), nn.Linear(D, F, bias=False)
        self.down_proj = nn.Linear(F, D, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class RMS(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(D))

    def forward(self, x):
        return x / (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt() * self.weight


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = RMS(), RMS()
        self.self_attn, self.mlp = Attn(), MLP()

    def forward(self, x, past):
        a, kv = self.self_attn(self.input_layernorm(x), past)
        x = x + a
        return x + self.mlp(self.post_attention_layernorm(x)), kv


class Decoder(nn.Module):
    """Llama-named modules, a per-layer KV cache, ``logits_to_keep``; positions only through causality."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(V, D)
        self.model.layers = nn.ModuleList(Layer() for _ in range(LAYERS))
        self.model.norm = RMS()
        self.lm_head = nn.Linear(D, V, bias=False)
        self.calls = []

    def forward(self, input_ids, past_key_values=None, use_cache=True, logits_to_keep=0, **options):
        self.calls.append(int(input_ids.shape[1]))
        x = self.model.embed_tokens(input_ids)
        past = []
        for l, layer in enumerate(self.model.layers):
            x, kv = layer(x, None if past_key_values is None else past_key_values[l])
            past.append(kv)
        logits = self.lm_head(self.model.norm(x))
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits, past_key_values=past)


def gguf_runner(tmp_path, monkeypatch):
    from wsp import gguf_runner as G
    tensors = checkpoint(str(tmp_path / "ckpt"))
    ws = Workspace.create_hf(str(tmp_path / "ws"), "tiny", str(tmp_path / "ckpt"))
    model = Decoder()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(tensors[name])
    r = object.__new__(G.GGUFRunner)
    r.ws, r.lock, r.max_ctx, r.device, r.model = ws, threading.RLock(), 64, torch.device("cpu"), model.eval()
    r._modules, r._named, r._saved, r._gain, r._pcache = dict(model.named_modules()), {}, [], {}, {}
    r._variant, r._extra = "current", None
    r._packed = []                                    # GGUFRunner.__init__ sets it; PackedLinear registers here
    monkeypatch.setattr(G, "require_lease", lambda ws, **kw: None)
    return r, model, G


def full(model, ids):
    with torch.no_grad():
        return model(torch.tensor([ids]), None, False, 0).logits[0]


def hooks_on(module):
    return len(module._forward_pre_hooks)


def test_gguf_cached_decode_pushes_every_position_once(tmp_path, monkeypatch):
    r, model, G = gguf_runner(tmp_path, monkeypatch)
    monkeypatch.setattr(G, "PREFILL_CHUNK", 3)                     # a prompt of 7 prefills in 3 chunks
    ws, down = r.ws, model.model.layers[1].mlp.down_proj
    assert ws.struct.layers[1].modules["mlp_out"] == "model.layers.1.mlp.down_proj"
    spec = [{"addr": "L1N4", "mode": "add", "dose": 3.0, "unit": "raw"}, {"addr": "L1N7", "mode": "add", "dose": 0.5},
            {"addr": "L0H1", "mode": "mul", "dose": 0}, {"addr": "blocks.0.mlp", "mode": "mul", "dose": 0.5}]
    wire, info = S.prepare(ws, [e for e in spec if e.get("unit") == "raw" or e["mode"] == "mul"])   # no calibration run here
    c = S.resolve(ws, wire)[1].c
    seen = []                                                       # (input before, input after) the steering pre-hook
    before = down.register_forward_pre_hook(lambda m, args: seen.append([args[0].clone()]))
    after = down.register_forward_hook(lambda m, args, out: seen[-1].append(args[0].clone()))
    prompt = [1, 5, 2, 7, 3, 3, 9]
    baseline = hooks_on(down)
    new = r.generate(prompt, max_new_tokens=6, steer=wire)
    assert hooks_on(down) == baseline and all(hooks_on(m) == 0 for m in (model.model.layers[0].self_attn.o_proj,
                                                                        model.model.layers[0].mlp.down_proj))
    # Every position was pushed once: 3 prefill chunks, then one call per decoded token but the last.
    assert model.calls == [3, 3, 1] + [1] * (len(new) - 1)
    assert sum(x.shape[1] for x, _ in seen) == len(prompt) + len(new) - 1
    for x, y in seen:
        assert torch.equal(y, x + c)
    # The cached decode equals a full recompute of the same steered model.
    with r._steering(wire, "current", None):
        logits = full(model, prompt + new)
    assert [int(logits[len(prompt) - 1 + i].argmax()) for i in range(len(new))] == new
    assert hooks_on(down) == baseline
    plain = r.generate(prompt, max_new_tokens=6)
    assert plain != new and [int(full(model, prompt + plain)[len(prompt) - 1 + i].argmax()) for i in range(len(plain))] == plain
    before.remove()
    after.remove()


def test_gguf_next_logits_with_steering_and_the_resid_unit(tmp_path, monkeypatch):
    r, model, G = gguf_runner(tmp_path, monkeypatch)
    ws = r.ws
    ids = [4, 1, 8, 2, 6]
    # no tokenizer in this checkpoint: the resid unit is refused with a clear reason; raw works
    with pytest.raises(S.SteeringError, match="no tokenizer"):
        S.prepare(ws, [{"addr": "L0N1", "mode": "add", "dose": 1}])
    wire = {"version": 1, "layers": {"0": {"add_resid": [[1], [0.75]]}}, "p50": {"0": 2.0}}
    got = r.next_logits(ids, steer=wire)
    norm = float(ws.read("model.layers.0.mlp.down_proj.weight")[:, 1].double().norm())
    assert norm == pytest.approx(float(model.model.layers[0].mlp.down_proj.weight.detach()[:, 1].double().norm()))

    def pushed(m, args):
        x = args[0].clone()
        x[..., 1] += 0.75 * 2.0 / norm
        return (x,)
    h = model.model.layers[0].mlp.down_proj.register_forward_pre_hook(pushed)
    expect = full(model, ids)[-1]
    h.remove()
    assert torch.allclose(got, expect, atol=1e-5) and not torch.allclose(got, full(model, ids)[-1], atol=1e-4)
    # head multiplier: zeroing head 1 equals zeroing its columns of o_proj in a copy
    got = r.next_logits(ids, steer=S.prepare(ws, [{"addr": "L1H1", "mode": "mul", "dose": 0}])[0])
    o = model.model.layers[1].self_attn.o_proj
    saved = o.weight.detach().clone()
    with torch.no_grad():
        o.weight[:, D // H:] = 0
    expect = full(model, ids)[-1]
    with torch.no_grad():
        o.weight.copy_(saved)
    assert torch.allclose(got, expect, atol=1e-5)
    assert all(hooks_on(m) == 0 for m in model.modules())
    # the probe uses next_logits on this runner
    ws._runner = r
    out = P.next_token(ws, ids, "current", S.prepare(ws, [{"addr": "L1N2", "mode": "mul", "dose": 0}])[0], 3)
    assert out["steered"] and out["kl"] > 0


def test_gguf_steering_hooks_are_removed_when_generation_is_refused(tmp_path, monkeypatch):
    r, model, G = gguf_runner(tmp_path, monkeypatch)
    calls = []

    def lease(ws, **kw):
        calls.append(1)
        if len(calls) > 3:
            raise RuntimeRefusal(event("lease_lost", "refused", "stopped for the test"))
    monkeypatch.setattr(G, "require_lease", lease)
    wire, _ = S.prepare(r.ws, [{"addr": "L1N4", "mode": "add", "dose": 3.0, "unit": "raw"}, {"addr": "L0H0", "mode": "mul", "dose": 2}])
    streamed = []
    with pytest.raises(RuntimeRefusal) as exc:
        r.generate([1, 2, 3], max_new_tokens=10, steer=wire, on_token=streamed.append)
    assert exc.value.tokens == streamed and streamed
    assert all(hooks_on(m) == 0 for m in model.modules())
    # A hook point the model lacks is refused before anything runs, and nothing stays registered.
    monkeypatch.setattr(G, "require_lease", lambda ws, **kw: None)
    r.ws.struct.layers[1].modules.pop("mlp_out")
    ran = len(model.calls)
    with pytest.raises(S.SteeringError, match="layer 1 has no MLP neurons"):
        r.generate([1, 2, 3], max_new_tokens=2, steer=wire)
    assert len(model.calls) == ran
    assert all(hooks_on(m) == 0 for m in model.modules())


class GenDecoder(Decoder):
    """Decoder plus a greedy ``generate`` shaped like transformers': one forward per new token over the
    KV cache, the streamer fed the prompt and then each token."""

    def __init__(self):
        super().__init__()
        self.generation_config = SimpleNamespace(pad_token_id=None, eos_token_id=0, num_beams=1)

    def generate(self, input_ids, attention_mask, pad_token_id, max_new_tokens, do_sample, streamer=None,
                 stopping_criteria=None):
        assert not do_sample and attention_mask.shape == input_ids.shape
        if streamer is not None:
            streamer.put(input_ids)
        out, ids = self(input_ids, None, True, 1), input_ids
        for i in range(max_new_tokens):
            token = out.logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, token], 1)
            if streamer is not None:
                streamer.put(token)
            if stopping_criteria is not None and all(bool(c(ids, None).all()) for c in stopping_criteria):
                break                                                   # as transformers: stop after this token
            if i + 1 < max_new_tokens:
                out = self(token, out.past_key_values, True, 1)
        return ids


def loaded(tensors, replace=None):
    model = GenDecoder()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_((replace or {}).get(name, tensors[name]))
    return model.eval()


def hf_runner(tmp_path):
    """An HFRunner over GenDecoder, built without transformers (its __init__ would load a checkpoint)."""
    tensors = checkpoint(str(tmp_path / "ckpt"))
    ws = Workspace.create_hf(str(tmp_path / "ws"), "tiny", str(tmp_path / "ckpt"))
    ws.tok = VocabTokenizer(list("abcdefghijkl"), "char")           # 12 symbols: the reference text can be measured
    model = loaded(tensors)
    r = object.__new__(HFRunner)
    r.ws, r.lock, r.device, r.dtype, r.model, r.max_ctx = ws, threading.RLock(), torch.device("cpu"), torch.float32, model, 64
    r._named, r._modules = dict(model.named_parameters()), dict(model.named_modules())
    r._pcache, r._saved, r._state, r._tok, r._gain = {}, [], "base", None, {}
    ws._runner = r
    return r, model, tensors


def greedy(model, prompt, n, push=None):
    """Full recompute per token (no cache), with an optional pre-hook on layer 1's down_proj."""
    h = model.model.layers[1].mlp.down_proj.register_forward_pre_hook(push) if push else None
    ids = list(prompt)
    try:
        for _ in range(n):
            ids.append(int(full(model, ids)[-1].argmax()))
    finally:
        if h is not None:
            h.remove()
    return ids[len(prompt):]


def test_transformers_generate_steers_the_variant_it_was_asked_for(tmp_path):
    """HFRunner.generate: the push is applied during model.generate (not only around it), and after
    calibration (which runs base) the weights are those of the variant + extra again."""
    r, model, tensors = hf_runner(tmp_path)
    ws = r.ws
    name = "model.layers.1.mlp.down_proj.weight"
    E = torch.randn(D, F, generator=torch.Generator().manual_seed(11)) * 1.5
    extra = {name: [Dl.block((range(D), range(F)), E)]}
    spec = [{"addr": "L1N4", "mode": "add", "dose": 4.0}]               # resid unit: generate calibrates first
    prompt, n = [1, 5, 2, 7], 8
    # Expected, independently: p50 of base layer 1 output (position 0 excluded), the column norm of base + extra.
    ref = S.reference_ids(ws)
    seen = {}
    base = loaded(tensors)
    h = base.model.layers[1].register_forward_hook(lambda m, a, out: seen.__setitem__("x", out[0][0, 1:].norm(dim=-1)))
    full(base, ref)
    h.remove()
    W = tensors[name] + E
    c = 4.0 * float(seen["x"].median()) / float(W[:, 4].double().norm())
    one = torch.zeros(F)
    one[4] = c
    push = lambda m, a: (a[0] + one,)                                    # noqa: E731
    current = loaded(tensors, {name: W})
    expect = greedy(current, prompt, n, push)
    assert expect != greedy(current, prompt, n)                          # the push matters...
    assert expect != greedy(base, prompt, n, push)                       # ...and so does which weights it runs on
    streamed = []
    got = r.generate(prompt, "current", extra, max_new_tokens=n, steer=spec, on_token=streamed.append)
    assert os.path.exists(os.path.join(ws.dir, S.CALIBRATION_FILE))      # it did calibrate, on the way
    assert got == expect and streamed == got
    assert all(hooks_on(m) == 0 for m in model.modules())
    # Once calibrated, the same again; without steer, the variant's own tokens.
    assert r.generate(prompt, "current", extra, max_new_tokens=n, steer=spec) == expect
    assert r.generate(prompt, "current", extra, max_new_tokens=n) == greedy(current, prompt, n)
    assert r.generate(prompt, "base", max_new_tokens=n) == greedy(base, prompt, n)


# ------------------------------------------------------------------ part 3: column norms from quantized bytes
def synthetic(qtype, rows, cols, seed=0):
    """Random packed bytes of a real GGML layout, with small finite fp16 block scales."""
    import gguf
    block, width = gguf.GGML_QUANT_SIZES[qtype]
    rng = np.random.default_rng(seed)
    if block == 1:
        return rng.standard_normal((rows, cols)).astype(np.float16).view(np.uint8)
    raw = rng.integers(0, 255, size=(rows, cols // block * width), dtype=np.uint8)
    scale = np.frombuffer(np.array([0.01, 0.01], dtype=np.float16).tobytes(), dtype=np.uint8)
    head = 2 if qtype == gguf.GGMLQuantizationType.Q8_0 else 4         # Q8_0: d; Q4_K: d, dmin
    raw.reshape(rows, -1, width)[:, :, :head] = scale[:head]
    return raw


def per_row(raw, rows, cols, qtype):
    """The decode GGUFStore.read did before: one dequantize call per row, over the blocks it needs."""
    import gguf
    block, width = gguf.GGML_QUANT_SIZES[qtype]
    blocks, inverse = np.unique(cols // block, return_inverse=True)
    out = np.empty((len(rows), len(cols)), np.float32)
    for i, row in enumerate(rows):
        out[i] = gguf.dequantize(raw[row].reshape(-1, width)[blocks], qtype).reshape(-1, block)[inverse, cols % block]
    return out


QTYPES = ("Q8_0", "Q4_K", "F16")


@pytest.mark.parametrize("qname", QTYPES)
def test_gather_columns_equals_the_per_row_decode(qname, monkeypatch):
    import gguf
    from wsp import gguf_store
    qtype = getattr(gguf.GGMLQuantizationType, qname)
    rows, cols = 24, 1024
    raw = synthetic(qtype, rows, cols)
    rng = np.random.default_rng(1)
    for chunk in (gguf_store.DECODE_CHUNK, 600):                         # one call, and several row chunks
        monkeypatch.setattr(gguf_store, "DECODE_CHUNK", chunk)
        for _ in range(5):
            r = rng.choice(rows, rng.integers(1, rows), replace=False)
            c = rng.integers(0, cols, rng.integers(1, 40))               # unsorted, repeats allowed
            out = np.empty((len(r), len(c)), np.float32)
            gguf_store.gather_columns(raw.reshape(rows, -1), r, c, qtype, out)
            assert np.array_equal(out, per_row(raw.reshape(rows, -1), r, c, qtype))


def fake_gguf(qname, rows, cols):
    """A GGUFStore over one synthetic down_proj tensor (logical [rows=d_model, cols=neurons]) and the
    workspace surface steering.column_norms uses."""
    import gguf
    from wsp.gguf_store import GGUFStore
    qtype = getattr(gguf.GGMLQuantizationType, qname)
    raw = synthetic(qtype, rows, cols)
    name = "model.layers.0.mlp.down_proj.weight"
    store = object.__new__(GGUFStore)
    store.where = {name: SimpleNamespace(shape=(cols, rows), tensor_type=qtype, data=raw)}
    store.row_permutations, store.column_permutations, store.architecture = {}, {}, "llama"

    class WS:
        calib, keep, reads = {}, {}, []

        def __init__(self):
            self.store = store
            self.sem = SimpleNamespace(mlp=lambda l: {"W_out": Ref(self, name, perm=(1, 0))})

        def shape(self, n):
            return store.shape(n)

        def read(self, n, region, variant="current", extra=None, cache=True):
            self.reads.append(region)
            return store.read(n, region)
    return WS(), raw, qtype, name


@pytest.mark.parametrize("qname,block", [("Q4_K", 256), ("Q8_0", 32)])
def test_column_norms_decode_only_needed_blocks_each_once(qname, block, monkeypatch):
    import gguf
    rows, cols = 16, 4096
    ws, raw, qtype, name = fake_gguf(qname, rows, cols)
    assert ws.store.read_block(name) == (1, block)
    decoded = []
    real = gguf.dequantize
    monkeypatch.setattr(gguf, "dequantize", lambda data, q: decoded.append(data.shape[0]) or real(data, q))
    monkeypatch.setattr(S, "READ_WINDOW", 4 * block * rows)              # at most 4 blocks of columns per read
    neurons = [5, 7, block + 1, 3 * block, 3 * block + 2, 4 * block - 1, 9 * block, 10 * block, 11 * block, 12 * block,
               13 * block, 14 * block + 3]
    norms, cost = S.column_norms(ws, 0, neurons)
    needed = {n // block for n in neurons}
    assert sum(decoded) == rows * len(needed)                            # every needed block once per row, nothing else
    covered = [set(range(c.start // block, (c.stop - 1) // block + 1)) for _, c in ws.reads]
    assert set().union(*covered) == needed and sum(map(len, covered)) == len(needed)
    assert all(len(c) * rows <= 4 * block * rows for _, c in ws.reads) and cost["reads"] == len(ws.reads) and cost["block"] == block
    full_cols = per_row(raw.reshape(rows, -1), np.arange(rows), np.array(neurons), qtype).astype(np.float64)
    assert np.allclose(norms.numpy(), np.linalg.norm(full_cols, axis=0), rtol=1e-12)
    # cached: nothing decoded again
    decoded.clear()
    assert S.column_norms(ws, 0, neurons[:3])[1]["cached"] == 3 and decoded == []
