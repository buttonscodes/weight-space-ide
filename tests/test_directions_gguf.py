"""Behaviour directions on a GGUF base: chat from the GGUF's own template, the workspace's generation config in the
GGUF generation loop, the edit as float32 patches over the immutable quantized file, and the adapter refusal.

Model-free: synthetic GGUF files written here with the gguf package from a random transformers Qwen2ForCausalLM
or, for the hybrid layout, Qwen3_5ForCausalLM (tests/gguf_fixtures.py), opened as the IDE opens any GGUF; a scripted model object behind GGUFRunner.generate
(logits from a table written in the test); transformers' own RepetitionPenaltyLogitsProcessor and AutoTokenizer as
the references. The tests marked "bridge" run the packed model and need WSP_GGML_BRIDGE (they skip without it).
"""
import json
import os
import re
import threading
from types import SimpleNamespace

import pytest
import torch

gguf = pytest.importorskip("gguf")

from wsp import delta as D
from wsp import directions as DR
from wsp import directions_ara as ARA
from wsp import directions_scores as DS
from wsp import gguf_runner as G
from wsp import probes as P
from wsp.gguf_store import MAX_READ, file_hash, open_gguf
from wsp.store import norm_index, region_size

from gguf_fixtures import CHATML, ENDOFTEXT, IM_END, IM_START, tiny_hybrid_gguf, tiny_qwen2_gguf

GOOD = ["abc", "abd", "hello", "zzz", "quiet", "fox"]
BAD = ["xyz", "qqq", "world", "mmmm", "bad"]
NO_TEMPLATE = "this GGUF carries no chat template (tokenizer.chat_template)"
bridge = pytest.mark.skipif(not os.environ.get("WSP_GGML_BRIDGE"), reason="Pinned native bridge required")


def opened(tmp_path, label="m", **kw):
    """A synthetic Qwen2 GGUF opened as the IDE opens one (open_gguf: config, converted tokenizer, base hash)."""
    path = tmp_path / f"{label}.gguf"
    tiny_qwen2_gguf(path, **kw)
    return open_gguf(str(tmp_path / f"root-{label}"), str(path)), path


# ------------------------------------------------------------------ Q1, Q2: chat from the GGUF's own template
def test_a_gguf_with_a_template_offers_chat(tmp_path):
    def gguf_dir(files):
        d = tmp_path / f"d{len(list(tmp_path.iterdir()))}"
        d.mkdir()
        for name, text in files.items():
            (d / name).write_text(text)
        return SimpleNamespace(kind="gguf", model_dir=str(d))
    assert P.chat_template(gguf_dir({"chat_template.jinja": CHATML})) == (True, "the GGUF's own chat template")
    assert P.chat_template(gguf_dir({"tokenizer_config.json": json.dumps({"chat_template": CHATML})})) == \
        (True, "the GGUF's own chat template")
    assert P.chat_template(gguf_dir({"tokenizer_config.json": json.dumps({"eos_token": "x"})})) == (False, NO_TEMPLATE)
    assert P.chat_template(gguf_dir({})) == (False, NO_TEMPLATE)
    assert P.chat_template(SimpleNamespace(kind="gguf", model_dir=str(tmp_path / "missing"))) == (False, NO_TEMPLATE)
    # A real open: the template is saved with the converted tokenizer, and the measurement's model-free check uses it.
    ws, _ = opened(tmp_path, "with")
    assert not os.path.exists(os.path.join(ws.model_dir, "tokenizer-error.txt"))
    assert P.chat_template(ws) == (True, "the GGUF's own chat template")
    assert DR.check_measure(ws, "m", GOOD, BAD, chat=True)["chat"] is True
    bare, _ = opened(tmp_path, "without", chat_template=None)
    assert not os.path.exists(os.path.join(bare.model_dir, "tokenizer-error.txt"))
    assert P.chat_template(bare) == (False, NO_TEMPLATE)
    with pytest.raises(DR.DirectionError) as e:
        DR.check_measure(bare, "m", GOOD, BAD, chat=True)
    assert str(e.value) == f"'as a chat message' needs the model's own chat template: {NO_TEMPLATE}"
    assert DR.check_measure(bare, "m", GOOD, BAD)["chat"] is False           # plain text is unchanged


def scripted_runner(ws):
    """A GGUFRunner without a model (nothing native is built): chat_prompt and hf_tokenizer only need the workspace."""
    r = object.__new__(G.GGUFRunner)
    r.ws, r._tok, r.lock = ws, None, threading.RLock()
    return r


def test_gguf_chat_renders_with_the_workspace_tokenizer(tmp_path):
    from transformers import AutoTokenizer
    ws, _ = opened(tmp_path)
    r = scripted_runner(ws)
    tok = AutoTokenizer.from_pretrained(ws.model_dir, local_files_only=True)
    for messages in ([{"role": "user", "content": "abc"}],
                     [{"role": "system", "content": "be quiet"}, {"role": "user", "content": "hello world"}]):
        want = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert r.chat_prompt(messages) == want
    assert want == "<|im_start|>system\nbe quiet<|im_end|>\n<|im_start|>user\nhello world<|im_end|>\n<|im_start|>assistant\n"
    # The rendering as a measurement or a score reads it: the template's control tokens stay whole.
    ws._runner = r
    ids = DR.render_ids(ws, ["abc"], "good", chat=True, chat_options={})[0]
    rendered = "<|im_start|>user\nabc<|im_end|>\n<|im_start|>assistant\n"
    assert ids == P.chat_encode(ws)(rendered) == ws.tok.encode(rendered, special=False)
    assert ids[0] == IM_START and ids.count(IM_START) == 2 and ids.count(IM_END) == 1
    assert ws.tok.decode(ids) == rendered


def test_gguf_directions_render_with_the_shared_renderer(tmp_path):
    """A GGUF base goes through the renderer every Directions path shares (wsp.directions_rendering): a template without
    a system role gets the system message folded into the user turn. A GGUF that asks for a beginning-of-text token
    (tokenizer.ggml.add_bos_token) on a byte-level vocabulary: transformers' conversion of the tokenizer adds none, so
    the ids hold the one the template prints (llama.cpp's own tokenizer would add a second; docs/DIRECTIONS.md)."""
    drops = "{%- set messages = messages | rejectattr('role', 'equalto', 'system') | list %}" + CHATML
    ws, _ = opened(tmp_path, "drops", chat_template=drops)
    ws._runner = scripted_runner(ws)
    ids = DR.render_ids(ws, ["abc"], "good", chat=True, system="be quiet", chat_options={})[0]
    rendered = "<|im_start|>user\nbe quiet\n\nabc<|im_end|>\n<|im_start|>assistant\n"
    assert ids == ws.tok.encode(rendered, special=False) and ws.tok.decode(ids) == rendered
    ws, _ = opened(tmp_path, "bos", chat_template="{{ bos_token }}" + CHATML, add_bos=True)
    ws._runner = scripted_runner(ws)
    ids = DR.render_ids(ws, ["abc"], "good", chat=True, chat_options={})[0]
    rendered = "<|endoftext|><|im_start|>user\nabc<|im_end|>\n<|im_start|>assistant\n"
    assert ids == ws.tok.encode(rendered, special=False) and ids[:2] == [ENDOFTEXT, IM_START]


# ------------------------------------------------------------------ Q3-Q5: the generation config in the GGUF loop
def test_the_repetition_penalty_is_transformers_own():
    from transformers import RepetitionPenaltyLogitsProcessor
    g = torch.Generator().manual_seed(5)
    for penalty in (1.3, 1.1, 0.7, 2.0):
        logits = torch.randn(50, generator=g) * 3
        ids = torch.randint(0, 50, (17,), generator=g).tolist() + [3, 3, 3]        # repeats count once
        before = logits.clone()
        got = G.GGUFRunner.repetition_penalty(logits, ids, penalty)
        want = RepetitionPenaltyLogitsProcessor(penalty)(torch.tensor([ids]), logits[None].clone())[0]
        assert got.dtype == torch.float32 and torch.equal(got, want)
        assert torch.equal(logits, before)                                         # the input is not changed
        assert (got != logits).any() and torch.equal(got[[i for i in range(50) if i not in ids]],
                                                     logits[[i for i in range(50) if i not in ids]])


V = 8


def table():
    """Next-token logits by the last token. Without a penalty the loop is 1, 2, 1, 2, ...; with repetition_penalty
    1.3 over the prompt [1, 2] and the response, 3 beats the prompt's 1, 4 beats the prompt's 2, and the listed end
    token 6 beats 1 and 3 (both seen)."""
    t = torch.full((V, V), -5.0)
    t[1, 2] = 1.0
    t[2, 1], t[2, 3] = 1.2, 1.0
    t[3, 2], t[3, 4] = 2.0, 1.6
    t[4, 6], t[4, 1], t[4, 3], t[4, 4] = 0.5, 0.6, 0.55, -0.1
    return t


def generator_runner(tmp_path, monkeypatch, generation=None, name="g"):
    """GGUFRunner.generate over a scripted model: its logits are table()[last token]; the configuration's own end
    token is 7 (never the best); ``generation`` is written as the workspace's model/generation_config.json."""
    d = tmp_path / name / "model"
    d.mkdir(parents=True)
    if generation is not None:
        (d / "generation_config.json").write_text(json.dumps(generation))
    r = object.__new__(G.GGUFRunner)
    r.lock, r.max_ctx, r.device = threading.RLock(), 64, torch.device("cpu")
    r.ws = SimpleNamespace(hf_config={"eos_token_id": 7}, model_dir=str(d), kind="gguf")
    r.use = lambda *a: None
    r.forward_options = lambda tokens: {}
    seq, calls = [], []
    t = table()

    def model(input_ids, past_key_values=None, use_cache=True, logits_to_keep=1):
        if past_key_values is None:
            seq.clear()
        seq.extend(input_ids[0].tolist())
        calls.append(list(seq))
        return SimpleNamespace(logits=t[seq[-1]].reshape(1, 1, V).clone(), past_key_values=object())
    r.model, r.calls = model, calls
    monkeypatch.setattr(G, "require_lease", lambda ws, **kw: None)
    return r


def reference(prompt, n, penalty=None, stops=(), temperature=0.0, seed=0):
    """The loop written out with transformers' processor applied to the whole sequence before every choice."""
    from transformers import RepetitionPenaltyLogitsProcessor
    seq, out, t = list(prompt), [], table()
    rng = torch.Generator().manual_seed(seed)
    for _ in range(n):
        logits = t[seq[-1]].clone()
        if penalty is not None:
            logits = RepetitionPenaltyLogitsProcessor(penalty)(torch.tensor([seq]), logits[None])[0]
        if temperature > 0:
            tok = int(torch.multinomial((logits / temperature).softmax(-1), 1, generator=rng))
        else:
            tok = int(logits.argmax())
        out.append(tok)
        seq.append(tok)
        if tok in stops:
            break
    return out


def test_gguf_generation_applies_the_workspace_generation_config(tmp_path, monkeypatch):
    prompt = [1, 2]
    # Without the file, or with inert values only: unchanged (plain argmax, the configuration's end token).
    for i, generation in enumerate((None, {"repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "do_sample": True,
                                           "top_p": 0.8, "temperature": 0.7})):
        r = generator_runner(tmp_path, monkeypatch, generation, f"plain{i}")
        assert r.generate(prompt, "base", max_new_tokens=6) == reference(prompt, 6) == [1, 2, 1, 2, 1, 2]
        assert r.generation_config() == {"applied": {}, "refused": []}
    # repetition_penalty 1.3 over prompt and response, and a stop on the file's eos_token_id (not the config's 7).
    for i, eos in enumerate(([5, 6], 6)):
        r = generator_runner(tmp_path, monkeypatch, {"repetition_penalty": 1.3, "eos_token_id": eos, "top_k": 20}, f"pen{i}")
        got = r.generate(prompt, "base", max_new_tokens=6)
        assert got == reference(prompt, 6, 1.3, stops={5, 6}) == [3, 4, 6]
        assert got != reference(prompt, 6) and got[0] != 1                    # the prompt's own ids are penalised
        assert [c[-1] for c in r.calls] == [2, 3, 4]                           # prefill, then one step per token
        assert r.generation_config()["applied"] == {"eos_token_id": eos, "repetition_penalty": 1.3}
    # Before sampling too: the same draws as sampling from the penalised logits.
    r = generator_runner(tmp_path, monkeypatch, {"repetition_penalty": 1.3}, "sampled")
    got = r.generate(prompt, "base", max_new_tokens=8, temperature=1.0, seed=11)
    assert got == reference(prompt, 8, 1.3, stops={7}, temperature=1.0, seed=11)
    # The scores' answer lists what the loop applies, for the runner here and for the model worker's.
    from wsp.remote_runner import RemoteRunner
    r = generator_runner(tmp_path, monkeypatch, {"repetition_penalty": 1.3, "eos_token_id": [5, 6]}, "listed")
    want = {"eos_token_id": [5, 6], "repetition_penalty": 1.3}
    for runner in (r, object.__new__(RemoteRunner)):
        ws = SimpleNamespace(kind="gguf", model_dir=r.ws.model_dir, runner=runner)
        block = DS.generation_block(ws, 4)
        assert block["settings"] == want and block["runner"] == type(runner).__name__
    assert DS.generation_block(SimpleNamespace(kind="gguf", model_dir=r.ws.model_dir, runner=r), 4)["generation"] == (
        "argmax loop to the configuration's end-of-text ids; from the workspace's generation_config.json: "
        "eos_token_id and repetition_penalty (other greedy keys refused)")


def test_an_unapplied_greedy_key_is_refused_by_name(tmp_path, monkeypatch):
    r = generator_runner(tmp_path, monkeypatch, {"no_repeat_ngram_size": 3, "repetition_penalty": 1.3}, "ngram")
    with pytest.raises(ValueError) as e:
        r.generate([1, 2], "base", max_new_tokens=4)
    assert str(e.value) == ("the generation config sets no_repeat_ngram_size=3, which the GGUF runner does not apply; "
                            "remove it from the workspace's model/generation_config.json or score on a transformers checkpoint")
    assert r.calls == []                                                       # refused before the model runs
    # Every other greedy key, by name, with a value that changes a response; the inert values are not refused.
    values = {"bad_words_ids": [[3]], "min_new_tokens": 2, "min_length": 4, "suppress_tokens": [3],
              "begin_suppress_tokens": [3], "forced_bos_token_id": 1, "forced_eos_token_id": 6,
              "sequence_bias": [[[3], 1.0]], "stop_strings": ["ab"]}
    assert set(values) | {"no_repeat_ngram_size", "eos_token_id", "repetition_penalty"} == set(DS.GREEDY_KEYS)
    for key, value in values.items():
        r = generator_runner(tmp_path, monkeypatch, {key: value}, key)
        with pytest.raises(ValueError, match="^" + re.escape(f"the generation config sets {key}={json.dumps(value)}, which")):
            r.generate([1, 2], "base", max_new_tokens=4)
        assert r.calls == []
    r = generator_runner(tmp_path, monkeypatch, {"min_length": 0, "min_new_tokens": None, "suppress_tokens": None}, "inert")
    assert r.generate([1, 2], "base", max_new_tokens=2) == [1, 2]


# ------------------------------------------------------------------ Q6, Q7: the edit on a packed base (bridge)
def gguf_session(tmp_path, monkeypatch, qtype):
    """A Session over a synthetic Qwen2 GGUF of ``qtype`` matrices, with every store read's size recorded."""
    from wsp.session import Session
    path = tmp_path / f"tiny-{qtype}.gguf"
    tiny_qwen2_gguf(path, qtype=qtype)
    digest = file_hash(path)
    sess = Session(str(tmp_path / "root"))
    ws = sess.open(str(path))
    assert ws.meta["base_sha256"] == digest
    monkeypatch.setattr(G, "require_lease", lambda ws, **kw: None)          # a synthetic model of about 20k values
    reads, real = [], ws.store.read

    def read(name, region=None):
        reads.append(region_size(norm_index((), ws.shape(name)) if region is None else region))
        return real(name, region)
    monkeypatch.setattr(ws.store, "read", read)
    return sess, ws, path, digest, reads


@bridge
@pytest.mark.parametrize("qtype", ["Q8_0", "Q4_0"])
def test_directions_on_a_synthetic_qwen2_gguf_are_fp32_patches(tmp_path, monkeypatch, qtype):
    sess, ws, path, digest, reads = gguf_session(tmp_path, monkeypatch, qtype)
    try:
        DR.measure(ws, "m", GOOD, BAD)
        assert file_hash(path) == digest
        dist = {"max_weight": 1.0, "max_weight_position": 1.0, "min_weight": 1.0, "min_weight_distance": 4.0}
        script = DR.script_for({"direction": "m", "weight_distributions": {c: dist for c in DR.COMPONENTS}})
        res = sess.apply(script, dry_run=False)
        assert res.get("committed"), res
        patch = ws.patches[-1]
        touched = sorted(patch.deltas)
        assert touched == sorted(f"model.layers.{l}.{m}.weight" for l in range(2) for m in ("self_attn.o_proj", "mlp.down_proj"))
        for name in touched:
            pieces = patch.deltas[name]
            assert len(pieces) == 1
            for t in (pieces[0].A, pieces[0].B, pieces[0].values):
                assert t is None or t.dtype == torch.float32
            dense = torch.zeros(ws.shape(name))
            D.add_to(dense, norm_index((), ws.shape(name)), pieces[0], 1.0)
            assert pieces[0].kind != "lowrank" or torch.equal(dense, pieces[0].A @ pieces[0].B)
            base, current = ws.read(name, (), "base"), ws.read(name, (), "current")
            # current = base + delta in float32, bit for bit (one piece: (0 + delta) is delta exactly)
            assert torch.equal(ws.read(name, (), "delta"), dense) and torch.equal(current, base + dense)
        # The packed product with the patch minus without is the float32 overlay x @ delta^T.
        name = "model.layers.1.self_attn.o_proj.weight"
        delta = ws.read(name, (), "delta")
        runner = ws.runner
        lin = runner.model.get_submodule(name[:-len(".weight")])
        x = torch.randn(1, 3, delta.shape[1], generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            runner.use("current")
            edited = lin(x)
            runner.use("base")
            plain = lin(x)
        want = x @ delta.T
        assert float((edited - plain - want).norm() / want.norm()) <= 1e-5
        assert file_hash(path) == digest and max(reads) <= MAX_READ
    finally:
        sess.unload()


@bridge
@pytest.mark.parametrize("qtype", ["Q8_0", "Q4_0"])
def test_ara_on_a_synthetic_gguf_reports_its_consistency(tmp_path, monkeypatch, qtype):
    sess, ws, path, digest, reads = gguf_session(tmp_path, monkeypatch, qtype)
    try:
        DR.measure(ws, "m", GOOD, BAD, module_io=True)
        params = {"method": "ara", "direction": "m", "lora_rank": 3, "n_optimization_steps": 2, "max_iter": 8}
        report = ARA.ablate(ws, params, add=lambda view, A, B: None)
        rows = report["components"]
        assert sorted((c["layer"], c["component"]) for c in rows) == sorted((l, c) for l in range(2) for c in DR.COMPONENTS)
        for c in rows:
            for m in (c["mismatch_good"], c["mismatch_bad"]):
                assert 0 < m <= ARA.MISMATCH_LIMIT == 0.05, c
        res = sess.apply(DR.script_for(params), dry_run=False)
        assert res.get("committed"), res
        assert all(t is None or t.dtype == torch.float32 for ps in ws.patches[-1].deltas.values() for p in ps
                   for t in (p.A, p.B, p.values))
        assert file_hash(path) == digest and max(reads) <= MAX_READ
    finally:
        sess.unload()


# ------------------------------------------------------------------ Q8: the linear-attention output on a hybrid GGUF (bridge)
def hybrid_session(tmp_path, monkeypatch, path, root):
    """A Session over the synthetic hybrid GGUF at ``path`` (tests/gguf_fixtures.py:tiny_hybrid_gguf), opened as the
    IDE opens any GGUF, with every store read's size recorded."""
    from wsp.session import Session
    sess = Session(str(tmp_path / root))
    ws = sess.open(str(path))
    monkeypatch.setattr(G, "require_lease", lambda ws, **kw: None)          # a synthetic model of about 100k values
    reads, real = [], ws.store.read

    def read(name, region=None):
        reads.append(region_size(norm_index((), ws.shape(name)) if region is None else region))
        return real(name, region)
    monkeypatch.setattr(ws.store, "read", read)
    return sess, ws, reads


HYBRID_ATTN = ["linear_attn.out_proj"] * 3 + ["self_attn.o_proj"]           # layers 0-2 linear attention, 3 full
HYBRID_TOUCHED = sorted([f"model.layers.{l}.{m}.weight" for l, m in enumerate(HYBRID_ATTN)] +
                        [f"model.layers.{l}.mlp.down_proj.weight" for l in range(4)])


@bridge
@pytest.mark.parametrize("qtype", ["Q8_0", "Q4_0"])
def test_a_hybrid_gguf_linear_output_is_captured_and_ablated(tmp_path, monkeypatch, qtype):
    path = tmp_path / f"tiny-hybrid-{qtype}.gguf"
    tiny_hybrid_gguf(path, qtype=qtype)
    digest = file_hash(path)
    # Capture and ARA: linear_attn.out_proj's input and output are captured from the packed product in layers 0-2.
    sess, ws, reads = hybrid_session(tmp_path, monkeypatch, path, "root-ara")
    try:
        assert ws.kind == "gguf" and ws.meta["base_sha256"] == digest and ws.struct.recognised
        out = "model.layers.1.linear_attn.out_proj.weight"
        assert out in ws.store.column_permutations                           # its columns are the grouped heads
        assert [[m["path"] for m in DR.modules(ws, l, "attn.o_proj")] for l in range(4)] == [[m] for m in HYBRID_ATTN]
        DR.measure(ws, "m", GOOD, BAD, module_io=True)
        info = json.load(open(os.path.join(ws.dir, "directions", "m.io", "io.json")))
        assert [[(e["component"], e["module"], e["source"], e["captured"]) for e in L["modules"]] for L in info["layers"]] == [
            [("attn.o_proj", m, "mixer" if l < 3 else "attn", True), ("mlp.down_proj", "mlp.down_proj", "mlp", True)]
            for l, m in enumerate(HYBRID_ATTN)]
        params = {"method": "ara", "direction": "m", "lora_rank": 3, "n_optimization_steps": 2, "max_iter": 8}
        rows = ARA.ablate(ws, params, add=lambda view, A, B: None)["components"]
        assert [(c["layer"], c["component"], c["module"]) for c in rows] == [
            (l, c, m if c == "attn.o_proj" else "mlp.down_proj") for l, m in enumerate(HYBRID_ATTN) for c in DR.COMPONENTS]
        for c in rows:
            for m in (c["mismatch_good"], c["mismatch_bad"]):
                assert 0 < m <= ARA.MISMATCH_LIMIT == 0.05, c
        res = sess.apply(DR.script_for(params), dry_run=False)
        assert res.get("committed"), res
        assert sorted(ws.patches[-1].deltas) == HYBRID_TOUCHED
        assert all(t is None or t.dtype == torch.float32 for ps in ws.patches[-1].deltas.values() for p in ps
                   for t in (p.A, p.B, p.values))
        assert file_hash(path) == digest and max(reads) <= MAX_READ
    finally:
        sess.unload()
    # Directional ablation: every layer's attention output is ablated, the mixer's included, as float32 patches the
    # packed runtime adds in the layer's own (logical) coordinates.
    sess, ws, reads = hybrid_session(tmp_path, monkeypatch, path, "root-abl")
    try:
        DR.measure(ws, "m", GOOD, BAD)
        dist = {"max_weight": 1.0, "max_weight_position": 1.0, "min_weight": 1.0, "min_weight_distance": 4.0}
        res = sess.apply(DR.script_for({"direction": "m", "weight_distributions": {c: dist for c in DR.COMPONENTS}}),
                         dry_run=False)
        assert res.get("committed"), res
        patch = ws.patches[-1]
        assert sorted(patch.deltas) == HYBRID_TOUCHED
        runner = ws.runner
        for l in range(3):
            name = f"model.layers.{l}.linear_attn.out_proj.weight"
            pieces = patch.deltas[name]
            assert len(pieces) == 1 and all(t is None or t.dtype == torch.float32
                                            for t in (pieces[0].A, pieces[0].B, pieces[0].values))
            dense = torch.zeros(ws.shape(name))
            D.add_to(dense, norm_index((), ws.shape(name)), pieces[0], 1.0)
            base, current = ws.read(name, (), "base"), ws.read(name, (), "current")
            assert float(dense.abs().max()) > 0
            assert torch.equal(ws.read(name, (), "delta"), dense) and torch.equal(current, base + dense)
            # The packed product with the patch minus without is the float32 overlay x @ delta^T.
            lin = runner.model.get_submodule(name[:-len(".weight")])
            x = torch.randn(1, 3, dense.shape[1], generator=torch.Generator().manual_seed(3 + l))
            with torch.no_grad():
                runner.use("current")
                edited = lin(x)
                runner.use("base")
                plain = lin(x)
            want = x @ dense.T
            assert float((edited - plain - want).norm() / want.norm()) <= 1e-5, name
        assert file_hash(path) == digest and max(reads) <= MAX_READ
    finally:
        sess.unload()


# ------------------------------------------------------------------ a reopened companion bundle computes the same bits
def test_the_overlay_gives_the_same_bits_whatever_the_factors_layout():
    """A committed low-rank piece holds its factors as transposed views; read back from its patch file it holds the
    same values contiguous. The run-time overlay gives the same output, bit for bit, for both (sizes of the 0.5B
    model's attention and MLP output matrices, ranks 3 and 50)."""
    from wsp.overlay import DeltaOverlay
    g = torch.Generator().manual_seed(0)
    for out_f, in_f, r in [(32, 32, 3), (896, 896, 50), (896, 4864, 50), (896, 4864, 3)]:
        A = (torch.randn(r, out_f, generator=g) * 0.05).T                 # [out, r], a transposed view
        B = (torch.randn(in_f, r, generator=g) * 0.05).T                  # [r, in]
        assert not A.is_contiguous() and not B.is_contiguous()
        region = norm_index((), (out_f, in_f))
        views = DeltaOverlay([(D.lowrank(region, A, B), 1.0)], out_f, in_f)
        loaded = DeltaOverlay([(D.lowrank(region, A.contiguous(), B.contiguous()), 1.0)], out_f, in_f)
        for n in (1, 8, 39):
            x = torch.randn(1, n, in_f, generator=g) * 3
            a, b = torch.zeros(1, n, out_f), torch.zeros(1, n, out_f)
            views.apply(x, a)
            loaded.apply(x, b)
            assert torch.equal(a, b), (out_f, in_f, r, n, float((a - b).abs().max()))


@bridge
@pytest.mark.parametrize("qtype", ["Q8_0", "Q4_0"])
def test_a_reopened_companion_bundle_gives_the_same_logits(tmp_path, monkeypatch, qtype):
    """The companion bundle's exactness after the comparison script's sequence (scripts/formats_gguf_compare.py,
    G-S9): an uncommitted candidate scored, a directional ablation committed and switched off, an ARA patch
    committed on top. Reopened in a fresh root with the same base, every touched matrix's packed output and the
    first-token logits of both variants are those of the workspace that made the patches, bit for bit."""
    from wsp.companion import export_bundle, import_bundle
    sess, ws, path, digest, _ = gguf_session(tmp_path, monkeypatch, qtype)
    try:
        DR.measure(ws, "m", GOOD, BAD, module_io=True)
        dist = {"max_weight": 1.0, "max_weight_position": 1.0, "min_weight": 1.0, "min_weight_distance": 4.0}
        ablation = DR.script_for({"direction": "m", "weight_distributions": {c: dist for c in DR.COMPONENTS},
                                  "row_normalization": "full", "full_normalization_lora_rank": 3})
        scored = DS.score(ws, {"state": "candidate", "script": ablation, "max_response_length": 2,
                               "scores": [{"plugin": "KLDivergence", "prompts": ["fox", "zzz"]}]})
        assert scored["scores"][0]["value"] > 0                            # the candidate ran, uncommitted
        res = sess.apply(ablation, dry_run=False)
        assert res.get("committed"), res
        ws.set_patch(res["patch"], enabled=False)
        res = sess.apply(DR.script_for({"method": "ara", "direction": "m", "lora_rank": 3}), dry_run=False)
        assert res.get("committed"), res
        assert [p.enabled for p in ws.patches] == [False, True]
        prompts = [ws.encode(t) for t in ("fox", "hello world", "abc")]
        g = torch.Generator().manual_seed(7)
        xs = {name: [torch.randn(1, n, ws.shape(name)[1], generator=g) for n in (1, 8)] for name in ws.touched()}

        def outputs(w):
            r = w.runner
            with r.lock, torch.no_grad():
                r.use("current")
                mods = {name: [r.model.get_submodule(name[:-len(".weight")])(x) for x in xs[name]] for name in xs}
            logits = {v: [torch.as_tensor(r.first_token_logits(ids, v)) for ids in prompts] for v in ("current", "base")}
            return mods, logits
        mods, logits = outputs(ws)
        (tmp_path / "export").mkdir()
        bundle = export_bundle(ws, str(tmp_path / "export" / "companion"))["path"]
        sess.unload()
        again = import_bundle(str(tmp_path / "reopened"), bundle, base_path=str(path))
        try:
            assert [p.enabled for p in again.patches] == [False, True] and sorted(again.touched()) == sorted(xs)
            mods2, logits2 = outputs(again)
            for name in xs:
                for a, b in zip(mods[name], mods2[name]):
                    assert torch.equal(a, b), (name, a.shape, float((a - b).abs().max()))
            for v in ("current", "base"):
                for a, b in zip(logits[v], logits2[v]):
                    assert torch.equal(a, b), (v, float((a - b).abs().max()))
        finally:
            again.runner.release()
        assert file_hash(path) == digest
    finally:
        sess.unload()


# ------------------------------------------------------------------ Q9: no PEFT adapter on a GGUF base
def test_the_adapter_export_is_refused_on_a_gguf_base(tmp_path):
    """The route on a real (synthetic) GGUF workspace; tests/test_adapter_export.py::test_refusals has the same text
    from wsp.adapter_export.plan on the fake quantized store."""
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    from wsp.session import Session
    path = tmp_path / "tiny.gguf"
    tiny_qwen2_gguf(path)
    digest = file_hash(path)
    sess = Session(str(tmp_path / "root"))
    ws = sess.open(str(path))
    name = "model.layers.0.self_attn.o_proj.weight"
    ws.commit({name: [D.lowrank(norm_index((), ws.shape(name)), torch.ones(32, 1) * 0.01, torch.ones(1, 32))]}, "m")
    c = TestClient(create_app(sess, "http://127.0.0.1:0"), raise_server_exceptions=False)
    text = ("AdapterError: this workspace's base is a quantized GGUF file: a PEFT adapter applies to a floating-point "
            "transformers checkpoint, and these patches were computed against the quantized weights, so the adapter's "
            "base would be the dequantized model, which the IDE never builds. Export the WSP companion bundle "
            "(GET /api/export), or apply the edit to the floating-point checkpoint the GGUF was made from")
    for body in ({"name": "a1"}, {"name": "a1", "dry_run": True}):
        r = c.post("/api/export_adapter", json=body)
        assert (r.status_code, r.json()) == (400, {"error": text})
    assert not os.path.exists(os.path.join(ws.dir, "export")) and file_hash(path) == digest
