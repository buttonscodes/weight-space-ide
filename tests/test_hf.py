"""Native Hugging Face layouts: tiny random models of five families, built offline.

The point of these tests is the *mapping*: that "head h", "neuron n" and the standard hook
points mean the same thing in every layout. Each claim is checked against the model's actual
behaviour (an edit to the weights must equal the corresponding intervention on activations),
not against the adapter's own bookkeeping.
"""
import json
import os
import sys

import pytest
import torch

transformers = pytest.importorskip("transformers")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wsp import addressing as A  # noqa: E402
from wsp import analysis as an  # noqa: E402
from wsp import chunks, edits, hierarchy, zoo  # noqa: E402
from wsp.session import Session  # noqa: E402
from wsp.wscript import run_script  # noqa: E402

V, D_MODEL = 40, 32
FAMILIES = ["llama", "gpt2", "neox", "phi3", "neo"]


def build(kind: str, directory: str):
    import transformers as T
    torch.manual_seed(0)
    if kind == "llama":
        m = T.LlamaForCausalLM(T.LlamaConfig(vocab_size=V, hidden_size=D_MODEL, intermediate_size=48, num_hidden_layers=2,
                                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
                                             tie_word_embeddings=False))
    elif kind == "gpt2":
        m = T.GPT2LMHeadModel(T.GPT2Config(vocab_size=V, n_positions=32, n_embd=D_MODEL, n_layer=2, n_head=4))
    elif kind == "neox":
        m = T.GPTNeoXForCausalLM(T.GPTNeoXConfig(vocab_size=V, hidden_size=D_MODEL, num_hidden_layers=2, num_attention_heads=4,
                                                 intermediate_size=64, max_position_embeddings=32))
    elif kind == "phi3":
        m = T.Phi3ForCausalLM(T.Phi3Config(vocab_size=V, hidden_size=D_MODEL, intermediate_size=48, num_hidden_layers=2,
                                           num_attention_heads=4, max_position_embeddings=32, pad_token_id=0, bos_token_id=1,
                                           eos_token_id=2))
    else:
        m = T.GPTNeoForCausalLM(T.GPTNeoConfig(vocab_size=V, hidden_size=D_MODEL, num_layers=2, num_heads=4,
                                               attention_types=[[["global"], 2]], max_position_embeddings=32))
    with torch.no_grad():                        # default init is tiny; make every component matter
        for name, p in m.named_parameters():
            if p.dim() >= 2:
                p.mul_(6.0)
            elif "bias" in name:
                p.add_(0.1 * torch.randn_like(p))
    m.save_pretrained(directory, safe_serialization=True)
    from tokenizers import Tokenizer, models, pre_tokenizers
    tk = Tokenizer(models.WordLevel({f"t{i}": i for i in range(V)}, unk_token="t0"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.save(os.path.join(directory, "tokenizer.json"))


@pytest.fixture(scope="module", params=FAMILIES)
def hf(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    build(request.param, str(root / "ckpt"))
    s = Session(str(root / "ws"))
    os.makedirs(s.root, exist_ok=True)
    s.ws = zoo.open_hf(s.root, str(root / "ckpt"), name="m")
    s.kind = request.param
    return s


PROMPT = "t3 t7 t11 t5 t9 t2"


def logits(ws, **kw):
    return ws.runner.run(ws.encode(PROMPT), **kw)[0][0]


def test_structure_is_recognised(hf):
    ws, s = hf.ws, hf.ws.struct
    assert s.recognised and s.n_layers == 2 and s.n_heads == 4 and s.d_head == 8 and s.d_model == D_MODEL and s.d_vocab == V
    assert s.n_kv_heads == (2 if hf.kind == "llama" else 4)
    assert all(s.has_heads(l) and s.has_neurons(l) for l in range(2))
    assert s.norm == ("rms" if hf.kind in ("llama", "phi3") else "ln")
    assert (s.pos is not None) == (hf.kind in ("gpt2", "neo"))
    assert s.tied == (hf.kind in ("gpt2", "neo"))
    floats = sum(ws.store.numel(n) for n in ws.names() if ws.editable(n))
    covered = set()
    for a in hierarchy.partition(ws, "model"):
        stack = [a]
        while stack:
            x = stack.pop()
            kids = hierarchy.partition(ws, x)
            if kids:
                stack += kids
                continue
            for name, idx in ws.resolve(x):
                key = (name, A.index_str(idx))
                assert key not in covered, key                     # heads / key-value groups / neuron runs never overlap
                covered.add(key)
    total = sum(A.count(ws, [(n, A.parse_index(i[1:-1]) if i else ())]) for n, i in covered if ws.editable(n))
    assert total == floats                                         # ... and together they are the whole checkpoint


def test_runner_matches_transformers(hf):
    ws = hf.ws
    ids = torch.tensor([ws.encode(PROMPT)])
    ref = transformers.AutoModelForCausalLM.from_pretrained(ws.model_dir).eval()
    with torch.no_grad():
        want = ref(ids).logits[0]
    assert torch.allclose(logits(ws), want, atol=1e-4)
    assert float(want.std()) > 0.05                                # the test model is not degenerate


@pytest.mark.parametrize("layer", [0, 1])
def test_a_head_is_the_same_head_in_weights_and_activations(hf, layer):
    """Zeroing head h's output weights == zeroing its mixed value in the forward pass, for every head."""
    ws = hf.ws
    base = logits(ws)
    for h in range(4):
        r = run_script(ws, f"head({layer}, {h}).O *= 0")

        def zero(x, _, h=h):
            x = x.clone()
            x[:, :, h] = 0
            return x

        by_weights = logits(ws, extra=r["deltas"])
        by_acts = logits(ws, hooks={f"blocks.{layer}.attn.hook_z": zero})
        assert torch.allclose(by_weights, by_acts, atol=2e-4), (hf.kind, layer, h)
        assert not torch.allclose(by_weights, base, atol=1e-3)


def test_values_are_shared_exactly_as_the_structure_says(hf):
    """Zeroing the value read of head h silences every head in its key-value group and no other."""
    ws, s = hf.ws, hf.ws.struct
    per = s.n_heads // s.n_kv_heads
    r = run_script(ws, "h = head(1, 0)\nh.V *= 0" + ("\nh.b_V *= 0" if "b_V" in s.layers[1].attn else ""))
    _, cache = ws.runner.run(ws.encode(PROMPT), extra=r["deltas"], names=["blocks.1.attn.hook_z"])
    z = cache["blocks.1.attn.hook_z"][0]
    norms = z.norm(dim=(0, 2))
    assert float(norms[:per].max()) < 1e-5 and float(norms[per:].min()) > 1e-3


def test_a_neuron_is_the_same_neuron_in_weights_and_activations(hf):
    ws, s = hf.ws, hf.ws.struct
    base = logits(ws)
    for n in (0, 5, s.d_mlp - 1):
        def zero(x, _, n=n):
            x = x.clone()
            x[..., n] = 0
            return x

        by_acts = logits(ws, hooks={"blocks.0.mlp.hook_post": zero})
        out = run_script(ws, f"neuron(0, {n}).w_out *= 0")
        assert torch.allclose(logits(ws, extra=out["deltas"]), by_acts, atol=2e-4), (hf.kind, n)
        gated = "W_gate" in s.layers[0].mlp
        read = run_script(ws, f"x = neuron(0, {n})\n" + ("x.up *= 0" if gated else "x.w_in *= 0\nx.b_in *= 0"))
        assert torch.allclose(logits(ws, extra=read["deltas"]), by_acts, atol=2e-4), (hf.kind, n, "read side")
        assert not torch.allclose(by_acts, base, atol=1e-4)


def test_attribution_adds_up_to_the_logit(hf):
    ws = hf.ws
    t = an.trace(ws, PROMPT)
    tgt = ws.tok.encode_token(t["target"])[0]
    const = float(ws.runner.unembed(torch.zeros(D_MODEL))[tgt])    # what the final norm's and the head's biases contribute
    assert abs(sum(d["value"] for d in t["dla"]) + const - t["dla_total"]) < 2e-3 * max(1.0, abs(t["dla_total"]))
    assert len([d for d in t["dla"] if d["kind"] == "head"]) == 8 and len(t["lens"]) == 3
    assert t["patterns"][0] is not None and abs(sum(t["patterns"][1][2][-1]) - 1) < 1e-2


def test_edits_are_lazy_exact_and_exportable(hf, tmp_path):
    ws = hf.ws
    base = logits(ws)
    r = hf.apply("head(0, 1).O *= 0.5\nrank1(mlp(1).W_out, randn(d_mlp, seed=1), 0.3 * randn(d_model, seed=2))\n"
                 "neuron(1, 3).w_out += 0.2 * unit(U('t7'))", enforce_scope=False)
    assert r["ok"] and r["committed"]
    assert r["stored"] < 1500 and any("lowrank" in c["kinds"] for c in r["changed"])
    cur = logits(ws)
    assert not torch.allclose(cur, base, atol=1e-3)
    assert torch.equal(logits(ws, variant="base"), base)             # switching back restores the weights bit for bit
    assert torch.allclose(logits(ws), cur, atol=1e-6)
    for name in ws.touched():                                        # the lazy overlay and the in-place runner agree
        assert torch.allclose(ws.runner.param(name).float(), ws.read(name), atol=1e-5)
    out = str(tmp_path / f"export-{hf.kind}")
    ws.export(out, log=lambda *_: None)
    for f in os.listdir(ws.model_dir):
        if f.endswith(".json"):
            with open(os.path.join(ws.model_dir, f)) as src, open(os.path.join(out, f), "w") as dst:
                dst.write(src.read())
    again = transformers.AutoModelForCausalLM.from_pretrained(out).eval()
    with torch.no_grad():
        assert torch.allclose(again(torch.tensor([ws.encode(PROMPT)])).logits[0], cur, atol=1e-4)
    for p in list(ws.patches):
        ws.delete_patch(p.id)
    assert torch.equal(logits(ws), base)


def test_rome_and_views_on_native_layouts(hf):
    ws = hf.ws
    top = ws.tok.token_str(int(logits(ws)[-1].argmax()))
    target = "t20" if top != "t20" else "t21"
    r = edits.rome(ws, PROMPT, "t9", target, layer=0, at="last")
    (name, pieces), = r["deltas"].items()
    assert pieces[0].kind == "lowrank" and r["p_after"] > r["p_before"]
    assert r["success"], (hf.kind, r["p_before"], r["p_after"])
    assert an.head_card(ws, 1, 2)["copying"]["sample"] == V
    card = an.neuron_card(ws, 0, 3)
    assert len(card["reads"]) == (2 if "W_gate" in ws.struct.layers[0].mlp else 1)
    w = an.wiring(ws)
    assert w["n_edges"] > 50 and len(w["window"]) == 2
    name = ws.struct.layers[0].mlp["W_out"].name
    assert an.matrix_view(ws, name)["shape"] == list(ws.shape(name))
    text = chunks.render(ws, ["L1H2"], "q")
    assert "head 1.2" in text and "format q" in text
    plan = chunks.plan(ws, ["blocks.0"], 6000, "q")
    assert all(c["est_tokens"] <= 6000 for c in plan["chunks"]) and len(plan["chunks"]) > 1
    kids = [c["addr"] for c in hierarchy.children(ws, "model")]
    assert "embed" in kids and "blocks.1" in kids and "raw:" in kids
    assert [c["label"] for c in hierarchy.children(ws, "raw:")]


def test_unrecognised_checkpoints_are_still_usable(tmp_path):
    """A checkpoint that is not a language model: no structure, but tensors can be browsed, edited and exported."""
    from safetensors.torch import save_file
    os.makedirs(tmp_path / "ckpt")
    save_file({"encoder.conv.weight": torch.randn(4, 3, 3, 3), "encoder.conv.bias": torch.randn(4), "head.fc.weight": torch.randn(5, 4),
               "steps": torch.tensor([7])}, str(tmp_path / "ckpt" / "model.safetensors"))
    s = Session(str(tmp_path / "ws"))
    os.makedirs(s.root)
    s.ws = ws = zoo.open_hf(s.root, str(tmp_path / "ckpt"), name="x")
    assert not ws.struct.recognised and not ws.can_run
    assert {c["label"] for c in hierarchy.children(ws, "model")} == {"encoder", "head", "steps"}
    r = s.apply('P("head.fc.weight")[2] *= 0\nhead.fc.weight[0, 1] = 3.0', enforce_scope=False)
    assert r["ok"] and r["committed"] and "not_run" in r
    assert float(ws.read("head.fc.weight", (2,)).abs().max()) == 0 and float(ws.read("head.fc.weight", (0, 1))) == 3.0
    assert not s.apply('P("steps")[0] = 1', enforce_scope=False)["ok"]            # integer tensors are not editable
    assert an.matrix_view(ws, "encoder.conv.weight")["shape"] == [36, 3]
    assert "head.fc.weight" in chunks.render(ws, ["raw:head"], "g3")
