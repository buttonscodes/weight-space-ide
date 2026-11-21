"""A transformers model run in bfloat16 (wsp.runner's patch application, wsp.memory_guard's forecast, the run
descriptions of wsp.directions and wsp.directions_scores).

Model-free: tiny random llama checkpoints built in the test's own directory as tests/test_hf.py builds them (saved in
float32 or, cast first, in bfloat16), and the tiny Qwen3.5 checkpoint of tests/layout_fixtures.py. Nothing is
downloaded. Every expectation is computed here from the model's own modules and weights (F.linear on the module's
bfloat16 weight, the norm weight read back from the checkpoint, output_hidden_states of the same model object), not
from the runner's bookkeeping. The float32 constants were recorded from the code before the bfloat16 change.
"""
import hashlib
import json
import os
import struct
import sys

import pytest
import torch
import torch.nn.functional as F

transformers = pytest.importorskip("transformers")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layout_fixtures import tiny_qwen35  # noqa: E402
from wsp import delta as Dl  # noqa: E402
from wsp import directions as DR  # noqa: E402
from wsp import directions_ara as ARA  # noqa: E402
from wsp import directions_scores as DS  # noqa: E402
from wsp import memory_guard as MG  # noqa: E402
from wsp import runner as RN  # noqa: E402
from wsp import runtime_layouts as RL  # noqa: E402
from wsp import zoo  # noqa: E402
from wsp.session import Session  # noqa: E402

V, D_MODEL, LAYERS = 40, 32, 2
PROMPT = "t3 t7 t11 t5 t9 t2"
O = "model.layers.0.self_attn.o_proj.weight"
DOWN = "model.layers.1.mlp.down_proj.weight"
NORM = "model.layers.0.input_layernorm.weight"
GOOD = ["t1 t2 t3", "t4 t5", "t6 t7 t8 t9", "t10 t11", "t12 t13 t14", "t15"]
BAD = ["t20 t21", "t22 t23 t24", "t25", "t26 t27 t28", "t29 t30"]
SMALL = {"lora_rank": 3, "n_optimization_steps": 2, "max_iter": 8}
SIDE_TEXT = "float32 side path on linear layers (PEFT's order); other tensors rounded once to bfloat16"

# Recorded from the code before this change (the forecast of the float32 llama workspace, unchanged rule
# n_params * 4), and the SHA-256 of every patched parameter after use("current") / use("current", extra) there.
OLD_FORECAST = {"weights": 72320, "runtime": 2147483648, "kv": 262144, "activations": 655360, "patches": 0,
                "temporary": 536943232, "eager_attention": 12582912, "total": 2697999616}
OLD_CURRENT_SHA256 = "ba606c974a6c2310c7223e3ed913424f5a88674f0df5078ae00ce4d6e9f228e4"
OLD_CANDIDATE_SHA256 = "81e72e2d5f18e8a9df310b24fc3ad3330ebc158a88e8e0b609269f77f58b6cd4"


def build_llama(directory, dtype=torch.float32):
    """The tiny llama of tests/test_hf.py (same sizes, seed and scaling), saved in ``dtype``."""
    import transformers as T
    torch.manual_seed(0)
    m = T.LlamaForCausalLM(T.LlamaConfig(vocab_size=V, hidden_size=D_MODEL, intermediate_size=48, num_hidden_layers=LAYERS,
                                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
                                         tie_word_embeddings=False))
    with torch.no_grad():                        # default init is tiny; make every component matter
        for name, p in m.named_parameters():
            if p.dim() >= 2:
                p.mul_(6.0)
            elif "bias" in name:
                p.add_(0.1 * torch.randn_like(p))
    m.to(dtype).save_pretrained(directory, safe_serialization=True)
    from tokenizers import Tokenizer, models, pre_tokenizers
    tk = Tokenizer(models.WordLevel({f"t{i}": i for i in range(V)}, unk_token="t0"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.save(os.path.join(directory, "tokenizer.json"))
    return str(directory)


def open_ws(root, ckpt_dtype=torch.float32):
    ckpt = build_llama(root / "ckpt", ckpt_dtype)
    s = Session(str(root / "ws"))
    os.makedirs(s.root, exist_ok=True)
    s.ws = zoo.open_hf(s.root, ckpt, name="m")
    return s.ws


def run_env(monkeypatch, dtype, mode=None):
    monkeypatch.setenv("WSP_RUN_DTYPE", dtype)
    if mode is None:
        monkeypatch.delenv("WSP_PATCH_APPLY", raising=False)
    else:
        monkeypatch.setenv("WSP_PATCH_APPLY", mode)


def full(shape):
    return tuple(range(n) for n in shape)


def float32_pieces(ws):
    """Pieces of every kind on a linear weight, a second linear weight and a norm weight (float32 deltas)."""
    g = torch.Generator().manual_seed(3)
    so, sd, sn = ws.shape(O), ws.shape(DOWN), ws.shape(NORM)
    return {
        O: [Dl.block(full(so), 0.3 * torch.randn(so, generator=g)),
            Dl.lowrank((range(2, 9), range(4, 30)), torch.randn(7, 2, generator=g), torch.randn(2, 26, generator=g)),
            Dl.Piece("sparse", full(so), values=torch.randn(5, generator=g),
                     coords=torch.tensor([[0, 1], [3, 3], [7, 30], [31, 0], [12, 12]]))],
        DOWN: [Dl.lowrank(full(sd), torch.randn(sd[0], 1, generator=g), torch.randn(1, sd[1], generator=g)),
               Dl.block((range(5, 6), range(0, sd[1])), torch.randn(1, sd[1], generator=g))],
        NORM: [Dl.block(full(sn), 0.01 * torch.randn(sn, generator=g))],
    }


def float32_extra(ws):
    g = torch.Generator().manual_seed(4)
    so, sn = ws.shape(O), ws.shape(NORM)
    return {O: [Dl.lowrank(full(so), torch.randn(so[0], 1, generator=g), torch.randn(1, so[1], generator=g))],
            NORM: [Dl.block((range(0, 8),), 0.02 * torch.randn(8, generator=g))]}


def commit_float32_patch(ws):
    """The float32 test's committed patch, at strength 0.7, through the workspace's one edit path."""
    p = ws.commit(float32_pieces(ws), "bf16 test patch")
    ws.set_patch(p.id, strength=0.7)
    return p


def digest(tensors: dict) -> str:
    h = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name].detach().contiguous().cpu()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(t.view(torch.uint8).reshape(-1).numpy().tobytes() if t.dtype != torch.bool else t.numpy().tobytes())
    return h.hexdigest()


def patched(runner, names=(O, DOWN, NORM)):
    return {n: runner.param(n).detach().clone() for n in names}


def module_io(runner, module, ids, variant, extra=None):
    """(input, output) of ``module`` in one run of ``ids``. The recorder is registered after the variant is set, so
    it runs after the side path's hook (forward hooks run in registration order), as a capture hook does."""
    runner.use(variant, extra)
    seen = {}
    h = module.register_forward_hook(lambda m, a, o: seen.update(x=a[0].detach().clone(), y=o.detach().clone()))
    try:
        runner.run(ids, variant, extra)
    finally:
        h.remove()
    return seen["x"], seen["y"]


def hook_count(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def header(directory) -> dict:
    """name -> shape, read from the safetensors files' own headers (not through the IDE's store)."""
    shapes = {}
    for f in sorted(os.listdir(directory)):
        if f.endswith(".safetensors"):
            with open(os.path.join(directory, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                meta = json.loads(fh.read(n))
            shapes.update({k: tuple(v["shape"]) for k, v in meta.items() if k != "__metadata__"})
    return shapes


# ------------------------------------------------------------------ B1-B4: how patches are applied in bfloat16
def test_the_side_path_is_pefts_order(tmp_path, monkeypatch):
    run_env(monkeypatch, "bfloat16")
    ws = open_ws(tmp_path, torch.bfloat16)
    r = ws.runner
    assert r.dtype == torch.bfloat16 and r.patch_apply == "side"
    W = ws.read(O, (), "base")
    delta = 0.05 * float(W.abs().mean()) * torch.randn(W.shape, generator=torch.Generator().manual_seed(1))
    extra = {O: [Dl.block(full(W.shape), delta)]}
    mod = r.model.model.layers[0].self_attn.o_proj
    ids = ws.encode(PROMPT)
    x, y = module_io(r, mod, ids, "current", extra)
    assert mod.weight.dtype == torch.bfloat16 and torch.equal(mod.weight.float(), W)      # the weight is untouched
    want = (F.linear(x, mod.weight).float() + x.float() @ delta.T).to(torch.bfloat16)
    assert y.dtype == torch.bfloat16 and torch.equal(y, want)
    xb, yb = module_io(r, mod, ids, "base")
    assert torch.equal(yb, F.linear(xb, mod.weight)) and not torch.equal(yb, y)
    # A capture hook on the patched module (registered by run, after use) sees the patched output.
    real = r._hook_target
    monkeypatch.setattr(r, "_hook_target",
                        lambda name: (mod, "post", "o_proj_out") if name == "blocks.0.o_proj_out" else real(name))
    _, cache = r.run(ids, "current", extra, names=["blocks.0.o_proj_out"])
    assert torch.equal(cache["blocks.0.o_proj_out"], want.float())
    # A committed patch at strength 0.5 takes the same path, scaled (0.5 is exact in every rounding).
    Wd = ws.read(DOWN, (), "base")
    dd = 0.05 * float(Wd.abs().mean()) * torch.randn(Wd.shape, generator=torch.Generator().manual_seed(5))
    p = ws.commit({DOWN: [Dl.block(full(Wd.shape), dd)]}, "down")
    ws.set_patch(p.id, strength=0.5)
    down = r.model.model.layers[1].mlp.down_proj
    x, y = module_io(r, down, ids, "current")
    assert torch.equal(down.weight.float(), Wd)
    assert torch.equal(y, (F.linear(x, down.weight).float() + x.float() @ (0.5 * dd).T).to(torch.bfloat16))


def test_a_weight_two_modules_share_is_changed_in_place(tmp_path, monkeypatch):
    """A tied embedding is also the output head's weight: a hook on the head alone would leave the embedding
    unpatched, so in bfloat16 it is changed in place (rounded once) like any tensor that is not one linear weight."""
    import transformers as T
    run_env(monkeypatch, "bfloat16")
    torch.manual_seed(0)
    m = T.LlamaForCausalLM(T.LlamaConfig(vocab_size=V, hidden_size=D_MODEL, intermediate_size=48, num_hidden_layers=1,
                                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
                                         tie_word_embeddings=True))
    m.to(torch.bfloat16).save_pretrained(tmp_path / "ckpt", safe_serialization=True)
    from tokenizers import Tokenizer, models, pre_tokenizers
    tk = Tokenizer(models.WordLevel({f"t{i}": i for i in range(V)}, unk_token="t0"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.save(str(tmp_path / "ckpt" / "tokenizer.json"))
    s = Session(str(tmp_path / "ws"))
    os.makedirs(s.root, exist_ok=True)
    ws = zoo.open_hf(s.root, str(tmp_path / "ckpt"), name="m")
    emb = "model.embed_tokens.weight"
    assert emb in ws.names() and "lm_head.weight" not in ws.names()
    w = ws.read(emb, (), "base")
    delta = 0.1 * torch.randn(w.shape, generator=torch.Generator().manual_seed(6))
    ws.commit({emb: [Dl.block(full(w.shape), delta)]}, "emb")
    r = ws.runner
    assert r.model.lm_head.weight is r.model.model.embed_tokens.weight
    r.use("current")
    assert hook_count(r.model) == 0 and torch.equal(r.param(emb), (w + delta).to(torch.bfloat16))
    r.use("base")
    assert torch.equal(r.param(emb).float(), w)


def test_a_change_below_half_a_bfloat16_step_survives_only_the_side_path(tmp_path, monkeypatch):
    """Each element of the delta is 1e-4 of its weight, with a random sign: below half a bfloat16 step of every weight
    (2^-9 of it at least), so in place every weight rounds back. (A delta exactly proportional to W would change no
    bfloat16 output in either mode: the output then moves by 1e-4 of itself, also below half a step.) The side path
    moves only output elements small enough for the change to pass half their step, so the prompt is long (30
    tokens, about 1000 output values)."""
    long_prompt = " ".join(f"t{i}" for i in range(1, 31))
    outputs = {}
    for mode in ("in_place", "side"):
        run_env(monkeypatch, "bfloat16", mode)
        ws = open_ws(tmp_path / mode, torch.bfloat16)
        r = ws.runner
        assert r.patch_apply == mode
        W = ws.read(O, (), "base")
        signs = torch.randint(0, 2, W.shape, generator=torch.Generator().manual_seed(2)).float() * 2 - 1
        extra = {O: [Dl.block(full(W.shape), 1e-4 * W * signs)]}
        mod = r.model.model.layers[0].self_attn.o_proj
        ids = ws.encode(long_prompt)
        assert len(ids) == 30
        _, base = module_io(r, mod, ids, "base")
        _, cand = module_io(r, mod, ids, "current", extra)
        if mode == "in_place":
            assert torch.equal(r.param(O).float(), W)                     # every element rounded back
        outputs[mode] = (base, cand)
    base, cand = outputs["in_place"]
    assert torch.equal(cand, base)
    base, cand = outputs["side"]
    assert not torch.equal(cand, base)


def test_restore_leaves_weights_and_hooks_as_they_were(tmp_path, monkeypatch):
    from wsp import overlay as OV
    run_env(monkeypatch, "bfloat16")
    ws = open_ws(tmp_path, torch.bfloat16)
    commit_float32_patch(ws)
    r = ws.runner
    before = {n: p.detach().clone() for n, p in r.model.named_parameters()}
    assert hook_count(r.model) == 0
    built = []

    class Counting(OV.DeltaOverlay):
        def __init__(self, *a, **k):
            built.append(a)
            super().__init__(*a, **k)
    monkeypatch.setattr(OV, "DeltaOverlay", Counting)
    ids = ws.encode(PROMPT)
    extra = float32_extra(ws)

    def unchanged():
        return all(torch.equal(p, before[n]) for n, p in r.model.named_parameters())
    r.run(ids, "base")
    assert unchanged() and hook_count(r.model) == 0
    r.run(ids, "current", extra)
    # one hook per patched linear module (o_proj: committed and candidate overlays; down_proj: committed)
    assert hook_count(r.model) == 2 and len(built) == 3
    assert torch.equal(r.param(O), before[O]) and torch.equal(r.param(DOWN), before[DOWN])    # linear: beside
    assert not torch.equal(r.param(NORM), before[NORM])                                       # norm: in place
    r.run(ids, "base", extra)
    assert unchanged() and hook_count(r.model) == 0
    r.run(ids, "current", extra)
    assert hook_count(r.model) == 2 and len(built) == 3                    # same pieces: the overlays are kept
    r.run(ids, "base")                         # without the candidate (as after a job): its copy is dropped
    assert unchanged() and hook_count(r.model) == 0 and (O, "extra") not in r._overlays
    r.run(ids, "current", extra)
    assert hook_count(r.model) == 2 and len(built) == 4                    # only the candidate's is built again
    r.use("current")
    assert hook_count(r.model) == 2 and len(built) == 4 and (O, "extra") not in r._overlays
    r.use("base")
    assert unchanged() and hook_count(r.model) == 0


def test_other_tensors_are_rounded_once(tmp_path, monkeypatch):
    run_env(monkeypatch, "bfloat16")
    ws = open_ws(tmp_path, torch.bfloat16)
    w = ws.read(NORM, (), "base")                                          # the bfloat16 values, exactly
    # Half a bfloat16 step of each weight and a little more: rounding the delta to bfloat16 first leaves exactly half
    # a step (a tie, rounded to even); one rounding of the sum goes a whole step up.
    step = torch.exp2(torch.floor(torch.log2(w.abs())) - 7)
    delta = 0.5 * step * (1 + 2.0 ** -12)
    once = (w + delta).to(torch.bfloat16)
    twice = (w + delta.to(torch.bfloat16).float()).to(torch.bfloat16)
    assert (once != twice).any()                                           # the test tells the two apart
    ws.commit({NORM: [Dl.block(full(w.shape), delta)]}, "norm")
    r = ws.runner
    r.use("current")
    assert r.param(NORM).dtype == torch.bfloat16 and torch.equal(r.param(NORM), once)
    r.use("base")
    assert torch.equal(r.param(NORM).float(), w)


# ------------------------------------------------------------------ B5-B6: what is measured in bfloat16
def test_bfloat16_captures_are_the_models_values(tmp_path, monkeypatch):
    run_env(monkeypatch, "bfloat16")
    ws = open_ws(tmp_path, torch.bfloat16)
    commit_float32_patch(ws)                     # side path on two linear weights, a norm weight in place
    r = ws.runner
    ids = ws.encode(PROMPT)
    got = {}
    for variant in ("base", "current"):
        got[variant] = DR.residuals(ws, [ids], variant)[0]
        r.use(variant)                           # the weights (and side hooks) the residuals ran on
        with torch.no_grad():
            hs = r.model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
        assert hs[0].dtype == torch.bfloat16 and len(hs) == LAYERS + 1
        want = torch.stack([h[0, -1].float() for h in hs])
        assert got[variant].shape == want.shape and torch.equal(got[variant], want), variant
    assert not torch.equal(got["base"], got["current"])


def test_the_consistency_check_in_bfloat16_is_small_but_not_zero(tmp_path, monkeypatch):
    run_env(monkeypatch, "bfloat16")
    ws = open_ws(tmp_path, torch.bfloat16)
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    report = ARA.ablate(ws, {"method": "ara", "direction": "m", **SMALL}, add=lambda view, A, B: None)
    assert not report["skipped"] and len(report["components"]) == 2 * LAYERS
    for c in report["components"]:
        for k in ("mismatch_good", "mismatch_bad"):
            assert 0 < c[k] <= 5e-3, (c["layer"], c["component"], k, c[k])


# ------------------------------------------------------------------ B7-B8: the forecast and the switch
def test_the_forecast_counts_loaded_tensors_at_the_run_dtype(tmp_path, monkeypatch):
    run_env(monkeypatch, "float32")
    assert MG.forecast(open_ws(tmp_path / "dense")) == OLD_FORECAST       # a dense float32 workspace: as before
    ckpt = tiny_qwen35(tmp_path / "q" / "ckpt")
    s = Session(str(tmp_path / "q" / "ws"))
    os.makedirs(s.root, exist_ok=True)
    ws = zoo.open_hf(s.root, ckpt, name="m")
    shapes = header(ckpt)
    text = sum(math_prod(v) for n, v in shapes.items() if not n.startswith(("model.visual.", "mtp.")))
    assert 0 < text < ws.n_params()                                       # the checkpoint has a vision tower
    monkeypatch.setenv("WSP_RUN_DTYPE", "bfloat16")
    assert MG.forecast(ws)["weights"] == RL.loaded_params(ws) * 2 == text * 2
    monkeypatch.setenv("WSP_RUN_DTYPE", "float32")
    assert MG.forecast(ws)["weights"] == text * 4
    # The runner's own memory check counts the same tensors: room for them (not for the vision tower too) loads.
    monkeypatch.setenv("WSP_RUN_DTYPE", "bfloat16")
    need = text * 2 * 1.15
    assert int(need) + 1 < ws.n_params() * 2 * 1.15
    monkeypatch.setattr(RN, "available_memory", lambda: int(need) + 1)
    r = RN.HFRunner(ws)
    assert r.dtype == torch.bfloat16
    r.release()
    monkeypatch.setattr(RN, "available_memory", lambda: int(need) - 1)
    with pytest.raises(RuntimeError, match="needs about"):
        RN.HFRunner(ws)


def math_prod(shape):
    n = 1
    for k in shape:
        n *= int(k)
    return n


def test_an_unknown_patch_mode_is_refused(tmp_path, monkeypatch):
    run_env(monkeypatch, "bfloat16", "sideways")
    ws = open_ws(tmp_path, torch.bfloat16)

    def loaded(*a, **k):
        raise AssertionError("the model was loaded before the switch was checked")
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", loaded)
    with pytest.raises(RuntimeError) as e:
        RN.HFRunner(ws)
    assert str(e.value) == "WSP_PATCH_APPLY must be in_place or side"
    for value, dtype, want in ((None, torch.bfloat16, "side"), (None, torch.float16, "side"), (None, torch.float32, "in_place"),
                               (None, torch.float64, "in_place"), ("in_place", torch.bfloat16, "in_place"),
                               ("side", torch.float32, "side")):
        if value is None:
            monkeypatch.delenv("WSP_PATCH_APPLY", raising=False)
        else:
            monkeypatch.setenv("WSP_PATCH_APPLY", value)
        assert RN.patch_apply(dtype) == want


# ------------------------------------------------------------------ B9-B11: float32 and GGUF unchanged
def test_float32_application_is_bitwise_unchanged(tmp_path, monkeypatch):
    from wsp.store import region_index
    run_env(monkeypatch, "float32")
    ws = open_ws(tmp_path)
    commit_float32_patch(ws)
    r = ws.runner
    assert r.dtype == torch.float32 and r.patch_apply == "in_place"

    def old(extra=None):
        """The code before this change, written out: each piece added in place to the float32 tensor, in order."""
        out = {}
        for n in (O, DOWN, NORM):
            t = ws.read(n, (), "base").clone()
            for piece, scale in list(ws.pieces(n)) + [(p, 1.0) for p in (extra or {}).get(n, [])]:
                if piece.kind == "sparse":
                    ix = tuple(piece.coords.T)
                    t[ix] += (scale * piece.values).to(torch.float32)
                else:
                    ix = region_index(piece.region)
                    t[ix] += (scale * piece.dense()).to(torch.float32)
            out[n] = t
        return out
    r.use("current")
    got, want = patched(r), old()
    assert all(torch.equal(got[n], want[n]) for n in want) and digest(got) == OLD_CURRENT_SHA256
    assert hook_count(r.model) == 0
    extra = float32_extra(ws)
    r.use("current", extra)
    got, want = patched(r), old(extra)
    assert all(torch.equal(got[n], want[n]) for n in want) and digest(got) == OLD_CANDIDATE_SHA256
    assert hook_count(r.model) == 0 and not r._overlays
    r.use("base")
    assert all(torch.equal(r.param(n), ws.read(n, (), "base")) for n in (O, DOWN, NORM))


@pytest.mark.parametrize("dtype,mode,apply", [("float32", None, None), ("float32", "in_place", None),
                                               ("float32", "side", "side"), ("bfloat16", None, None),
                                               ("bfloat16", "side", None), ("bfloat16", "in_place", "in_place")])
def test_float32_runner_description_and_kl_key_are_unchanged(tmp_path, monkeypatch, dtype, mode, apply):
    run_env(monkeypatch, dtype, mode)
    ws = open_ws(tmp_path, torch.bfloat16 if dtype == "bfloat16" else torch.float32)
    ws.runner
    want = {"class": "HFRunner", "dtype": f"torch.{dtype}", "device": "cpu"}
    key_runner = {"class": "HFRunner", "dtype": f"torch.{dtype}", "device": "cpu", "backend": None}
    if apply is not None:
        want["apply"] = key_runner["apply"] = apply
    got = DR.runner_description(ws)
    assert json.dumps(got) == json.dumps(want)                              # same keys, same order
    assert DS.reference_key(ws, [[1, 2]])[1]["runner"] == key_runner
    block = DS.generation_block(ws, 16)
    if dtype == "float32" and apply is None:
        assert set(block) == {"runner", "where", "generation", "first_token", "max_response_length", "settings"}
    elif dtype == "bfloat16" and apply is None:
        assert block["patches"] == SIDE_TEXT
    elif dtype == "bfloat16":
        assert block["patches"] == "added to the weights in place, rounded once to bfloat16"
    else:
        assert block["patches"] == "float32 side path on linear layers (PEFT's order); other tensors rounded once to float32"


def test_the_overlay_names_still_import_from_gguf_runner():
    from wsp import gguf_runner as G
    from wsp import overlay as OV
    for name in ("DeltaOverlay", "piece_key", "delta_matmul", "GATHER_MIN", "_axis", "_take", "_put", "_index", "_spans",
                 "_owned"):
        assert getattr(G, name) is getattr(OV, name), name


# ------------------------------------------------------------------ the attention kernel (bfloat16: the default)
PATTERN_NAMES = [f"blocks.{l}.attn.hook_pattern" for l in range(LAYERS)]


def reference(ckpt, dtype, eager):
    """The same checkpoint loaded by transformers alone: with eager attention, or without attn_implementation."""
    kw = {"attn_implementation": "eager"} if eager else {}
    return transformers.AutoModelForCausalLM.from_pretrained(str(ckpt), dtype=dtype, **kw).eval()


def ref_logits(model, ids):
    """(the full logits of a plain forward pass, the first-token logits as HFRunner.first_token_logits computes them)."""
    t = torch.tensor([ids])
    with torch.no_grad():
        full_ = model(input_ids=t).logits[0].float()
        first = model(input_ids=t, attention_mask=torch.ones_like(t), use_cache=False, logits_to_keep=1).logits[0, -1].float()
    return full_, first


def attn_settings(model):
    """Every config of the model and what it says about attention: what a pattern run must leave as it was."""
    out, todo, seen = [], [model.config] + [getattr(m, "config", None) for m in model.modules()], set()
    while todo:
        c = todo.pop()
        if c is None or id(c) in seen:
            continue
        seen.add(id(c))
        out.append((type(c).__name__, c._attn_implementation, c.__dict__.get("_attn_was_changed", "absent")))
        todo += [getattr(c, k, None) for k in getattr(c, "sub_configs", None) or ()]
    return sorted(out, key=repr)


PROMPTS = ["t3 t7 t11 t5 t9 t2", "t1 t2 t3 t4", "t30 t12 t8 t8 t19 t25 t6 t2 t2 t14"]


def test_bfloat16_runs_on_the_default_attention_and_float32_on_eager(tmp_path, monkeypatch):
    for dtype, name in ((torch.bfloat16, "bfloat16"), (torch.float32, "float32")):
        run_env(monkeypatch, name)
        ws = open_ws(tmp_path / name, dtype)
        r = ws.runner
        ref = reference(tmp_path / name / "ckpt", dtype, eager=(dtype == torch.float32))
        if dtype == torch.bfloat16:
            assert ref.config._attn_implementation != "eager" and r.attention == ref.config._attn_implementation
        else:
            assert r.attention == "eager" == ref.config._attn_implementation
        assert r.model.config._attn_implementation == r.attention
        for text in PROMPTS:
            ids = ws.encode(text)
            full_, first = ref_logits(ref, ids)
            assert torch.equal(r.first_token_logits(ids, "base"), first), (name, text)
            assert torch.equal(r.logits(ids, "base"), full_), (name, text)


def test_the_two_bfloat16_kernels_differ_on_the_fixture(tmp_path):
    ckpt = build_llama(tmp_path / "ckpt", torch.bfloat16)
    eager, default = reference(ckpt, torch.bfloat16, True), reference(ckpt, torch.bfloat16, False)
    assert eager.config._attn_implementation == "eager" != default.config._attn_implementation
    tok_ids = [[int(t[1:]) for t in text.split()] for text in PROMPTS]
    differ = [not torch.equal(ref_logits(eager, ids)[0], ref_logits(default, ids)[0]) for ids in tok_ids]
    assert any(differ)                    # so the bit-equality of the previous test tells the two kernels apart


def test_a_bfloat16_pattern_run_uses_eager_and_switches_back(tmp_path, monkeypatch):
    run_env(monkeypatch, "bfloat16")
    ws = open_ws(tmp_path, torch.bfloat16)
    r = ws.runner
    eager, default = reference(tmp_path / "ckpt", torch.bfloat16, True), reference(tmp_path / "ckpt", torch.bfloat16, False)
    before = attn_settings(r.model)
    ids = ws.encode(PROMPT)
    logits, cache = r.run(ids, "base", names=PATTERN_NAMES)
    with torch.no_grad():
        want = eager(input_ids=torch.tensor([ids]), output_attentions=True)
    assert sorted(cache) == PATTERN_NAMES
    for l in range(LAYERS):
        assert torch.equal(cache[f"blocks.{l}.attn.hook_pattern"], want.attentions[l].float()), l
    assert torch.equal(logits[0], want.logits[0].float())                 # the pattern pass ran on eager
    assert attn_settings(r.model) == before and r.model.config._attn_implementation == r.attention != "eager"
    full_, first = ref_logits(default, ids)
    assert torch.equal(r.logits(ids, "base"), full_) and torch.equal(r.first_token_logits(ids, "base"), first)


def test_an_error_in_a_pattern_run_still_switches_back(tmp_path, monkeypatch):
    run_env(monkeypatch, "bfloat16")
    ws = open_ws(tmp_path, torch.bfloat16)
    r = ws.runner
    default = reference(tmp_path / "ckpt", torch.bfloat16, False)
    before = attn_settings(r.model)
    real, seen = r.model.forward, []

    def fail_once(*a, **k):
        if not seen:
            seen.append(r.model.config._attn_implementation)
            raise RuntimeError("forward failed")
        return real(*a, **k)
    monkeypatch.setattr(r.model, "forward", fail_once)
    ids = ws.encode(PROMPT)
    with pytest.raises(RuntimeError, match="forward failed"):
        r.run(ids, "base", names=PATTERN_NAMES)
    assert seen == ["eager"]                                              # it had switched for the pass
    assert attn_settings(r.model) == before and r.model.config._attn_implementation == r.attention != "eager"
    assert torch.equal(r.logits(ids, "base"), ref_logits(default, ids)[0])


def test_a_float32_pattern_run_is_unchanged(tmp_path, monkeypatch):
    run_env(monkeypatch, "float32")
    ws = open_ws(tmp_path)
    r = ws.runner
    assert r.attention == "eager"

    def switched(*a, **k):
        raise AssertionError("a float32 model is already on eager: nothing to switch")
    monkeypatch.setattr(r.model, "set_attn_implementation", switched)
    ref = reference(tmp_path / "ckpt", torch.float32, True)
    ids = ws.encode(PROMPT)
    logits, cache = r.run(ids, "base", names=PATTERN_NAMES)
    with torch.no_grad():
        want = ref(input_ids=torch.tensor([ids]), output_attentions=True)
    assert torch.equal(logits[0], want.logits[0].float())
    for l in range(LAYERS):
        assert torch.equal(cache[f"blocks.{l}.attn.hook_pattern"], want.attentions[l].float()), l


def test_a_bfloat16_pattern_trace_maps_the_full_attention_layer_on_hybrid(tmp_path, monkeypatch):
    run_env(monkeypatch, "bfloat16")
    ckpt = tiny_qwen35(tmp_path / "ckpt")
    s = Session(str(tmp_path / "ws"))
    os.makedirs(s.root, exist_ok=True)
    ws = zoo.open_hf(s.root, ckpt, name="m")
    r = ws.runner
    eager, default = reference(ckpt, torch.bfloat16, True), reference(ckpt, torch.bfloat16, False)
    assert r.attention == default.config._attn_implementation != "eager"
    before = attn_settings(r.model)
    ids = ws.encode(PROMPT)
    _, cache = r.run(ids, "base", names=[f"blocks.{l}.attn.hook_pattern" for l in range(4)])
    assert sorted(cache) == ["blocks.3.attn.hook_pattern"]                # the one full-attention layer
    seen = {}
    h = eager.model.layers[3].self_attn.register_forward_hook(lambda m, a, o: seen.__setitem__("w", o[1]))
    try:
        with torch.no_grad():
            eager(torch.tensor([ids]), output_attentions=True)
    finally:
        h.remove()
    assert tuple(cache["blocks.3.attn.hook_pattern"].shape) == (1, 2, len(ids), len(ids))
    assert torch.equal(cache["blocks.3.attn.hook_pattern"], seen["w"].float())
    assert attn_settings(r.model) == before and r.model.config._attn_implementation == r.attention
    assert torch.equal(r.logits(ids, "base"), ref_logits(default, ids)[0])
