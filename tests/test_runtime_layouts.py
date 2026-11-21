"""Checkpoint names -> the names of the model transformers loads (wsp.runtime_layouts), and the runner's use of them.

The tiny Qwen3.5 checkpoints are built offline in the test's own directory (tests/layout_fixtures.py); no model is
downloaded. The claims are checked against what transformers actually loads, not against the table itself.
"""
import json
import math
import os
import struct
import sys

import pytest
import torch

transformers = pytest.importorskip("transformers")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layout_fixtures import tiny_qwen35  # noqa: E402
from wsp import runtime_layouts as RL  # noqa: E402
from wsp import zoo  # noqa: E402
from wsp.runner import HFRunner  # noqa: E402
from wsp.session import Session  # noqa: E402

# The published Qwen3.5-4B config.json, the keys that matter here (typed in; the multimodal layout).
M4_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "tie_word_embeddings": True,
    "image_token_id": 248056, "video_token_id": 248057, "vision_start_token_id": 248053, "vision_end_token_id": 248054,
    "text_config": {"model_type": "qwen3_5_text", "hidden_size": 2560, "num_hidden_layers": 32, "num_attention_heads": 16,
                    "num_key_value_heads": 4, "head_dim": 256, "full_attention_interval": 4, "attn_output_gate": True,
                    "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128, "mtp_num_hidden_layers": 1, "vocab_size": 248320},
    "vision_config": {"model_type": "qwen3_5", "depth": 24, "hidden_size": 1024, "out_hidden_size": 2560},
}
MOE_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",
    "text_config": {"model_type": "qwen3_5_moe_text", "num_experts": 256, "num_experts_per_tok": 8,
                    "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512},
    "vision_config": {"model_type": "qwen3_5_moe"},
}
VISUAL = "model.visual.blocks.0.attn.qkv.weight"
PROMPT = "t3 t7 t11 t5 t9 t2"


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


def text_names(shapes) -> list:
    return [n for n in shapes if not n.startswith(("model.visual.", "mtp."))]


def _open(root, moe):
    ckpt = tiny_qwen35(root / "ckpt", moe=moe)
    s = Session(str(root / "ws"))
    os.makedirs(s.root, exist_ok=True)
    s.ws = zoo.open_hf(s.root, ckpt, name="m")
    s.moe, s.ckpt = moe, ckpt
    return s


@pytest.fixture(scope="module")
def dense(tmp_path_factory):
    return _open(tmp_path_factory.mktemp("dense"), False)


@pytest.fixture(scope="module")
def moe(tmp_path_factory):
    return _open(tmp_path_factory.mktemp("moe"), True)


def _runner_finds_every_text_tensor(tiny, runner):
    """HFRunner (float32) on the checkpoint: every text tensor is found with the checkpoint's shape and values; a
    vision tensor is refused with its reason."""
    ws = tiny.ws
    assert runner.model.dtype == torch.float32
    names = text_names(header(tiny.ckpt))
    assert any(".linear_attn." in n for n in names) and any(".self_attn." in n for n in names)
    for n in names:
        p = runner.param(n)
        assert tuple(p.shape) == tuple(ws.shape(n)), n
        assert torch.equal(p, ws.read(n, (), "base")), n                    # the checkpoint's own values
    with pytest.raises(KeyError) as e:
        runner.param(VISUAL)
    assert e.value.args[0] == (f"checkpoint tensor {VISUAL} cannot be changed at run time: {VISUAL} belongs to the "
                               f"vision tower, which the text model the IDE runs does not load")


def test_the_text_model_type_is_read_from_text_config():
    assert RL.text_model_type({"model_type": "llama"}) == "llama"
    assert RL.text_config({"model_type": "llama"}) == {"model_type": "llama"}
    assert RL.text_model_type(M4_CONFIG) == "qwen3_5_text"
    assert RL.text_config(M4_CONFIG) is M4_CONFIG["text_config"]
    assert RL.text_model_type(MOE_CONFIG) == "qwen3_5_moe_text"
    assert RL.text_model_type({"model_type": "qwen3_5", "text_config": {"hidden_size": 8}}) == "qwen3_5"
    assert RL.text_model_type({"model_type": "x", "text_config": "not a dict"}) == "x"
    assert RL.text_model_type({}) == ""
    assert set(RL.HYBRID_TEXT_TYPES) == {"qwen3_5_text", "qwen3_5_moe_text"}


def test_the_name_table_imports_no_torch():
    import ast
    tree = ast.parse(open(RL.__file__).read())
    top = {a.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for a in node.names}
    top |= {(node.module or "").split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert top and "torch" not in top and "transformers" not in top


def test_language_model_names_map_to_the_loaded_model():
    assert RL.runtime_name(M4_CONFIG, "model.language_model.layers.3.self_attn.o_proj.weight", (2560, 4096)) == \
        ("model.layers.3.self_attn.o_proj.weight", ())
    assert RL.runtime_name(M4_CONFIG, "model.language_model.embed_tokens.weight", (248320, 2560)) == \
        ("model.embed_tokens.weight", ())
    assert RL.runtime_name(M4_CONFIG, "lm_head.weight", (248320, 2560)) == ("lm_head.weight", ())
    assert RL.module_path(M4_CONFIG, "model.language_model.layers.3") == "model.layers.3"
    assert RL.module_path(M4_CONFIG, "model.language_model.layers.0.linear_attn.out_proj") == \
        "model.layers.0.linear_attn.out_proj"
    assert RL.module_path(M4_CONFIG, "lm_head") == "lm_head"
    # A text-only Qwen3.5 (a GGUF base's config, or a saved Qwen3_5ForCausalLM) already has the runtime names.
    flat = M4_CONFIG["text_config"]
    assert RL.runtime_name(flat, "model.layers.1.linear_attn.out_proj.weight", (2560, 4096)) == \
        ("model.layers.1.linear_attn.out_proj.weight", ())
    assert RL.module_path(flat, "model.layers.1") == "model.layers.1"


def test_vision_and_mtp_tensors_are_skipped_with_their_reason(monkeypatch):
    for cfg in (M4_CONFIG, MOE_CONFIG):
        with pytest.raises(RL.Skipped) as e:
            RL.runtime_name(cfg, VISUAL, (3072, 1024))
        assert e.value.reason == (f"{VISUAL} belongs to the vision tower, which the text model the IDE runs does not load")
        assert isinstance(e.value, KeyError)
        with pytest.raises(RL.Skipped) as e:
            RL.runtime_name(cfg, "mtp.fc.weight", (2560, 5120))
        assert e.value.reason == ("mtp.fc.weight belongs to the multi-token prediction head, which the text model the IDE "
                                  "runs does not load")
    # The skip prefixes are tested on the checkpoint name, not on the name after the prefix rule.
    monkeypatch.setitem(RL.LAYOUTS, "test_layout", {"prefix": (("a.", "b."), ("c.", "d.")),
                                                    "skip": (("a.", "part a"), ("d.", "part d")), "experts": ()})
    cfg = {"model_type": "test_layout"}
    with pytest.raises(RL.Skipped, match="^a.w belongs to part a,"):
        RL.runtime_name(cfg, "a.w", (1,))
    assert RL.runtime_name(cfg, "c.w", (1,)) == ("d.w", ())


def test_expert_tensors_map_to_slices_of_the_fused_parameters():
    I, d = 8, 32
    pre = "model.language_model.layers.0.mlp"
    assert RL.runtime_name(MOE_CONFIG, f"{pre}.experts.3.down_proj.weight", (d, I)) == \
        ("model.layers.0.mlp.experts.down_proj", (3,))
    assert RL.runtime_name(MOE_CONFIG, f"{pre}.experts.3.gate_proj.weight", (I, d)) == \
        ("model.layers.0.mlp.experts.gate_up_proj", (3, slice(0, I)))
    assert RL.runtime_name(MOE_CONFIG, f"{pre}.experts.3.up_proj.weight", (I, d)) == \
        ("model.layers.0.mlp.experts.gate_up_proj", (3, slice(I, 2 * I)))
    assert RL.runtime_name(MOE_CONFIG, f"{pre}.experts.12.up_proj.weight", (I, d)) == \
        ("model.layers.0.mlp.experts.gate_up_proj", (12, slice(I, 2 * I)))
    # The shared expert and the routers are ordinary modules: only the prefix rule applies.
    for leaf in ("shared_expert.down_proj.weight", "gate.weight", "shared_expert_gate.weight"):
        assert RL.runtime_name(MOE_CONFIG, f"{pre}.{leaf}", (d, I)) == (f"model.layers.0.mlp.{leaf}", ())
    for model_type in ("qwen3_moe", "qwen2_moe"):
        cfg = {"model_type": model_type}
        assert RL.runtime_name(cfg, "model.layers.1.mlp.experts.0.up_proj.weight", (I, d)) == \
            ("model.layers.1.mlp.experts.gate_up_proj", (0, slice(I, 2 * I)))
        assert RL.runtime_name(cfg, "model.layers.1.mlp.experts.0.down_proj.weight", (d, I)) == \
            ("model.layers.1.mlp.experts.down_proj", (0,))
    # A dense Qwen3.5 has no expert rules.
    assert RL.runtime_name(M4_CONFIG, f"{pre}.experts.3.down_proj.weight", (d, I)) == \
        ("model.layers.0.mlp.experts.3.down_proj.weight", ())


def test_an_unknown_model_type_is_identity():
    cfg = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
    for name in ("model.layers.0.self_attn.o_proj.weight", "lm_head.weight", "model.language_model.layers.0.mlp.down_proj.weight",
                 VISUAL, "mtp.fc.weight", "model.layers.0.mlp.experts.3.down_proj.weight"):
        assert RL.runtime_name(cfg, name, (4, 4)) == (name, ())
        assert RL.runtime_name({}, name, (4, 4)) == (name, ())
    assert RL.module_path(cfg, "model.language_model.layers.0") == "model.language_model.layers.0"
    assert RL.module_path(M4_CONFIG, None) is None


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_loaded_params_leave_out_skipped_tensors(kind, request):
    tiny = request.getfixturevalue(kind)
    ws = tiny.ws
    shapes = header(tiny.ckpt)
    want = sum(math.prod(shapes[n]) for n in text_names(shapes))
    assert RL.loaded_params(ws) == want
    assert RL.loaded_params(ws) == (36_848 if tiny.moe else 39_280)            # pinned: the tiny model's count
    visual = sum(math.prod(s) for n, s in shapes.items() if n.startswith("model.visual."))
    assert visual > 0 and ws.n_params() == want + visual
    assert len(text_names(shapes)) < len(shapes)


def test_the_ide_finds_every_text_tensor_of_a_hybrid_checkpoint(dense, monkeypatch):
    monkeypatch.setenv("WSP_RUN_DTYPE", "float32")
    ws = dense.ws
    runner = HFRunner(ws)
    try:
        _runner_finds_every_text_tensor(dense, runner)
        assert ws.struct.n_layers == 4
        for l, L in enumerate(ws.struct.layers):
            assert runner.module(L.modules["layer"]) is runner.model.model.layers[l]
    finally:
        runner.release()


def test_the_ide_finds_every_expert_tensor_of_a_moe_checkpoint(moe, monkeypatch):
    """Each routed expert's checkpoint tensor is its slice of transformers' fused parameter (values equal, so gate is
    the first half of gate_up_proj and up the second), and a patch on one changes exactly that slice and is undone.
    The layer structure of a MoE checkpoint is recognised by wsp.arch only later; this checks the names alone."""
    monkeypatch.setenv("WSP_RUN_DTYPE", "float32")
    ws = moe.ws
    runner = HFRunner(ws)
    try:
        _runner_finds_every_text_tensor(moe, runner)
        name = "model.language_model.layers.2.mlp.experts.3.up_proj.weight"
        fused = runner.model.model.layers[2].mlp.experts.gate_up_proj
        assert tuple(fused.shape) == (4, 16, 32) and runner.param(name).data_ptr() == fused[3, 8:].data_ptr()
        before = fused.detach().clone()
        piece = ws.ref(name).lowrank_piece(torch.randn(8, generator=torch.Generator().manual_seed(1)),
                                           torch.randn(32, generator=torch.Generator().manual_seed(2)))
        runner.use("current", extra={name: [piece]})
        after = fused.detach().clone()
        assert torch.equal(after[3, 8:], before[3, 8:] + piece.dense())
        outside = torch.ones_like(after, dtype=torch.bool)
        outside[3, 8:] = False
        assert torch.equal(after[outside], before[outside])
        runner.use("base")
        assert torch.equal(fused.detach(), before)
    finally:
        runner.release()


def test_a_patch_on_a_hybrid_checkpoint_changes_the_model_as_its_weights_say(tmp_path, monkeypatch):
    monkeypatch.setenv("WSP_RUN_DTYPE", "float32")
    ckpt = tiny_qwen35(tmp_path / "ckpt")
    s = Session(str(tmp_path / "ws"))
    os.makedirs(s.root)
    s.ws = ws = zoo.open_hf(s.root, ckpt, name="m")
    name = "model.language_model.layers.1.linear_attn.out_proj.weight"
    ids = ws.encode(PROMPT)
    base = ws.runner.run(ids)[0][0]
    r = s.apply(f'rank1(P("{name}"), randn({ws.shape(name)[0]}, seed=1), randn({ws.shape(name)[1]}, seed=2))',
                enforce_scope=False)
    assert r["ok"] and r["committed"], r
    assert ws.touched() == [name] and [p.kind for p, _ in ws.pieces(name)] == ["lowrank"]
    current = ws.runner.run(ids)[0][0]
    ref = transformers.Qwen3_5ForCausalLM.from_pretrained(ckpt, dtype=torch.float32, attn_implementation="eager").eval()
    assert type(ref) is transformers.Qwen3_5ForCausalLM
    with torch.no_grad():
        want_base = ref(torch.tensor([ids])).logits[0]
        ref.model.layers[1].linear_attn.out_proj.weight.copy_(ws.read(name))
        want = ref(torch.tensor([ids])).logits[0]
    torch.testing.assert_close(base, want_base, atol=1e-5, rtol=0)
    torch.testing.assert_close(current, want, atol=1e-5, rtol=0)
    assert float((want - want_base).abs().max()) > 1e-2                         # the patch matters
    assert torch.equal(ws.runner.run(ids, variant="base")[0][0], base)          # and switching back restores it
