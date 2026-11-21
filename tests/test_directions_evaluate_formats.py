"""Scoring a modified checkpoint (wsp.directions_evaluate) on the Qwen3.5 layouts of wsp.runtime_layouts and in a
bfloat16 run (wsp.runner's side path), model-free.

The tiny random Qwen3.5 checkpoints of tests/layout_fixtures.py (dense hybrid and MoE; both carry a vision tower,
``model.visual.*``, as the published multimodal checkpoints do), the tiny llama of tests/test_directions_bf16.py, and
the tiny model and helpers of tests/test_directions_evaluate.py. Modified copies are written here. Every expectation
is typed from the decisions (the note texts, which tensors are listed) or computed independently of the code under
test: logits and KL divergences of ``transformers`` models loaded here from the modified checkpoint and the base,
tensor differences read from the files, and the fused expert parameter compared with the files' expert tensors.

Tolerances: the candidate's (and the base reference's) raw first-token logits against the transformers reference's
with ``atol=1e-5, rtol=0``, as tests/test_directions_layouts.py compares the runner's logits with a transformers
reference on these fixtures; the KL divergence per prompt against the float64 KL of the two transformers models within
``1e-6 + 1e-4 KL``, the E5 bound of scripts/heretic_compare_evaluate.py (see assert_matches_reference for why not
ET13's ``rel=1e-5``).
"""
import json
import os
import shutil
import struct
import sys

import pytest
import torch

transformers = pytest.importorskip("transformers")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safetensors.torch import load_file, save_file  # noqa: E402

import wsp.runner  # noqa: E402
from layout_fixtures import EXPERTS, tiny_qwen35  # noqa: E402
from wsp import directions_evaluate as DE  # noqa: E402
from wsp import zoo  # noqa: E402
from wsp.session import Session  # noqa: E402

from test_directions_evaluate import hfpair, merged_change, modified, score_body  # noqa: E402
from test_steering import make_ws  # noqa: E402

PRE = "model.language_model.layers"
PROMPTS = ["t3 t7 t11 t5", "t9 t2 t4", "t1 t1 t8 t30 t6"]
KL = [{"plugin": "KLDivergence", "prompts": PROMPTS}]
VISION = "the vision tower"
MTP = "the multi-token prediction head"
TOKENIZER_NOTE = ("the modified checkpoint has no tokenizer file; it is scored with the base's tokenizer, as Heretic's "
                  "evaluate_model does")
# The compare keys of an answer before this change (test_directions_evaluate.py ET2 and the route section's example).
OLD_COMPARE = {"tensors", "differ", "changed", "of", "rows", "dtypes", "tied", "seconds", "held", "config",
               "generation_config", "tokenizer"}
OLD_ANSWER = {"check", "path", "version", "seconds", "compare", "rendering", "generation", "base", "modified", "changes",
              "datasets", "notes", "done", "total"}


@pytest.fixture(autouse=True)
def _float32(monkeypatch):
    monkeypatch.setenv("WSP_RUN_DTYPE", "float32")
    monkeypatch.delenv("WSP_PATCH_APPLY", raising=False)


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    root = tmp_path_factory.mktemp("ckpt")
    return {"dense": tiny_qwen35(root / "dense"), "moe": tiny_qwen35(root / "moe", moe=True)}


# ------------------------------------------------------------------ helpers (independent of the code under test)
def not_loaded_note(names, parts):
    """The X1 note, typed from the decision (one tensor: the singular text with its one part)."""
    k = len(names)
    if k == 1:
        (part,) = parts
        return (f"1 differing tensor belongs to a part the text model does not load ({part}) and is not scored: "
                f"{names[0]}")
    shown = ", ".join(names[:8]) + (f" and {k - 8} more" if k > 8 else "")
    return (f"{k} differing tensors belong to parts the text model does not load ({'; '.join(parts)}) and are not "
            f"scored: {shown}")


def side_note(dtype):
    """The X3 note, typed from the decision."""
    return (f"this run is in {dtype}: the checkpoint's changes are applied as float32 differences beside the {dtype} "
            f"base weights (the side path), not as the checkpoint's weights rounded to {dtype}, which is what loading it "
            f"would give")


def header(directory) -> dict:
    """name -> shape from the safetensors files' own headers."""
    shapes = {}
    for f in sorted(os.listdir(directory)):
        if f.endswith(".safetensors"):
            with open(os.path.join(directory, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                meta = json.loads(fh.read(n))
            shapes.update({k: tuple(v["shape"]) for k, v in meta.items() if k != "__metadata__"})
    return shapes


def weights_file(directory):
    files = [f for f in os.listdir(directory) if f.endswith(".safetensors")]
    assert files == ["model.safetensors"], files
    return os.path.join(directory, "model.safetensors")


def changed_copy(src, dst, change):
    """A copy of checkpoint directory ``src`` (config, generation config, tokenizer, template) whose tensors are
    ``change(name, tensor)``; -> (the directory, the names whose stored values differ, read back from both files)."""
    shutil.copytree(src, dst)
    before = load_file(weights_file(src))
    after = {n: change(n, t).contiguous() for n, t in before.items()}
    save_file(after, weights_file(dst), metadata={"format": "pt"})
    again = load_file(weights_file(dst))
    differ = sorted(n for n in before if not torch.equal(before[n], again[n]))
    return str(dst), differ


def scaling(factors):
    return lambda n, t: t * factors[n] if n in factors else t


def session(root, ckpt):
    s = Session(str(root))
    os.makedirs(s.root, exist_ok=True)
    s.ws = zoo.open_hf(s.root, ckpt, name="m")
    return s


def client(sess):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    return TestClient(create_app(sess, "http://127.0.0.1:0"), raise_server_exceptions=False)


def record_logits(runner, on_candidate=None):
    """Every first-token logits call of the runner: (variant, with a candidate?, ids, logits); ``on_candidate(extra)``
    runs while a candidate is applied (just after its call)."""
    seen = []
    real = runner.first_token_logits

    def rec(ids, variant="current", extra=None):
        out = real(ids, variant, extra)
        if extra and on_candidate is not None:
            on_candidate(extra)
        seen.append((variant, bool(extra), list(ids), out.detach().clone()))
        return out
    runner.first_token_logits = rec
    return seen


def reference(directory):
    return transformers.AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32,
                                                             attn_implementation="eager").eval()


def ref_logits(model, ids):
    with torch.no_grad():
        return model(torch.tensor([ids])).logits[0, -1].float()


def kl(base_logits, other_logits):
    lb = torch.log_softmax(base_logits.double(), -1)
    lm = torch.log_softmax(other_logits.double(), -1)
    return float((lb.exp() * (lb - lm)).sum())


def score(sess, path, **kw):
    r = client(sess).post("/api/directions/evaluate", json={"path": path, "max_response_length": 2, "scores": KL, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def assert_matches_reference(ws, out, seen, base_dir, other_dir):
    """The candidate's first-token logits are those of transformers running ``other_dir`` and the base reference's
    those of transformers running ``base_dir`` (atol 1e-5, rtol 0); each prompt's KL divergence is the float64 KL of
    the two transformers models within E5's bound of scripts/heretic_compare_evaluate.py, 1e-6 + 1e-4 KL (the
    checkpoint scored as a change against the merged model run directly). ET13's rel 1e-5 alone is below the scorer's
    float32 kl_div rounding for a KL under about 0.01 (seen here: 3.7e-8 at KL 6.4e-4, 1.0e-7 at KL 3.2e-3, with the
    logits within 1.2e-7 of the reference's)."""
    mb, mm = reference(base_dir), reference(other_dir)
    base = [x for x in seen if x[0] == "base"]
    cand = [x for x in seen if x[0] == "current" and x[1]]
    assert len(base) == len(cand) == len(PROMPTS)
    moved = 0.0
    for k, p in enumerate(PROMPTS):
        ids = ws.encode(p)
        lb, lm = ref_logits(mb, ids), ref_logits(mm, ids)
        assert base[k][2] == cand[k][2] == ids
        torch.testing.assert_close(base[k][3], lb, atol=1e-5, rtol=0)
        torch.testing.assert_close(cand[k][3], lm, atol=1e-5, rtol=0)
        got, want = out["modified"]["scores"][0]["per_prompt"][k], kl(lb, lm)
        assert want > 1e-5 and abs(got - want) <= 1e-6 + 1e-4 * want, (k, got, want)
        moved = max(moved, float((lm - lb).abs().max()))
    assert moved > 1e-3                                                     # the change matters


# ------------------------------------------------------------------ F1: the dense hybrid layout
def test_f1_a_hybrid_checkpoint_scores_as_transformers_runs_it(tmp_path, ckpts):
    out_proj, down = f"{PRE}.0.linear_attn.out_proj.weight", f"{PRE}.2.mlp.down_proj.weight"
    path, differ = changed_copy(ckpts["dense"], tmp_path / "merged", scaling({out_proj: 1.5, down: 0.5}))
    assert differ == sorted([out_proj, down])
    sess = session(tmp_path / "root", ckpts["dense"])
    ws = sess.ws
    seen = record_logits(ws.runner)
    out = score(sess, path)
    c = out["compare"]
    assert sorted(r["param"] for r in c["rows"]) == differ and c["differ"] == 2
    assert c["not_loaded"] == [] and c["tied"] == [] and c["held"]["tensors"] == 2
    assert c["reconstruction"] == {"inexact": 0, "max_abs": 0.0}
    assert out["modified"]["state"] == "candidate" and out["notes"] == []
    assert_matches_reference(ws, out, seen, ckpts["dense"], path)


# ------------------------------------------------------------------ F2: routed experts are slices of one parameter
def test_f2_routed_experts_are_applied_to_their_slices_only(tmp_path, ckpts):
    e0, e2 = f"{PRE}.1.mlp.experts.0.down_proj.weight", f"{PRE}.1.mlp.experts.2.down_proj.weight"
    shared = f"{PRE}.1.mlp.shared_expert.down_proj.weight"
    path, differ = changed_copy(ckpts["moe"], tmp_path / "merged", scaling({e0: 1.5, e2: 0.5, shared: 1.5}))
    assert differ == sorted([e0, e2, shared])
    other = load_file(weights_file(path))
    sess = session(tmp_path / "root", ckpts["moe"])
    ws = sess.ws
    r = ws.runner
    r.use("base")
    layer = r.model.model.layers[1].mlp
    base_down = {l: r.model.model.layers[l].mlp.experts.down_proj.detach().clone() for l in range(4)}
    base_gate_up = {l: r.model.model.layers[l].mlp.experts.gate_up_proj.detach().clone() for l in range(4)}
    base_shared = layer.shared_expert.down_proj.weight.detach().clone()
    during = []

    def check_applied(extra):
        assert sorted(extra) == differ                                     # one piece list per differing tensor
        fused = layer.experts.down_proj
        for e in range(EXPERTS):
            if e in (0, 2):
                assert torch.equal(fused[e], other[f"{PRE}.1.mlp.experts.{e}.down_proj.weight"]), e
            else:
                assert torch.equal(fused[e], base_down[1][e]), e           # bit-equal to the base
        for l in (0, 2, 3):
            assert torch.equal(r.model.model.layers[l].mlp.experts.down_proj, base_down[l]), l
        for l in range(4):
            assert torch.equal(r.model.model.layers[l].mlp.experts.gate_up_proj, base_gate_up[l]), l
        assert torch.equal(layer.shared_expert.down_proj.weight, other[shared])
        during.append(True)
    seen = record_logits(r, check_applied)
    out = score(sess, path)
    assert len(during) == len(PROMPTS)
    c = out["compare"]
    assert sorted(x["param"] for x in c["rows"]) == differ and c["not_loaded"] == [] and c["held"]["tensors"] == 3
    assert c["reconstruction"] == {"inexact": 0, "max_abs": 0.0} and out["notes"] == []
    assert_matches_reference(ws, out, seen, ckpts["moe"], path)
    # Afterwards the fused parameter is the base's again.
    for l in range(4):
        assert torch.equal(r.model.model.layers[l].mlp.experts.down_proj, base_down[l]), l
        assert torch.equal(r.model.model.layers[l].mlp.experts.gate_up_proj, base_gate_up[l]), l
    assert torch.equal(layer.shared_expert.down_proj.weight, base_shared)


# ------------------------------------------------------------------ F3: a tensor the text model does not load
def visual_names(ckpt):
    return sorted(n for n in header(ckpt) if n.startswith("model.visual."))


def test_f3_a_vision_tower_tensor_is_compared_but_not_scored(tmp_path, ckpts):
    visual = "model.visual.blocks.0.attn.qkv.weight"
    assert visual in visual_names(ckpts["dense"])                        # the fixture's vision tower
    text = f"{PRE}.3.self_attn.o_proj.weight"
    path, differ = changed_copy(ckpts["dense"], tmp_path / "merged", scaling({visual: 1.5, text: 0.5}))
    assert differ == sorted([visual, text])
    sess = session(tmp_path / "root", ckpts["dense"])
    ws = sess.ws
    seen = record_logits(ws.runner)
    out = score(sess, path)                                              # nothing raises
    c = out["compare"]
    assert sorted(r["param"] for r in c["rows"]) == differ and c["differ"] == 2
    assert c["changed"] == sum(r["changed"] for r in c["rows"]) and c["changed"] > header(ckpts["dense"])[text][0]
    assert c["not_loaded"] == [visual] and c["held"]["tensors"] == 1
    assert out["notes"] == [not_loaded_note([visual], [VISION])] == [
        "1 differing tensor belongs to a part the text model does not load (the vision tower) and is not scored: "
        "model.visual.blocks.0.attn.qkv.weight"]
    assert out["modified"]["state"] == "candidate"
    cands = [x for x in seen if x[1]]
    assert cands                                                         # the text tensor is applied
    assert_matches_reference(ws, out, seen, ckpts["dense"], path)        # a reference that does not load the tower

    # Only the vision tower differs: nothing is applied, the scores are the base's.
    only, differ = changed_copy(ckpts["dense"], tmp_path / "only", scaling({visual: 1.5}))
    assert differ == [visual]
    seen.clear()
    out = score(sess, only)
    c = out["compare"]
    assert [r["param"] for r in c["rows"]] == [visual] and c["differ"] == 1
    assert c["not_loaded"] == [visual] and c["held"]["tensors"] == 0
    assert out["notes"] == [not_loaded_note([visual], [VISION])]
    assert out["modified"]["state"] == "current" and not any(x[1] for x in seen)
    kl_score = out["modified"]["scores"][0]
    assert kl_score["value"] == 0.0 and kl_score["per_prompt"] == [0.0] * len(PROMPTS)
    assert out["changes"][0]["change"] == 0.0
    mb = reference(ckpts["dense"])
    for (variant, _, ids, got) in [x for x in seen if x[0] == "current"]:
        torch.testing.assert_close(got, ref_logits(mb, ids), atol=1e-5, rtol=0)


# ------------------------------------------------------------------ F4: check mode lists it and loads no model
def test_f4_check_lists_tensors_not_loaded_without_a_model(tmp_path, ckpts, monkeypatch):
    visual = "model.visual.blocks.0.attn.qkv.weight"
    text = f"{PRE}.3.self_attn.o_proj.weight"
    path, differ = changed_copy(ckpts["dense"], tmp_path / "merged", scaling({visual: 1.5, text: 0.5}))
    sess = session(tmp_path / "root", ckpts["dense"])
    ws = sess.ws
    loads = []

    def boom(w):
        loads.append(w.name)
        raise AssertionError("a model was loaded")
    monkeypatch.setattr(wsp.runner, "make_runner", boom)
    out = DE.evaluate(ws, {"path": path, "check": True})
    assert out["check"] is True and "base" not in out
    assert out["compare"]["not_loaded"] == [visual] and out["compare"]["held"]["tensors"] == 1
    assert sorted(r["param"] for r in out["compare"]["rows"]) == differ
    assert out["notes"] == [not_loaded_note([visual], [VISION])]
    r = client(sess).post("/api/directions/evaluate", json={"path": path, "check": True})
    assert r.status_code == 200, r.text
    assert r.json()["compare"]["not_loaded"] == [visual] and r.json()["notes"] == out["notes"]
    assert ws._runner is None and loads == []
    # Two tensors of one part: the plural text, with that one part.
    proj = "model.visual.blocks.0.attn.proj.weight"
    assert proj in visual_names(ckpts["dense"])
    pair, differ = changed_copy(ckpts["dense"], tmp_path / "pair", scaling({visual: 1.5, proj: 0.5}))
    assert differ == sorted([visual, proj])
    out = DE.evaluate(ws, {"path": pair, "check": True})
    assert out["compare"]["not_loaded"] == sorted([visual, proj])
    assert out["notes"] == [not_loaded_note(sorted([visual, proj]), [VISION])] == [
        "2 differing tensors belong to parts the text model does not load (the vision tower) and are not scored: "
        "model.visual.blocks.0.attn.proj.weight, model.visual.blocks.0.attn.qkv.weight"]
    # Every tensor of the vision tower differs: at most 8 names, then the rest counted.
    names = visual_names(ckpts["dense"])
    assert len(names) > 8
    every, differ = changed_copy(ckpts["dense"], tmp_path / "every", lambda n, t: t + 0.25 if n in names else t)
    assert differ == names
    out = DE.evaluate(ws, {"path": every, "check": True})
    assert out["compare"]["not_loaded"] == names and out["compare"]["held"]["tensors"] == 0
    assert out["notes"] == [not_loaded_note(names, [VISION])]
    assert out["notes"][0].endswith(f" and {len(names) - 8} more")
    # Two parts, two reasons: a base with a multi-token prediction tensor (written here; the fixture has none).
    with_mtp = tmp_path / "with_mtp"
    shutil.copytree(ckpts["dense"], with_mtp)
    t = load_file(weights_file(with_mtp))
    t["mtp.fc.weight"] = torch.randn(32, 64, generator=torch.Generator().manual_seed(9))
    save_file(t, weights_file(with_mtp), metadata={"format": "pt"})
    two, differ = changed_copy(with_mtp, tmp_path / "two",
                               scaling({visual: 1.5, "mtp.fc.weight": 0.5, text: 0.5}))
    assert differ == sorted([visual, "mtp.fc.weight", text])
    s2 = session(tmp_path / "root2", str(with_mtp))
    out = DE.evaluate(s2.ws, {"path": two, "check": True})
    assert out["compare"]["not_loaded"] == [visual, "mtp.fc.weight"] and out["compare"]["held"]["tensors"] == 1
    assert out["notes"] == [not_loaded_note([visual, "mtp.fc.weight"], [VISION, MTP])] == [
        "2 differing tensors belong to parts the text model does not load (the vision tower; the multi-token "
        "prediction head) and are not scored: model.visual.blocks.0.attn.qkv.weight, mtp.fc.weight"]
    assert s2.ws._runner is None and loads == []


# ------------------------------------------------------------------ F5: a bfloat16 run says what is scored
def test_f5_a_bfloat16_run_says_the_changes_take_the_side_path(tmp_path, monkeypatch):
    from test_directions_bf16 import O, build_llama
    base = build_llama(tmp_path / "ckpt", torch.bfloat16)
    path, differ = changed_copy(base, tmp_path / "merged", scaling({O: 1.5}))
    assert differ == [O]
    monkeypatch.setenv("WSP_RUN_DTYPE", "bfloat16")
    bf = session(tmp_path / "bf", base)
    checked = DE.evaluate(bf.ws, {"path": path, "check": True})            # a check never has the note
    assert checked["notes"] == [] and bf.ws._runner is None
    out = score(bf, path)
    assert bf.ws.runner.patch_apply == "side" and bf.ws.runner.dtype == torch.bfloat16
    assert out["notes"] == [side_note("bfloat16")]
    assert out["modified"]["state"] == "candidate"
    monkeypatch.setenv("WSP_RUN_DTYPE", "float32")
    f32 = session(tmp_path / "f32", base)
    out = score(f32, path)
    assert f32.ws.runner.patch_apply == "in_place" and out["notes"] == []
    assert not any("this run is in" in n for n in out["notes"])


# ------------------------------------------------------------------ F6: a plain checkpoint's answer is as before
def test_f6_a_plain_checkpoint_answer_is_unchanged(tmp_path):
    sess = Session(str(tmp_path))
    make_ws(tmp_path / "rnd")
    sess.open("rnd")
    ws = sess.ws
    path = modified(tmp_path, ws, merged_change)
    check = DE.evaluate(ws, {"path": path, "check": True})               # no tokenizer file yet: the old note
    assert set(check) == {"check", "path", "input", "version", "compare", "notes", "seconds"}   # input: H4b, H4c
    assert set(check["compare"]) == OLD_COMPARE | {"not_loaded"} and check["compare"]["not_loaded"] == []
    assert check["notes"] == [TOKENIZER_NOTE]
    shutil.copy(os.path.join(ws.dir, "tokenizer_meta.json"), path)
    out = DE.evaluate(ws, score_body(path))
    assert set(out) == OLD_ANSWER | {"input"}
    assert set(out["compare"]) == OLD_COMPARE | {"reconstruction", "not_loaded"} and out["compare"]["not_loaded"] == []
    assert out["notes"] == []
    # A transformers layout without a runtime name table (llama): nothing is skipped.
    s, w, base, merged = hfpair(tmp_path / "hf")
    out = DE.evaluate(w, {"path": merged, "max_response_length": 2,
                          "scores": [{"plugin": "KLDivergence", "prompts": ["t3 t7", "t1"]}]})
    assert set(out) == OLD_ANSWER | {"input"} and out["compare"]["not_loaded"] == []
    assert [r["param"] for r in out["compare"]["rows"]] == ["model.layers.0.self_attn.o_proj.weight"]
    assert out["notes"] == []
