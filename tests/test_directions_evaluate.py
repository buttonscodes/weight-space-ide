"""Scoring a modified checkpoint against its base (wsp.directions_evaluate, Heretic's evaluate_model), model-free.

Tiny random models of this project (Config + init_params, no training), a modified copy of their weights written
by the test, the fake quantized store of tests/selection_fixtures.py, and tiny random transformers checkpoints
built in the test's own directory by tests/test_hf.py:build. Every expectation is computed here independently of
the code under test: tensor differences from the files themselves, greedy loops and KL divergences from
`wsp.model.forward`, and transformers' own `generate` and logits for the transformers pair.
"""
import json
import math
import os
import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file

import wsp.runner
from wsp import delta as Dl
from wsp import directions_evaluate as DE
from wsp import directions_scores as DS
from wsp import edit_limit
from wsp import workspace as W
from wsp.directions import DirectionError
from wsp.session import Session
from wsp.store import Store, region_size

from test_directions_scores import events, greedy, kl64, params_of
from test_steering import make_ws


@pytest.fixture()
def ws(tmp_path):
    return make_ws(tmp_path / "rnd")


@pytest.fixture()
def sess(tmp_path):
    make_ws(tmp_path / "rnd")
    s = Session(str(tmp_path))
    s.open("rnd")
    return s


def modified(tmp_path, ws, change=None, name="merged", dtypes=None):
    """<tmp>/<name>/model.safetensors from every base tensor, ``change(name, t)`` applied (None drops a tensor)."""
    d = tmp_path / name
    os.makedirs(d, exist_ok=True)
    out = {}
    for n in ws.names():
        t = ws.store.read(n)
        t = change(n, t) if change is not None else t
        if t is not None:
            out[n] = t.contiguous()
    for n, dt in (dtypes or {}).items():
        out[n] = out[n].to(dt)
    save_file(out, str(d / "model.safetensors"), metadata={"format": "pt"})
    return str(d)


def scaled(which, factor):
    return lambda n, t: t * factor if n == which else t


def hfpair(tmp_path, *, tied=False, change="model.layers.0.self_attn.o_proj.weight", factor=1.5):
    """A tiny random llama (tests/test_hf.py:build) and a copy whose ``change`` tensor is scaled by ``factor``
    (config, generation config and tokenizer copied); the base opened in Session(<tmp>/root)."""
    pytest.importorskip("transformers")
    from test_hf import build
    base, merged = tmp_path / "base", tmp_path / "merged"
    build("llama", str(base))
    if tied:
        with open(base / "config.json") as f:
            cfg = json.load(f)
        cfg["tie_word_embeddings"] = True
        with open(base / "config.json", "w") as f:
            json.dump(cfg, f, indent=1)
    shutil.copytree(base, merged)
    tensors = load_file(str(base / "model.safetensors"))
    if tied:
        assert "lm_head.weight" in tensors
        tensors.pop("lm_head.weight")
    tensors[change] = (tensors[change] * factor).contiguous()
    save_file(tensors, str(merged / "model.safetensors"), metadata={"format": "pt"})
    s = Session(str(tmp_path / "root"))
    os.makedirs(s.root, exist_ok=True)
    w = s.open(str(base))
    return s, w, str(base), str(merged)


def refused(ws, body, text):
    with pytest.raises(DirectionError) as e:
        DE.check_evaluate(ws, body)
    assert str(e.value) == text
    return e


PROMPTS = ["abc", "hello"]
ENTRIES = [{"plugin": "KeywordRate", "prompts": PROMPTS, "print_responses": True},
           {"plugin": "KLDivergence", "prompts": ["ab", "zz"]},
           {"plugin": "ResponseLength", "prompts": ["quiet"], "print_responses": True}]


def merged_change(n, t):
    if n == "blocks.1.mlp.W_out":
        return t * 1.5
    if n == "blocks.0.attn.W_O":
        return t * 0.5
    return t


def strip(scores):
    return [{k: v for k, v in s.items() if k != "reference"} for s in scores]


def score_body(path, **kw):
    return {"path": path, "max_response_length": 4, "scores": ENTRIES, **kw}


# ------------------------------------------------------------------ ET1. the checkpoint scores as itself
def test_the_modified_checkpoint_scores_as_itself_against_the_base(tmp_path, sess):
    ws = sess.ws
    path = modified(tmp_path, ws, merged_change)
    shutil.copy(os.path.join(ws.dir, "tokenizer_meta.json"), path)
    out = DE.evaluate(ws, score_body(path))
    assert out["check"] is False and out["path"] == path and out["version"] == ws.version
    assert out["modified"]["state"] == "candidate" and out["base"]["state"] == "base"
    assert [r["param"] for r in out["compare"]["rows"]] == ["blocks.0.attn.W_O", "blocks.1.mlp.W_out"]
    assert out["compare"]["reconstruction"] == {"inexact": 0, "max_abs": 0.0} and out["notes"] == []
    # The merged weights opened as their own workspace and scored as its base weights.
    merged = load_file(os.path.join(path, "model.safetensors"))
    make_ws(tmp_path / "two" / "rnd", params=merged)
    two = Session(str(tmp_path / "two"))
    two.open("rnd")
    there = DS.score(two.ws, {"state": "base", "max_response_length": 4, "scores": ENTRIES})
    got = strip(out["modified"]["scores"])
    assert [got[0], got[2]] == [strip(there["scores"])[0], strip(there["scores"])[2]]    # KL there is 0 by definition
    pb, pm = params_of(ws, "base"), params_of(two.ws, "base")
    want = [greedy(pm, ws.cfg, ws.encode(t), 4) for t in PROMPTS]
    assert [r["response"] for r in out["modified"]["scores"][0]["responses"]] == [ws.tok.decode(x) for x in want]
    kl = out["modified"]["scores"][1]
    assert kl["value"] > 1e-4
    for got, t in zip(kl["per_prompt"], ["ab", "zz"]):
        assert got == pytest.approx(kl64(pb, pm, ws.cfg, ws.encode(t)), rel=1e-6)
    # The base's scores are the workspace's own base scores.
    here = DS.score(ws, {"state": "base", "max_response_length": 4, "scores": ENTRIES})
    assert strip(out["base"]["scores"]) == strip(here["scores"])
    assert out["base"]["scores"][1]["display"] == "0 (by definition)"
    assert [r["response"] for r in out["base"]["scores"][0]["responses"]] == \
        [ws.tok.decode(greedy(pb, ws.cfg, ws.encode(t), 4)) for t in PROMPTS]
    assert out["changes"] == [{"name": b["name"], "base": b["value"], "modified": m["value"],
                               "change": m["value"] - b["value"]}
                              for b, m in zip(out["base"]["scores"], out["modified"]["scores"])]
    assert [c["name"] for c in out["changes"]] == ["Refusals", "KL divergence", "Mean response length"]
    assert out["changes"][1]["change"] == kl["value"]
    assert out["rendering"] == {"chat": False, "system": None, "prefix": None, "chat_options": None}
    assert out["generation"]["runner"] == "FunctionalRunner" and out["datasets"] == [None, None, None]
    assert out["done"] == out["total"] == 2 + 1 + 2 + 2           # the reference, responses (abc, hello, quiet), KL
    assert set(out) == {"check", "path", "input", "version", "seconds", "compare", "rendering", "generation", "base",
                        "modified", "changes", "datasets", "notes", "done", "total"}
    assert out["input"] == {"kind": "checkpoint"}
    # A checkpoint equal to the current weights scores as current (no pieces).
    same = modified(tmp_path, ws, name="same")
    eq = DE.evaluate(ws, score_body(same))
    assert eq["modified"]["state"] == "current" and eq["compare"]["held"]["tensors"] == 0
    assert strip(eq["modified"]["scores"]) == strip(eq["base"]["scores"][:1]) + \
        [{**strip(eq["base"]["scores"])[1], "value": 0.0, "display": "0.0000", "by_definition": False,
          "per_prompt": [0.0, 0.0]}] + strip(eq["base"]["scores"][2:])
    assert "the modified checkpoint's weights equal the base's in every tensor" in eq["notes"]


# ------------------------------------------------------------------ ET3. patches do not change what is scored
def test_patches_do_not_change_what_is_scored(tmp_path, sess):
    ws = sess.ws
    path = modified(tmp_path, ws, merged_change)
    shutil.copy(os.path.join(ws.dir, "tokenizer_meta.json"), path)
    before = DE.evaluate(ws, score_body(path))
    assert sess.apply("mlp(0).W_out *= 3")["committed"]
    after = DE.evaluate(ws, score_body(path))
    assert after["compare"]["rows"] == before["compare"]["rows"]            # still against the base
    assert after["compare"]["held"]["tensors"] == 3 and before["compare"]["held"]["tensors"] == 2
    assert after["compare"]["held"]["weights"] == before["compare"]["held"]["weights"] + ws.store.numel("blocks.0.mlp.W_out")
    assert strip(after["base"]["scores"]) == strip(before["base"]["scores"])
    a, b = after["modified"]["scores"], before["modified"]["scores"]
    assert strip([a[0], a[2]]) == strip([b[0], b[2]])                       # the same responses
    # Taking the patched tensor back to the base is other - current in float32: within one rounding, said in N7.
    rec = after["compare"]["reconstruction"]
    assert 0 <= rec["max_abs"] <= 2 ** -23 * float(ws.read("blocks.0.mlp.W_out", (), "base").abs().max())
    for x, y in zip(a[1]["per_prompt"], b[1]["per_prompt"]):
        assert x == pytest.approx(y, rel=1e-5)
    assert "1 patch(es) are enabled in this workspace: the modified checkpoint is scored as itself (its weights replace " \
           "the current ones for this scoring), against the base weights" in after["notes"]
    if rec["inexact"]:
        assert f"{rec['inexact']} weights of the modified checkpoint come out differently by at most " \
               f"{rec['max_abs']:.6g} once applied as a change in float32" in after["notes"]
    assert not any("patch(es)" in n for n in before["notes"])
    assert ws.touched() == ["blocks.0.mlp.W_out"]                           # the patch is still there


# ------------------------------------------------------------------ ET4. nothing is written or committed
def tree(directory, skip=()):
    import hashlib
    out = {}
    for root, dirs, files in os.walk(directory):
        rel = os.path.relpath(root, directory)
        if any(rel == s or rel.startswith(s + os.sep) for s in skip):
            continue
        for f in files:
            p = os.path.join(root, f)
            st = os.stat(p)
            with open(p, "rb") as fh:
                out[os.path.join(rel, f)] = (st.st_size, st.st_mtime_ns, hashlib.sha256(fh.read()).hexdigest())
        out[rel + os.sep] = sorted(x for x in dirs if os.path.normpath(os.path.join(rel, x)) not in skip)
    return out


def test_nothing_is_written_or_committed(tmp_path, sess):
    ws = sess.ws
    assert sess.apply("attn(1).W_O *= 2")["committed"]
    path = modified(tmp_path, ws, merged_change)
    kl = os.path.join("directions", "kl_reference")
    before = (ws.version, len(ws.patches), tree(ws.dir, skip=("directions",)), tree(path))
    before_patches = sorted(os.listdir(os.path.join(ws.dir, "patches")))
    mtime = os.stat(os.path.join(ws.dir, "workspace.json")).st_mtime_ns
    DE.evaluate(ws, score_body(path))
    DE.evaluate(ws, {"path": path, "check": True})
    assert (ws.version, len(ws.patches), tree(ws.dir, skip=("directions",)), tree(path)) == before
    assert sorted(os.listdir(os.path.join(ws.dir, "patches"))) == before_patches
    assert os.stat(os.path.join(ws.dir, "workspace.json")).st_mtime_ns == mtime
    d = os.path.join(ws.dir, "directions")
    assert os.listdir(d) == ["kl_reference"] and os.listdir(os.path.join(ws.dir, kl))
    # The runner holds the current weights again.
    p = ws.runner.params("current")
    for n in ws.names():
        assert torch.equal(p[n], ws.read(n, (), "current"))


# ------------------------------------------------------------------ ET2. the weight comparison
def test_the_weight_comparison(tmp_path, ws):
    def change(n, t):
        if n == "blocks.1.mlp.W_out":
            return t * 1.5
        if n == "blocks.0.mlp.b_in":
            t = t.clone()
            t[3] += 0.25
            return t
        return t
    path = modified(tmp_path, ws, change, dtypes={"blocks.0.attn.W_Q": torch.bfloat16})
    out = DE.evaluate(ws, {"path": path, "check": True})
    assert set(out) == {"check", "path", "input", "version", "compare", "notes", "seconds"}
    assert out["input"] == {"kind": "checkpoint"}
    assert out["check"] is True and out["path"] == path and out["version"] == ws.version
    c = out["compare"]
    assert "reconstruction" not in c
    base = load_file(os.path.join(ws.dir, "base.safetensors"))
    other = {k: v.float() for k, v in load_file(os.path.join(path, "model.safetensors")).items()}
    want = []
    for n in ws.names():                                   # the workspace's (natural) order
        b, o = base[n], other[n]
        k = int((b != o).sum())
        if k:
            d = o.double() - b.double()
            norm, base_sq = math.sqrt(float((d * d).sum())), float((b.double() ** 2).sum())
            want.append({"param": n, "changed": k, "of": b.numel(), "norm": norm, "max_abs": float(d.abs().max()),
                         "relative": norm / math.sqrt(base_sq) if base_sq else None})
    assert [r["param"] for r in want] == ["blocks.0.attn.W_Q", "blocks.0.mlp.b_in", "blocks.1.mlp.W_out"]
    assert [r["changed"] for r in want][1:] == [1, 960]
    assert len(c["rows"]) == len(want)
    for got, w in zip(c["rows"], want):
        assert got["param"] == w["param"] and got["changed"] == w["changed"] and got["of"] == w["of"]
        assert got["norm"] == pytest.approx(w["norm"], rel=1e-12) and got["max_abs"] == w["max_abs"]
        assert got["relative"] == (None if w["relative"] is None else pytest.approx(w["relative"], rel=1e-12))
    assert c["rows"][1]["max_abs"] == pytest.approx(0.25, abs=1e-6)
    assert c["rows"][1]["relative"] is None and c["rows"][2]["relative"] == pytest.approx(0.5, rel=1e-6)  # a zero base
    assert c["tensors"] == 38 and c["differ"] == 3 and c["of"] == sum(t.numel() for t in base.values())
    assert c["changed"] == sum(r["changed"] for r in want)
    assert c["dtypes"] == {"base": {"F32": 38}, "modified": {"BF16": 1, "F32": 37}} and c["tied"] == []
    assert c["held"] == {"tensors": 3, "weights": sum(r["of"] for r in want), "bytes": 4 * sum(r["of"] for r in want),
                         "applied_bytes": 12 * sum(r["of"] for r in want), "fits": True, "why": None}
    assert c["config"] == {"same": True, "other": []} and c["generation_config"] == {"same": True}
    assert c["tokenizer"] == {"prompts": 0, "differ": None, "template_same": True}
    assert out["notes"] == ["the modified checkpoint has no tokenizer file; it is scored with the base's tokenizer, "
                            "as Heretic's evaluate_model does"]
    # Equal weights: no rows, and the note.
    same = modified(tmp_path, ws, name="same")
    shutil.copy(os.path.join(ws.dir, "tokenizer_meta.json"), same)
    s = DE.evaluate(ws, {"path": same, "check": True})
    assert s["compare"]["differ"] == 0 and s["compare"]["rows"] == [] and s["compare"]["changed"] == 0
    assert s["compare"]["held"]["tensors"] == 0 and s["compare"]["tokenizer"]["differ"] == 0
    assert s["notes"] == ["the modified checkpoint's weights equal the base's in every tensor"]
    # A stop before the first tensor, and the progress of pass 1.
    seen = []
    assert DE.evaluate(ws, {"path": path, "check": True}, stopped=lambda: True) == {"stopped": True, "done": 0, "total": 38}
    DE.evaluate(ws, {"path": path, "check": True}, progress=lambda *a: seen.append(a))
    assert seen == [("compare", k, 38) for k in range(1, 39)]


# ------------------------------------------------------------------ ET5. refusals (before any weight read)
def test_refusals(tmp_path, ws, monkeypatch):
    from selection_fixtures import quant_workspace
    good = modified(tmp_path, ws, scaled("blocks.1.mlp.W_out", 1.5))
    q = quant_workspace(tmp_path / "q")
    refused(q, {"path": good, "check": True},
            "scoring a modified checkpoint needs a floating-point base: this workspace's base is GGUF (quantized) and "
            "is never dequantized to compare it with another checkpoint")
    refused(q, {}, "scoring a modified checkpoint needs a floating-point base: this workspace's base is GGUF (quantized) "
                   "and is never dequantized to compare it with another checkpoint")
    with monkeypatch.context() as m:
        m.setattr(Store, "remote", property(lambda self: True))
        refused(ws, {"path": good, "check": True}, "the open model's weights are still on the Hub; pull them first (the "
                                                   "comparison reads every base tensor)")
    with monkeypatch.context() as m:
        real = ws.store.dtype
        m.setattr(ws.store, "dtype", lambda n: "I32" if n == "blocks.0.ln1.w" else real(n))
        refused(ws, {"path": good, "check": True}, "the open model's tensor blocks.0.ln1.w is stored as I32; only a "
                                                   "checkpoint of floating-point tensors can be compared")
    # The body (E-H4 to E-H7).
    refused(ws, [], "the request must be an object")
    refused(ws, {"path": good, "check": True, "state": "base"},
            "state: unknown field; the fields are path, check, scores, chat, system, prefix, chat_options, "
            "max_response_length and stream")
    refused(ws, {"check": True}, "path: required: the absolute path of the modified checkpoint's directory")
    refused(ws, {"path": 3, "check": True}, "path must be text")
    refused(ws, {"path": "/" + "a" * 4096, "check": True}, "path is longer than 4096 characters")
    refused(ws, {"path": good, "check": "yes"}, "check must be true or false")
    refused(ws, {"path": good}, "scores: required (unless check is true)")
    refused(ws, {"path": good, "check": False}, "scores: required (unless check is true)")
    # The path (E-H8 to E-H16).
    refused(ws, {"path": "merged", "check": True},
            "path must be an absolute path of a local directory; a Hub id is not looked up (give its snapshot directory "
            "in the Hugging Face cache)")
    refused(ws, {"path": "org/model", "check": True},
            "path must be an absolute path of a local directory; a Hub id is not looked up (give its snapshot directory "
            "in the Hugging Face cache)")
    nope = str(tmp_path / "nope")
    refused(ws, {"path": nope, "check": True}, f"path: no such directory: {nope}")
    gguf = tmp_path / "m.Q4_K.gguf"
    gguf.write_bytes(b"GGUF")
    refused(ws, {"path": str(gguf), "check": True},
            "path names a GGUF file; the modified model must be a directory of floating-point safetensors files (a GGUF "
            "model is not read as floating-point weights)")
    refused(ws, {"path": os.path.join(good, "model.safetensors"), "check": True},
            "path names a file; give the checkpoint's directory")
    adapter = modified(tmp_path, ws, name="adapter")
    with open(os.path.join(adapter, "adapter_config.json"), "w") as f:
        json.dump({"peft_type": "LORA"}, f)
    refused(ws, {"path": adapter, "check": True},
            "path holds more than one kind of model: a PEFT adapter (adapter_config.json); checkpoint weights "
            "(model.safetensors); give the directory of one")
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "config.json").write_text("{}")
    refused(ws, {"path": str(empty), "check": True},
            "path holds no model the IDE reads: no PEFT adapter (adapter_config.json), no workspace (workspace.json) and "
            "no safetensors weights (model.safetensors, an index, or *.safetensors files); it holds 1 entry: config.json")
    no_weights = "path has no safetensors weights (model.safetensors, an index, or *.safetensors files)"
    sharded = tmp_path / "sharded"
    sharded.mkdir()
    shutil.copy(os.path.join(good, "model.safetensors"), sharded / "model-00001-of-00002.safetensors")
    with open(sharded / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}, f)
    refused(ws, {"path": str(sharded), "check": True},
            "the checkpoint's index names model-00002-of-00002.safetensors, which is not in path")
    with open(sharded / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": {}}, f)
    refused(ws, {"path": str(sharded), "check": True}, no_weights)
    with open(sharded / "model.safetensors.index.json", "w") as f:       # a sharded checkpoint is read by its index
        json.dump({"weight_map": {"a": "model-00001-of-00002.safetensors"}}, f)
    assert DE.evaluate(ws, {"path": str(sharded), "check": True})["compare"]["differ"] == 1
    itself = "path is the open workspace itself: its patches are scored against its base by the scores (state " \
             "current), not here"
    refused(ws, {"path": ws.dir, "check": True}, itself)                  # a workspace directory is read as one (H4c)
    os.symlink(ws.dir, tmp_path / "link")
    refused(ws, {"path": str(tmp_path / "link"), "check": True}, itself)
    own = tmp_path / "own"                                                # the base's own files, as a checkpoint
    own.mkdir()
    os.symlink(os.path.join(ws.dir, "base.safetensors"), own / "model.safetensors")
    refused(ws, {"path": str(own), "check": True},
            "path holds the open workspace's own base checkpoint: there is nothing to compare")
    # E-H17: quantization declared.
    quant = modified(tmp_path, ws, name="quant")
    with open(os.path.join(quant, "config.json"), "w") as f:
        json.dump({"quantization": {"bits": 4, "group_size": 64}}, f)
    refused(ws, {"path": quant, "check": True},
            "the modified checkpoint's config.json declares quantization (quantization); its integer storage is not "
            "read as floating-point weights")
    # Names, shapes, dtypes, values (E-H18 to E-H21, E-H24).
    lacks = modified(tmp_path, ws, lambda n, t: None if n in ("blocks.0.ln1.b", "unembed.b_U") else t, name="lacks")
    refused(ws, {"path": lacks, "check": True},
            "the modified checkpoint lacks 2 tensor(s) of the base, for example blocks.0.ln1.b; it is not the same "
            "architecture")
    extra = modified(tmp_path, ws, name="extra")
    t = load_file(os.path.join(extra, "model.safetensors"))
    t["zz.extra.weight"] = torch.ones(3)
    save_file(t, os.path.join(extra, "model.safetensors"))
    refused(ws, {"path": extra, "check": True},
            "the modified checkpoint has 1 tensor(s) the base does not, for example zz.extra.weight; it is not the same "
            "architecture")
    shape = modified(tmp_path, ws, lambda n, t: t.T.contiguous() if n == "embed.W_E" else t, name="shape")
    refused(ws, {"path": shape, "check": True}, "embed.W_E has shape [24, 26] in the modified checkpoint and [26, 24] in "
                                                "the base")
    ints = modified(tmp_path, ws, name="ints", dtypes={"blocks.0.ln1.w": torch.int32})
    refused(ws, {"path": ints, "check": True},
            "blocks.0.ln1.w is stored as I32 in the modified checkpoint; only floating-point tensors are compared (a "
            "quantized checkpoint is not dequantized)")

    def nan(n, t):
        if n == "blocks.1.attn.b_O":
            t = t.clone()
            t[5] = float("nan")
        return t
    bad = modified(tmp_path, ws, nan, name="nan")
    DE.check_evaluate(ws, {"path": bad, "check": True})          # the values are read by pass 1, not by the checks
    with pytest.raises(DirectionError) as e:
        DE.evaluate(ws, {"path": bad, "check": True})
    assert str(e.value) == "blocks.1.attn.b_O contains NaN or inf in the modified checkpoint"
    # The scores are checked as POST /api/directions/scores checks them (its own refusals).
    with pytest.raises(DirectionError, match=r"^max_response_length must be an integer from 1 to 2048$"):
        DE.check_evaluate(ws, {"path": good, "max_response_length": 0, "scores": [{"plugin": "KeywordRate",
                                                                                     "prompts": ["abc"]}]})
    with pytest.raises(DirectionError, match=r"^scores\[0\]\.plugin must be one of "):
        DE.check_evaluate(ws, {"path": good, "check": True, "scores": [{"plugin": "Nope", "prompts": ["abc"]}]})
    assert ws._runner is None
    # The memory forecast (E-H25, E-H26): before pass 2 and before the model.
    asked = []
    scores = [{"plugin": "KeywordRate", "prompts": ["abc"]}]
    with monkeypatch.context() as m:
        m.setattr(edit_limit, "_memory", lambda w, scratch: (asked.append(scratch), ("memory", 10, {}))[1])
        assert DE.evaluate(ws, {"path": good, "check": True})["compare"]["held"]["fits"] is True   # 960 <= SMALL_EDIT
        assert asked == []
    big = ("the modified checkpoint differs from the current weights in 1 tensors (960 weights); scoring it holds them "
           "as FP32 changes applied to the loaded model, about less than 1 MiB at 12 bytes per weight, and memory allows "
           "10 weights now (1.0 GiB available, 512 MiB kept free by the Admission headroom reserve). Free memory on the "
           "machine, or score a checkpoint that changes fewer tensors")
    with monkeypatch.context() as m:
        m.setattr(edit_limit, "SMALL_EDIT", 0)
        m.setattr(edit_limit, "_memory", lambda w, scratch: (asked.append(scratch), (
            "memory", 10, {"available": 2 ** 30, "preferred": 2 ** 29, "scratch": 0}))[1])
        with pytest.raises(DirectionError) as e:
            DE.evaluate(ws, {"path": good, "max_response_length": 2, "scores": scores})
        assert str(e.value) == big and asked == [DE.SCRATCH] == [4 * 4 * 16_000_000]
        held = DE.evaluate(ws, {"path": good, "check": True})["compare"]["held"]
        assert held == {"tensors": 1, "weights": 960, "bytes": 3840, "applied_bytes": 11520, "fits": False, "why": big}
        m.setattr(edit_limit, "_memory", lambda w, scratch: ("memory", None, {"reason": "telemetry off"}))
        with pytest.raises(DirectionError) as e:
            DE.evaluate(ws, {"path": good, "max_response_length": 2, "scores": scores})
        assert str(e.value) == ("memory could not be measured (telemetry off), so the modified checkpoint's 960 changed "
                                "weights are not loaded")
        m.setattr(edit_limit, "_memory", lambda w, scratch: ("memory", 960, {"available": 2 ** 30, "preferred": 0}))
        assert DE.evaluate(ws, {"path": good, "check": True})["compare"]["held"]["fits"] is True   # exactly enough
    assert ws._runner is None and not os.path.exists(os.path.join(ws.dir, "directions"))


def test_refusals_of_a_transformers_pair(tmp_path):
    s, w, base, merged = hfpair(tmp_path)
    ok = DE.check_evaluate(w, {"path": merged, "check": True})
    assert ok["config"] == {"same": True, "other": []} and ok["generation_config"] == {"same": True}
    refused(w, {"path": base, "check": True},
            "path holds the open workspace's own base checkpoint: there is nothing to compare")
    with open(os.path.join(merged, "config.json")) as f:
        cfg = json.load(f)
    gen_path = os.path.join(merged, "generation_config.json")
    gen = {}
    if os.path.exists(gen_path):
        with open(gen_path) as f:
            gen = json.load(f)
    os.rename(os.path.join(merged, "config.json"), os.path.join(merged, "config.moved"))
    refused(w, {"path": merged, "check": True}, "path has no config.json")
    os.rename(os.path.join(merged, "config.moved"), os.path.join(merged, "config.json"))

    def write_cfg(**kw):
        with open(os.path.join(merged, "config.json"), "w") as f:
            json.dump({**cfg, **kw}, f)
    write_cfg(quantization_config={"quant_method": "bitsandbytes", "load_in_4bit": True})
    refused(w, {"path": merged, "check": True},
            "the modified checkpoint's config.json declares quantization (bitsandbytes); its integer storage is not "
            "read as floating-point weights")
    write_cfg(hidden_act="gelu")
    refused(w, {"path": merged, "check": True},
            "the modified checkpoint's config.json differs from the base's in hidden_act (base \"silu\", modified "
            "\"gelu\"): it would run differently from the configuration the IDE runs it with, the base's")
    theta = float(cfg["rope_theta"] if cfg.get("rope_theta") is not None else cfg["rope_parameters"]["rope_theta"])
    write_cfg(rope_theta=theta * 2)
    refused(w, {"path": merged, "check": True},
            f"the modified checkpoint's config.json differs from the base's in rope_theta (base {json.dumps(theta)}, "
            f"modified {json.dumps(theta * 2)}): it would run differently from the configuration the IDE runs it with, "
            f"the base's")
    write_cfg()
    with open(gen_path, "w") as f:
        json.dump({**gen, "eos_token_id": 7}, f)
    refused(w, {"path": merged, "check": True},
            f"the modified checkpoint's generation_config.json differs from the base's in eos_token_id (base "
            f"{json.dumps(gen.get('eos_token_id'))}, modified 7): Heretic's evaluate_model would generate with the "
            f"modified checkpoint's settings and the IDE would use the base's; this is not supported yet")
    with open(gen_path, "w") as f:
        json.dump({**gen, "repetition_penalty": 1.3}, f)
    refused(w, {"path": merged, "check": True},
            "the modified checkpoint's generation_config.json differs from the base's in repetition_penalty (base null, "
            "modified 1.3): Heretic's evaluate_model would generate with the modified checkpoint's settings and the IDE "
            "would use the base's; this is not supported yet")
    with open(gen_path, "w") as f:                                      # a key outside GREEDY_KEYS is not compared
        json.dump({**gen, "transformers_version": "0.0.1", "temperature": 0.7}, f)
    assert DE.check_evaluate(w, {"path": merged, "check": True})["generation_config"] == {"same": True}
    os.remove(gen_path)                                                 # a missing file is {}
    assert not os.path.exists(gen_path)
    if any(gen.get(k) is not None for k in DE.DS.GREEDY_KEYS):
        with pytest.raises(DirectionError, match="generation_config.json differs from the base's in "):
            DE.check_evaluate(w, {"path": merged, "check": True})
    assert w._runner is None


# ------------------------------------------------------------------ ET6. check loads no model
def test_check_loads_no_model(tmp_path, sess, monkeypatch):
    ws = sess.ws
    loads = []

    def boom(w):
        loads.append(w.name)
        raise AssertionError("a model was loaded")
    monkeypatch.setattr(wsp.runner, "make_runner", boom)
    path = modified(tmp_path, ws, scaled("blocks.0.attn.W_O", 0.5))
    out = DE.evaluate(ws, {"path": path, "check": True, "max_response_length": 3,
                           "scores": [{"plugin": "KeywordRate", "prompts": ["abc"]},
                                      {"plugin": "KLDivergence", "prompts": ["ab", "zz"]}]})
    assert out["check"] is True and "base" not in out and "modified" not in out and "changes" not in out
    assert out["compare"]["held"]["tensors"] == 1 and out["compare"]["held"]["fits"] is True
    assert [r["param"] for r in out["compare"]["rows"]] == ["blocks.0.attn.W_O"]
    assert out["compare"]["tokenizer"] == {"prompts": 3, "differ": None, "template_same": True}
    assert ws._runner is None and loads == []
    assert not os.path.exists(os.path.join(ws.dir, "directions", "kl_reference"))
    # Through the route: no model either.
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    c = TestClient(create_app(sess, "http://127.0.0.1:0"), raise_server_exceptions=False)
    r = c.post("/api/directions/evaluate", json={"path": path, "check": True})
    assert r.status_code == 200, r.text
    assert r.json()["check"] is True and r.json()["compare"]["differ"] == 1 and "base" not in r.json()
    assert ws._runner is None and loads == []
    r = c.post("/api/directions/evaluate", json={"path": path})               # scoring needs the model: refused first
    assert r.status_code == 400 and r.json() == {"error": "DirectionError: scores: required (unless check is true)"}
    r = c.post("/api/directions/evaluate", json={"path": path, "max_response_length": 2,
                                                  "scores": [{"plugin": "KeywordRate", "prompts": ["abc"]}]})
    assert r.status_code == 400 and "a model was loaded" in r.json()["error"] and ws._runner is None
    assert loads and set(loads) == {"rnd"}
    loads.clear()
    r = c.post("/api/directions/evaluate", json={"path": "merged", "max_response_length": 2,   # checked before the model
                                                  "scores": [{"plugin": "KeywordRate", "prompts": ["abc"]}]})
    assert r.status_code == 400 and r.json()["error"].startswith("DirectionError: path must be an absolute path")
    assert ws._runner is None and loads == []


# ------------------------------------------------------------------ ET7. a tied head absent from the checkpoint
def test_a_tied_head_absent_from_the_checkpoint(tmp_path):
    s, w, base, merged = hfpair(tmp_path, tied=True)
    assert w.struct.tied is True and "lm_head.weight" in w.names()
    out = DE.evaluate(w, {"path": merged, "check": True})
    c = out["compare"]
    assert c["tied"] == ["lm_head.weight"] and c["tensors"] == len(w.names()) - 1
    assert [r["param"] for r in c["rows"]] == ["model.layers.0.self_attn.o_proj.weight"]
    assert c["held"]["tensors"] == 1
    assert "lm_head.weight is not in the modified checkpoint; the base ties it to the token embedding, so it is not " \
           "compared" in out["notes"]
    # Untied, the same file is missing a tensor of the base.
    s2, w2, base2, merged2 = hfpair(tmp_path / "untied")
    t = load_file(os.path.join(merged2, "model.safetensors"))
    t.pop("lm_head.weight")
    save_file(t, os.path.join(merged2, "model.safetensors"), metadata={"format": "pt"})
    refused(w2, {"path": merged2, "check": True},
            "the modified checkpoint lacks 1 tensor(s) of the base, for example lm_head.weight; it is not the same "
            "architecture")


# ------------------------------------------------------------------ ET8. the config table
LLAMA = {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"], "hidden_size": 896, "num_hidden_layers": 3,
         "num_attention_heads": 14, "num_key_value_heads": 2, "intermediate_size": 4864, "vocab_size": 151936,
         "max_position_embeddings": 32768, "rms_norm_eps": 1e-06, "hidden_act": "silu", "tie_word_embeddings": True,
         "rope_theta": 1000000.0, "rope_scaling": None, "sliding_window": 32768, "use_sliding_window": False,
         "torch_dtype": "bfloat16", "transformers_version": "4.43.1", "use_cache": True, "max_window_layers": 21}


def test_the_config_table():
    C = DE.compare_config
    # A newer transformers re-serialises: rope_theta into rope_parameters, sliding_window dropped (it is off), and
    # layer_types written out. None of it changes how the model runs.
    newer = {k: v for k, v in LLAMA.items() if k not in ("rope_theta", "rope_scaling", "sliding_window", "torch_dtype")}
    newer.update(rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"}, dtype="float32",
                 transformers_version="4.57.0", layer_types=["full_attention"] * 3)
    assert C(LLAMA, newer) == {"same": True, "other": []}
    assert C(newer, LLAMA) == {"same": True, "other": []}
    # use_sliding_window false with and without sliding_window; true with a different window is refused.
    assert C(LLAMA, {**LLAMA, "sliding_window": 4096}) == {"same": True, "other": []}
    with pytest.raises(DirectionError) as e:
        C({**LLAMA, "use_sliding_window": True}, {**LLAMA, "use_sliding_window": True, "sliding_window": 4096})
    assert str(e.value) == ("the modified checkpoint's config.json differs from the base's in sliding_window (base 32768, "
                            "modified 4096): it would run differently from the configuration the IDE runs it with, the "
                            "base's")
    # layer_types absent against all full_attention; a sliding layer is refused.
    assert C({**LLAMA, "layer_types": ["full_attention"] * 3}, LLAMA)["same"] is True
    with pytest.raises(DirectionError, match=r" in layer_types \(base \[\"full_attention\", \"sliding_attention\", "):
        C({**LLAMA, "layer_types": ["full_attention", "sliding_attention", "full_attention"]}, LLAMA)
    # A different rope_theta, written either way.
    with pytest.raises(DirectionError) as e:
        C(LLAMA, {**newer, "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}})
    assert str(e.value) == ("the modified checkpoint's config.json differs from the base's in rope_theta (base 1000000.0, "
                            "modified 10000.0): it would run differently from the configuration the IDE runs it with, the "
                            "base's")
    # rope scaling: its type and its other values, from rope_scaling or rope_parameters.
    yarn = {**LLAMA, "rope_scaling": {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}}
    yarn_new = {**newer, "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "yarn", "factor": 4.0,
                                             "original_max_position_embeddings": 32768}}
    assert C(yarn, yarn_new)["same"] is True
    with pytest.raises(DirectionError, match=r" in rope_type \(base \"yarn\", modified \"default\"\)"):
        C(yarn, LLAMA)
    with pytest.raises(DirectionError, match=r" in rope_rest \(base \{\"factor\": 4.0, "):
        C(yarn, {**yarn, "rope_scaling": {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 32768}})
    # Integers and floats normalised; num_key_value_heads absent = num_attention_heads.
    assert C({**LLAMA, "hidden_size": 896.0, "rms_norm_eps": 1e-6}, LLAMA)["same"] is True
    mha = {k: v for k, v in LLAMA.items() if k != "num_key_value_heads"}
    assert C(mha, {**LLAMA, "num_key_value_heads": 14})["same"] is True
    with pytest.raises(DirectionError, match=r" in num_key_value_heads \(base 14, modified 2\)"):
        C(mha, LLAMA)
    # Another name for the same row (GPT-2's), and each row refused with its normalised values.
    assert C({"model_type": "gpt2", "n_embd": 32, "n_layer": 2, "n_head": 4},
             {"model_type": "gpt2", "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4})["same"] is True
    b = {**LLAMA, "head_dim": 64, "attention_bias": False}
    for key, value, row, was in (("model_type", "llama", "model_type", '"qwen2"'),
                                 ("architectures", ["LlamaForCausalLM"], "architectures", '["Qwen2ForCausalLM"]'),
                                 ("hidden_size", 1024, "hidden_size", "896"),
                                 ("num_hidden_layers", 4, "num_hidden_layers", "3"),
                                 ("num_attention_heads", 7, "num_attention_heads", "14"),
                                 ("num_key_value_heads", 7, "num_key_value_heads", "2"),
                                 ("head_dim", 128, "head_dim", "64"),
                                 ("intermediate_size", 1, "intermediate_size", "4864"),
                                 ("vocab_size", 3, "vocab_size", "151936"),
                                 ("max_position_embeddings", 8, "max_position_embeddings", "32768"),
                                 ("rms_norm_eps", 1e-5, "norm_eps", "1e-06"),
                                 ("hidden_act", "gelu", "hidden_act", '"silu"'),
                                 ("tie_word_embeddings", False, "tie_word_embeddings", "true"),
                                 ("attention_bias", True, "attention_bias", "false")):
        with pytest.raises(DirectionError) as e:
            C(b, {**b, key: value})
        assert str(e.value) == (f"the modified checkpoint's config.json differs from the base's in {row} (base {was}, "
                                f"modified {json.dumps(value)}): it would run differently from the configuration the IDE "
                                f"runs it with, the base's"), key
    # A differing key that changes nothing the IDE runs, and a row present on one side only: note N6.
    other = C(LLAMA, {**LLAMA, "max_window_layers": 28, "bos_token_id": 1, "head_dim": 64, "_name_or_path": "x"})
    assert other == {"same": False, "other": ["bos_token_id", "head_dim", "max_window_layers"]}
    # text_config: the rows again, prefixed.
    vlm = {"model_type": "gemma3", "architectures": ["Gemma3ForConditionalGeneration"],
           "text_config": {**LLAMA, "model_type": "gemma3_text"}, "vision_config": {"hidden_size": 1152}}
    assert C(vlm, vlm)["same"] is True
    with pytest.raises(DirectionError) as e:
        C(vlm, {**vlm, "text_config": {**vlm["text_config"], "rope_theta": 10000.0}})
    assert str(e.value).startswith("the modified checkpoint's config.json differs from the base's in text_config.rope_theta "
                                   "(base 1000000.0, modified 10000.0)")
    assert C(vlm, {**vlm, "text_config": {**vlm["text_config"], "max_window_layers": 2}})["other"] == \
        ["text_config.max_window_layers"]
    assert C(vlm, {**vlm, "vision_config": {"hidden_size": 1}})["other"] == ["vision_config"]


def test_the_config_note_lists_at_most_eight_keys(tmp_path):
    s, w, base, merged = hfpair(tmp_path)
    with open(os.path.join(merged, "config.json")) as f:
        cfg = json.load(f)
    cfg.update({f"k{i}": i for i in range(10)})
    with open(os.path.join(merged, "config.json"), "w") as f:
        json.dump(cfg, f)
    out = DE.evaluate(w, {"path": merged, "check": True})
    assert out["compare"]["config"] == {"same": False, "other": sorted(f"k{i}" for i in range(10))}
    assert "the modified checkpoint's config.json also differs in k0, k1, k2, k3, k4, k5, k6, k7 and 2 more; these do " \
           "not change how the IDE runs it" in out["notes"]


# ------------------------------------------------------------------ ET9. tokenizer and template notes
def test_tokenizer_and_template_notes(tmp_path):
    s, w, base, merged = hfpair(tmp_path)
    body = {"path": merged, "check": True, "max_response_length": 2,
            "scores": [{"plugin": "KeywordRate", "prompts": ["t3 t5 t7", "t1 t2"]},
                       {"plugin": "KLDivergence", "prompts": ["t5", "t1 t2"]}]}
    out = DE.evaluate(w, body)                               # identical tokenizer and no template on either side
    assert out["compare"]["tokenizer"] == {"prompts": 3, "differ": 0, "template_same": True} and out["notes"] == []
    tp = os.path.join(merged, "tokenizer.json")
    with open(tp) as f:
        tj = json.load(f)
    tj["model"]["vocab"]["t5x"] = tj["model"]["vocab"].pop("t5")         # one entry changed: t5 is now unknown
    with open(tp, "w") as f:
        json.dump(tj, f)
    with open(os.path.join(merged, "chat_template.jinja"), "w") as f:
        f.write("{% for m in messages %}{{ m.content }}{% endfor %}")
    out = DE.evaluate(w, body)
    assert out["compare"]["tokenizer"] == {"prompts": 3, "differ": 2, "template_same": False}
    assert out["notes"] == [
        "the modified checkpoint's tokenizer encodes 2 of the 3 prompt texts differently from the base's; it is scored "
        "with the base's tokenizer, as Heretic's evaluate_model does",
        "the modified checkpoint's chat template differs from the base's; prompts are rendered with the base's "
        "template, as Heretic's evaluate_model does"]
    # The template from tokenizer_config.json, on both sides: equal content, no note.
    os.remove(os.path.join(merged, "chat_template.jinja"))
    for d in (base, merged):
        with open(os.path.join(d, "tokenizer_config.json"), "w") as f:
            json.dump({"chat_template": [{"name": "default", "template": "{{ x }}"}]}, f)
    os.remove(tp)
    out = DE.evaluate(w, body)
    assert out["compare"]["tokenizer"] == {"prompts": 3, "differ": None, "template_same": True}
    assert out["notes"] == ["the modified checkpoint has no tokenizer file; it is scored with the base's tokenizer, as "
                            "Heretic's evaluate_model does"]
    assert DE.template_text(base) == json.dumps([{"name": "default", "template": "{{ x }}"}], sort_keys=True)
    assert w._runner is None


# ------------------------------------------------------------------ ET11. slabs give what whole tensors give
def dense_sums(ws, deltas):
    out = {}
    for n, ps in (deltas or {}).items():
        t = torch.zeros(ws.shape(n))
        whole = tuple(range(s) for s in ws.shape(n))
        for p in ps:
            Dl.add_to(t, whole, p)
        out[n] = t
    return out


def test_slabs_equal_whole_tensors(tmp_path, sess, monkeypatch):
    ws = sess.ws
    assert sess.apply("mlp(0).W_out *= 3")["committed"]

    def change(n, t):
        if n == "blocks.1.mlp.W_out":
            return t * 1.5
        if n == "blocks.0.attn.W_Q":
            t = t.clone()
            t[1, 5, 3] = 2.0
            return t
        return t
    path = modified(tmp_path, ws, change)

    def run():
        checked = DE.check_evaluate(ws, {"path": path, "check": True})
        compared = DE.compare_weights(ws, checked["other"])
        names = DE.held_names(ws, checked["other"], compared)
        deltas, rec = DE.build_candidate(ws, checked["other"], names)
        return compared, names, deltas, rec
    whole = run()
    scored = DE.evaluate(ws, score_body(path))
    monkeypatch.setattr(W, "STREAM_ELEMS", 7)
    monkeypatch.setattr(DE, "STREAM_ELEMS", 7)
    reads = []
    real = Store.read
    monkeypatch.setattr(Store, "read", lambda self, n, region=None: (reads.append(region), real(self, n, region))[1])
    slabs = run()
    assert reads and max(region_size(r) for r in reads if r is not None) == 1 * 24 * 12   # one row of W_Q at most
    assert [{k: v for k, v in r.items() if k not in ("norm", "relative")} for r in slabs[0]["rows"]] == \
        [{k: v for k, v in r.items() if k not in ("norm", "relative")} for r in whole[0]["rows"]]
    for a, b in zip(slabs[0]["rows"], whole[0]["rows"]):
        assert a["norm"] == pytest.approx(b["norm"], rel=1e-12) and a["relative"] == pytest.approx(b["relative"], rel=1e-12)
    assert [r["param"] for r in whole[0]["rows"]] == ["blocks.0.attn.W_Q", "blocks.1.mlp.W_out"]
    assert whole[1] == slabs[1] == ["blocks.0.attn.W_Q", "blocks.0.mlp.W_out", "blocks.1.mlp.W_out"]
    ds, dw = dense_sums(ws, slabs[2]), dense_sums(ws, whole[2])
    assert sorted(ds) == sorted(dw) == whole[1]
    for n in dw:
        assert torch.equal(ds[n], dw[n])
    # current + pieces = the checkpoint, to the reconstruction the answer reports (counted here again).
    other = load_file(os.path.join(path, "model.safetensors"))
    inexact, worst = 0, 0.0
    for n in whole[1]:
        back = ws.read(n, (), "current") + dw[n]
        inexact += int((back != other[n]).sum())
        worst = max(worst, float((back - other[n]).abs().max()))
    assert whole[3] == slabs[3] == {"inexact": inexact, "max_abs": worst}
    assert torch.equal(ws.read("blocks.1.mlp.W_out", (), "current") + dw["blocks.1.mlp.W_out"],
                       other["blocks.1.mlp.W_out"])
    again = DE.evaluate(ws, score_body(path))
    assert strip(again["base"]["scores"]) == strip(scored["base"]["scores"])
    assert strip(again["modified"]["scores"]) == strip(scored["modified"]["scores"])
    assert again["compare"]["reconstruction"] == scored["compare"]["reconstruction"]


# ------------------------------------------------------------------ ET12. the HF runner holds the checkpoint's weights
def test_the_hf_runner_holds_the_checkpoints_weights(tmp_path):
    s, w, base, merged = hfpair(tmp_path)
    checked = DE.check_evaluate(w, {"path": merged, "check": True})
    compared = DE.compare_weights(w, checked["other"])
    names = DE.held_names(w, checked["other"], compared)
    deltas, rec = DE.build_candidate(w, checked["other"], names)
    assert names == ["model.layers.0.self_attn.o_proj.weight"] and sorted(deltas) == names
    assert rec == {"inexact": 0, "max_abs": 0.0}
    want, was = load_file(os.path.join(merged, "model.safetensors")), load_file(os.path.join(base, "model.safetensors"))
    r = w.runner
    assert type(r).__name__ == "HFRunner"
    r.use("current", deltas)
    for n in w.names():
        if n in deltas:
            assert torch.equal(r.param(n).data, want[n].float()), n
            assert not torch.equal(r.param(n).data, was[n].float())
    DS._release(w, deltas)
    for n in names:
        assert torch.equal(r.param(n).data, was[n].float())
    # Through the whole request: scored, then the base weights again.
    out = DE.evaluate(w, {"path": merged, "max_response_length": 3,
                          "scores": [{"plugin": "KLDivergence", "prompts": ["t3 t7", "t1"]}]})
    assert out["compare"]["reconstruction"] == {"inexact": 0, "max_abs": 0.0}
    assert out["modified"]["state"] == "candidate" and out["generation"]["runner"] == "HFRunner"
    for n in names:
        assert torch.equal(w.runner.param(n).data, was[n].float())


# ------------------------------------------------------------------ ET13. a transformers pair matches transformers
def test_a_transformers_pair_matches_transformers(tmp_path):
    from transformers import AutoModelForCausalLM
    s, w, base, merged = hfpair(tmp_path)
    prompts = ["t3 t7 t11 t5", "t9 t2 t4", "t1 t1 t8 t30 t6"]
    M = 6
    seen = []
    real = w.runner.generate

    def record(ids, variant="current", extra=None, **kw):
        out = real(ids, variant, extra, **kw)
        seen.append((variant, extra is not None, list(ids), list(out)))
        return out
    w.runner.generate = record
    out = DE.evaluate(w, {"path": merged, "max_response_length": M,
                          "scores": [{"plugin": "KeywordRate", "prompts": prompts, "print_responses": True,
                                      "keyword_markers": ["t5"]},
                                     {"plugin": "KLDivergence", "prompts": prompts}]})
    mb = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).eval()
    mm = AutoModelForCausalLM.from_pretrained(merged, dtype=torch.float32).eval()
    base_runs = [x for x in seen if x[0] == "base"]
    cand_runs = [x for x in seen if x[0] == "current" and x[1]]
    assert len(base_runs) == len(cand_runs) == 3 and len(seen) == 6
    for k, t in enumerate(prompts):
        ids = torch.tensor([w.encode(t)])
        one = torch.ones_like(ids)
        with torch.no_grad():
            gm = mm.generate(ids, attention_mask=one, max_new_tokens=M, do_sample=False, pad_token_id=0)[0, ids.shape[1]:]
            gb = mb.generate(ids, attention_mask=one, max_new_tokens=M, do_sample=False, pad_token_id=0)[0, ids.shape[1]:]
            lb = torch.log_softmax(mb(ids).logits[0, -1].double(), -1)
            lm = torch.log_softmax(mm(ids).logits[0, -1].double(), -1)
        assert cand_runs[k][2] == ids[0].tolist() and cand_runs[k][3] == gm.tolist()
        assert base_runs[k][3] == gb.tolist()
        assert out["modified"]["scores"][0]["responses"][k]["tokens"] == len(gm)
        assert out["base"]["scores"][0]["responses"][k]["tokens"] == len(gb)
        kl = float((lb.exp() * (lb - lm)).sum())
        assert kl > 1e-6
        assert out["modified"]["scores"][1]["per_prompt"][k] == pytest.approx(kl, rel=1e-5)
    assert any(x[3] != y[3] for x, y in zip(base_runs, cand_runs))       # the change shows in the responses
    assert out["base"]["scores"][1]["display"] == "0 (by definition)"


# ------------------------------------------------------------------ ET10. the route, its events and a stop
class Clock:
    """The server's clock, a second further on at every reading, so every progress event is published."""

    def __init__(self):
        import time
        self.real, self.t = time, 0.0

    def monotonic(self):
        self.t += 1.0
        return self.t

    def __getattr__(self, name):
        return getattr(self.real, name)


def test_the_route_events_and_stop(tmp_path, sess, monkeypatch):
    import wsp.server
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    ws = sess.ws
    app = create_app(sess, "http://127.0.0.1:0")
    c = TestClient(app, raise_server_exceptions=False)
    path = modified(tmp_path, ws, merged_change)
    monkeypatch.setattr(wsp.server, "time", Clock())
    q = sess.subscribe()
    r = c.post("/api/directions/evaluate", json={**score_body(path), "stream": "ev-1"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["modified"]["state"] == "candidate" and [s["name"] for s in out["modified"]["scores"]] == \
        ["Refusals", "KL divergence", "Mean response length"]
    ev = events(q, "direction_scores")
    assert all(e["stream"] == "ev-1" and e["workspace"] == "rnd" for e in ev)
    assert ev[0]["state"] == "scoring" and ev[0]["phase"] is None and ev[-1]["state"] == "scored"
    assert ev[-1]["done"] == ev[-1]["total"] == out["done"] == out["total"]
    phases = []
    for e in ev[1:-1]:
        assert e["state"] == "scoring"
        if not phases or phases[-1] != e["phase"]:
            phases.append(e["phase"])
    assert phases == ["compare", "reference", "base_responses", "responses", "logits"]
    compare = [(e["done"], e["total"]) for e in ev if e.get("phase") == "compare"]
    assert compare == [(k, 38) for k in range(1, 39)]
    assert [(e["done"], e["total"]) for e in ev if e.get("phase") == "base_responses"] == [(3, 5), (4, 5), (5, 5)]
    assert c.get("/api/directions").json()["requests"] == []
    # The check mode publishes the comparison's progress too.
    r = c.post("/api/directions/evaluate", json={"path": path, "check": True, "stream": "ev-2"})
    ev = events(q, "direction_scores")
    assert r.status_code == 200 and ev[-1]["state"] == "scored" and {e.get("phase") for e in ev} == {None, "compare"}
    # A stop sent before the request arrives: nothing runs.
    assert c.post("/api/generation/stop", json={"stream": "early"}).json()["found"] is False
    r = c.post("/api/directions/evaluate", json={**score_body(path), "stream": "early"})
    assert r.json() == {"stopped": True, "done": 0, "total": 38}
    assert events(q, "direction_scores")[-1]["state"] == "stopped"
    # A stop between the two evaluations: after the base's responses.
    real = DS.evaluate

    def base_then_stop(w, prepared, state, *a, **kw):
        out = real(w, prepared, state, *a, **kw)
        if state == "base":
            app.state.generation_stops.stop("mid")
        return out
    monkeypatch.setattr(DS, "evaluate", base_then_stop)
    r = c.post("/api/directions/evaluate", json={**score_body(path), "stream": "mid"})
    assert r.json() == {"stopped": True, "done": 0, "total": 5}      # the checkpoint's step; the reference was cached
    ev = events(q, "direction_scores")
    assert ev[-1]["state"] == "stopped" and ev[-2]["phase"] == "base_responses"
    monkeypatch.setattr(DS, "evaluate", real)
    assert ws.runner.params("current")["blocks.1.mlp.W_out"].equal(ws.read("blocks.1.mlp.W_out", (), "current"))
    # A refusal is a 400 with its text and publishes nothing.
    r = c.post("/api/directions/evaluate", json={"path": "merged", "check": True, "stream": "bad"})
    assert r.status_code == 400 and r.json()["error"] == (
        "DirectionError: path must be an absolute path of a local directory; a Hub id is not looked up (give its "
        "snapshot directory in the Hugging Face cache)")
    assert events(q, "direction_scores") == []
    # A failure part-way publishes failed.
    monkeypatch.setattr(DS, "generate_response", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("the unit failed")))
    r = c.post("/api/directions/evaluate", json=score_body(path))
    assert r.status_code == 400 and "the unit failed" in r.json()["error"]
    last = events(q, "direction_scores")[-1]
    assert last["state"] == "failed" and "the unit failed" in last["error"] and last["phase"] == "compare"  # reference cached
    assert c.get("/api/directions").json()["requests"] == []


# ------------------------------------------------------------------ a tied head that both checkpoints hold
def test_a_tied_head_in_both_checkpoints(tmp_path):
    """A tied base whose file holds lm_head.weight (equal to the embedding, as a tied model saves it) and a tied
    fine-tune that changed the embedding and saved the head with it: the head is the embedding in the loaded model,
    so it is not compared and gets no piece; the embedding's change reaches the head through the tie."""
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM
    from test_hf import build
    base, merged = tmp_path / "base", tmp_path / "merged"
    build("llama", str(base))
    with open(base / "config.json") as f:
        cfg = json.load(f)
    cfg["tie_word_embeddings"] = True
    with open(base / "config.json", "w") as f:
        json.dump(cfg, f, indent=1)
    t = load_file(str(base / "model.safetensors"))
    t["lm_head.weight"] = t["model.embed_tokens.weight"].clone()
    save_file(t, str(base / "model.safetensors"), metadata={"format": "pt"})
    shutil.copytree(base, merged)
    g = torch.Generator().manual_seed(4)
    E = t["model.embed_tokens.weight"]
    new = (E + 0.5 * E.abs().mean() * torch.randn(E.shape, generator=g)).contiguous()
    t["model.embed_tokens.weight"], t["lm_head.weight"] = new, new.clone()
    save_file(t, str(merged / "model.safetensors"), metadata={"format": "pt"})
    s = Session(str(tmp_path / "root"))
    os.makedirs(s.root, exist_ok=True)
    w = s.open(str(base))
    assert w.struct.tied is True and "lm_head.weight" in w.names()
    prompts = ["t3 t7 t11 t5", "t9 t2 t4", "t1 t1 t8 t30 t6"]
    M = 6
    seen = []
    real = w.runner.generate

    def record(ids, variant="current", extra=None, **kw):
        out = real(ids, variant, extra, **kw)
        seen.append((variant, extra, list(out)))
        return out
    w.runner.generate = record
    out = DE.evaluate(w, {"path": str(merged), "max_response_length": M,
                          "scores": [{"plugin": "KeywordRate", "prompts": prompts, "print_responses": True},
                                     {"plugin": "KLDivergence", "prompts": prompts}]})
    c = out["compare"]
    assert c["tied"] == ["lm_head.weight"] and c["tensors"] == len(w.names()) - 1
    assert [r["param"] for r in c["rows"]] == ["model.embed_tokens.weight"]
    assert c["held"]["tensors"] == 1 and c["held"]["weights"] == E.numel()
    cands = [x for x in seen if x[0] == "current" and x[1]]
    assert cands and all(sorted(x[1]) == ["model.embed_tokens.weight"] for x in cands)      # no piece for the head
    assert "lm_head.weight is in both checkpoints; the base ties it to the token embedding, so the model does not use " \
           "it and it is not compared" in out["notes"]
    assert not any("is not in the modified checkpoint" in n or "but not in the base" in n for n in out["notes"])
    mm = AutoModelForCausalLM.from_pretrained(str(merged), dtype=torch.float32).eval()
    assert torch.equal(mm.lm_head.weight, new)                                   # transformers ties them too
    for k, p in enumerate(prompts):
        ids = torch.tensor([w.encode(p)])
        with torch.no_grad():
            gm = mm.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=M, do_sample=False,
                             pad_token_id=0)[0, ids.shape[1]:]
            lb = torch.log_softmax(w.runner.model(ids).logits[0, -1].double(), -1)       # restored to the base
            lm = torch.log_softmax(mm(ids).logits[0, -1].double(), -1)
        assert cands[k][2] == gm.tolist()
        assert out["modified"]["scores"][0]["responses"][k]["tokens"] == len(gm)
        kl = float((lb.exp() * (lb - lm)).sum())
        assert kl > 1e-6 and out["modified"]["scores"][1]["per_prompt"][k] == pytest.approx(kl, rel=1e-5)
    # The base model of the runner is the base again (the tie intact).
    assert torch.equal(w.runner.model.lm_head.weight, E) and w.runner.model.lm_head.weight is \
        w.runner.model.model.embed_tokens.weight
