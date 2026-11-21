"""The other inputs of scoring a modified model (H4b: a PEFT LoRA adapter; H4c: another workspace), model-free.

Tiny random transformers checkpoints built in the test's own directory (tests/test_hf.py:build) and tiny random
models of this project (tests/test_steering.py:make_ws). Expectations are independent of the code under test: an
adapter's merged weights are computed here from PEFT's formula (weight + transpose(lora_B @ lora_A) * scaling, the
scaling alpha / r or alpha / sqrt(r)) and written as a checkpoint, which the existing checkpoint path scores; a
workspace's own scoring of its edit (directions_scores, state current) is what scoring it from another workspace must
give; file trees are hashed before and after.
"""
import hashlib
import json
import math
import os
import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file

from wsp import adapter_export as AE
from wsp import delta as Dl
from wsp import directions_evaluate as DE
from wsp import directions_scores as DS
from wsp.directions import DirectionError
from wsp.session import Session

from test_steering import make_ws

pytest.importorskip("transformers")
PREFIX = "base_model.model."
O0, D1 = "model.layers.0.self_attn.o_proj", "model.layers.1.mlp.down_proj"
PROMPTS = ["t3 t7 t11 t5", "t9 t2 t4", "t1 t1 t8 t30 t6"]
SCORES = [{"plugin": "KeywordRate", "prompts": PROMPTS, "print_responses": True, "keyword_markers": ["t5"]},
          {"plugin": "KLDivergence", "prompts": PROMPTS}]


def base_dir(tmp_path, kind="llama"):
    from test_hf import build
    d = tmp_path / f"base-{kind}"
    if not d.exists():
        build(kind, str(d))
    return str(d)


def opened(root, base):
    s = Session(str(root))
    os.makedirs(s.root, exist_ok=True)
    return s, s.open(base)


def config(**kw):
    """An adapter_config.json as PEFT 0.21.2 writes it for a plain LoRA (the keys Heretic's export has)."""
    c = {"peft_type": "LORA", "r": 2, "lora_alpha": 2, "target_modules": [O0, D1], "bias": "none", "use_dora": False,
         "use_rslora": False, "fan_in_fan_out": False, "init_lora_weights": True, "modules_to_save": None,
         "rank_pattern": {}, "alpha_pattern": {}, "exclude_modules": None, "layers_to_transform": None,
         "layers_pattern": None, "lora_bias": False, "lora_dropout": 0, "target_parameters": None,
         "task_type": "CAUSAL_LM", "base_model_name_or_path": None, "loftq_config": {}, "megatron_config": None}
    c.update(kw)
    return c


def write_adapter(d, cfg, factors, dtype=torch.float32, extra=None):
    """factors: {module: (lora_A [r, in], lora_B [out, r])}."""
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "adapter_config.json"), "w") as f:
        json.dump(cfg, f)
    t = {}
    for m, (a, b) in factors.items():
        t[f"{PREFIX}{m}.lora_A.weight"] = a.to(dtype).contiguous()
        t[f"{PREFIX}{m}.lora_B.weight"] = b.to(dtype).contiguous()
    t.update(extra or {})
    save_file(t, os.path.join(d, "adapter_model.safetensors"), metadata={"format": "pt"})
    return str(d)


def factors_for(base, spec, seed=0):
    """spec: {module: r} -> {module: (A, B)} random factors of the module's tensor shape (B not zero)."""
    g = torch.Generator().manual_seed(seed)
    w = load_file(os.path.join(base, "model.safetensors"))
    out = {}
    for m, r in spec.items():
        o, i = w[f"{m}.weight"].shape
        out[m] = (torch.randn(r, i, generator=g) * 0.3, torch.randn(o, r, generator=g) * 0.3)
    return out


def merged_checkpoint(tmp_path, base, factors, scaling, transposed=(), name="merged"):
    """PEFT's merge written here: weight.float() + transpose(B @ A) * scaling, saved float32 beside the base's files."""
    d = tmp_path / name
    shutil.copytree(base, d)
    w = {k: v.float() for k, v in load_file(os.path.join(base, "model.safetensors")).items()}
    for m, (a, b) in factors.items():
        p = b.float() @ a.float()
        p = p.T if m in transposed else p
        w[f"{m}.weight"] = (w[f"{m}.weight"] + p * scaling[m]).contiguous()
    save_file(w, str(d / "model.safetensors"), metadata={"format": "pt"})
    return str(d), w


def body(path, **kw):
    return {"path": path, "max_response_length": 4, "scores": SCORES, **kw}


def strip(scores):
    return [{k: v for k, v in s.items() if k != "reference"} for s in scores]


def refused(ws, path, text):
    with pytest.raises(DirectionError) as e:
        DE.check_evaluate(ws, {"path": path, "check": True})
    assert str(e.value) == text
    return e


def tree(directory):
    out = {}
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            out[os.path.relpath(os.path.join(root, d), directory) + "/"] = None
        for f in files:
            p = os.path.join(root, f)
            st = os.stat(p)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, directory)] = (st.st_size, st.st_mtime_ns, hashlib.sha256(fh.read()).hexdigest())
    return out


# ------------------------------------------------------------------ H4b. an adapter scores as PEFT's merge of it
def test_an_adapter_scores_as_its_merge(tmp_path):
    base = base_dir(tmp_path)
    s, w = opened(tmp_path / "root", base)
    f = factors_for(base, {O0: 2, D1: 3})
    cfg = config(r=2, lora_alpha=4, rank_pattern={"down_proj": 3}, alpha_pattern={"mlp.down_proj": 6})
    ad = write_adapter(tmp_path / "adapter", cfg, f)
    scaling = {O0: 4 / 2, D1: 6 / 3}
    ck, want = merged_checkpoint(tmp_path, base, f, scaling)
    checked = DE.check_evaluate(w, {"path": ad, "check": True})
    other = checked["other"]
    for n in w.names():
        assert torch.equal(other.store.read(n), want[n]), n
    before = tree(ad)
    a, c = DE.evaluate(w, body(ad)), DE.evaluate(w, body(ck))
    assert tree(ad) == before
    assert a["input"]["kind"] == "adapter" and c["input"] == {"kind": "checkpoint"}
    assert [(m["module"], m["param"], m["r"], m["alpha"], m["scaling"], m["transposed"], m["dtype"])
            for m in a["input"]["modules"]] == [(O0, O0 + ".weight", 2, 4, 2.0, False, "F32"),
                                                (D1, D1 + ".weight", 3, 6, 2.0, False, "F32")]
    assert a["compare"]["rows"] == c["compare"]["rows"] and [r["param"] for r in a["compare"]["rows"]] == \
        [O0 + ".weight", D1 + ".weight"]
    assert a["compare"]["dtypes"]["modified"] == {"F32": len(w.names())}           # the base is float32 here
    assert strip(a["modified"]["scores"]) == strip(c["modified"]["scores"])
    assert strip(a["base"]["scores"]) == strip(c["base"]["scores"])
    assert a["modified"]["scores"][1]["value"] > 0 and a["notes"] == [] and a["check"] is False
    chk = DE.evaluate(w, {"path": ad, "check": True})
    assert chk["input"] == a["input"] and chk["compare"]["rows"] == a["compare"]["rows"]


def test_scaling_dtypes_and_fan_in_fan_out(tmp_path):
    base = base_dir(tmp_path)
    s, w = opened(tmp_path / "root", base)
    f = factors_for(base, {O0: 4, D1: 4}, seed=3)
    W = {k: v.float() for k, v in load_file(os.path.join(base, "model.safetensors")).items()}
    # use_rslora: alpha / sqrt(r); factors stored in bfloat16 are upcast first (PeftModel.from_pretrained's autocast);
    # fan_in_fan_out true on an nn.Linear is reset by PEFT, so B @ A is not transposed.
    bf = {m: (a.to(torch.bfloat16), b.to(torch.bfloat16)) for m, (a, b) in f.items()}
    ad = write_adapter(tmp_path / "rs", config(r=4, lora_alpha=3, use_rslora=True, fan_in_fan_out=True), bf,
                       dtype=torch.bfloat16)
    other = DE.check_evaluate(w, {"path": ad, "check": True})["other"]
    for m, (a, b) in bf.items():
        want = W[m + ".weight"] + (b.float() @ a.float()) * (3 / math.sqrt(4))
        assert torch.equal(other.store.read(m + ".weight"), want)
    assert other.input["modules"][0]["dtype"] == "BF16" and other.input["modules"][0]["scaling"] == 1.5
    # A region read (pass 2 reads slabs) is the region of the whole change.
    got = other.store.read(O0 + ".weight", (range(3, 9), range(0, 32)))
    assert torch.equal(got, other.store.read(O0 + ".weight")[3:9])


def test_target_selection_as_peft_matches(tmp_path):
    base = base_dir(tmp_path)
    s, w = opened(tmp_path / "root", base)
    f = factors_for(base, {O0: 2})
    ok = {"suffix": config(target_modules=["o_proj"]), "regex": config(target_modules=r".*layers\.0\.self_attn\.o_proj"),
          "layers": config(target_modules=["o_proj"], layers_to_transform=[0]),
          "exclude": config(target_modules=["o_proj"], exclude_modules=["layers.1.self_attn.o_proj"])}
    for name, cfg in ok.items():
        ad = write_adapter(tmp_path / name, cfg, f)
        assert [m["module"] for m in DE.check_evaluate(w, {"path": ad, "check": True})["other"].input["modules"]] == [O0]
    not_selected = {"layers": config(target_modules=["o_proj"], layers_to_transform=[1]),
                    "exclude": config(target_modules=["o_proj"], exclude_modules=["layers.0.self_attn.o_proj"]),
                    "regex": config(target_modules=r"o_proj"), "list": config(target_modules=[D1])}
    for name, cfg in not_selected.items():
        ad = write_adapter(tmp_path / f"no-{name}", cfg, f)
        refused(w, ad, f"the adapter's file has factors for {O0}, which its target_modules do not select: PEFT would "
                       f"not load them")


def test_a_conv1d_module_is_transposed(tmp_path):
    base = base_dir(tmp_path, "gpt2")
    s, w = opened(tmp_path / "root", base)
    names = w.names()
    t = next(n for n in names if n.endswith("h.0.attn.c_attn.weight"))
    m = t[:-len(".weight")]
    i, o = w.shape(t)                                       # a Conv1D weight is stored [in, out]
    g = torch.Generator().manual_seed(1)
    a, b = torch.randn(2, i, generator=g), torch.randn(o, 2, generator=g)
    ad = write_adapter(tmp_path / "adapter", config(target_modules=[m], lora_alpha=4), {m: (a, b)})
    other = DE.check_evaluate(w, {"path": ad, "check": True})["other"]
    want = w.store.read(t) + (b @ a).T * 2.0
    assert torch.equal(other.store.read(t), want) and other.input["modules"][0]["transposed"] is True
    bad = write_adapter(tmp_path / "bad", config(target_modules=[m]), {m: (torch.randn(2, o, generator=g),
                                                                           torch.randn(i, 2, generator=g))})
    with pytest.raises(DirectionError, match=r"\(a Conv1D weight, stored \[in, out\]\)"):
        DE.check_evaluate(w, {"path": bad, "check": True})


def test_pass_1_reads_only_the_adapted_tensors(tmp_path):
    base = base_dir(tmp_path)
    s, w = opened(tmp_path / "root", base)
    ad = write_adapter(tmp_path / "adapter", config(target_modules=[O0]), factors_for(base, {O0: 2}))
    other = DE.check_evaluate(w, {"path": ad, "check": True})["other"]
    seen = []
    real = other.store.read
    other.store.read = lambda n, r=(): (seen.append(n), real(n, r))[1]
    out = DE.compare_weights(w, other)
    assert set(seen) == {O0 + ".weight"} and out["differ"] == 1
    assert out["tensors"] == len(w.names()) and out["of"] == sum(w.store.numel(n) for n in w.names())


def test_adapter_refusals(tmp_path):
    from selection_fixtures import quant_workspace
    base = base_dir(tmp_path)
    s, w = opened(tmp_path / "root", base)
    f = factors_for(base, {O0: 2})
    k = [0]

    def ad(cfg=None, factors=None, **kw):
        k[0] += 1
        return write_adapter(tmp_path / f"a{k[0]}", config(target_modules=[O0]) if cfg is None else cfg,
                             f if factors is None else factors, **kw)
    cases = [
        (config(peft_type="IA3"), "the adapter's peft_type is \"IA3\"; only a LoRA adapter (peft_type \"LORA\") is read"),
        (config(use_dora=True), "the adapter's use_dora is true: a DoRA adapter (a magnitude vector beside the factors), "
                                "not a plain LoRA"),
        (config(modules_to_save=["lm_head"]), "the adapter's modules_to_save is [\"lm_head\"]: whole modules saved beside "
                                              "the factors, not a plain LoRA"),
        (config(lora_bias=True), "the adapter's lora_bias is true: a bias trained on lora_B, not a plain LoRA"),
        (config(bias="lora_only"), "the adapter's bias is \"lora_only\": it trains bias terms, not a plain LoRA (bias "
                                   "\"none\")"),
        (config(init_lora_weights="pissa"), "the adapter's init_lora_weights is \"pissa\": only true and \"gaussian\" are "
                                            "read (lora_B starts at zero, so a selected module the file lacks changes "
                                            "nothing)"),
        (config(init_lora_weights=False), "the adapter's init_lora_weights is false: only true and \"gaussian\" are read "
                                          "(lora_B starts at zero, so a selected module the file lacks changes nothing)"),
        (config(target_modules="all-linear"), "the adapter's target_modules is \"all-linear\", which PEFT expands from the "
                                              "loaded model's layers; give an adapter whose target_modules lists its "
                                              "modules"),
        (config(r=3), f"the adapter's module {O0} has factors of rank 2, but its config gives it r 3"),
        (config(r=0), "the adapter's r is 0; it must be a positive integer"),
    ]
    for cfg, text in cases:
        refused(w, ad(cfg), text)
    a, b = f[O0]
    refused(w, ad(factors={"model.layers.0.self_attn.nope": (a, b)}, cfg=config(target_modules=["nope"])),
            "the adapter's target module model.layers.0.self_attn.nope maps to no tensor of this model (no "
            "model.layers.0.self_attn.nope.weight)")
    refused(w, ad(factors={"model.layers.0.self_attn.k_proj": (a, b)}, cfg=config(target_modules=["k_proj"])),
            "the adapter's module model.layers.0.self_attn.k_proj changes model.layers.0.self_attn.k_proj.weight by a "
            "[32, 32] product, but model.layers.0.self_attn.k_proj.weight has shape [16, 32] in this model")
    refused(w, ad(extra={f"{PREFIX}{O0}.lora_magnitude_vector": torch.ones(32)}),
            f"the adapter's adapter_model.safetensors holds {PREFIX}{O0}.lora_magnitude_vector, which is not a LoRA factor "
            f"(base_model.model.<module>.lora_A.weight or .lora_B.weight)")
    one = ad()
    t = load_file(os.path.join(one, "adapter_model.safetensors"))
    t.pop(f"{PREFIX}{O0}.lora_B.weight")
    save_file(t, os.path.join(one, "adapter_model.safetensors"))
    refused(w, one, f"the adapter's module {O0} has no lora_B factor")
    refused(w, ad(dtype=torch.float64), f"the adapter's module {O0} has factors stored as F64 and F64; only F32, BF16 and "
                                        f"F16 factors are read")
    pick = ad()
    os.rename(os.path.join(pick, "adapter_model.safetensors"), os.path.join(pick, "adapter_model.bin"))
    refused(w, pick, "the adapter's weights are adapter_model.bin (a pickle), which is not read; save the adapter with "
                     "safetensors (adapter_model.safetensors)")
    os.remove(os.path.join(pick, "adapter_model.bin"))
    refused(w, pick, "the adapter has no adapter_model.safetensors")
    (tmp_path / "a-bad-json").mkdir()
    (tmp_path / "a-bad-json" / "adapter_config.json").write_text("[]")
    refused(w, str(tmp_path / "a-bad-json"), "the adapter's adapter_config.json cannot be read: it is not an object")
    # A GGUF base is refused before the path is looked at, as for a checkpoint; a bundled model has no PEFT module.
    q = quant_workspace(tmp_path / "q")
    refused(q, one, "scoring a modified checkpoint needs a floating-point base: this workspace's base is GGUF (quantized) "
                    "and is never dequantized to compare it with another checkpoint")
    mine = make_ws(tmp_path / "rnd")
    refused(mine, ad(), "a PEFT adapter applies to a transformers checkpoint; this workspace is one of this project's own "
                        "models")


def test_a_tied_head_target_is_refused(tmp_path):
    from test_hf import build
    d = tmp_path / "tied"
    build("llama", str(d))
    cfg = json.loads((d / "config.json").read_text())
    cfg["tie_word_embeddings"] = True
    (d / "config.json").write_text(json.dumps(cfg))
    s, w = opened(tmp_path / "root", str(d))
    o, i = w.shape("lm_head.weight") if "lm_head.weight" in w.names() else w.shape("model.embed_tokens.weight")
    ad = write_adapter(tmp_path / "ad", config(target_modules=["lm_head"]), {"lm_head": (torch.ones(2, i), torch.ones(o, 2))})
    if "lm_head.weight" in w.names():
        refused(w, ad, "the adapter's target module lm_head is the output head, which this model ties to the token "
                       "embedding")
    else:
        refused(w, ad, "the adapter's target module lm_head maps to no tensor of this model (no lm_head.weight)")


def test_the_ide_export_round_trips(tmp_path):
    """An adapter the IDE exports from a workspace's patches, scored from a second workspace over the same base, reads
    as that workspace's current weights (bit for bit for block pieces, whose factors are one-hot)."""
    base = base_dir(tmp_path)
    sa, wa = opened(tmp_path / "a", base)
    assert sa.apply(f"P('{O0}.weight')[0:4] *= 1.5")["committed"]
    AE.export(wa, "x")
    ad = os.path.join(wa.dir, "export", "adapters", "x")
    sb, wb = opened(tmp_path / "b", base)
    other = DE.check_evaluate(wb, {"path": ad, "check": True})["other"]
    assert torch.equal(other.store.read(O0 + ".weight"), wa.read(O0 + ".weight", (), "current"))
    assert DE.evaluate(wb, {"path": ad, "check": True})["compare"]["differ"] == 1


# ------------------------------------------------------------------ H4c. another workspace
def committed(sess):
    """Three patches: a low-rank piece, a block at strength 0.5, a disabled one."""
    ws = sess.ws
    g = torch.Generator().manual_seed(5)
    o, i = ws.shape(O0 + ".weight")
    p1 = ws.commit({O0 + ".weight": [Dl.lowrank(((range(o)), range(i)), torch.randn(o, 2, generator=g) * 0.2,
                                                torch.randn(2, i, generator=g))]}, "low rank")
    p2 = ws.commit({D1 + ".weight": [Dl.block((range(0, 8), range(ws.shape(D1 + ".weight")[1])),
                                              torch.randn(8, ws.shape(D1 + ".weight")[1], generator=g))]}, "block")
    ws.set_patch(p2.id, strength=0.5)
    p3 = ws.commit({D1 + ".weight": [Dl.block((range(8, 9), range(ws.shape(D1 + ".weight")[1])),
                                              torch.ones(1, ws.shape(D1 + ".weight")[1]))]}, "off")
    ws.set_patch(p3.id, enabled=False)
    return [p1.id, p2.id], p3.id


def test_a_workspace_over_the_same_base_scores_as_its_own_edit(tmp_path):
    base = base_dir(tmp_path)
    sa, wa = opened(tmp_path / "a", base)
    on, off = committed(sa)
    own = DS.score(wa, {"state": "current", "max_response_length": 4, "scores": SCORES})
    own_base = DS.score(wa, {"state": "base", "max_response_length": 4, "scores": SCORES})
    sb, wb = opened(tmp_path / "b", base)
    before = tree(wa.dir)
    out = DE.evaluate(wb, body(wa.dir))
    chk = DE.evaluate(wb, {"path": wa.dir, "check": True})
    assert tree(wa.dir) == before                           # nothing in the other workspace changes, nothing is added
    inp = out["input"]
    assert inp == {"kind": "workspace", "name": wa.name, "workspace_kind": "hf", "same_base": True,
                   "patches": [{"id": on[0], "message": "low rank", "strength": 1.0},
                               {"id": on[1], "message": "block", "strength": 0.5}], "disabled": 1,
                   "candidate": "patches"} and chk["input"] == inp
    assert strip(out["modified"]["scores"]) == strip(own["scores"])            # exactly, responses and per-prompt KL
    assert strip(out["base"]["scores"]) == strip(own_base["scores"])
    assert out["modified"]["scores"][1]["value"] > 0
    assert [r["param"] for r in out["compare"]["rows"]] == [O0 + ".weight", D1 + ".weight"]
    d = wa.read(D1 + ".weight", (), "current") - wa.read(D1 + ".weight", (), "base")
    assert out["compare"]["rows"][1]["changed"] == int((d != 0).sum()) == 8 * d.shape[1]
    assert out["compare"]["reconstruction"] == {"inexact": 0, "max_abs": 0.0} and out["notes"] == []
    assert out["compare"]["dtypes"]["modified"] == {"F32": len(wb.names())}


def test_patches_on_the_open_workspace_score_the_difference(tmp_path):
    base = base_dir(tmp_path)
    sa, wa = opened(tmp_path / "a", base)
    committed(sa)
    ck = tmp_path / "a-export"
    shutil.copytree(base, ck)
    os.remove(ck / "model.safetensors")
    wa.export(str(ck))                                      # the other workspace's logical weights as a checkpoint
    sb, wb = opened(tmp_path / "b", base)
    assert sb.apply(f"P('{O0}.weight')[1:3] *= 3")["committed"]
    out, via = DE.evaluate(wb, body(wa.dir)), DE.evaluate(wb, body(str(ck)))
    assert out["input"]["candidate"] == "difference" and out["input"]["same_base"] is True
    assert out["compare"]["rows"] == via["compare"]["rows"]
    assert out["compare"]["reconstruction"] == via["compare"]["reconstruction"]
    assert strip(out["modified"]["scores"]) == strip(via["modified"]["scores"])


def test_a_workspace_over_another_base_is_compared_as_a_checkpoint(tmp_path):
    a, b = make_ws(tmp_path / "one" / "rnd"), make_ws(tmp_path / "two" / "rnd")   # equal weights, different files
    sa = Session(str(tmp_path / "one"))
    sa.open("rnd")
    assert sa.apply("mlp(1).W_out *= 1.5")["committed"]
    ck = tmp_path / "ck"
    ck.mkdir()
    sa.ws.export(str(ck))
    shutil.copy(os.path.join(sa.ws.dir, "tokenizer_meta.json"), ck)
    sb = Session(str(tmp_path / "two"))
    sb.open("rnd")
    out, via = DE.evaluate(sb.ws, body(sa.ws.dir, max_response_length=3,
                                       scores=[{"plugin": "KLDivergence", "prompts": ["ab", "zz"]}])), \
        DE.evaluate(sb.ws, body(str(ck), max_response_length=3, scores=[{"plugin": "KLDivergence", "prompts": ["ab", "zz"]}]))
    assert out["input"]["same_base"] is False and out["input"]["candidate"] == "difference"
    assert out["input"]["workspace_kind"] == "wsp"
    assert out["notes"] == ["the workspace at path is over another base (its weight files are not this workspace's): "
                            "its base with its 1 enabled patch(es) is compared with this base as a checkpoint"]
    assert out["compare"]["rows"] == via["compare"]["rows"] and [r["param"] for r in out["compare"]["rows"]] == \
        ["blocks.1.mlp.W_out"]
    assert strip(out["modified"]["scores"]) == strip(via["modified"]["scores"])
    del a, b


def test_workspace_refusals(tmp_path):
    base = base_dir(tmp_path)
    sa, wa = opened(tmp_path / "a", base)
    sb, wb = opened(tmp_path / "b", base)
    refused(wb, wb.dir, "path is the open workspace itself: its patches are scored against its base by the scores (state "
                        "current), not here")
    refused(wb, wa.dir, "path is a workspace over the open workspace's own base with no enabled patch: there is nothing "
                        "to compare")
    gg = tmp_path / "gguf-ws"
    (gg / "patches").mkdir(parents=True)
    (gg / "workspace.json").write_text(json.dumps({"kind": "gguf", "name": "q", "gguf_path": str(tmp_path / "none.gguf")}))
    before = tree(gg)
    refused(wb, str(gg), "path is a GGUF workspace: its base is quantized and is never dequantized to compare it with this "
                         "base")
    assert tree(gg) == before
    hub = tmp_path / "hub-ws"
    (hub / "patches").mkdir(parents=True)
    (hub / "workspace.json").write_text(json.dumps({"kind": "hf", "name": "h", "model_dir": "model",
                                                    "weights": ["model.safetensors"], "urls": ["https://example.invalid/x"]}))
    refused(wb, str(hub), "the workspace at path has weights still on the Hub; nothing is fetched (pull them in that "
                          "workspace first)")
    odd = tmp_path / "odd-ws"
    odd.mkdir()
    (odd / "workspace.json").write_text(json.dumps({"kind": "mlx"}))
    refused(wb, str(odd), "path is a workspace of kind \"mlx\", which is not read")
    (odd / "workspace.json").write_text("[1]")
    refused(wb, str(odd), "the workspace's workspace.json cannot be read: it is not an object")
    (odd / "workspace.json").write_text(json.dumps({"kind": "hf", "name": "h", "model_dir": "missing"}))
    with pytest.raises(DirectionError, match=r"^the workspace at path cannot be read: "):
        DE.check_evaluate(wb, {"path": str(odd), "check": True})
    # A GGUF base on this side is refused first, whatever path names.
    from selection_fixtures import quant_workspace
    refused(quant_workspace(tmp_path / "q"), wa.dir,
            "scoring a modified checkpoint needs a floating-point base: this workspace's base is GGUF (quantized) and is "
            "never dequantized to compare it with another checkpoint")


# ------------------------------------------------------------------ which kind of directory
def test_the_kind_of_directory(tmp_path):
    base = base_dir(tmp_path)
    s, w = opened(tmp_path / "root", base)
    f = factors_for(base, {O0: 2})
    both = write_adapter(tmp_path / "both", config(target_modules=[O0]), f)
    shutil.copy(os.path.join(base, "model.safetensors"), both)
    refused(w, both, "path holds more than one kind of model: a PEFT adapter (adapter_config.json); checkpoint weights "
                     "(model.safetensors); give the directory of one")
    wsa = write_adapter(tmp_path / "wsa", config(target_modules=[O0]), f)
    (tmp_path / "wsa" / "workspace.json").write_text("{}")
    refused(w, wsa, "path holds more than one kind of model: a PEFT adapter (adapter_config.json); a workspace "
                    "(workspace.json); give the directory of one")
    wsc = tmp_path / "wsc"
    wsc.mkdir()
    (wsc / "workspace.json").write_text("{}")
    for n in ("a.safetensors", "b.safetensors", "c.safetensors", "d.safetensors"):
        (wsc / n).write_bytes(b"")
    refused(w, str(wsc), "path holds more than one kind of model: a workspace (workspace.json); checkpoint weights "
                         "(a.safetensors, b.safetensors, c.safetensors and 1 more); give the directory of one")
    lead = "path holds no model the IDE reads: no PEFT adapter (adapter_config.json), no workspace (workspace.json) and " \
           "no safetensors weights (model.safetensors, an index, or *.safetensors files); "
    empty = tmp_path / "empty"
    empty.mkdir()
    refused(w, str(empty), lead + "it is empty")
    (empty / "config.json").write_text("{}")
    refused(w, str(empty), lead + "it holds 1 entry: config.json")
    run = tmp_path / "run"
    for d in ("adapter", "checkpoints", "dump", "model"):
        (run / d).mkdir(parents=True)
    for n in ("config.toml", "a.log", "b.log", "c.log", "d.log", "adapter_model.safetensors"):
        (run / n).write_text("x")
    refused(w, str(run), lead + "it holds 10 entries: a.log, adapter/, adapter_model.safetensors, b.log, c.log, "
                                "checkpoints/, config.toml, d.log and 2 more")
    # A workspace of this project's own layout without workspace.json is a workspace (wsp.session.is_workspace);
    # base.safetensors alone is a checkpoint's weights.
    from wsp import directions_evaluate_inputs as EI
    old = tmp_path / "old"
    old.mkdir()
    (old / "config.json").write_text("{}")
    (old / "base.safetensors").write_bytes(b"")
    assert EI.detect(str(old)) == "workspace"
    os.remove(old / "config.json")
    assert EI.detect(str(old)) == "checkpoint"


def test_an_adapter_naming_another_base_is_a_note(tmp_path):
    base = base_dir(tmp_path)
    s, w = opened(tmp_path / "root", base)
    f = factors_for(base, {O0: 2})
    w.meta["repo"] = "org/base"                            # in memory only: the Hub repository the base came from
    other = write_adapter(tmp_path / "other", config(target_modules=[O0], base_model_name_or_path="org/other"), f)
    same = write_adapter(tmp_path / "same", config(target_modules=[O0], base_model_name_or_path="org/base"), f)
    assert DE.evaluate(w, {"path": other, "check": True})["notes"] == [
        "the adapter names org/other as its base (base_model_name_or_path); it is applied to the open workspace's base, "
        "org/base"]
    assert DE.evaluate(w, {"path": same, "check": True})["notes"] == []
