"""The patches as a LoRA adapter in PEFT's format (wsp.adapter_export), model-free.

The workspace is the llama-named checkpoint of tests/test_steering_worker.py (o_proj [8, 8], down_proj [8, 10], two
layers, untied lm_head) opened as an `hf` workspace; no model runs. Every expectation reads the written files back
with safetensors and json and rebuilds the change as PEFT does, `lora_B @ lora_A * (alpha_m / r_m)` with the
module's rank and alpha resolved by PEFT's own rule written out here (the first key k of rank_pattern with
re.match(rf"(.*\\.)?({k})$", module), else r), and compares it with the workspace's own delta
(`ws.read(name, (), "delta")`). `peft` is not imported.
"""
import hashlib
import json
import os
import re

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import wsp.store
from wsp import adapter_export as AX
from wsp import arch
from wsp import delta as D
from wsp import directions as DR
from wsp import directions_search as DSR
from wsp.store import norm_index
from wsp.workspace import Workspace
from wsp.wscript import run_script

from test_directions import save_random
from test_steering_worker import D as HD, LAYERS, checkpoint

MODULES = sorted(f"model.layers.{l}.{p}" for l in range(LAYERS) for p in ("self_attn.o_proj", "mlp.down_proj"))
O0, O1 = "model.layers.0.self_attn.o_proj.weight", "model.layers.1.self_attn.o_proj.weight"
DN0, DN1 = "model.layers.0.mlp.down_proj.weight", "model.layers.1.mlp.down_proj.weight"


@pytest.fixture()
def hfws(tmp_path):
    checkpoint(str(tmp_path / "ckpt"))
    return Workspace.create_hf(str(tmp_path / "ws"), "tiny", str(tmp_path / "ckpt"))


def dists(w=0.9):
    return {c: {"max_weight": w, "max_weight_position": 0, "min_weight": 0.6, "min_weight_distance": 1} for c in DR.COMPONENTS}


def ablation(ws, mode="none", rank=3):
    save_random(ws, "r", n_layers=LAYERS, d=HD, seed=7)
    deltas = run_script(ws, DR.script_for({"direction": "r", "weight_distributions": dists(), "row_normalization": mode,
                                           "full_normalization_lora_rank": rank}))["deltas"]
    return ws.commit(deltas, "m")


def whole(ws, name):
    return norm_index((), ws.shape(name))


def ints(*shape, seed=0):
    """Small integers as float32: every product and sum of them is exact, in any order."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(-3, 4, shape, generator=g).to(torch.float32)


def read(directory):
    cfg = json.load(open(os.path.join(directory, "adapter_config.json")))
    tensors = load_file(os.path.join(directory, "adapter_model.safetensors"))
    return cfg, tensors


def peft_value(cfg, module, which):
    """PEFT's rank / alpha of one module: the first rank_pattern (alpha_pattern) key k with
    re.match(rf"(.*\\.)?({k})$", module), else the config's r (lora_alpha)."""
    pattern = cfg["rank_pattern"] if which == "r" else cfg["alpha_pattern"]
    for k in pattern:
        if re.match(rf"(.*\.)?({k})$", module):
            return pattern[k]
    return cfg["r"] if which == "r" else cfg["lora_alpha"]


def factors(tensors, module):
    return tensors[f"base_model.model.{module}.lora_B.weight"], tensors[f"base_model.model.{module}.lora_A.weight"]


def rebuilt(cfg, tensors, module):
    B, A = factors(tensors, module)
    return B @ A * (peft_value(cfg, module, "alpha") / peft_value(cfg, module, "r"))


def relative(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def typed_config(r, modules, rank_pattern=None, base=None, revision=None):
    rp = rank_pattern or {}
    return {"alpha_pattern": dict(rp), "base_model_name_or_path": base, "bias": "none", "exclude_modules": None,
            "fan_in_fan_out": False, "inference_mode": True, "init_lora_weights": True, "layers_pattern": None,
            "layers_to_transform": None, "lora_alpha": r, "lora_bias": False, "lora_dropout": 0.0,
            "modules_to_save": None, "peft_type": "LORA", "r": r, "rank_pattern": dict(rp), "revision": revision,
            "target_modules": sorted(modules), "target_parameters": None, "task_type": "CAUSAL_LM", "use_dora": False,
            "use_rslora": False}


# ------------------------------------------------------------------ X1
def test_one_ablation_patch_is_a_uniform_rank_one_adapter(hfws):
    ws = hfws
    p = ablation(ws, "none")
    out = AX.export(ws, "a1")
    d = os.path.join(ws.dir, "export", "adapters", "a1")
    assert out["dir"] == d and out["written"] and not out["dry_run"]
    cfg, tensors = read(d)
    assert cfg == typed_config(1, MODULES)
    text = open(os.path.join(d, "adapter_config.json")).read()
    assert text == json.dumps(cfg, indent=2, sort_keys=True) and not text.endswith("\n")     # PEFT's own form
    assert sorted(tensors) == sorted(f"base_model.model.{m}.lora_{ab}.weight" for m in MODULES for ab in "AB")
    with safe_open(os.path.join(d, "adapter_model.safetensors"), "pt") as f:
        assert f.metadata() == {"format": "pt"}
    for m in MODULES:
        out_, in_ = ws.shape(m + ".weight")
        B, A = factors(tensors, m)
        assert A.shape == (1, in_) and B.shape == (out_, 1) and A.dtype == B.dtype == torch.float32
        want = ws.read(m + ".weight", (), "delta")
        assert relative(rebuilt(cfg, tensors, m), want) <= 1e-6, m
    assert out["layout"] == "uniform" and out["r"] == 1 and out["rank_pattern"] == {}
    assert [e["id"] for e in out["patches"]] == [p.id]


# ------------------------------------------------------------------ X2
def test_full_keeps_its_rank(hfws):
    ws = hfws
    ablation(ws, "full", rank=3)
    out = AX.export(ws, "a1")
    cfg, tensors = read(out["dir"])
    assert cfg["r"] == cfg["lora_alpha"] == 3 and cfg["rank_pattern"] == {} and cfg["alpha_pattern"] == {}
    assert [e["rank"] for e in out["modules"]] == [3] * len(MODULES)
    for m in MODULES:
        B, A = factors(tensors, m)
        assert B.shape[1] == A.shape[0] == 3
        assert relative(rebuilt(cfg, tensors, m), ws.read(m + ".weight", (), "delta")) <= 1e-6


# ------------------------------------------------------------------ X3
def test_block_and_sparse_pieces_come_back_exactly(hfws):
    ws = hfws
    g = torch.Generator().manual_seed(1)
    wide = D.block((range(1, 4), range(0, 8)), torch.randn(3, 8, generator=g))           # m = 3 <= n = 8
    tall = D.block((range(0, 8), range(2, 4)), torch.randn(8, 2, generator=g))           # m = 8 > n = 2
    row = D.block((5, range(0, 10)), torch.randn(10, generator=g))                       # an int axis: one row
    coords = torch.tensor([[0, 1], [2, 5], [2, 7]])
    sparse = D.Piece("sparse", whole(ws, O1), values=torch.randn(3, generator=g), coords=coords)
    ws.commit({O0: [wide], DN0: [tall], DN1: [row], O1: [sparse]}, "m")
    out = AX.export(ws, "a1", layout="per_module")
    cfg, tensors = read(out["dir"])
    ranks = {e["tensor"]: e["rank"] for e in out["modules"]}
    assert ranks == {O0: 3, DN0: 2, DN1: 1, O1: 2}                 # m, n, one row, min(|R| = 2, |C| = 3)
    assert {e["tensor"]: [pc["kind"] for pc in e["pieces"]] for e in out["modules"]} == \
        {O0: ["block"], DN0: ["block"], DN1: ["block"], O1: ["sparse"]}
    assert all(e["exact"] for e in out["modules"])
    for name in (O0, DN0, DN1, O1):
        m = AX.module_of(name)
        B, A = factors(tensors, m)
        assert B.shape[1] == A.shape[0] == ranks[name] == peft_value(cfg, m, "r")
        assert torch.equal(rebuilt(cfg, tensors, m), ws.read(name, (), "delta")), name


# ------------------------------------------------------------------ X4
def two_patches(ws):
    """Patch 1: rank 1 on o_proj of layer 1 and rank 1 on down_proj of layer 0; patch 2: rank 2 on a part of the same
    o_proj, at strength 0.5; patch 3 (switched off): rank 1 on o_proj of layer 0."""
    A1, B1, C1, E1 = ints(8, 1, seed=1), ints(1, 8, seed=2), ints(8, 1, seed=3), ints(1, 10, seed=4)
    A2, B2 = ints(4, 2, seed=5), ints(2, 6, seed=6)
    p1 = ws.commit({O1: [D.lowrank(whole(ws, O1), A1, B1)], DN0: [D.lowrank(whole(ws, DN0), C1, E1)]}, "one")
    p2 = ws.commit({O1: [D.lowrank((range(2, 6), range(1, 7)), A2, B2)]}, "two")
    ws.set_patch(p2.id, strength=0.5)
    p3 = ws.commit({O0: [D.lowrank(whole(ws, O0), ints(8, 1, seed=7), ints(1, 8, seed=8))]}, "three")
    ws.set_patch(p3.id, enabled=False)
    return (p1, p2, p3), (A1, B1, A2, B2)


def test_patches_concatenate_and_strengths_fold(hfws):
    ws = hfws
    (p1, p2, p3), (A1, B1, A2, B2) = two_patches(ws)
    out = AX.export(ws, "a1")
    cfg, tensors = read(out["dir"])
    assert [e["id"] for e in out["patches"]] == [p1.id, p2.id]                     # the switched-off one is absent
    assert cfg["target_modules"] == sorted([AX.module_of(O1), AX.module_of(DN0)])
    assert {e["tensor"]: e["rank"] for e in out["modules"]} == {O1: 3, DN0: 1}
    # auto: the ranks differ, so per_module, with escaped keys and alpha_pattern equal to rank_pattern
    assert out["layout"] == "per_module" and cfg["r"] == cfg["lora_alpha"] == 3
    assert cfg["rank_pattern"] == {re.escape(AX.module_of(DN0)): 1} == cfg["alpha_pattern"]
    assert "\\." in next(iter(cfg["rank_pattern"]))
    B, A = factors(tensors, AX.module_of(O1))
    assert B.shape == (8, 3) and A.shape == (3, 8)
    # patch 1 first (stack order), then patch 2 with its strength folded into lora_B, its B as lora_A's rows
    assert torch.equal(B[:, :1], A1) and torch.equal(A[:1], B1)
    assert torch.equal(B[2:6, 1:3], 0.5 * A2) and not B[[0, 1, 6, 7], 1:3].any()
    assert torch.equal(A[1:3, 1:7], B2) and not A[1:3, [0, 7]].any()
    for name in (O1, DN0):
        assert torch.equal(rebuilt(cfg, tensors, AX.module_of(name)), ws.read(name, (), "delta")), name
    assert all(e["exact"] for e in out["modules"])                                 # 0.5 is a power of two
    # The switched-off patch asked for by id: included at its strength, with the note.
    out = AX.plan(ws, patches=[p3.id, p1.id])
    assert [e["id"] for e in out["patches"]] == [p1.id, p3.id]                     # stack order, not request order
    assert f"patch {p3.id} is switched off in the workspace; the adapter includes it at strength 1.0" in out["notes"]


# ------------------------------------------------------------------ X5
def test_the_three_layouts(hfws):
    ws = hfws
    two_patches(ws)
    want = {name: ws.read(name, (), "delta") for name in (O1, DN0)}
    uni = AX.export(ws, "u", layout="uniform")
    cfg, tensors = read(uni["dir"])
    assert uni["layout"] == "uniform" and cfg["r"] == 3 and cfg["rank_pattern"] == {} == cfg["alpha_pattern"]
    B, A = factors(tensors, AX.module_of(DN0))
    assert B.shape == (8, 3) and A.shape == (3, 10)
    assert torch.equal(B[:, 1:], torch.zeros(8, 2)) and torch.equal(A[1:], torch.zeros(2, 10))     # exact zeros
    assert not torch.signbit(B[:, 1:]).any() and not torch.signbit(A[1:]).any()
    for name in (O1, DN0):
        assert torch.equal(rebuilt(cfg, tensors, AX.module_of(name)), want[name])
    per = AX.export(ws, "p", layout="per_module")
    cfg, tensors = read(per["dir"])
    names = [n[:-len(".weight")] for n in ws.names() if n.endswith(".weight") and len(ws.shape(n)) == 2]
    assert cfg["rank_pattern"]
    for k in cfg["rank_pattern"]:
        assert [m for m in names if re.match(rf"(.*\.)?({k})$", m)] == [AX.module_of(DN0)]
    for name in (O1, DN0):
        m = AX.module_of(name)
        B, A = factors(tensors, m)
        assert B.shape[1] == A.shape[0] == peft_value(cfg, m, "r") == {O1: 3, DN0: 1}[name]
        assert torch.equal(rebuilt(cfg, tensors, m), want[name])
    assert AX.plan(ws)["layout"] == "per_module"                                   # auto, ranks differ
    p1 = ws.patches[0]
    assert AX.plan(ws, patches=[p1.id])["layout"] == "uniform"                     # auto, ranks equal (1 and 1)
    assert AX.plan(ws, patches=[p1.id], layout="per_module")["rank_pattern"] == {}


# ------------------------------------------------------------------ X6
def test_a_strength_that_is_not_a_power_of_two_is_marked(hfws):
    ws = hfws
    g = torch.Generator().manual_seed(2)
    p = ws.commit({O1: [D.lowrank(whole(ws, O1), torch.randn(8, 2, generator=g), torch.randn(2, 8, generator=g))]}, "m")
    ws.set_patch(p.id, strength=0.3)
    out = AX.export(ws, "a1")
    cfg, tensors = read(out["dir"])
    assert [e["exact"] for e in out["modules"]] == [False]
    assert out["modules"][0]["pieces"] == [{"patch": p.id, "kind": "lowrank", "rank": 2, "strength": 0.3}]
    assert (f"patch {p.id} has strength 0.3: its factors are scaled, which rounds each value once (the IDE scales the "
            f"product instead)") in out["notes"]
    assert relative(rebuilt(cfg, tensors, AX.module_of(O1)), ws.read(O1, (), "delta")) <= 1e-6
    ws.set_patch(p.id, strength=-2.0)                                              # a power of two: exact again
    assert [e["exact"] for e in AX.plan(ws)["modules"]] == [True]


# ------------------------------------------------------------------ X7
def refused(fn, *a, **kw):
    with pytest.raises(AX.AdapterError) as e:
        fn(*a, **kw)
    return str(e.value)


def extra_checkpoint(directory, extra):
    """The fixture checkpoint with more tensors in its file."""
    t = checkpoint(str(directory))
    save_file({**t, **extra}, os.path.join(str(directory), "model.safetensors"))


def test_refusals(hfws, tmp_path, monkeypatch):
    ws = hfws
    from test_steering import make_ws
    from selection_fixtures import quant_workspace
    # A1, A2, A3: the kind of workspace
    assert refused(AX.plan, make_ws(tmp_path / "own")) == (
        "a PEFT adapter is for a transformers checkpoint; this is one of this project's own models: use the existing "
        "export (GET /api/export)")
    q = quant_workspace(tmp_path / "q")
    q.commit({O1: [D.lowrank(whole(q, O1), torch.ones(16, 1), torch.ones(1, 16))]}, "m")
    text = refused(AX.plan, q)
    assert "companion" in text and text == (
        "this workspace's base is a quantized GGUF file: a PEFT adapter applies to a floating-point transformers "
        "checkpoint, and these patches were computed against the quantized weights, so the adapter's base would be the "
        "dequantized model, which the IDE never builds. Export the WSP companion bundle (GET /api/export), or apply the "
        "edit to the floating-point checkpoint the GGUF was made from")
    extra_checkpoint(tmp_path / "ck-int", {"model.extra_ids": torch.arange(4)})
    packed = Workspace.create_hf(str(tmp_path / "ws-int"), "int", str(tmp_path / "ck-int"))
    assert refused(AX.plan, packed) == ("this checkpoint stores packed or scaled weights (I64 in model.extra_ids), which "
                                        "the IDE does not decode; no adapter can be exported")
    # A4, A4b, the 404, A13
    assert refused(AX.plan, ws) == "nothing to export: no patch is enabled"
    p = ws.commit({O1: [D.lowrank(whole(ws, O1), torch.ones(8, 1), torch.ones(1, 8))]}, "m")
    assert refused(AX.plan, ws, patches=[]) == "nothing to export: patches is empty"
    ws.set_patch(p.id, strength=0)
    assert refused(AX.plan, ws) == "nothing to export: no patch is enabled"
    assert refused(AX.plan, ws, patches=[p.id]) == f"patch {p.id} has strength 0: it changes nothing"
    ws.set_patch(p.id, strength=1)
    with pytest.raises(AX.AdapterMissing) as e:
        AX.plan(ws, patches=["9999-00000"])
    assert str(e.value) == "no patch '9999-00000'" and isinstance(e.value, AX.AdapterError)
    for bad in ("x", [1], [p.id, p.id], {"a": 1}):
        assert refused(AX.plan, ws, patches=bad) == "patches must be null or a list of distinct patch ids"
    # A9, A10, A12
    assert refused(AX.plan, ws, layout="flat") == "layout must be 'auto', 'uniform' or 'per_module'"
    for bad in ("", "a\nb", "x" * 257, 5, "a\x7fb"):
        assert refused(AX.plan, ws, base_model_name_or_path=bad) == (
            "base_model_name_or_path must be text of 1..256 characters without control characters, or null")
    assert AX.plan(ws, base_model_name_or_path="x" * 256)["base_model_name_or_path"] == "x" * 256
    for bad in ("", "a b", ".hidden", "x" * 65, None, 3):
        assert refused(AX.export, ws, bad) == "name must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
    # A8
    AX.export(ws, "a1")
    assert refused(AX.export, ws, "a1") == ("an adapter named 'a1' exists in this workspace; pass replace, or choose "
                                            "another name")
    # A7: the safetensors size plus 1 MiB against the free space less the reserve
    need = AX.plan(ws)["bytes"] + (1 << 20)
    monkeypatch.setattr(DR, "_free_bytes", lambda path: DR.DISK_RESERVE + need - 1)
    assert refused(AX.export, ws, "a2") == (
        f"the adapter needs {need:,} bytes; the workspace's file system has {DR.DISK_RESERVE + need - 1:,} bytes free, "
        f"of which {DR.DISK_RESERVE:,} stay free")
    assert not os.path.exists(os.path.join(ws.dir, "export", "adapters", "a2"))
    monkeypatch.setattr(DR, "_free_bytes", lambda path: DR.DISK_RESERVE + need)
    assert AX.export(ws, "a2")["written"]
    # A6: more rank than min(out, in) = 8
    ws.commit({O0: [D.lowrank(whole(ws, O0), torch.ones(8, 1) * (i + 1), torch.ones(1, 8)) for i in range(9)]}, "nine")
    assert refused(AX.plan, ws) == (f"{O0} would need rank 9 in the adapter, above min(out, in) = 8: export fewer "
                                    f"patches at once")
    # A11: a sparse piece's box above STREAM_ELEMS
    q2 = Workspace.create_hf(str(tmp_path / "ws2"), "tiny", str(tmp_path / "ckpt"))
    sp = q2.commit({O1: [D.Piece("sparse", whole(q2, O1), values=torch.ones(3), coords=torch.tensor([[0, 1], [2, 5], [2, 7]]))]}, "s")
    monkeypatch.setattr(AX, "STREAM_ELEMS", 5)
    assert refused(AX.plan, q2) == (f"patch {sp.id} changes {O1} at 3 scattered weights spread over 2 rows and 3 "
                                    f"columns: too wide to factor exactly")
    monkeypatch.setattr(AX, "STREAM_ELEMS", 6)
    assert AX.plan(q2)["modules"][0]["rank"] == 2
    # A5: a norm, a bias, the token embedding
    extra_checkpoint(tmp_path / "ck-bias", {"model.layers.0.self_attn.q_proj.bias": torch.zeros(HD)})
    wb = Workspace.create_hf(str(tmp_path / "ws-bias"), "bias", str(tmp_path / "ck-bias"))
    tail = ": a LoRA adapter holds changes of linear layers' weight matrices only"
    for name, piece, reason in (
            ("model.layers.0.input_layernorm.weight", D.block((range(0, HD),), torch.ones(HD)), "a normalisation weight"),
            ("model.layers.0.self_attn.q_proj.bias", D.block((range(0, HD),), torch.ones(HD)), "a bias"),
            ("model.embed_tokens.weight", D.block((3, range(0, HD)), torch.ones(HD)),
             "the token embedding (PEFT's embedding adapters are not written by this export)")):
        bad = wb.commit({name: [piece]}, "bad")
        assert refused(AX.plan, wb) == f"patch {bad.id} changes {name}, {reason}{tail}"
        wb.delete_patch(bad.id)
    # the untied unembedding is a linear layer
    wb.commit({"lm_head.weight": [D.block((3, range(0, HD)), torch.ones(HD))]}, "head")
    assert [e["module"] for e in AX.plan(wb)["modules"]] == ["lm_head"]


# ------------------------------------------------------------------ X8
def snapshot(ws):
    pdir = os.path.join(ws.dir, "patches")
    files = sorted((fn, os.stat(os.path.join(pdir, fn)).st_mtime_ns) for fn in os.listdir(pdir))
    meta = os.path.join(ws.dir, "workspace.json")
    return (ws.version, files, open(meta).read(), os.stat(meta).st_mtime_ns,
            [(p.id, p.enabled, p.strength) for p in ws.patches])


def test_the_export_never_changes_the_workspace(hfws):
    ws = hfws
    (p1, p2, p3), _ = two_patches(ws)
    before = snapshot(ws)
    AX.plan(ws)
    AX.plan(ws, patches=[p3.id])
    assert AX.request(ws, {"name": "a1", "dry_run": True})["written"] is False
    assert not os.path.exists(os.path.join(ws.dir, "export"))                       # a dry run writes no directory
    assert snapshot(ws) == before
    AX.export(ws, "a1")
    AX.export(ws, "a1", patches=[p3.id], replace=True)
    assert snapshot(ws) == before
    assert sorted(os.listdir(os.path.join(ws.dir, "export", "adapters"))) == ["a1"]


# ------------------------------------------------------------------ X9
def test_base_model_name_or_path(hfws, tmp_path):
    ws = hfws
    ablation(ws, "none")
    note = ("no Hub repository is known for this checkpoint, so base_model_name_or_path is null: pass it, or load the "
            "base yourself and give PeftModel.from_pretrained this directory")
    out = AX.export(ws, "local")
    cfg, _ = read(out["dir"])
    assert cfg["base_model_name_or_path"] is None and cfg["revision"] is None and out["notes"][0] == note
    ckpt = str(tmp_path / "ckpt")
    for fn in os.listdir(out["dir"]):
        blob = open(os.path.join(out["dir"], fn), "rb").read()
        for path in {ckpt, os.path.realpath(ckpt), str(tmp_path), os.path.realpath(str(tmp_path))}:
            assert path.encode() not in blob, fn
    sha = "0123456789abcdef0123456789abcdef01234567"
    hub = Workspace.create_hf(str(tmp_path / "ws-hub"), "tiny", ckpt, repo="org/tiny-model", sha=sha)
    ablation(hub, "none")
    out = AX.export(hub, "hub")
    cfg, _ = read(out["dir"])
    assert (cfg["base_model_name_or_path"], cfg["revision"]) == ("org/tiny-model", sha) and note not in out["notes"]
    assert (out["base_model_name_or_path"], out["revision"]) == ("org/tiny-model", sha)
    given = AX.plan(hub, base_model_name_or_path="someone/base")
    assert (given["base_model_name_or_path"], given["revision"]) == ("someone/base", None)
    given = AX.plan(ws, base_model_name_or_path="someone/base")
    assert (given["base_model_name_or_path"], given["revision"]) == ("someone/base", None) and note not in given["notes"]


# ------------------------------------------------------------------ X10
def test_the_route(tmp_path):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    from wsp.session import Session
    checkpoint(str(tmp_path / "ckpt"))
    Workspace.create_hf(str(tmp_path / "root" / "tiny"), "tiny", str(tmp_path / "ckpt"))
    sess = Session(str(tmp_path / "root"))
    ws = sess.open("tiny")
    c = TestClient(create_app(sess, "http://127.0.0.1:0"), raise_server_exceptions=False)
    r = c.post("/api/export_adapter", json={"name": "a1", "dry_run": False})
    assert (r.status_code, r.json()) == (400, {"error": "AdapterError: nothing to export: no patch is enabled"})
    (p1, p2, p3), _ = two_patches(ws)
    r = c.post("/api/export_adapter", json={"name": "a1", "format": "zip"})
    assert (r.status_code, r.json()) == (400, {"error": "AdapterError: format: unknown field; the fields are name, "
                                                        "patches, layout, base_model_name_or_path, replace and dry_run"})
    r = c.post("/api/export_adapter", json={"name": "a1", "patches": ["9999-00000"]})
    assert (r.status_code, r.json()) == (404, {"error": "AdapterError: no patch '9999-00000'"})
    for body in ({"name": "a1", "dry_run": "yes"}, {"name": "a1", "replace": 1}):
        assert c.post("/api/export_adapter", json=body).status_code == 400
    assert c.post("/api/export_adapter", json={"dry_run": False}).json() == {
        "error": "AdapterError: name must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"}
    keys = {"dry_run", "written", "files", "layout", "r", "rank_pattern", "bytes", "modules", "patches",
            "base_model_name_or_path", "revision", "family", "validated_family", "notes"}
    dry = c.post("/api/export_adapter", json={"name": "a1", "patches": None, "layout": "auto",
                                              "base_model_name_or_path": None, "replace": False, "dry_run": True})
    assert dry.status_code == 200 and set(dry.json()) == keys
    assert dry.json()["written"] is False and dry.json()["files"] == [] and dry.json()["dry_run"] is True
    assert not os.path.exists(os.path.join(ws.dir, "export"))
    r = c.post("/api/export_adapter", json={"name": "a1"})
    out = r.json()
    assert r.status_code == 200 and set(out) == keys | {"dir"} and out["written"] is True and out["dry_run"] is False
    assert out["dir"] == os.path.join(ws.dir, "export", "adapters", "a1")
    assert {k: v for k, v in out.items() if k not in ("dry_run", "written", "files", "dir")} == \
        {k: v for k, v in dry.json().items() if k not in ("dry_run", "written", "files")}
    assert [f["name"] for f in out["files"]] == ["adapter_config.json", "adapter_model.safetensors", "wsp_adapter.json"]
    for f in out["files"]:
        path = os.path.join(out["dir"], f["name"])
        assert f["sha256"] == hashlib.sha256(open(path, "rb").read()).hexdigest() and f["bytes"] == os.path.getsize(path)
    assert out["bytes"] == os.path.getsize(os.path.join(out["dir"], "adapter_model.safetensors"))
    r = c.post("/api/export_adapter", json={"name": "a1", "patches": [p3.id]})
    assert (r.status_code, r.json()) == (400, {"error": "AdapterError: an adapter named 'a1' exists in this workspace; "
                                                        "pass replace, or choose another name"})
    r = c.post("/api/export_adapter", json={"name": "a1", "patches": [p3.id], "replace": True})
    assert r.status_code == 200 and [e["id"] for e in r.json()["patches"]] == [p3.id]
    cfg, _ = read(out["dir"])
    assert cfg["target_modules"] == [AX.module_of(O0)]                              # the new adapter replaced the old
    assert sorted(os.listdir(os.path.join(ws.dir, "export", "adapters"))) == ["a1"]


# ------------------------------------------------------------------ X11
def test_a_failed_write_leaves_nothing(hfws, monkeypatch):
    ws = hfws
    ablation(ws, "none")
    parent = os.path.join(ws.dir, "export", "adapters")
    real = wsp.store.write_safetensors

    def boom(*a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(wsp.store, "write_safetensors", boom)
    with pytest.raises(OSError, match="disk full"):
        AX.export(ws, "a1")
    assert os.listdir(parent) == []
    monkeypatch.setattr(wsp.store, "write_safetensors", real)
    AX.export(ws, "a1")
    old = {fn: open(os.path.join(parent, "a1", fn), "rb").read() for fn in os.listdir(os.path.join(parent, "a1"))}
    monkeypatch.setattr(wsp.store, "write_safetensors", boom)
    with pytest.raises(OSError, match="disk full"):
        AX.export(ws, "a1", replace=True)
    assert os.listdir(parent) == ["a1"]
    assert {fn: open(os.path.join(parent, "a1", fn), "rb").read() for fn in os.listdir(os.path.join(parent, "a1"))} == old
    # a failure after the factors were written (the provenance file) leaves nothing either
    monkeypatch.setattr(wsp.store, "write_safetensors", real)
    monkeypatch.setattr(AX, "base_files", lambda ws: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        AX.export(ws, "a2")
    assert os.listdir(parent) == ["a1"]


# ------------------------------------------------------------------ X12
def test_the_target_table():
    s = arch.Structure()
    assert AX.target_reason(arch.Part("o.weight", "W_O", ("resid", "headdh")), s) is None
    assert AX.target_reason(arch.Part("o.weight", "W_O", ("headdh", "resid")), s) == (
        "a Conv1D weight stored [in, out] (GPT-2 style); PEFT needs fan_in_fan_out for it, which this export does not write")
    assert AX.target_reason(arch.Part("fc.weight", "W_in", ("neuron", "resid")), s) is None
    assert AX.target_reason(arch.Part("fc.weight", "W_in", ("resid", "neuron")), s) == AX.CONV1D
    assert AX.target_reason(arch.Part("q.weight", "W_Q", ("headdh", "resid")), s) is None
    assert AX.target_reason(arch.Part("d.weight", "W_out", ("resid", "neuron")), s) is None
    assert AX.target_reason(arch.Part("d.weight", "W_out", ("neuron", "resid")), s) == AX.CONV1D
    head = arch.Part("lm_head.weight", "W_U", ("token", "resid"))
    assert AX.target_reason(head, s) is None
    assert AX.target_reason(head, arch.Structure(tied=True)) == "the unembedding tied to the token embedding"
    assert AX.target_reason(arch.Part("b", "b_O", ("resid",)), s) == "a bias"
    assert AX.target_reason(arch.Part("n", "norm_b", ("resid",)), s) == "a normalisation weight"
    assert AX.target_reason(arch.Part("p", "W_pos", ("pos", "resid")), s) == "a position embedding"
    assert AX.target_reason(arch.Part("x", "W_mystery", ("resid", "neuron")), s) == "not a matrix the IDE recognises in this model"


# ------------------------------------------------------------------ X13
def test_wsp_adapter_json(hfws):
    ws = hfws
    (p1, p2, p3), _ = two_patches(ws)
    out = AX.export(ws, "a1")
    d = out["dir"]
    prov = json.load(open(os.path.join(d, "wsp_adapter.json")))
    cfg, _ = read(d)
    assert prov["version"] == 1 and prov["format"] == "peft-lora" and prov["peft_validated"] == "0.21.2"
    assert prov["heretic"] == DSR.HERETIC and prov["workspace"] == "tiny"
    assert [e["digest"] for e in prov["patches"]] == [DSR.piece_digest(ws.patch(p.id).deltas) for p in (p1, p2)]
    assert [(e["id"], e["strength"], e["enabled"], e["record"]) for e in prov["patches"]] == \
        [(p1.id, 1.0, True, False), (p2.id, 0.5, True, False)]
    for fn in ("adapter_config.json", "adapter_model.safetensors"):
        path = os.path.join(d, fn)
        assert prov["files"][fn] == {"bytes": os.path.getsize(path), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    assert sorted(prov["files"]) == ["adapter_config.json", "adapter_model.safetensors"]
    assert out["bytes"] == os.path.getsize(os.path.join(d, "adapter_model.safetensors"))
    assert [f["name"] for f in out["files"]] == ["adapter_config.json", "adapter_model.safetensors", "wsp_adapter.json"]
    for f in out["files"]:
        path = os.path.join(d, f["name"])
        assert (f["bytes"], f["sha256"]) == (os.path.getsize(path), hashlib.sha256(open(path, "rb").read()).hexdigest())
    assert [e["module"] for e in prov["modules"]] == cfg["target_modules"]
    for e in prov["modules"]:
        assert e["rank"] == peft_value(cfg, e["module"], "r") and e["tensor"] == e["module"] + ".weight"
        assert e["shape"] == list(ws.shape(e["tensor"]))
    assert prov["modules"] == out["modules"] and prov["layout"] == out["layout"] == "per_module"
    assert prov["r"] == cfg["r"] == 3
    assert prov["base"] == {"repo": None, "sha": None, "files": [
        {"name": "model.safetensors", "bytes": os.path.getsize(ws.store.files[0].path), "sha256": None}]}
    assert prov["family"] == "llama" and prov["validated_family"] is False and out["validated_family"] is False
    family = ("this model type (llama) has not been loaded with PEFT by the IDE's acceptance; PEFT refuses an adapter "
              "whose module names do not exist")
    assert family in prov["notes"] and family in out["notes"] and prov["notes"] == out["notes"]
    assert open(os.path.join(d, "wsp_adapter.json")).read() == json.dumps(prov, indent=1, sort_keys=True)


def test_a_record_and_a_cached_base_hash_are_read_when_present(hfws):
    """The record capture's files (written by the reproduction record): a patch's record says what it was made on,
    and the base hash cache gives the base files' SHA-256 while their size and modification time are unchanged."""
    ws = hfws
    (p1, p2, p3), _ = two_patches(ws)
    records = os.path.join(ws.dir, "directions", "records")
    os.makedirs(records)
    made_on = [[p1.id, 1.0, p1.created, p1.stored], [p3.id, 1.0, p3.created, p3.stored]]
    json.dump({"version": 1, "patch": {"id": p2.id, "created": p2.created}, "made_on": made_on},
              open(os.path.join(records, f"{p2.id}.json"), "w"))
    json.dump({"version": 1, "patch": {"id": p1.id, "created": p1.created + 1}, "made_on": [["0000-00000", 1, 0, 1]]},
              open(os.path.join(records, f"{p1.id}.json"), "w"))                   # stale: another patch had this id
    f = ws.store.files[0].path
    st = os.stat(f)
    json.dump({"files": [["model.safetensors", st.st_size, st.st_mtime_ns, "ab" * 32]]},
              open(os.path.join(records, "base.json"), "w"))
    out = AX.export(ws, "a1")
    assert [(e["id"], e["record"]) for e in out["patches"]] == [(p1.id, False), (p2.id, True)]
    assert f"patch {p2.id} was made on top of patches {p3.id}, which this adapter does not include" in out["notes"]
    assert not any("0000-00000" in n for n in out["notes"])
    prov = json.load(open(os.path.join(out["dir"], "wsp_adapter.json")))
    assert prov["base"]["files"] == [{"name": "model.safetensors", "bytes": st.st_size, "sha256": "ab" * 32}]
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1000))                      # touched: the cached hash is stale
    assert AX.base_files(ws) == [{"name": "model.safetensors", "bytes": st.st_size, "sha256": None}]
    assert AX.plan(ws, patches=[p2.id, p3.id])["notes"].count(
        f"patch {p2.id} was made on top of patches {p1.id}, which this adapter does not include") == 1


# ------------------------------------------------------------------ PEFT matches module names on their ending
def peft_rank(rank_pattern, r, module):
    """PEFT's get_pattern_key, written out: the first key k of rank_pattern (in its order) with
    re.match(rf"(.*\\.)?({k})$", module) gives the rank, else r."""
    for k in rank_pattern:
        if re.match(rf"(.*\.)?({k})$", module):
            return rank_pattern[k]
    return r


DOWN = "model.layers.1.mlp.down_proj"
VDOWN = "vision.model.layers.1.mlp.down_proj"


def test_pattern_collision_on_typed_module_lists():
    # a (rank 1, a pattern entry since r = 3) is an ending of m (rank 3): PEFT would give m rank 1
    assert AX.pattern_collision({DOWN: 1, VDOWN: 3, "x.o_proj": 3}, 3) == (DOWN, VDOWN)
    # m has a key of its own but a's key sorts first: m would still get a's rank
    assert AX.pattern_collision({DOWN: 1, VDOWN: 2, "x.o_proj": 3}, 3) == (DOWN, VDOWN)
    assert AX.pattern_collision({DOWN: 1, VDOWN: 1, "x.o_proj": 3}, 3) is None            # equal ranks
    assert AX.pattern_collision({DOWN: 3, VDOWN: 1}, 3) is None                           # a at the default rank
    assert AX.pattern_collision({DOWN: 1, "vmodel.layers.1.mlp.down_proj": 3}, 3) is None  # no dot before the end


def test_rank_patterns_resolve_to_each_modules_rank_under_peft_rule():
    colliding = {DOWN: 1, VDOWN: 3, "model.layers.0.self_attn.o_proj": 2}
    # Without the fix: the escaped keys of the per-module layout give VDOWN the rank of DOWN under PEFT's rule.
    naive = {re.escape(m): k for m, k in sorted(colliding.items()) if k != 3}
    assert peft_rank(naive, 3, VDOWN) == 1 != colliding[VDOWN]
    # auto: resolved to uniform with the note; every module then has r columns (padded), which PEFT reads as r.
    resolved, r, rank_pattern, notes = AX._layout(colliding, "auto")
    assert (resolved, r, rank_pattern) == ("uniform", 3, {})
    assert notes == [f"module names {DOWN} and {VDOWN} end alike and have different ranks, which PEFT's rank patterns "
                     f"cannot tell apart: the uniform layout is used"]
    assert all(peft_rank(rank_pattern, r, m) == r for m in colliding)
    with pytest.raises(AX.AdapterError) as e:
        AX._layout(colliding, "per_module")
    assert str(e.value) == (f"layout 'per_module' cannot be written: PEFT matches a rank pattern on the end of a module's "
                            f"name, and {VDOWN} ends in .{DOWN} with another rank; use 'uniform' or 'auto'")
    # No collision: per_module, keys sorted by module name, each module resolved to its own rank by PEFT's rule.
    fine = {DOWN: 3, VDOWN: 1, "model.layers.0.self_attn.o_proj": 2, "lm_head": 1}
    for layout in ("auto", "per_module"):
        resolved, r, rank_pattern, notes = AX._layout(fine, layout)
        assert resolved == "per_module" and r == 3 and notes == []
        assert list(rank_pattern) == [re.escape(m) for m in sorted(fine) if fine[m] != 3]
        for m in fine:
            assert peft_rank(rank_pattern, r, m) == fine[m], m


def test_modules_peft_also_attaches_to():
    targets = [DOWN, "model.layers.0.self_attn.o_proj"]
    tensors = {DOWN + ".weight": (8, 10),                                   # a target itself
               VDOWN + ".weight": (4, 4),                                   # ends in .<DOWN>: reported
               "vision.model.layers.0.self_attn.o_proj.weight": (4, 4),     # ends in .<o_proj target>: reported
               "norm." + DOWN + ".weight": (4,),                            # 1-D: not a linear layer
               VDOWN + ".bias": (4, 4),                                     # not a weight
               "vmodel.layers.1.mlp.down_proj.weight": (4, 4)}              # no dot before the ending
    assert AX.also_matched(targets, tensors) == [("model.layers.0.self_attn.o_proj",
                                                  "vision.model.layers.0.self_attn.o_proj"), (DOWN, VDOWN)]
    pairs = [(DOWN, f"v{i}.{DOWN}") for i in range(7)]
    notes = AX._also_notes(pairs)
    assert len(notes) == 6 and notes[-1] == "and 2 more"
    assert notes[0] == (f"module v0.{DOWN} is not changed by these patches but its name ends in .{DOWN}: PEFT will "
                        f"attach an empty adapter to it too (it changes nothing) and warn that its weights are missing")


def test_a_checkpoint_module_ending_like_a_target_is_noted(tmp_path):
    extra_checkpoint(tmp_path / "ck", {"extra.model.layers.1.self_attn.o_proj.weight": torch.zeros(4, 4)})
    ws = Workspace.create_hf(str(tmp_path / "ws"), "tiny", str(tmp_path / "ck"))
    ws.commit({O1: [D.lowrank(whole(ws, O1), ints(8, 1, seed=1), ints(1, 8, seed=2))]}, "m")
    out = AX.plan(ws)
    assert ("module extra.model.layers.1.self_attn.o_proj is not changed by these patches but its name ends in "
            ".model.layers.1.self_attn.o_proj: PEFT will attach an empty adapter to it too (it changes nothing) and warn "
            "that its weights are missing") in out["notes"]
    assert [e["module"] for e in out["modules"]] == [AX.module_of(O1)]


# ------------------------------------------------------------------ the checkpoint export keeps out of export/adapters
def test_the_checkpoint_export_refuses_the_adapters_name(tmp_path):
    from fastapi.testclient import TestClient
    from wsp.server import create_app
    from wsp.session import Session
    checkpoint(str(tmp_path / "ckpt"))
    Workspace.create_hf(str(tmp_path / "root" / "tiny"), "tiny", str(tmp_path / "ckpt"))
    sess = Session(str(tmp_path / "root"))
    ws = sess.open("tiny")
    ws.commit({O1: [D.lowrank(whole(ws, O1), ints(8, 1, seed=1), ints(1, 8, seed=2))]}, "m")
    c = TestClient(create_app(sess, "http://127.0.0.1:0"), raise_server_exceptions=False)
    r = c.post("/api/export_checkpoint", json={"name": "adapters"})
    assert (r.status_code, r.json()) == (400, {"error": "the name 'adapters' is taken by the adapter exports "
                                                        "(<workspace>/export/adapters); choose another name"})
    assert not os.path.exists(os.path.join(ws.dir, "export"))
    r = c.post("/api/export_checkpoint", json={"name": "x"})
    assert r.status_code == 200
    out = r.json()
    assert out["dir"] == os.path.join(ws.dir, "export", "x") and out["files"] == [os.path.join(out["dir"], "model.safetensors")]
    assert torch.equal(load_file(out["files"][0])[O1], ws.read(O1, (), "current").to(torch.float32))
    assert os.listdir(os.path.join(ws.dir, "export")) == ["x"]
