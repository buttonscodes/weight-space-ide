"""Behaviour directions on hybrid (linear-attention) and mixture-of-experts layouts (wsp.directions: COMPONENT_SOURCES,
modules; wsp.directions_ara; wsp.arch's mixer / expert / shared parts; wsp.runner's hook kinds).

Model-free: the tiny random Qwen3.5 checkpoints of tests/layout_fixtures.py (dense hybrid and MoE, built and saved by
transformers in the test's own directory), the tiny random model of tests/test_steering.py, the llama-named fake of
tests/test_steering_worker.py, a hand-written safetensors checkpoint and a synthetic GGUF file. No model is downloaded.
Every expectation is computed independently of the code under test: the ablation formulas written out here, captures
from forward hooks registered here on the loaded model's own modules, the fused expert parameters of a reference model
edited here, and initial values drawn here. The dense constants of L10 and L14 were recorded from the code before
this change (the same calls on the same fixtures) and are pinned here.
"""
import hashlib
import json
import math
import os
import re
import struct
import sys

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wsp.runner  # noqa: E402
from layout_fixtures import EXPERT_WIDTH, EXPERTS, tiny_qwen35  # noqa: E402
from wsp import arch  # noqa: E402
from wsp import directions as DR  # noqa: E402
from wsp import directions_ara as ARA  # noqa: E402
from wsp import directions_record as REC  # noqa: E402
from wsp import directions_search as DSR  # noqa: E402
from wsp import zoo  # noqa: E402
from wsp.session import Session  # noqa: E402
from wsp.store import region_index  # noqa: E402
from wsp.workspace import Workspace  # noqa: E402
from wsp.wscript import run_script  # noqa: E402

from test_steering import make_ws  # noqa: E402

D = 32                                    # the tiny Qwen3.5's d_model
LAYERS = 4
GOOD = ["t1 t2 t3", "t4 t5 t6 t7", "t8 t9", "t10 t11 t12"]
BAD = ["t20 t21", "t22 t23 t24", "t25 t26 t27 t28", "t29"]
SMALL = {"lora_rank": 3, "n_optimization_steps": 2, "max_iter": 8}
PRE = "model.language_model.layers"
MIXER_NOTE = ("3 layers use a token mixer without attention heads (linear attention); its output projection is the "
              "layer's attn.o_proj for directions")
MOE_NOTE = ("4 layers have 4 routed experts and a shared expert; directions treat each expert's down projection as a "
            "module of mlp.down_proj")
ROUTED_REFUSAL = ("layer 0 mlp.down_proj: ARA cannot fit routed experts yet (an expert's inputs exist only for the "
                  "prompts routed to it); pass expert_modules=\"shared\" to fit the shared expert only")

# Recorded from the code before this change (scratch script, the same calls on the same fixtures; float32, CPU).
BEFORE = {
    "wsp": {"io_json": "872822ff4008afc0668d61dceac5ed6e4cd40f18a892acd983808e8375dc4cf4",
            "pieces_ara": "de35a076783f51bdb3101d548b9722efb3dbf1cd2139a97ae257a31ca4bc24a3",
            "pieces_full": "164313b19e272ed5f90ea08846eea1051a725c1fd3543d9a4a7c0c8c64195cee",
            "pieces_none": "81eb74639c77479ea0f26bb1e72ce32198190104678129e0c2eb5fc54c9d05c7",
            "pieces_pre": "5dde42b0589a21630899ccc6d68f106588c06a3efdc5768f37962b2cf480c434",
            "plan_ara": "2f8989029cf0372ada84efc28f6bc5c03dc19210604c962f9245bf85c7f4a3f7",
            "plan_full": "dc3c99cbc984f3eca4c8a25709ca6c93d1ab65566a4f9c7731f080858f3944e8",
            "plan_none": "bf3f756ba5904bf4d10642af38be429d064f4c3c8adca3713e8769496d5d5361",
            "plan_pre": "015f898c7625af3235927c1d2a27db4d76bb7b381e5a9c1f4834e0d508f7b420"},
    "llama": {"io_json": "46911eba4be4bdff52d4186aadee83b6cc1c615eb6b6a0d53d32d2b7f4c5d0ad",
              "pieces_ara": "7741ce465138a9a7ca388b6602a3bb8265e5ed4f2e477535219871a9049126a0",
              "pieces_full": "b6863f418be4716587cd351bb9c201c3765ddfc6abac9a25d3d8296a89823a7d",
              "pieces_none": "9e9a8198da20ebd8e318f068a0401db3b4156f0e64029fb1e2d1440f3c3ccc7b",
              "pieces_pre": "8548c854253707bed285a96b815f3036a487a51f1e76d7c8c851062a227ca7d7",
              "plan_ara": "18c9609064a1d199e98f7d9036b8a08769e705ceb57c01ee755339f87fb00de5",
              "plan_full": "2c87b17edf80fcc90f8df481a79e7f866b08023a87ed605a8d32d82ca71be8c8",
              "plan_none": "400a1776fbea39d45b4f9b6a6a6a4e701444bb0d8e592e2c279ea924937bf821",
              "plan_pre": "8602513ea2d98886e448fa5d9f45c7e23e83478f2e348f3b316e2d8de6ce4366"},
}


@pytest.fixture(autouse=True)
def _float32(monkeypatch):
    monkeypatch.setenv("WSP_RUN_DTYPE", "float32")


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    root = tmp_path_factory.mktemp("ckpt")
    return {"dense": tiny_qwen35(root / "dense"), "moe": tiny_qwen35(root / "moe", moe=True)}


def session(tmp_path, ckpt):
    s = Session(str(tmp_path / "ws"))
    os.makedirs(s.root, exist_ok=True)
    s.ws = zoo.open_hf(s.root, ckpt, name="m")
    return s


# ------------------------------------------------------------------ independent references (written out here)
def unit(x):
    return x / x.norm()


def ref_direction(good, bad, k):
    r = unit(bad[k].double() - good[k].double())
    g = unit(good[k].double())
    return unit(r - (r @ g) * g)


def plain_ref(V, v, w):
    return V - w * torch.outer(V @ v, v)


def pre_ref(V, v, w):
    n = V.norm(dim=0)
    Wn = V / torch.where(n > 0, n, torch.ones_like(n))
    return (Wn - w * torch.outer(Wn @ v, v)) * n


def full_exact(V, v, w):
    """Normalise columns, ablate, renormalise each column, restore the original norms -> the difference."""
    n = V.norm(dim=0)
    Wn = V / torch.where(n > 0, n, torch.ones_like(n))
    Wn = Wn - w * torch.outer(Wn @ v, v)
    m = Wn.norm(dim=0)
    Wn = Wn / torch.where(m > 0, m, torch.ones_like(m))
    return torch.where(n > 0, Wn * n, V) - V


def draws(seed, r, sizes):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.init.kaiming_uniform_(torch.empty(r, n), a=math.sqrt(5), generator=g) for n in sizes]


def record_initial_values(monkeypatch):
    got = []

    def fake_fit(data, A0, B0, p, where=""):
        got.append((where, A0.clone()))
        return A0, torch.zeros_like(B0), {"loss_before": 0.0, "loss_after": 0.0, "scale_min": 1.0, "scale_max": 1.0}
    monkeypatch.setattr(ARA, "fit", fake_fit)
    return got


def save_random(ws, name, seed=0):
    """A saved measurement of random means (no model run), as tests/test_directions.py makes one."""
    s = ws.struct
    g = torch.Generator().manual_seed(seed)
    good, bad = torch.randn(s.n_layers + 1, s.d_model, generator=g), torch.randn(s.n_layers + 1, s.d_model, generator=g)
    meta = {"version": 1, "name": name, "created": 0, "variant": "base", "n_layers": s.n_layers, "d_model": s.d_model,
            "counts": {"good": 1, "bad": 1}, "tokens": {"good": 1, "bad": 1}, "options": {}, "prompts_sha256": {}, "model": {}}
    DR.save(ws, name, good, bad, meta, {"good": ["a"], "bad": ["b"]})
    return good, bad


def dist(position, distance, w=0.9, low=0.6):
    return {"max_weight": w, "max_weight_position": position, "min_weight": low, "min_weight_distance": distance}


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def io_dir(ws, name):
    return os.path.join(ws.dir, "directions", name + ".io")


def stored(ws, name, label, layer):
    return torch.from_numpy(np.load(os.path.join(io_dir(ws, name), f"{label}.L{layer}.npy")))


def header(directory) -> dict:
    shapes = {}
    for f in sorted(os.listdir(directory)):
        if f.endswith(".safetensors"):
            with open(os.path.join(directory, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                meta = json.loads(fh.read(n))
            shapes.update({k: tuple(v["shape"]) for k, v in meta.items() if k != "__metadata__"})
    return shapes


def hooked(model, specs):
    """Forward hooks on the model's own modules: specs [(key, module, 'in' | 'out', pick)] -> (seen, handles)."""
    seen, handles = {}, []
    for key, mod, which, pick in specs:
        if which == "in":
            handles.append(mod.register_forward_pre_hook(lambda m, a, key=key, pick=pick: seen.__setitem__(key, pick(a[0]).clone())))
        else:
            handles.append(mod.register_forward_hook(lambda m, a, o, key=key, pick=pick: seen.__setitem__(
                key, pick(o[0] if isinstance(o, tuple) else o).clone())))
    return seen, handles


# ------------------------------------------------------------------ L1-L4: the hybrid layout
def test_a_hybrid_checkpoint_has_an_attention_output_in_every_layer(tmp_path, ckpts):
    ws = session(tmp_path, ckpts["dense"]).ws
    s = ws.struct
    assert s.n_layers == LAYERS and s.recognised
    mods = [DR.modules(ws, l, "attn.o_proj") for l in range(LAYERS)]
    assert [[m["path"] for m in ms] for ms in mods] == [["linear_attn.out_proj"]] * 3 + [["self_attn.o_proj"]]
    assert [ms[0]["source"] for ms in mods] == ["mixer"] * 3 + ["attn"]
    assert [[tuple(v.shape) for v in ms[0]["views"]] for ms in mods] == [[(D, D)]] * 4            # [d_in, d_model]
    assert all(ms[0]["hooks"] == ("mixer.hook_in", "mixer.hook_out") for ms in mods[:3])
    assert [m["path"] for l in range(LAYERS) for m in DR.modules(ws, l, "mlp.down_proj")] == ["mlp.down_proj"] * 4
    # The mixer is a registered part (tensor descriptions name it) and its tensor stays reachable by the layer's name.
    name = f"{PRE}.0.linear_attn.out_proj.weight"
    assert [(p.role, p.roles) for p in s.parts[name]] == [("W_O", ("resid", "other"))]
    assert s.layers[0].mixer["W_O"].name == name and name in s.layers[0].other
    assert s.layers[0].modules["mixer_out"] == f"{PRE}.0.linear_attn.out_proj"
    assert MIXER_NOTE in s.notes and MIXER_NOTE in s.describe()
    # The hybrid rules read the text model's type (text_config.model_type of a multimodal checkpoint): the linear
    # layers' block output is the mixer, and the full layer's q_proj holds query | gate rows per head.
    assert ws.hf_config["model_type"] == "qwen3_5" and ws.hf_config["text_config"]["model_type"] == "qwen3_5_text"
    assert [s.layers[l].modules["attn"] for l in range(3)] == [f"{PRE}.{l}.linear_attn" for l in range(3)]
    assert s.layers[3].attn["W_Q"].head == (2 * 16, 0, 16, 2)
    save_random(ws, "r")
    plan = DR.plan(ws, {"direction": "r", "weight_distributions": {"attn.o_proj": dist(1.5, 4)}})
    rows = [r for r in plan["rows"] if r["component"] == "attn.o_proj"]
    assert len(rows) == 4 and all(r["skipped"] is None for r in rows)
    assert [r["module"] for r in rows] == ["linear_attn.out_proj"] * 3 + ["self_attn.o_proj"]
    assert plan["totals"] == {"layers": 4, "components": 4, "views": 4, "stored": 4 * (D + D), "skipped": 0}


@pytest.mark.parametrize("mode", ["none", "pre", "full"])
def test_the_ablation_of_a_linear_attention_output_equals_the_formula(tmp_path, ckpts, mode):
    ws = session(tmp_path, ckpts["dense"]).ws
    good, bad = save_random(ws, "r", seed=2)
    name = f"{PRE}.1.linear_attn.out_proj.weight"
    params = {"direction": "r", "weight_distributions": {"attn.o_proj": dist(1.0, 0.5)}, "row_normalization": mode,
              "full_normalization_lora_rank": D}                       # rank d: the full change exactly
    deltas = run_script(ws, DR.script_for(params))["deltas"]
    assert set(deltas) == {name}                                      # only layer 1's out_proj
    V = ws.read(name).T.double()                                      # [value_dim, d]: what the module writes
    v, w = ref_direction(good, bad, 2), 0.9
    got = ws.read(name, (), "current", deltas).T.double()
    want = {"none": plain_ref, "pre": pre_ref, "full": lambda V, v, w: V + full_exact(V, v, w)}[mode](V, v, w)
    assert torch.allclose(got, want, atol=2e-5, rtol=1e-5), float((got - want).abs().max())
    assert float((got - V).abs().max()) > 1e-2
    if mode == "full":
        assert torch.allclose(got.norm(dim=0), V.norm(dim=0), rtol=1e-5)


def test_hybrid_module_io_is_each_modules_input_and_output(tmp_path, ckpts):
    ws = session(tmp_path, ckpts["dense"]).ws
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    info = json.load(open(os.path.join(io_dir(ws, "m"), "io.json")))
    assert info["version"] == 2 == DR.IO_VERSION
    row = (D + D) + (48 + D)
    for L in info["layers"]:
        attn = "linear_attn.out_proj" if L["layer"] < 3 else "self_attn.o_proj"
        assert L["row"] == row
        assert L["modules"] == [
            {"component": "attn.o_proj", "module": attn, "source": "mixer" if L["layer"] < 3 else "attn", "d_in": D,
             "captured": True, "why": None, "x": 0, "y": D},
            {"component": "mlp.down_proj", "module": "mlp.down_proj", "source": "mlp", "d_in": 48, "captured": True,
             "why": None, "x": 2 * D, "y": 2 * D + 48}]
    model = ws.runner.model
    specs = []
    for l, layer in enumerate(model.model.layers):
        last = lambda t: t[0, -1]  # noqa: E731
        if l < 3:
            specs += [((l, "xa"), layer.linear_attn.out_proj, "in", last), ((l, "ya"), layer.linear_attn.out_proj, "out", last)]
        else:
            specs += [((l, "xa"), layer.self_attn.o_proj, "in", last), ((l, "ya"), layer.self_attn, "out", last)]
        specs += [((l, "xm"), layer.mlp.down_proj, "in", last), ((l, "ym"), layer.mlp, "out", last)]
    seen, handles = hooked(model, specs)
    try:
        for label, texts in (("good", GOOD), ("bad", BAD)):
            for i, text in enumerate(texts):
                with torch.no_grad():
                    model(torch.tensor([ws.encode(text)]))
                for l in range(LAYERS):
                    want = torch.cat([seen[(l, k)] for k in ("xa", "ya", "xm", "ym")])
                    assert torch.allclose(stored(ws, "m", label, l)[i], want, atol=1e-6), (label, i, l)
    finally:
        for h in handles:
            h.remove()
    # Every module's saved input reproduces its saved output through the matrix (no bias in these modules).
    for l in range(LAYERS):
        for label in ("good", "bad"):
            a = stored(ws, "m", label, l)
            for e in info["layers"][l]["modules"]:
                V = ws.read(f"{PRE}.{l}.{e['module']}.weight").T
                X, Y = a[:, e["x"]:e["x"] + e["d_in"]], a[:, e["y"]:e["y"] + D]
                assert ARA.mismatch(X @ V, torch.zeros(D), Y) <= 1e-6, (l, e["module"], label)


def test_ara_on_a_hybrid_model_fits_every_module_in_range(tmp_path, ckpts, monkeypatch):
    sess = session(tmp_path, ckpts["dense"])
    ws = sess.ws
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    params = {"method": "ara", "direction": "m", **SMALL, "start_layer_index": 1, "steer_bad_behavior_weight": 0.5,
              "seed": 11}
    plan = DR.plan(ws, params)
    assert [(r["layer"], r["component"], r["module"], r["skipped"]) for r in plan["rows"]] == [
        (1, "attn.o_proj", "linear_attn.out_proj", None), (1, "mlp.down_proj", "mlp.down_proj", None),
        (2, "attn.o_proj", "linear_attn.out_proj", None), (2, "mlp.down_proj", "mlp.down_proj", None),
        (3, "attn.o_proj", "self_attn.o_proj", None), (3, "mlp.down_proj", "mlp.down_proj", None)]
    want = {f"{PRE}.{l}.{m}.weight" for l, m in ((1, "linear_attn.out_proj"), (2, "linear_attn.out_proj"),
                                                  (3, "self_attn.o_proj"))} | {f"{PRE}.{l}.mlp.down_proj.weight" for l in (1, 2, 3)}
    res = run_script(ws, DR.script_for(params))
    assert set(res["deltas"]) == want and all(p.kind == "lowrank" for ps in res["deltas"].values() for p in ps)
    for line in res["output"].splitlines()[1:]:
        assert float(re.search(r"mismatch good (\S+) ", line).group(1)) <= 1e-5, line
    # One initial draw per fitted module, in fit order, from one generator.
    got = record_initial_values(monkeypatch)
    run_script(ws, DR.script_for({**params, "start_layer_index": 0}))
    assert [w for w, _ in got] == [f"layer {l} {c}" for l in range(LAYERS) for c in ("attn.o_proj", "mlp.down_proj")]
    for (w, A0), ref in zip(got, draws(11, 3, [D, 48] * LAYERS)):
        assert torch.equal(A0, ref), w
    monkeypatch.undo()
    done = sess.apply(DR.script_for(params))                        # the edit path: committed, recorded
    assert done["ok"] and done["committed"] and set(ws.touched()) == want


# ------------------------------------------------------------------ L5-L9: the mixture-of-experts layout
def test_a_moe_checkpoint_is_saved_per_expert(ckpts):
    names = header(ckpts["moe"])
    for l in range(LAYERS):
        for e in range(EXPERTS):
            for leaf in ("gate_proj", "up_proj", "down_proj"):
                assert f"{PRE}.{l}.mlp.experts.{e}.{leaf}.weight" in names
        assert names[f"{PRE}.{l}.mlp.experts.0.down_proj.weight"] == (D, EXPERT_WIDTH)
        assert names[f"{PRE}.{l}.mlp.shared_expert.down_proj.weight"] == (D, EXPERT_WIDTH)
    assert not any(".mlp.experts.down_proj" in n or "gate_up_proj" in n for n in names)


def test_a_moe_stack_whose_first_layer_is_linear_attention_is_recognised():
    """The layer stack is chosen by counting rule hits in its lowest layer. In a Qwen3.5 MoE checkpoint that layer is
    linear attention with routed and shared experts: none of its tensors match the dense attention or MLP rules, so
    the mixer, expert and shared-expert rules must count too, or no stack is found (n_layers 0)."""
    d, i, e = 8, 4, 2
    sh = {f"{PRE[:-7]}.embed_tokens.weight": (16, d), f"{PRE[:-7]}.norm.weight": (d,), "lm_head.weight": (16, d),
          "model.visual.blocks.0.attn.qkv.weight": (3 * d, d), "model.visual.blocks.0.mlp.linear_fc1.weight": (d, d)}
    for l in range(2):
        p = f"{PRE}.{l}"
        if l == 0:
            sh |= {f"{p}.linear_attn.in_proj_qkv.weight": (3 * d, d), f"{p}.linear_attn.out_proj.weight": (d, d)}
        else:
            sh |= {f"{p}.self_attn.{x}_proj.weight": (d, d) for x in "qkvo"}
        sh |= {f"{p}.mlp.gate.weight": (e, d), f"{p}.mlp.shared_expert_gate.weight": (1, d)}
        sh |= {f"{p}.mlp.experts.{k}.{x}_proj.weight": ((d, i) if x == "down" else (i, d)) for k in range(e) for x in ("gate", "up", "down")}
        sh |= {f"{p}.mlp.shared_expert.{x}_proj.weight": ((d, i) if x == "down" else (i, d)) for x in ("gate", "up", "down")}
    first = [n[len(PRE) + 3:] for n in sh if n.startswith(f"{PRE}.0.")]
    assert not any(r.match(rest) for rest in first for r, _ in arch._ATTN_RULES + arch._MLP_RULES)   # the finding
    cfg = {"model_type": "qwen3_5_moe", "text_config": {"model_type": "qwen3_5_moe_text", "num_attention_heads": 1}}
    s = arch.detect(sh, cfg)
    assert s.recognised and s.n_layers == 2 and s.layers[0].prefix == f"{PRE}.0"
    assert [len(L.experts) for L in s.layers] == [2, 2] and all("W_out" in L.shared for L in s.layers)
    assert "W_O" in s.layers[0].mixer and s.has_heads(1)


def test_moe_structure_lists_routed_and_shared_experts(tmp_path, ckpts):
    ws = session(tmp_path, ckpts["moe"]).ws
    s = ws.struct
    assert s.recognised and s.n_layers == LAYERS
    roles = {"W_gate": ("gate_proj", ("neuron", "resid")), "W_up": ("up_proj", ("neuron", "resid")),
             "W_out": ("down_proj", ("resid", "neuron"))}
    for l, L in enumerate(s.layers):
        p = f"{PRE}.{l}.mlp"
        assert len(L.experts) == EXPERTS
        for e, ex in enumerate(L.experts):
            assert {r: (part.name, part.roles) for r, part in ex.items()} == {
                r: (f"{p}.experts.{e}.{leaf}.weight", rr) for r, (leaf, rr) in roles.items()}
            assert all(s.parts[part.name] == [part] for part in ex.values())
        assert {r: part.name for r, part in L.shared.items()} == {r: f"{p}.shared_expert.{leaf}.weight" for r, (leaf, _) in roles.items()}
        for router in (f"{p}.gate.weight", f"{p}.shared_expert_gate.weight"):
            assert router in L.other and router not in s.parts
        assert L.modules["experts"] == f"{p}.experts" and L.modules["shared_out"] == f"{p}.shared_expert.down_proj"
        assert L.mlp == {} and not s.has_neurons(l)
    assert MOE_NOTE in s.notes and MIXER_NOTE in s.notes
    mods = DR.modules(ws, 1, "mlp.down_proj")
    assert [(m["key"], m["path"], m["routed"]) for m in mods] == [
        (f"experts.{e}", f"mlp.experts.{e}.down_proj", True) for e in range(EXPERTS)] + [("shared", "mlp.shared_expert.down_proj", False)]
    assert all([tuple(v.shape) for v in m["views"]] == [(EXPERT_WIDTH, D)] for m in mods)
    assert [m["path"] for m in DR.modules(ws, 1, "mlp.down_proj", "shared")] == ["mlp.shared_expert.down_proj"]


def test_an_expert_index_gap_is_not_recognised():
    d, i = 8, 4
    sh = {"model.embed_tokens.weight": (16, d), "model.norm.weight": (d,)}
    for l in range(2):
        p = f"model.layers.{l}"
        sh |= {f"{p}.self_attn.{x}_proj.weight": (d, d) for x in "qkvo"}
        for k in ((0, 1) if l == 0 else (0, 2)):
            sh |= {f"{p}.mlp.experts.{k}.{x}_proj.weight": ((d, i) if x == "down" else (i, d)) for x in ("gate", "up", "down")}
    s = arch.detect(sh, {"model_type": "qwen3_moe", "num_attention_heads": 1})
    assert len(s.layers[0].experts) == 2 and s.layers[1].experts == []
    assert "layer 1: expert indices are not 0..1; its experts are not recognised" in s.notes
    assert "model.layers.1.mlp.experts.2.down_proj.weight" in s.layers[1].other
    assert "model.layers.1.mlp.experts.2.down_proj.weight" not in s.parts


def moe_reference(ckpt):
    ref = transformers.AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.float32, attn_implementation="eager").eval()
    assert type(ref).__name__ == "Qwen3_5MoeForCausalLM"
    return ref


def test_the_ablation_of_every_expert_reaches_the_fused_runtime(tmp_path, ckpts):
    sess = session(tmp_path, ckpts["moe"])
    ws = sess.ws
    save_random(ws, "r", seed=4)
    params = {"direction": "r", "weight_distributions": {"mlp.down_proj": dist(1.0, 0.5, w=1.0)}}   # layer 1 only
    plan = DR.plan(ws, params)
    rows = [r for r in plan["rows"] if r["layer"] == 1]
    assert [(r["module"], r.get("modules"), r["views"], r["stored"]) for r in rows] == [
        ("mlp.experts.*.down_proj", EXPERTS, EXPERTS, EXPERTS * (EXPERT_WIDTH + D)),
        ("mlp.shared_expert.down_proj", None, 1, EXPERT_WIDTH + D)]
    assert plan["totals"]["components"] == 2 and plan["totals"]["modules"] == EXPERTS + 1
    script = DR.script_for(params)
    assert "expert_modules" not in script
    done = sess.apply(script)
    assert done["ok"] and done["committed"], done
    names = [f"{PRE}.1.mlp.experts.{e}.down_proj.weight" for e in range(EXPERTS)] + [f"{PRE}.1.mlp.shared_expert.down_proj.weight"]
    assert sorted(ws.touched()) == sorted(names)
    with open(REC.record_path(ws, done["patch"])) as f:                   # the record lists every module it changed
        rec = json.load(f)
    assert [c["module"] for c in rec["calls"][0]["result"]["components"]] == [n[len(PRE) + 3:-7] for n in names]
    assert "expert_modules" not in rec["calls"][0]["parameters"] and "note" not in rec["calls"][0]["heretic"]
    runner = ws.runner
    runner.use("base")
    base = {l: runner.model.model.layers[l].mlp.experts.down_proj.detach().clone() for l in range(LAYERS)}
    gate_up = {l: runner.model.model.layers[l].mlp.experts.gate_up_proj.detach().clone() for l in range(LAYERS)}
    ids = ws.encode("t3 t7 t11 t5 t9 t2")
    base_logits = runner.run(ids, variant="base")[0][0]
    current = runner.run(ids)[0][0]                                          # applies the committed patch
    fused = runner.model.model.layers[1].mlp.experts.down_proj
    for e in range(EXPERTS):
        want = base[1][e].clone()
        for piece, scale in ws.pieces(names[e]):
            want[region_index(piece.region)] += (scale * piece.dense()).to(want.dtype)
        assert torch.equal(fused[e], want), e                               # exactly its own delta, nothing else
        assert not torch.equal(fused[e], base[1][e])
    for l in (0, 2, 3):
        assert torch.equal(runner.model.model.layers[l].mlp.experts.down_proj, base[l]), l
    for l in range(LAYERS):
        assert torch.equal(runner.model.model.layers[l].mlp.experts.gate_up_proj, gate_up[l]), l
    ref = moe_reference(ckpts["moe"])
    with torch.no_grad():
        torch.testing.assert_close(ref(torch.tensor([ids])).logits[0], base_logits, atol=1e-5, rtol=0)
        for e in range(EXPERTS):
            ref.model.layers[1].mlp.experts.down_proj[e].copy_(ws.read(names[e]))
        ref.model.layers[1].mlp.shared_expert.down_proj.weight.copy_(ws.read(names[-1]))
        want = ref(torch.tensor([ids])).logits[0]
    torch.testing.assert_close(current, want, atol=1e-5, rtol=0)
    assert float((want - base_logits).abs().max()) > 1e-3                    # the edit matters
    assert torch.equal(runner.run(ids, variant="base")[0][0], base_logits)


@pytest.mark.parametrize("mode", ["pre", "full"])
def test_each_expert_is_normalised_on_its_own(tmp_path, ckpts, mode):
    """Row norms (pre, full) are per module (Heretic normalises each module's W on its own): pooled norms over the
    layer's experts would be another change."""
    ws = session(tmp_path, ckpts["moe"]).ws
    good, bad = save_random(ws, "r", seed=5)
    params = {"direction": "r", "weight_distributions": {"mlp.down_proj": dist(2.0, 0.5, w=0.8)}, "row_normalization": mode,
              "full_normalization_lora_rank": D}
    deltas = run_script(ws, DR.script_for(params))["deltas"]
    v, w = ref_direction(good, bad, 3), 0.8
    names = [f"{PRE}.2.mlp.experts.{e}.down_proj.weight" for e in range(EXPERTS)] + [f"{PRE}.2.mlp.shared_expert.down_proj.weight"]
    assert sorted(deltas) == sorted(names)
    ref = pre_ref if mode == "pre" else (lambda V, v, w: V + full_exact(V, v, w))
    Vs = [ws.read(n).T.double() for n in names]                              # each [I, d]
    for n, V in zip(names, Vs):
        got = ws.read(n, (), "current", deltas).T.double()
        assert torch.allclose(got, ref(V, v, w), atol=2e-5, rtol=1e-5), n
    pooled = ref(torch.cat(Vs[:EXPERTS]), v, w)
    mine = torch.cat([ref(V, v, w) for V in Vs[:EXPERTS]])
    assert not torch.allclose(pooled, mine, atol=1e-3)                       # the test can tell the two apart


def test_expert_modules_shared_leaves_routed_experts_alone(tmp_path, ckpts):
    ws = session(tmp_path, ckpts["moe"]).ws
    save_random(ws, "r")
    params = {"direction": "r", "weight_distributions": {"mlp.down_proj": dist(1.5, 4)}, "expert_modules": "shared"}
    seen = []
    rep = DR.ablate(ws, params, add=lambda view, A, B: seen.append(view.name))
    assert sorted(set(seen)) == [f"{PRE}.{l}.mlp.shared_expert.down_proj.weight" for l in range(LAYERS)]
    assert [c["module"] for c in rep["components"]] == ["mlp.shared_expert.down_proj"] * LAYERS
    plan = DR.plan(ws, params)
    assert [r["module"] for r in plan["rows"]] == ["mlp.shared_expert.down_proj"] * LAYERS and "modules" not in plan["totals"]
    # The script names the field only when it is not the default; normalize keeps the default out the same way.
    assert DR.script_for(params).endswith(', full_normalization_lora_rank=3, expert_modules="shared")')
    for p in ({**params, "expert_modules": "all"}, {k: v for k, v in params.items() if k != "expert_modules"}):
        assert "expert_modules" not in DR.script_for(p) and "expert_modules" not in DR.normalize(p)
    assert DR.normalize(params)["expert_modules"] == "shared"
    ara = {"method": "ara", "direction": "r", **SMALL}
    assert DR.script_for({**ara, "expert_modules": "shared"}).endswith(', seed=0, expert_modules="shared")')
    assert "expert_modules" not in DR.script_for({**ara, "expert_modules": "all"}) and "expert_modules" not in ARA.normalize(ara)
    for bad in ("some", None, True, ["shared"]):
        with pytest.raises(DR.DirectionError) as e:
            DR.plan(ws, {**params, "expert_modules": bad})
        assert str(e.value) == 'expert_modules: must be "all" or "shared"'
        with pytest.raises(DR.DirectionError) as e:
            ARA.normalize({**ara, "expert_modules": bad})
        assert str(e.value) == 'expert_modules: must be "all" or "shared"'
    dense = session(tmp_path / "d", ckpts["dense"]).ws
    save_random(dense, "r")
    for p in (params, {**ara, "expert_modules": "shared"}):
        with pytest.raises(DR.DirectionError) as e:
            DR.plan(dense, p)
        assert str(e.value) == 'expert_modules: "shared" changes nothing: this model has no routed experts'
    # The record: Heretic's parameters never hold the field; "shared" adds the note.
    for method, p in (("ablation", DR.normalize(params)), ("ara", ARA.normalize({**ara, "expert_modules": "shared"}))):
        h, _ = REC._heretic(ws, {"function": "ablate_direction" if method == "ablation" else "ablate_ara", "method": method,
                                 "parameters": p})
        assert "expert_modules" not in h["parameters"] and "expert_modules" not in h["settings"]
        assert h["note"] == HERETIC_NOTE[method]
        h, _ = REC._heretic(ws, {"function": "ablate_ara", "method": "ara", "parameters": ARA.normalize(ara)})
        assert "note" not in h


def test_ara_refuses_routed_experts_and_fits_the_shared_one(tmp_path, ckpts):
    ws = session(tmp_path, ckpts["moe"]).ws
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    info = DR.load_io(ws, "m")
    for L in info["layers"]:
        attn = "linear_attn.out_proj" if L["layer"] < 3 else "self_attn.o_proj"
        assert L["row"] == 2 * D + EXPERT_WIDTH + D
        assert [(e["module"], e["captured"], e.get("x"), e["why"], e.get("modules")) for e in L["modules"]] == [
            (attn, True, 0, None, None),
            ("mlp.experts.*.down_proj", False, None, "routed experts are not captured: ARA does not fit them yet", EXPERTS),
            ("mlp.shared_expert.down_proj", True, 2 * D, None, None)]
    ara = {"method": "ara", "direction": "m", **SMALL, "steer_bad_behavior_weight": 0.5}
    for call, prefix in ((lambda: DR.plan(ws, ara), ""), (lambda: run_script(ws, DR.script_for(ara)), "line 1: "),
                         (lambda: ARA.ablate(ws, ara, add=lambda *a: pytest.fail("a piece before the refusal")), "")):
        with pytest.raises(Exception) as e:
            call()
        assert str(e.value) == prefix + ROUTED_REFUSAL
    # The shared expert's capture is its down projection's own input and output (flat [tokens, width] activations).
    model = ws.runner.model
    flat_last = lambda t: t[-1]  # noqa: E731
    specs = [((l, k), layer.mlp.shared_expert.down_proj, k, flat_last) for l, layer in enumerate(model.model.layers)
             for k in ("in", "out")]
    seen, handles = hooked(model, specs)
    try:
        for label, texts in (("good", GOOD), ("bad", BAD)):
            for i, text in enumerate(texts):
                with torch.no_grad():
                    model(torch.tensor([ws.encode(text)]))
                for l in range(LAYERS):
                    got = stored(ws, "m", label, l)[i]
                    assert torch.allclose(got[2 * D:2 * D + EXPERT_WIDTH], seen[(l, "in")], atol=1e-6)
                    assert torch.allclose(got[2 * D + EXPERT_WIDTH:], seen[(l, "out")], atol=1e-6)
    finally:
        for h in handles:
            h.remove()
    shared = {**ara, "expert_modules": "shared"}
    pieces = []
    rep = ARA.ablate(ws, shared, add=lambda view, A, B: pieces.append(view.name))
    mods = [(c["layer"], c["component"], c["module"]) for c in rep["components"]]
    assert mods == [(l, c, m) for l in range(LAYERS) for c, m in (
        ("attn.o_proj", "linear_attn.out_proj" if l < 3 else "self_attn.o_proj"), ("mlp.down_proj", "mlp.shared_expert.down_proj"))]
    assert not any(".experts." in n for n in pieces)
    for c in rep["components"]:
        assert c["mismatch_good"] <= 1e-6 and c["mismatch_bad"] <= 1e-6, c
        assert c["loss_after"] <= c["loss_before"]
    # A flat hook point takes one prompt at a time.
    ids = ws.encode(GOOD[0])
    with pytest.raises(ValueError, match=r"^hook point blocks\.0\.shared\.hook_in takes one prompt at a time$"):
        ws.runner.run(torch.tensor([ids, ids]), names=["blocks.0.shared.hook_in"])
    _, cache = ws.runner.run(ids, names=["blocks.0.shared.hook_in"])
    assert tuple(cache["blocks.0.shared.hook_in"].shape) == (1, len(ids), EXPERT_WIDTH)


# ------------------------------------------------------------------ L10, L11: old measurements, Heretic's order
def test_a_version_1_module_io_measurement_still_fits(tmp_path):
    ws = make_ws(tmp_path / "rnd")
    DR.measure(ws, "m", ["abc", "abd", "hello", "zzz", "quiet", "fox"], ["xyz", "qqq", "world", "mmmm", "bad"], module_io=True)
    path = os.path.join(io_dir(ws, "m"), "io.json")
    info = json.load(open(path))
    assert info["version"] == 2 and [[e["module"] for e in L["modules"]] for L in info["layers"]] == [["attn.W_O", "mlp.W_out"]] * 2
    v1 = {**info, "version": 1, "layers": [{k: v for k, v in L.items() if k != "modules"} for L in info["layers"]]}
    assert sha(v1) == BEFORE["wsp"]["io_json"]                    # exactly what the code before this change wrote
    script = DR.script_for({"method": "ara", "direction": "m", **SMALL, "steer_bad_behavior_weight": 0.5})
    as_v2 = DSR.piece_digest(run_script(ws, script)["deltas"])
    with open(path, "w") as f:
        json.dump(v1, f, indent=1)
    read = DR.load_io(ws, "m")
    assert read["version"] == 1 and [[(e["component"], e["module"], e["x"]) for e in L["modules"]] for L in read["layers"]] == \
        [[("attn.o_proj", "attn.W_O", 0), ("mlp.down_proj", "mlp.W_out", 24 + 24)]] * 2
    as_v1 = DSR.piece_digest(run_script(ws, script)["deltas"])
    assert as_v1 == as_v2 == BEFORE["wsp"]["pieces_ara"]
    with open(path, "w") as f:
        json.dump({**v1, "version": 3}, f, indent=1)
    with pytest.raises(DR.DirectionError, match="measure it again with module_io"):
        DR.load_io(ws, "m")


HERETIC_ORDER = {               # heretic/model.py at 64e9991, get_layer_modules: the modules tried, in order
    "attn.o_proj": ["self_attn.o_proj", "linear_attn.out_proj", "self_attn.o_b_proj", "conv.out_proj", "self_attn.out_proj"],
    "mlp.down_proj": ["mlp.down_proj", "mlp.experts", "mlp.shared_experts", "mlp.shared_expert", "block_sparse_moe.experts",
                      "feed_forward.w2", "feed_forward.experts", "shared_mlp.output_linear", "moe.experts"],
}
HERETIC_NAME = {"attn": "self_attn.o_proj", "mixer": "linear_attn.out_proj", "mlp": "mlp.down_proj",
                "experts": "mlp.experts", "shared": "mlp.shared_expert"}


def both_mixers(tmp_path):
    """A hand-written checkpoint whose layer 0 has both a softmax attention output and a linear-attention output
    projection, and a dense MLP: the only way to see the order of two modules of one component in one layer."""
    from safetensors.torch import save_file
    d, f, v = 8, 12, 6
    g = torch.Generator().manual_seed(7)
    w = lambda *shape: torch.randn(*shape, generator=g)  # noqa: E731
    t = {"model.embed_tokens.weight": w(16, d), "model.norm.weight": torch.ones(d), "lm_head.weight": w(16, d)}
    for l in range(2):
        p = f"model.layers.{l}"
        t |= {f"{p}.self_attn.{x}_proj.weight": w(d, d) for x in "qkvo"}
        t |= {f"{p}.mlp.gate_proj.weight": w(f, d), f"{p}.mlp.up_proj.weight": w(f, d), f"{p}.mlp.down_proj.weight": w(d, f)}
    t["model.layers.0.linear_attn.out_proj.weight"] = w(d, v)
    ckpt = tmp_path / "both"
    os.makedirs(ckpt)
    save_file(t, str(ckpt / "model.safetensors"))
    json.dump({"model_type": "llama", "architectures": ["LlamaForCausalLM"], "vocab_size": 16, "hidden_size": d,
               "intermediate_size": f, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
               "head_dim": 4}, open(ckpt / "config.json", "w"))
    return Workspace.create_hf(str(tmp_path / "ws"), "both", str(ckpt)), (d, f, v)


def test_the_source_table_is_in_heretics_module_order(tmp_path, monkeypatch):
    # The table, against Heretic's order for the sources both have; the derived names keep their earlier values.
    for comp, sources in DR.COMPONENT_SOURCES.items():
        ours = [HERETIC_NAME[s["group"]] for s in sources]
        assert ours == [m for m in HERETIC_ORDER[comp] if m in ours], comp
    assert DR.COMPONENTS == {"attn.o_proj": ("attn", "W_O"), "mlp.down_proj": ("mlp", "W_out")}
    assert DR.IO_HOOKS == {"attn.o_proj": ("attn.hook_z", "hook_attn_out"), "mlp.down_proj": ("mlp.hook_post", "hook_mlp_out")}
    assert DR.BIASES == {"attn.o_proj": "b_O", "mlp.down_proj": "b_out"}
    assert DR.EXPERT_MODULES == ("all", "shared")
    # Two modules of one component in one layer: every loop takes them in the table's order.
    ws, (d, f, v) = both_mixers(tmp_path)
    assert [m["path"] for m in DR.modules(ws, 0, "attn.o_proj")] == ["self_attn.o_proj", "linear_attn.out_proj"]
    layout = DR.io_layout(ws)
    assert [(e["component"], e["module"], e["x"], e["y"]) for e in layout[0]["modules"]] == [
        ("attn.o_proj", "self_attn.o_proj", 0, d), ("attn.o_proj", "linear_attn.out_proj", 2 * d, 2 * d + v),
        ("mlp.down_proj", "mlp.down_proj", 2 * d + v + d, 2 * d + v + d + f)]
    assert layout[0]["components"]["attn.o_proj"]["x"] == 0 and layout[0]["row"] == 2 * d + v + d + f + d
    save_random(ws, "r")
    plan = DR.plan(ws, {"direction": "r", "weight_distributions": {c: dist(0.5, 2) for c in DR.COMPONENTS}})
    assert [(r["layer"], r["component"], r["module"]) for r in plan["rows"]] == [
        (0, "attn.o_proj", "self_attn.o_proj"), (0, "attn.o_proj", "linear_attn.out_proj"), (0, "mlp.down_proj", "mlp.down_proj"),
        (1, "attn.o_proj", "self_attn.o_proj"), (1, "mlp.down_proj", "mlp.down_proj")]
    rep = DR.ablate(ws, {"direction": "r", "weight_distributions": {c: dist(0.5, 2) for c in DR.COMPONENTS}},
                    add=lambda *a: None)
    assert [(c["layer"], c["module"]) for c in rep["components"]] == [(r["layer"], r["module"]) for r in plan["rows"]]
    # ARA on saved inputs and outputs (written here: random inputs, outputs through the matrix read whole).
    counts = {"good": 3, "bad": 3}
    writer = DR._IoWriter(ws, "q", layout, counts)
    g = torch.Generator().manual_seed(3)
    for label, n in counts.items():
        parts = []
        for L in layout:
            for e in L["modules"]:
                V = ws.read(f"model.layers.{L['layer']}.{e['module']}.weight").T
                X = torch.randn(n, V.shape[0], generator=g) + (1.0 if label == "bad" else 0.0)
                parts += [X, X @ V]
        writer.write(label, 0, torch.cat(parts, 1))
    meta = {"version": 1, "name": "q", "created": 0, "variant": "base", "n_layers": 2, "d_model": d, "counts": counts,
            "tokens": {}, "options": {}, "prompts_sha256": {}, "model": {}, "module_io": writer.finish()}
    DR.save(ws, "q", torch.zeros(3, d), torch.zeros(3, d), meta, {"good": ["a"] * 3, "bad": ["b"] * 3}, io_dir=writer.dir)
    got = record_initial_values(monkeypatch)
    ARA.ablate(ws, {"method": "ara", "direction": "q", "lora_rank": 2, "seed": 5}, add=lambda *a: None)
    assert [w for w, _ in got] == ["layer 0 attn.o_proj self_attn.o_proj", "layer 0 attn.o_proj linear_attn.out_proj",
                                   "layer 0 mlp.down_proj", "layer 1 attn.o_proj", "layer 1 mlp.down_proj"]
    for (w, A0), ref in zip(got, draws(5, 2, [d, v, f, d, f])):            # one draw per module, in this order
        assert torch.equal(A0, ref), w


# ------------------------------------------------------------------ L12, L13
def raw_moe_gguf(path):
    """A synthetic GGUF of architecture qwen35moe with the tensor names of a published Qwen3.6-35B-A3B GGUF, 2 blocks
    (0 linear attention, 1 full attention), 4 experts (3-D packed expert tensors), tiny shapes, F32."""
    import gguf
    d, e, i, v = 8, 4, 4, 6
    w = gguf.GGUFWriter(str(path), "qwen35moe")
    w.add_uint32("qwen35moe.block_count", 2)
    w.add_uint32("qwen35moe.expert_count", e)
    z = lambda *shape: np.zeros(shape, dtype=np.float32)  # noqa: E731
    t = {"token_embd.weight": z(16, d), "output_norm.weight": z(d), "output.weight": z(16, d)}
    for b in range(2):
        p = f"blk.{b}"
        t |= {f"{p}.attn_norm.weight": z(d), f"{p}.post_attention_norm.weight": z(d),
              f"{p}.ffn_gate_inp.weight": z(e, d), f"{p}.ffn_gate_exps.weight": z(e, i, d), f"{p}.ffn_up_exps.weight": z(e, i, d),
              f"{p}.ffn_down_exps.weight": z(e, d, i), f"{p}.ffn_gate_inp_shexp.weight": z(1, d),
              f"{p}.ffn_gate_shexp.weight": z(i, d), f"{p}.ffn_up_shexp.weight": z(i, d), f"{p}.ffn_down_shexp.weight": z(d, i)}
        if b == 0:
            t |= {f"{p}.attn_qkv.weight": z(3 * v, d), f"{p}.attn_gate.weight": z(v, d), f"{p}.ssm_a": z(2),
                  f"{p}.ssm_alpha.weight": z(2, d), f"{p}.ssm_beta.weight": z(2, d), f"{p}.ssm_conv1d.weight": z(3 * v, 4),
                  f"{p}.ssm_dt.bias": z(2), f"{p}.ssm_norm.weight": z(3), f"{p}.ssm_out.weight": z(d, v)}
        else:
            t |= {f"{p}.attn_q.weight": z(2 * d, d), f"{p}.attn_k.weight": z(d, d), f"{p}.attn_v.weight": z(d, d),
                  f"{p}.attn_output.weight": z(d, d), f"{p}.attn_q_norm.weight": z(4), f"{p}.attn_k_norm.weight": z(4)}
    for name, a in t.items():
        w.add_tensor(name, a)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return t


def test_a_raw_name_gguf_layout_is_refused_by_directions(tmp_path, monkeypatch):
    from wsp.gguf_store import open_gguf
    names = raw_moe_gguf(tmp_path / "moe.gguf")
    ws = open_gguf(str(tmp_path / "root"), str(tmp_path / "moe.gguf"))
    assert ws.kind == "gguf" and sorted(ws.names()) == sorted(names) and not ws.struct.recognised
    built = []
    monkeypatch.setattr(wsp.runner, "make_runner", lambda w: built.append(w) or pytest.fail("a model was loaded"))
    want = ("directions need a recognised decoder layout; this model's tensors are available by raw name only (No "
            "validated runtime mapping for qwen35moe; raw-name import only.)")
    wd = {"mlp.down_proj": dist(0.0, 1.0)}
    for call in (lambda: DR.plan(ws, {"direction": "m", "weight_distributions": wd}),
                 lambda: DR.prepare(ws, {"direction": "m", "weight_distributions": wd}),
                 lambda: DR.plan(ws, {"method": "ara", "direction": "m"}),
                 lambda: ARA.prepare(ws, {"method": "ara", "direction": "m"}),
                 lambda: DR.measure(ws, "m", ["a"], ["b"]),
                 lambda: DR.measure(ws, "m", ["a"], ["b"], module_io=True)):
        with pytest.raises(DR.DirectionError) as e:
            call()
        assert str(e.value) == want
    assert built == [] and ws._runner is None


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_the_search_offers_both_components_on_hybrid_and_moe(tmp_path, ckpts, kind):
    ws = session(tmp_path, ckpts[kind]).ws
    assert DSR._model_components(ws) == ["attn.o_proj", "mlp.down_proj"]
    # Before the source table only the legacy groups counted: a MoE model has no dense MLP in any layer.
    assert any(DR.component_views(ws, l, "attn.o_proj") is not None for l in range(3))
    if kind == "moe":
        assert all(ws.struct.layers[l].mlp == {} for l in range(LAYERS))
        assert [tuple(v.shape) for v in DR.component_views(ws, 0, "mlp.down_proj")] == [(EXPERT_WIDTH, D)]


# ------------------------------------------------------------------ L14: dense models are unchanged
def _without_new_keys(plan):
    """The plan without what this change added (each row's module and modules, the totals' modules): the rest must be
    exactly what the code before it gave."""
    out = dict(plan)
    out["rows"] = [{k: v for k, v in r.items() if k not in ("module", "modules")} for r in plan["rows"]]
    out["totals"] = {k: v for k, v in plan["totals"].items() if k != "modules"}
    return out


def _dense_digests(ws, good, bad):
    DR.measure(ws, "m", good, bad, module_io=True)
    out, modules = {}, {}
    dists = {"attn.o_proj": dist(0.0, 2.0, w=1.0, low=0.5), "mlp.down_proj": dist(1.0, 1.0, w=0.8, low=0.2)}
    for mode in ("none", "pre", "full"):
        params = {"direction": "m", "weight_distributions": dists, "row_normalization": mode}
        plan = DR.plan(ws, params)
        modules[mode] = [r["module"] for r in plan["rows"]]
        assert "modules" not in plan["totals"]
        out[f"plan_{mode}"] = sha(_without_new_keys(plan))
        out[f"pieces_{mode}"] = DSR.piece_digest(run_script(ws, DR.script_for(params))["deltas"])
    ara = {"method": "ara", "direction": "m", **SMALL, "steer_bad_behavior_weight": 0.5}
    plan = DR.plan(ws, ara)
    modules["ara"] = [r["module"] for r in plan["rows"]]
    out["plan_ara"] = sha(_without_new_keys(plan))
    out["pieces_ara"] = DSR.piece_digest(run_script(ws, DR.script_for(ara))["deltas"])
    info = json.load(open(os.path.join(io_dir(ws, "m"), "io.json")))
    out["io_json"] = sha({**info, "version": 1, "layers": [{k: v for k, v in L.items() if k != "modules"} for L in info["layers"]]})
    return out, modules


def test_dense_models_are_unchanged(tmp_path):
    ws = make_ws(tmp_path / "rnd")
    got, modules = _dense_digests(ws, ["abc", "abd", "hello", "zzz", "quiet", "fox"], ["xyz", "qqq", "world", "mmmm", "bad"])
    assert got == BEFORE["wsp"]
    assert modules["none"] == ["attn.W_O", "mlp.W_out"] * 2 and modules["ara"] == ["attn.W_O", "mlp.W_out"] * 2
    from test_steering_worker import hf_runner
    r, _, _ = hf_runner(tmp_path / "hf")
    got, modules = _dense_digests(r.ws, ["abc", "fed", "gig"], ["kji", "lal", "bb", "hhh"])
    assert got == BEFORE["llama"]
    assert modules["none"] == ["self_attn.o_proj", "mlp.down_proj"] * 2


# ------------------------------------------------------------------ the script, the worker, traces, the adapter export
# Recorded from the code before the interpreter took expert_modules: SHA-256 of the weight script's output text of
# _dense_digests' four calls (none, pre, full, ARA) on the two dense fixtures.
BEFORE_OUTPUT = {
    "wsp": {"none": "05cb89c972deda310343833f2b0f4ff8630555972aa3e3ad6fe65c7ec799b31d",
            "pre": "cd223f4b82d9f2b9e10d349f647d449557f3f98197a85ceb24107d28646d3ea5",
            "full": "bcdad544b8f647a2b22fb388f29852f0c05432e80dfed4e5b4f1985df02c3063",
            "ara": "938f9e5cf5e72e085b5b8b66dcd8819b56fa73c130934f53668897ed660926d8"},
    "llama": {"none": "a62b53addd6bd7f20a27cd185267e74a2cb155a4aae171898f4fa42385250a05",
              "pre": "807a49504b2c20202076b47c2a9a38af1959e0f07250b0bfc5ed5b0ae6c239bc",
              "full": "5a206a8d81c5ff98e788a7efe5fd261200e416d66d63bb42b873ecfa222875e2",
              "ara": "66aa4475145dd206306958c3450d3efc0f5bee58dbdd57a0a2edbfab208283dc"},
}
EXPERTS_REFUSED = ("transformers loads a layer's routed experts as one fused parameter, so a LoRA adapter has no module "
                   "to attach to; export the checkpoint instead")
MULTIMODAL_NOTE = ("the adapter's module names are the checkpoint's (model.language_model...): load it on the model class "
                   "the checkpoint names, not on the text-only model")
# What Heretic at 64e9991 was seen to do on tiny mixture-of-experts models (scripts/heretic_moe_probe.py), by method
HERETIC_NOTE = {
    "ablation": ("expert_modules is an IDE field; Heretic at 64e9991 under transformers 5.x ablates only the shared "
                 "expert of such a model (it does not list the fused routed experts as modules)"),
    "ara": ("expert_modules is an IDE field; Heretic at 64e9991 under transformers 5.x lists no routed experts of such a "
            "model, and its ARA raises while capturing a shared expert's inputs (it reads them as [prompt, position, "
            "width]; they are [tokens, width]), so it fits such a model only when it has no shared expert"),
}


def _outputs(ws, good, bad):
    DR.measure(ws, "m", good, bad, module_io=True)
    dists = {"attn.o_proj": dist(0.0, 2.0, w=1.0, low=0.5), "mlp.down_proj": dist(1.0, 1.0, w=0.8, low=0.2)}
    out = {mode: run_script(ws, DR.script_for({"direction": "m", "weight_distributions": dists,
                                                 "row_normalization": mode}))["output"] for mode in ("none", "pre", "full")}
    out["ara"] = run_script(ws, DR.script_for({"method": "ara", "direction": "m", **SMALL,
                                               "steer_bad_behavior_weight": 0.5}))["output"]
    return {k: hashlib.sha256(v.encode()).hexdigest() for k, v in out.items()}


def test_dense_script_output_is_unchanged(tmp_path):
    assert _outputs(make_ws(tmp_path / "rnd"), ["abc", "abd", "hello", "zzz", "quiet", "fox"],
                    ["xyz", "qqq", "world", "mmmm", "bad"]) == BEFORE_OUTPUT["wsp"]
    from test_steering_worker import hf_runner
    r, _, _ = hf_runner(tmp_path / "hf")
    assert _outputs(r.ws, ["abc", "fed", "gig"], ["kji", "lal", "bb", "hhh"]) == BEFORE_OUTPUT["llama"]


def _record(ws, pid):
    with open(REC.record_path(ws, pid)) as f:
        return json.load(f)


def shared_names(layers=range(LAYERS)):
    return [f"{PRE}.{l}.mlp.shared_expert.down_proj.weight" for l in layers]


def expert_names(layers=range(LAYERS)):
    return [f"{PRE}.{l}.mlp.experts.{e}.down_proj.weight" for l in layers for e in range(EXPERTS)]


def test_a_shared_ablation_through_the_script_commits_only_shared_expert_tensors(tmp_path, ckpts):
    sess = session(tmp_path, ckpts["moe"])
    ws = sess.ws
    save_random(ws, "r")
    params = {"direction": "r", "weight_distributions": {"mlp.down_proj": dist(1.5, 4)}, "expert_modules": "shared"}
    script = DR.script_for(params)
    done = sess.apply(script)
    assert done["ok"] and done["committed"], done
    assert sorted(ws.touched()) == sorted(shared_names())
    assert done["output"].splitlines()[0] == (
        "ablate_direction('r'): 4 components in 4 layers, 160 values stored, row_normalization none, 0 skipped")
    rec = _record(ws, done["patch"])
    (call,) = rec["calls"]
    assert call["parameters"]["expert_modules"] == "shared" and rec["rebuildable"] is True and rec["script"] == script
    assert "expert_modules" not in call["heretic"]["parameters"] and "expert_modules" not in call["heretic"]["settings"]
    assert call["heretic"]["note"] == HERETIC_NOTE["ablation"]
    assert [c["module"] for c in call["result"]["components"]] == ["mlp.shared_expert.down_proj"] * LAYERS


def test_the_default_ablation_through_the_script_commits_every_expert_and_the_shared_one(tmp_path, ckpts):
    sess = session(tmp_path, ckpts["moe"])
    ws = sess.ws
    save_random(ws, "r")
    params = {"direction": "r", "weight_distributions": {"mlp.down_proj": dist(1.5, 4)}, "row_normalization": "full",
              "full_normalization_lora_rank": 2}
    done = sess.apply(DR.script_for(params))
    assert done["ok"] and done["committed"], done
    assert sorted(ws.touched()) == sorted(expert_names() + shared_names())
    lines = done["output"].splitlines()
    # 5 modules per layer in 2 rows (the routed experts, the shared expert): the line names both counts.
    assert lines[0].startswith("ablate_direction('r'): 20 modules (8 components) in 4 layers, ")
    # More than one module of mlp.down_proj in a layer: each per-module line names its module.
    assert [re.match(r"  L(\d) mlp\.down_proj (\S+): weight", x).groups() for x in lines[1:]] == [
        (str(l), m) for l in range(LAYERS) for m in [f"mlp.experts.{e}.down_proj" for e in range(EXPERTS)]
        + ["mlp.shared_expert.down_proj"]]
    rec = _record(ws, done["patch"])
    assert "expert_modules" not in rec["calls"][0]["parameters"] and "note" not in rec["calls"][0]["heretic"]


def test_ara_through_the_script_fits_the_shared_expert_and_refuses_the_default(tmp_path, ckpts):
    sess = session(tmp_path, ckpts["moe"])
    ws = sess.ws
    DR.measure(ws, "m", GOOD, BAD, module_io=True)
    ara = {"method": "ara", "direction": "m", **SMALL, "steer_bad_behavior_weight": 0.5}
    refused = sess.apply(DR.script_for(ara))
    assert not refused["ok"] and refused["error"] == "line 1: " + ROUTED_REFUSAL and ws.patches == []
    done = sess.apply(DR.script_for({**ara, "expert_modules": "shared"}))
    assert done["ok"] and done["committed"], done
    mixers = [f"{PRE}.{l}.linear_attn.out_proj.weight" for l in range(3)] + [f"{PRE}.3.self_attn.o_proj.weight"]
    assert sorted(ws.touched()) == sorted(mixers + shared_names()) and len(ws.patches) == 1
    out = done["output"].splitlines()
    assert out[0].startswith("ablate_ara('m'): 8 components in 4 layers, ")
    assert [x.split(":")[0] for x in out[1:]] == [f"  L{l} {c}" for l in range(LAYERS) for c in ("attn.o_proj", "mlp.down_proj")]
    rec = _record(ws, done["patch"])
    assert rec["calls"][0]["heretic"]["note"] == HERETIC_NOTE["ara"]
    assert "expert_modules" not in rec["calls"][0]["heretic"]["parameters"]


def test_the_worker_reports_every_hook_kind_the_runner_has(tmp_path):
    """The model worker's load job lists the hook points its runner has (RemoteRunner.available reads that list):
    every kind of the runner's table, and attention patterns."""
    from test_worker_protocol import FakeRunner, Harness
    from wsp.runner import HOOK_KINDS

    class EveryHook(FakeRunner):
        def available(self, name):
            return True
    h = Harness(tmp_path, make_runner=EveryHook)
    try:
        got = h.load()["value"]["available"]
        kinds = set(HOOK_KINDS) | {"attn.hook_pattern"}
        assert {"mixer.hook_in", "mixer.hook_out", "shared.hook_in", "shared.hook_out"} <= kinds
        assert sorted(got) == sorted(f"blocks.{l}.{k}" for l in range(2) for k in kinds)
    finally:
        h.shutdown()


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_a_pattern_trace_maps_the_full_attention_layer_on_hybrid_and_moe(tmp_path, ckpts, kind):
    ws = session(tmp_path, ckpts[kind]).ws
    ids = ws.encode("t3 t7 t11 t5 t9 t2")
    _, cache = ws.runner.run(ids, names=[f"blocks.{l}.attn.hook_pattern" for l in range(LAYERS)])
    assert sorted(cache) == ["blocks.3.attn.hook_pattern"]                   # the one full-attention layer
    ref = transformers.AutoModelForCausalLM.from_pretrained(ckpts[kind], dtype=torch.float32, attn_implementation="eager").eval()
    seen = {}
    h = ref.model.layers[3].self_attn.register_forward_hook(lambda m, a, o: seen.__setitem__("w", o[1]))
    try:
        with torch.no_grad():
            ref(torch.tensor([ids]), output_attentions=True)
    finally:
        h.remove()
    assert tuple(cache["blocks.3.attn.hook_pattern"].shape) == (1, 2, len(ids), len(ids))
    torch.testing.assert_close(cache["blocks.3.attn.hook_pattern"], seen["w"].float(), atol=1e-6, rtol=0)


def _export_answer(ws):
    from wsp import adapter_export as AX
    try:
        return AX.plan(ws)
    except AX.AdapterError as e:
        return str(e)


def test_the_adapter_export_refuses_routed_expert_tensors(tmp_path, ckpts):
    sess = session(tmp_path, ckpts["moe"])
    ws = sess.ws
    save_random(ws, "r")
    done = sess.apply(DR.script_for({"direction": "r", "weight_distributions": {"mlp.down_proj": dist(1.5, 4)}}))
    assert done["ok"] and done["committed"]
    answer = _export_answer(ws)
    assert isinstance(answer, str), answer                                  # refused, not planned
    m = re.match(r"^patch \S+ changes (\S+), (.+): a LoRA adapter holds changes of linear layers' weight matrices only$", answer)
    assert m and m.group(1) in expert_names() and m.group(2) == EXPERTS_REFUSED, answer
    from wsp import adapter_export as AX
    for name in expert_names([0]) + [f"{PRE}.0.mlp.experts.1.{x}_proj.weight" for x in ("gate", "up")]:
        assert AX.tensor_reason(ws, name) == EXPERTS_REFUSED, name
    for name in shared_names([0]) + [f"{PRE}.0.linear_attn.out_proj.weight", f"{PRE}.0.mlp.shared_expert.up_proj.weight"]:
        assert AX.tensor_reason(ws, name) is None, name


def test_with_shared_the_adapter_targets_are_the_shared_experts_and_the_mixers(tmp_path, ckpts):
    sess = session(tmp_path, ckpts["moe"])
    ws = sess.ws
    save_random(ws, "r")
    params = {"direction": "r", "expert_modules": "shared",
              "weight_distributions": {"attn.o_proj": dist(1.0, 1.5), "mlp.down_proj": dist(1.5, 4)}}  # attn: layers 0-2
    done = sess.apply(DR.script_for(params))
    assert done["ok"] and done["committed"], done
    answer = _export_answer(ws)
    assert isinstance(answer, dict), answer
    want = sorted([f"{PRE}.{l}.linear_attn.out_proj" for l in range(3)] + [n[:-len(".weight")] for n in shared_names()])
    assert sorted(e["module"] for e in answer["modules"]) == want
    assert answer["notes"].count(MULTIMODAL_NOTE) == 1
    # A checkpoint without text_config gets no such note (the llama-named fake).
    from test_steering_worker import hf_runner
    from wsp import adapter_export as AX
    r, _, _ = hf_runner(tmp_path / "hf")
    save_random(r.ws, "r")
    res = run_script(r.ws, DR.script_for({"direction": "r", "weight_distributions": {"mlp.down_proj": dist(0.0, 4)}}))
    r.ws.commit(res["deltas"], "ablation")
    assert MULTIMODAL_NOTE not in AX.plan(r.ws)["notes"]
