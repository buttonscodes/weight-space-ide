"""End-to-end checks on throwaway workspaces (no network, no trained models)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wsp import addressing as A  # noqa: E402
from wsp import analysis as an  # noqa: E402
from wsp import chunks, delta as D, edits, hierarchy, mcp, zoo  # noqa: E402
from wsp.model import Config, forward, init_params, param_shapes  # noqa: E402
from wsp.session import Session  # noqa: E402
from wsp.store import Store, norm_index, region_index, region_shape, write_safetensors  # noqa: E402
from wsp.workspace import Workspace  # noqa: E402
from wsp.wscript import ScriptError, run_script  # noqa: E402


@pytest.fixture()
def sess(tmp_path):
    s = Session(str(tmp_path))
    zoo.build_blank_reverse(str(tmp_path / "blank-reverse"))
    zoo.build_blank_modadd(str(tmp_path / "blank-modadd"))
    return s


@pytest.fixture()
def rnd(tmp_path):
    """A random 2-layer model with LayerNorm and a char vocabulary."""
    from wsp.tokenizer import VocabTokenizer
    tok = VocabTokenizer(list("abcdefghij"), "char")
    cfg = Config(d_vocab=10, n_ctx=12, d_model=16, n_layers=2, n_heads=2, d_mlp=32, name="rnd")
    ws = Workspace.create(str(tmp_path / "rnd"), cfg, tok, init_params(cfg, seed=1, std=0.3))
    ws.write_json("tests/t.json", {"name": "t", "cases": [{"type": "next_token", "prompt": "abc", "expect": "d"},
                                                          {"type": "preserve", "prompt": "abca", "max_kl": 0.01}]})
    s = Session(str(tmp_path))
    s.open("rnd")
    return s


def params(ws, variant="current"):
    return {n: ws.read(n, (), variant) for n in ws.names()}


# ------------------------------------------------------------------------------ lazy store and deltas
def test_store_reads_regions_without_loading(tmp_path):
    rng = np.random.default_rng(0)
    arrs = {"a.weight": rng.normal(size=(40, 30)).astype(np.float32), "b.bias": rng.normal(size=(7,)).astype(np.float16),
            "c": rng.normal(size=(3, 4, 5)).astype(np.float32)}
    bf = torch.randn(6, 5).to(torch.bfloat16)
    items = [(k, {"float32": "F32", "float16": "F16"}[str(v.dtype)], v.shape) for k, v in arrs.items()] + [("d.bf", "BF16", (6, 5))]
    blob = {**{k: v.tobytes() for k, v in arrs.items()}, "d.bf": bf.view(torch.int16).numpy().tobytes()}
    write_safetensors(str(tmp_path / "m.safetensors"), items, blob.__getitem__)
    st = Store([str(tmp_path / "m.safetensors")])
    assert st.shape("a.weight") == (40, 30) and st.dtype("d.bf") == "BF16" and st.total() == 1200 + 7 + 60 + 30
    for name, idx in (("a.weight", (slice(3, 9), 4)), ("a.weight", (slice(0, 40, 3), slice(2, 20, 5))), ("c", (1, slice(None), 2)), ("b.bias", (3,))):
        got = st.read(name, norm_index(idx, st.shape(name)))
        assert torch.allclose(got, torch.as_tensor(np.asarray(arrs[name][idx], dtype=np.float32)))
    assert torch.equal(st.read("d.bf"), bf.float())
    total = torch.cat([t for _, t in st.row_blocks("a.weight", max_elems=100)])
    assert torch.equal(total, torch.from_numpy(arrs["a.weight"]))
    from safetensors.torch import load_file                      # the writer produces files the reference reader accepts
    assert torch.equal(load_file(str(tmp_path / "m.safetensors"))["d.bf"], bf)


def test_delta_pieces_match_dense_arithmetic():
    rng = np.random.default_rng(1)
    shape = (6, 7, 5)

    def axis(n, allow_int=True):
        if allow_int and rng.random() < 0.3:
            return int(rng.integers(n))
        a = int(rng.integers(n))
        return slice(a, int(rng.integers(a + 1, n + 1)), 1 if rng.random() < 0.7 else 2)

    for _ in range(150):
        dense, pieces = torch.zeros(shape), []
        for _ in range(3):
            kind = rng.choice(["block", "lowrank", "sparse"])
            if kind == "block":
                reg = norm_index(tuple(axis(n) for n in shape), shape)
                v = torch.randn(region_shape(reg))
                pieces.append(D.block(reg, v))
                dense[region_index(reg)] += v
            elif kind == "lowrank":
                ax = sorted(rng.choice(3, 2, replace=False))
                reg = norm_index(tuple(axis(shape[d], False) if d in ax else int(rng.integers(shape[d])) for d in range(3)), shape)
                m, n = region_shape(reg)
                a, b = torch.randn(m, 2), torch.randn(2, n)
                pieces.append(D.lowrank(reg, a, b))
                dense[region_index(reg)] += a @ b
            else:
                c = torch.unique(torch.stack([torch.randint(0, n, (4,)) for n in shape], 1), dim=0)
                v = torch.randn(c.shape[0])
                pieces.append(D.Piece("sparse", norm_index((), shape), values=v, coords=c))
                dense[tuple(c.T)] += v
        req = norm_index(tuple(axis(n) for n in shape), shape)
        out = torch.zeros(region_shape(req))
        for p in pieces:
            D.add_to(out, req, p)
        assert torch.allclose(out, dense[region_index(req)], atol=1e-5)
        merged = torch.zeros(shape)
        for p in D.compact(pieces, shape):
            D.add_to(merged, norm_index((), shape), p)
        assert torch.allclose(merged, dense, atol=1e-5)
        index, tensors = D.pack({"t": pieces})
        again = torch.zeros(region_shape(req))
        for p in D.unpack(index, tensors)["t"]:
            D.add_to(again, req, p)
        assert torch.allclose(again, out)


# ------------------------------------------------------------------------------ this project's own models
def test_hand_written_programs(sess):
    for name, script in (("blank-reverse", "reverse.wscript"), ("blank-modadd", "modadd.wscript")):
        sess.open(name)
        assert sess.tests()["passed"] == 0                       # all-zero weights: every case is a tie, so a failure
        r = sess.apply(open(os.path.join(ROOT, "examples", script)).read(), enforce_scope=True)
        assert r["ok"] and r["committed"], r
        assert r["tests"]["after"][0] == r["tests"]["after"][1]
        assert r["stored"] < 2000                                # stored as what changed, not as whole tensors


def test_forward_hooks_and_cache(rnd):
    ws = rnd.ws
    toks = torch.tensor(ws.tok.encode("abcabc"))
    cache = {}
    p = params(ws)
    a = forward(p, ws.cfg, toks, cache=cache, per_head=True)
    assert cache["blocks.1.attn.hook_pattern"].shape == (1, 2, 6, 6)
    assert torch.allclose(cache["blocks.0.attn.hook_result"].sum(2) + p["blocks.0.attn.b_O"], cache["blocks.0.hook_attn_out"], atol=1e-5)
    b = forward(p, ws.cfg, toks, hooks={"blocks.0.hook_mlp_out": lambda x, n: torch.zeros_like(x)})
    assert not torch.allclose(a, b)
    c = forward(p, ws.cfg, torch.tensor(ws.tok.encode("abcabd")))     # causal: a later token cannot change earlier logits
    assert torch.allclose(a[0, :5], c[0, :5], atol=1e-5)
    assert torch.allclose(ws.runner.run(toks)[0], a, atol=1e-5)


def test_addresses_cover_the_model_exactly(rnd):
    ws = rnd.ws
    n = ws.n_params()
    assert A.count(ws, ws.resolve("model")) == n
    assert hierarchy.node(ws, "model", kind="model")["n_params"] == n
    assert sum(c["n_params"] for c in hierarchy.children(ws, "model")) == n
    assert ws.resolve("L1H0")[0] == ("blocks.1.attn.W_Q", (0,))
    assert A.count(ws, ws.resolve("L0N3")) == 16 + 1 + 16
    assert A.count(ws, ws.resolve("blocks.*.mlp")) == 2 * (16 * 32 * 2 + 32 + 16)
    assert A.count(ws, ws.resolve("blocks.0.mlp.W_out[0:4]")) == 4 * 16
    with pytest.raises(KeyError):
        ws.resolve("blocks.0.nothing")
    seen = set()

    def walk(addr):
        for c in hierarchy.partition(ws, addr) or []:
            walk(c)
        if not hierarchy.partition(ws, addr):
            for name, idx in ws.resolve(addr):
                key = (name, A.index_str(idx))
                assert key not in seen
                seen.add(key)

    walk("model")                                                   # the chunk partition never lists a slice twice


def test_patches_are_additive_and_reversible(rnd):
    ws = rnd.ws
    base = params(ws)
    r1 = rnd.apply("blocks[0].mlp.W_out[3] *= 0")
    r2 = rnd.apply("embed.W_E[tok('a')] += 0.5 * unit(U('b'))")
    assert r1["committed"] and r2["committed"]
    assert float(ws.read("blocks.0.mlp.W_out", (3,)).abs().sum()) == 0
    ws.set_patch(r1["patch"], strength=0.5)
    assert torch.allclose(ws.read("blocks.0.mlp.W_out", (3,)), 0.5 * base["blocks.0.mlp.W_out"][3], atol=1e-6)
    ws.set_patch(r1["patch"], enabled=False)
    assert torch.equal(ws.read("blocks.0.mlp.W_out"), base["blocks.0.mlp.W_out"])
    again = Workspace.load(ws.dir)                                  # persisted as pieces
    for k in base:
        assert torch.allclose(again.read(k), ws.read(k), atol=1e-7)
    ws.delete_patch(r1["patch"])
    ws.delete_patch(r2["patch"])
    assert all(torch.equal(ws.read(k), base[k]) for k in base)
    assert rnd.apply("blocks[1].attn.W_O[0] *= 2")["ok"]
    assert rnd.apply("revert(head(1, 0))")["ok"]
    assert torch.allclose(ws.read("blocks.1.attn.W_O"), base["blocks.1.attn.W_O"], atol=1e-6)
    assert ws.changed("blocks.1.attn.W_O")[0] == 0 or float(ws.read("blocks.1.attn.W_O", (), "delta").abs().max()) < 1e-6


def test_scope_is_enforced(rnd):
    rnd.set_task({"scope": ["blocks.0.mlp.neuron.5"], "suite": "t"})
    assert not rnd.apply("blocks[0].mlp.W_out[6] *= 0", enforce_scope=True)["ok"]
    assert not rnd.apply("embed.W_E *= 0.9", enforce_scope=True)["ok"]
    assert not rnd.apply("rank1(mlp(0).W_out, onehot(5, d_mlp) + onehot(6, d_mlp), ones(d_model))", enforce_scope=True)["ok"]
    ok = rnd.apply("blocks[0].mlp.W_out[5] *= 0\nblocks[0].mlp.b_in[5] = -1\nrank1(mlp(0).W_out, onehot(5, d_mlp), ones(d_model))",
                   enforce_scope=True)
    assert ok["ok"] and ok["committed"]
    assert len(rnd.ws.patches) == 1


@pytest.mark.parametrize("bad", ["import os", "x = embed.W_E.__class__", "open('/etc/passwd')", "embed.W_E.numpy()",
                                 "(lambda: 1)()", "blocks[9].mlp.W_in *= 2", "embed.W_E[0] = zeros(3)",
                                 "x = [].__class__", "def f(): pass", "while True: pass", "neuron(0, 99).w_out *= 0",
                                 "x = 'a'.__len__()", "exec('1')", "embed.W_E[0] = 1/0 * ones(16)", "head(0, 0).Z *= 2"])
def test_scripts_cannot_escape(rnd, bad):
    r = rnd.apply(bad)
    assert not r["ok"], bad
    assert len(rnd.ws.patches) == 0


def test_script_language(rnd):
    ws = rnd.ws
    r = run_script(ws, """
ks = [act('ab' + c, 'blocks.0.mlp.hook_post') for c in ['a', 'b'] for i in range(2) if i == 0]
K = stack(ks, 0)
xs = []
xs.append(norm(K))
Uc, S, Vh = svd(blocks[0].mlp.W_out)
blocks[0].mlp.W_out -= S[0] * outer(Uc[:, 0], Vh[0])
orthogonalize(U('c'))
for h, f in [(0, 0.5), (1, 2.0)]:
    head(1, h).O *= f
w = neuron(1, 4).w_out
w *= 0
print(len(ks), K.shape, f"{float(xs[0]):.2f}", cossim(U('a'), U('b')), w.shape)
""")
    assert "2 (2, 32)" in r["output"] and "(16,)" in r["output"]
    assert set(r["deltas"]) >= {"blocks.0.mlp.W_out", "blocks.1.attn.W_O", "embed.W_E", "blocks.1.mlp.W_out"}
    cand = {k: ws.read(k, (), "current", r["deltas"]) for k in ws.names()}
    c = ws.read("unembed.W_U")[:, ws.tok.encode("c")[0]]
    assert float((cand["embed.W_E"] @ c).abs().max()) < 1e-4          # direction removed from the embeddings
    assert float(cand["blocks.1.mlp.W_out"][4].abs().max()) < 1e-6    # zeroed on top of an earlier low-rank piece
    assert torch.allclose(cand["blocks.1.attn.W_O"][1], 2 * ws.read("blocks.1.attn.W_O")[1] - 2 * torch.outer(ws.read("blocks.1.attn.W_O")[1] @ (c / c.norm()), c / c.norm()), atol=1e-4)
    with pytest.raises(ScriptError):
        run_script(ws, "embed.W_E[tok('ab')] = 0")


def test_chunk_plan_partitions_and_respects_budget(rnd):
    ws = rnd.ws
    n = ws.n_params()
    for budget, fmt in ((3000, "q"), (20000, "g3"), (2000, "q")):
        plan = chunks.plan(ws, ["model"], budget, fmt)
        assert sum(c["n_params"] for c in plan["chunks"]) == n
        assert all(c["est_tokens"] <= budget for c in plan["chunks"])
        mask = {k: torch.zeros(ws.shape(k), dtype=torch.int32) for k in ws.names()}
        for c in plan["chunks"]:
            for a in c["addrs"]:
                for name, idx in ws.resolve(a):
                    m = an.as_matrix(mask[name][idx]) if c.get("rows") else mask[name][idx]
                    if c.get("rows"):
                        m[c["rows"][0]:c["rows"][1]] += 1
                    else:
                        m += 1
        assert all(bool((m == 1).all()) for m in mask.values())     # every weight in exactly one chunk
        text = chunks.render_chunk(ws, plan["chunks"][0])
        assert chunks.est_text(text) <= plan["chunks"][0]["est_tokens"] * 1.25
    assert "format q: weight = integer *" in chunks.render(ws, ["L0H1"], "q")
    for fmt in chunks.FORMATS:
        assert chunks.render(ws, ["blocks.0.mlp.W_in"], fmt)


def test_q_format_round_trips_through_a_script(rnd):
    """What the programmer reads in format q can be written back and reproduces the tensor to ~1/999 of its range."""
    ws = rnd.ws
    text = chunks.render_tensor(ws, "blocks.0.attn.W_Q", (1,), "q")
    scale = float(text.split("integer * ")[1].split()[0])
    rows = [[int(x) for x in line.split(": ")[1].split()] for line in text.splitlines() if line.startswith("r")]
    w = ws.read("blocks.0.attn.W_Q", (1,))
    r = rnd.apply(f"blocks[0].attn.W_Q[1] = {scale} * tensor({json.dumps(rows)})")
    assert r["ok"]
    assert float((ws.read("blocks.0.attn.W_Q", (1,)) - w).abs().max()) <= scale * 0.51


def test_analysis_views(rnd):
    ws = rnd.ws
    m = an.matrix_view(ws, "blocks.0.mlp.W_in", max_h=8, max_w=8)
    assert m["grid"] == [8, 8] and m["factor"] == [2, 4]
    c = an.cell_address(ws, "blocks.0.mlp.W_in[:, 5]", 0, 7)
    assert c["script"] == "blocks[0].mlp.W_in[7, 5]" and c["current"] == float(ws.read("blocks.0.mlp.W_in")[7, 5])
    assert an.cell_address(ws, "blocks.1.attn.W_O", 8 + 3, 2)["index"] == [1, 3, 2]
    assert an.svd_view(ws, "blocks.0.mlp.W_out")["rank"] == 16
    assert an.head_card(ws, 0, 1)["norms"]["Q"] > 0 and an.neuron_card(ws, 1, 3)["mechanism"]
    w = an.wiring(ws)
    assert w["n_edges"] > 0 and all(e["score"] > 0 for e in w["edges"])
    t = an.trace(ws, "abcab")
    tgt = ws.tok.encode(t["target"])[0]
    const = float(ws.read("unembed.b_U")[tgt]) + float(ws.read("ln_final.b") @ ws.read("unembed.W_U")[:, tgt])
    assert abs(sum(d["value"] for d in t["dla"]) + const - t["dla_total"]) < 1e-3      # attribution adds up to the logit
    p = an.patching(ws, "abcab", "abdab")
    assert len(p["heads"]) == 2 and len(p["by_position"][0]["grid"][0]) == 5


def test_streamed_views_agree_with_dense(rnd, monkeypatch):
    """Force the large-tensor code paths (slabs, streamed SVD, sampled statistics) on a small tensor and compare."""
    ws = rnd.ws
    dense = an.matrix_view(ws, "blocks.0.mlp.W_in", max_h=8, max_w=8)
    sv = an.svd_view(ws, "blocks.0.mlp.W_out")
    from wsp import workspace as W
    monkeypatch.setattr(an, "DENSE_ELEMS", 10)
    monkeypatch.setattr(an, "STREAM_ELEMS", 64)
    monkeypatch.setattr(W, "STREAM_ELEMS", 64)
    monkeypatch.setattr(an, "EXACT_ELEMS", 50)
    streamed = an.matrix_view(ws, "blocks.0.mlp.W_in", max_h=8, max_w=8)
    assert streamed["grid"] == dense["grid"] and streamed["data"] == dense["data"]
    sv2 = an.svd_view(ws, "blocks.0.mlp.W_out")
    assert sv2["approx"] and np.allclose(sv2["singular_values"][:8], sv["singular_values"][:8], rtol=1e-3)
    v = torch.randn(16)
    ref = ws.sem.mlp(0)["W_out"]
    assert torch.allclose(ref.contract(v, axis=1), ws.read("blocks.0.mlp.W_out") @ v, atol=1e-5)
    assert torch.allclose(ref.contract(torch.randn(32, 3), axis=0).shape == (16, 3) and torch.zeros(1), torch.zeros(1))
    st = an.region_stats(ws, "blocks.0.mlp")
    assert st["sampled"] and st["current"]["n"] == 16 * 32 * 2 + 32 + 16


def test_rome_inserts_a_fact_as_two_vectors(rnd):
    ws = rnd.ws
    before = ws.tok.token_str(int(ws.logits("abc")[-1].argmax()))
    target = "d" if before != "d" else "e"
    r = edits.rome(ws, "abc", "c", target)
    assert r["success"] and r["p_after"] > 0.5
    (name, pieces), = r["deltas"].items()
    assert name == f"blocks.{r['layer']}.mlp.W_out" and pieces[0].kind == "lowrank" and pieces[0].numel() == 32 + 16


def test_file_route(rnd, tmp_path):
    ws = rnd.ws
    edits.checkout(ws, ["L0N2"], str(tmp_path / "w"))
    f = tmp_path / "w" / "blocks.0.mlp.W_out.npy"
    arr = np.load(f)
    arr[2] *= 0
    np.save(f, arr)
    d = edits.read_files(ws, str(tmp_path / "w"))
    assert list(d) == ["blocks.0.mlp.W_out"] and sum(p.stats()["changed"] for p in d["blocks.0.mlp.W_out"]) == 16
    assert d["blocks.0.mlp.W_out"][0].numel() == 16                  # trimmed to the row that changed


def test_tools_and_mcp_protocol(rnd):
    backend = mcp.LocalBackend(rnd)
    init = mcp.handle(backend, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
    assert init["result"]["protocolVersion"] == "2025-06-18" and "tools" in init["result"]["capabilities"]
    assert mcp.handle(backend, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    names = [t["name"] for t in mcp.handle(backend, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]]
    assert {"overview", "plan_chunks", "read_chunk", "read_weights", "edit", "run_tests", "finish"} <= set(names)

    def call(name, **args):
        r = mcp.handle(backend, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": name, "arguments": args}})
        return r["result"]["content"][0]["text"], r["result"]["isError"]

    for name, args in (("overview", {}), ("list_nodes", {"address": "blocks.0", "depth": 2}), ("plan_chunks", {"budget_tokens": 4000}),
                       ("read_chunk", {"chunk_id": 0}), ("read_weights", {"address": "L0H1"}), ("inspect", {"address": "L1N3"}),
                       ("inspect", {"address": "blocks.0.mlp.W_out", "what": "svd"}), ("inspect", {"what": "wiring"}),
                       ("inspect", {"expr": "U('a') - U('b')"}), ("vocab", {"text": "abc"}), ("run", {"prompt": "abc", "max_new_tokens": 3}),
                       ("trace", {"prompt": "abcab"}), ("run_tests", {}), ("patches", {})):
        text, err = call(name, **args)
        assert not err and text, (name, text)
    text, err = call("read_weights", address="model", max_tokens=500)
    assert "plan_chunks" in text                                   # refuses, and says how to narrow
    text, err = call("edit", script="blocks[0].mlp.W_out[1] *= 0", message="ablate", dry_run=True)
    assert "DRY RUN" in text and len(rnd.ws.patches) == 0
    text, err = call("edit", script="blocks[0].mlp.W_out[1] *= 0", message="ablate")
    assert "COMMITTED" in text and rnd.ws.patches[0].author == "programmer"
    text, err = call("edit", script="import os")
    assert text.startswith("ERROR") and not err
    assert call("edit", nonsense=1)[1] is True                     # bad arguments -> isError, not a crash
    assert mcp.handle(backend, {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "nope"}})["error"]["code"] == -32602
    assert "weight script" in rnd.brief().lower()


def test_stdio_server_without_ide_reports_cleanly():
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}},
            {"jsonrpc": "2.0", "id": 2, "method": "ping"}]
    out = subprocess.run([sys.executable, "-m", "wsp.mcp", "--url", "http://127.0.0.1:9"], cwd=ROOT, text=True,
                         input="\n".join(json.dumps(m) for m in msgs) + "\n", capture_output=True, timeout=30)
    lines = [json.loads(x) for x in out.stdout.splitlines()]
    assert lines[0]["result"]["serverInfo"]["name"] == "wsp" and lines[1]["result"] == {}


def test_converters_match_reference_layout():
    """GPT-2 converter: a state dict in the checkpoint layout gives the same logits as a direct forward."""
    torch.manual_seed(0)
    d, H, L, V, T = 8, 2, 1, 11, 6
    sd = {"wte.weight": torch.randn(V, d), "wpe.weight": torch.randn(T, d), "ln_f.weight": torch.rand(d) + 0.5, "ln_f.bias": torch.randn(d),
          "h.0.ln_1.weight": torch.rand(d) + 0.5, "h.0.ln_1.bias": torch.randn(d), "h.0.ln_2.weight": torch.rand(d) + 0.5, "h.0.ln_2.bias": torch.randn(d),
          "h.0.attn.c_attn.weight": torch.randn(d, 3 * d), "h.0.attn.c_attn.bias": torch.randn(3 * d),
          "h.0.attn.c_proj.weight": torch.randn(d, d), "h.0.attn.c_proj.bias": torch.randn(d),
          "h.0.mlp.c_fc.weight": torch.randn(d, 4 * d), "h.0.mlp.c_fc.bias": torch.randn(4 * d),
          "h.0.mlp.c_proj.weight": torch.randn(4 * d, d), "h.0.mlp.c_proj.bias": torch.randn(d)}
    cfg, p = zoo.convert_gpt2(sd, {"n_embd": d, "n_layer": L, "n_head": H, "n_positions": T, "vocab_size": V, "layer_norm_epsilon": 1e-5})
    toks = torch.tensor([[1, 4, 2, 9, 3]])
    ours = forward(p, cfg, toks)
    import torch.nn.functional as F
    x = sd["wte.weight"][toks] + sd["wpe.weight"][:5]
    h = F.layer_norm(x, (d,), sd["h.0.ln_1.weight"], sd["h.0.ln_1.bias"], 1e-5)
    q, k, v = (h @ sd["h.0.attn.c_attn.weight"] + sd["h.0.attn.c_attn.bias"]).split(d, dim=-1)
    sp = lambda t: t.view(1, 5, H, d // H).transpose(1, 2)  # noqa: E731
    att = (sp(q) @ sp(k).transpose(-1, -2)) / (d // H) ** 0.5
    att = att.masked_fill(torch.ones(5, 5, dtype=torch.bool).triu(1), float("-inf")).softmax(-1)
    y = (att @ sp(v)).transpose(1, 2).reshape(1, 5, d)
    x = x + y @ sd["h.0.attn.c_proj.weight"] + sd["h.0.attn.c_proj.bias"]
    h = F.layer_norm(x, (d,), sd["h.0.ln_2.weight"], sd["h.0.ln_2.bias"], 1e-5)
    x = x + F.gelu(h @ sd["h.0.mlp.c_fc.weight"] + sd["h.0.mlp.c_fc.bias"], approximate="tanh") @ sd["h.0.mlp.c_proj.weight"] + sd["h.0.mlp.c_proj.bias"]
    ref = F.layer_norm(x, (d,), sd["ln_f.weight"], sd["ln_f.bias"], 1e-5) @ sd["wte.weight"].T
    assert torch.allclose(ours, ref, atol=1e-4)
    assert set(p) == set(param_shapes(cfg))


def test_remote_store_reads_by_range(tmp_path):
    """A checkpoint behind HTTP: the header is all that is fetched to open it; regions come by range request."""
    import http.server
    import threading

    rng = np.random.default_rng(3)
    big = rng.normal(size=(4000, 4096)).astype(np.float32)          # 62 MB: above the fetch-whole threshold
    small = rng.normal(size=(50, 8)).astype(np.float32)
    path = tmp_path / "srv" / "model.safetensors"
    os.makedirs(path.parent)
    write_safetensors(str(path), [("big.weight", "F32", big.shape), ("small.weight", "F32", small.shape)],
                      {"big.weight": big.tobytes(), "small.weight": small.tobytes()}.__getitem__)
    served = {"bytes": 0, "requests": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            a, b = self.headers["Range"].split("=")[1].split("-")
            with open(path, "rb") as f:
                f.seek(int(a))
                data = f.read(int(b) - int(a) + 1)
            served["bytes"] += len(data)
            served["requests"] += 1
            self.send_response(206)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/model.safetensors"
        st = Store([url], str(tmp_path / "cache"))
        assert st.remote and st.shape("big.weight") == (4000, 4096) and served["bytes"] < 2000      # opening = the header
        row = st.read("big.weight", norm_index((1234,), big.shape))
        assert torch.equal(row, torch.from_numpy(big[1234])) and served["bytes"] < 20_000            # one row = 16 KB
        blk = st.read("big.weight", norm_index((slice(10, 20), slice(5, 9)), big.shape))
        assert torch.equal(blk, torch.from_numpy(big[10:20, 5:9])) and not st.local("big.weight")
        assert torch.equal(st.read("small.weight"), torch.from_numpy(small)) and st.local("small.weight")
        before = served["bytes"]
        st.read("small.weight")
        assert served["bytes"] == before                                                             # cached on disk
        st.fetch("big.weight")
        assert st.local("big.weight")
        before = served["bytes"]
        assert torch.equal(st.read("big.weight", norm_index((slice(None), 7), big.shape)), torch.from_numpy(big[:, 7]))
        assert served["bytes"] == before
        again = Store([url], str(tmp_path / "cache"))                                                # reopening is offline
        assert served["bytes"] == before and again.total() == big.size + small.size
    finally:
        srv.shutdown()
