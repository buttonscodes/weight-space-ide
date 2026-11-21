"""The HTTP API end to end, on one of this project's own models and on a native Hugging Face layout."""
import os
import sys
from urllib.parse import parse_qsl, urlsplit

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from wsp import zoo  # noqa: E402
from wsp.server import create_app  # noqa: E402
from wsp.session import Session  # noqa: E402


def client(sess):
    return TestClient(create_app(sess, "http://127.0.0.1:0"), raise_server_exceptions=False)


def ok(r):
    assert r.status_code == 200, r.text[:400]
    return r.json() if "json" in r.headers.get("content-type", "") else r.text


def scoped(c):
    status = ok(c.get("/api/status"))
    return {"workspace_id": status["workspace_id"], "version": status["version"]}


def scoped_get(c, path):
    # TestClient's explicit params replace the URL query, so preserve both sets.
    return c.get(path, params={**dict(parse_qsl(urlsplit(path).query)), **scoped(c)})


def test_api_on_a_bundled_model(tmp_path):
    zoo.build_blank_modadd(str(tmp_path / "blank-modadd"))
    sess = Session(str(tmp_path))
    sess.open("blank-modadd")
    c = client(sess)
    st = ok(c.get("/api/state"))
    assert st["open"]["name"] == "blank-modadd" and st["status"]["tests"] is None and st["status"]["can_run"]
    assert [k["addr"] for k in ok(c.get("/api/tree?addr=model"))["children"]][:2] == ["embed", "pos_embed"]
    r = ok(c.post("/api/edit", json={"script": open(os.path.join(ROOT, "examples", "modadd.wscript")).read(), "message": "by hand"}))
    assert r["committed"] and r["tests"]["after"][0] == 49 and r["status"]["tests"] is None
    assert ok(c.post("/api/run", json={"prompt": "3 + 5 =", "max_new_tokens": 1}))["current"]["next"][0][0] == "1"
    node = ok(scoped_get(c, "/api/node?addr=L0N10&analysis=1"))
    assert node["kind"] == "unit" and node["neuron"]["mechanism"] and node["stats"]["changed"] > 0
    m = ok(scoped_get(c, "/api/matrix?addr=blocks.0.mlp.W_in&variant=delta"))
    assert m["shape"] == [32, 49]
    assert ok(c.get("/api/cell?addr=blocks.0.mlp.W_in&r=13&c=0"))["current"] == 1.0
    assert ok(scoped_get(c, "/api/wiring"))["n_edges"] > 0
    assert len(ok(c.post("/api/trace", json={"prompt": "3 + 5 ="}))["dla"]) > 2
    plan = ok(c.post("/api/plan", json={"selection": ["model"], "budget": 4000, "format": "q"}))
    assert "format q" in ok(c.get(f"/api/chunk?id={plan['chunks'][-1]['id']}"))
    pid = ok(c.get("/api/patches"))["patches"][0]["id"]
    assert ok(c.post("/api/patch", json={"id": pid, "action": "set", "enabled": False}))["status"]["tests"] is None
    assert ok(c.post("/api/patch", json={"id": pid, "action": "set", "enabled": True}))["status"]["tests"] is None
    assert c.get("/api/export").status_code == 200
    assert "weight script" in ok(c.get("/api/brief")).lower()
    tools = ok(c.get("/api/agent/tools"))["tools"]
    out = ok(c.post("/api/agent/call", json={"name": "overview", "arguments": {}}))
    assert len(tools) == 17 and "blank-modadd" in out["text"] and not out["isError"]
    rpc = ok(c.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "run_tests", "arguments": {}}}))
    assert "49/49" in rpc["result"]["content"][0]["text"]
    assert c.post("/api/edit", json={"script": "import os"}).json()["ok"] is False


def test_api_on_a_native_checkpoint(tmp_path):
    pytest.importorskip("transformers")
    from test_hf import PROMPT, build
    build("llama", str(tmp_path / "ckpt"))
    sess = Session(str(tmp_path / "ws"))
    os.makedirs(sess.root)
    sess.open(str(tmp_path / "ckpt"))                               # a local directory as the model name
    ws = sess.ws
    ws.write_json("tests/t.json", {"name": "t", "cases": [{"type": "preserve", "prompt": PROMPT, "max_kl": 10.0}]})
    sess.set_task({"scope": ["blocks.*.mlp", "blocks.*.attn"], "suite": "t"})
    c = client(sess)
    st = ok(c.get("/api/state"))
    assert st["open"]["recognised"] and st["open"]["family"] == "llama" and st["open"]["config"]["n_kv_heads"] == 2
    assert st["status"]["can_run"] and st["status"]["runner_loaded"] is False and st["status"]["tests"] is None
    assert ws._runner is None                                        # drawing the page did not load the model
    kids = ok(c.get("/api/tree?addr=blocks.1.attn"))["children"]
    assert [k["label"] for k in kids[:4]] == ["head 0", "head 1", "head 2", "head 3"] and kids[0]["thumb"].endswith("o_proj.weight[:, 0:8]")
    assert ok(scoped_get(c, "/api/node?addr=L1H2&analysis=1"))["head"]["kv_group"] == 1
    assert len(ok(scoped_get(c, "/api/node?addr=L0N3&analysis=1"))["neuron"]["reads"]) == 2
    assert ok(scoped_get(c, "/api/matrix?addr=model.layers.0.mlp.down_proj.weight"))["shape"] == [32, 48]
    assert ok(scoped_get(c, "/api/svd?addr=model.layers.0.mlp.down_proj.weight"))["rank"] == 32
    assert ok(scoped_get(c, "/api/wiring"))["window"] == [0, 1]
    assert ws._runner is None                                        # ... nor did any weight view
    r = ok(c.post("/api/edit", json={"script": "head(1, 2).O *= 0\nrank1(mlp(0).W_out, onehot(5, d_mlp), ones(d_model))",
                                     "message": "two edits", "enforce_scope": True}))
    assert r["committed"] and r["tests"]["after"] == [1, 1] and ws._runner is not None      # testing the edit did
    assert r["n_changed"] == 32 * 8 + 32 and r["stored"] <= 32 * 8 + 33                      # one head's columns + one neuron's row
    st2 = ok(c.get("/api/status"))
    assert st2["runner_loaded"] and st2["tests"] is None and st2["changed"] == 32 * 8 + 32
    run = ok(c.post("/api/run", json={"prompt": PROMPT, "max_new_tokens": 3}))
    assert run["base"]["next"] != run["current"]["next"]
    chat = ok(c.post("/api/chat", json={"messages": [{"role": "user", "content": "t1 t2"}], "template": {"user": "t3 ", "assistant": "t4 "},
                                        "max_new_tokens": 4}))
    assert chat["chat_template"] is False and isinstance(chat["current"], str)
    tr = ok(c.post("/api/trace", json={"prompt": PROMPT}))
    assert len([d for d in tr["dla"] if d["kind"] == "head"]) == 8
    pt = ok(c.post("/api/patching", json={"clean": PROMPT, "corrupt": PROMPT.replace("t9", "t8")}))
    assert pt["per_position"] and len(pt["heads"]) == 2
    out = ok(c.post("/api/agent/call", json={"name": "overview", "arguments": {}}))["text"]
    assert "self_attn.q_proj.weight" in out and "family llama" in out
    out = ok(c.post("/api/agent/call", json={"name": "read_weights", "arguments": {"address": "L1N5", "format": "g3"}}))["text"]
    assert 'P("model.layers.1.mlp.down_proj.weight")[:, 5]' in out and "gate read" in out
    exp = ok(c.post("/api/export_checkpoint", json={"name": "x"}))
    import transformers
    again = transformers.AutoModelForCausalLM.from_pretrained(exp["dir"]).eval()
    with torch.no_grad():
        assert torch.allclose(again(torch.tensor([ws.encode(PROMPT)])).logits[0], ws.logits(PROMPT), atol=1e-4)


def test_a_model_too_large_to_run_is_still_editable(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    from test_hf import build
    from wsp import runner
    build("gpt2", str(tmp_path / "ckpt"))
    sess = Session(str(tmp_path / "ws"))
    os.makedirs(sess.root)
    sess.open(str(tmp_path / "ckpt"))
    monkeypatch.setattr(runner, "available_memory", lambda: 1000)   # pretend there is 1 kB of RAM left
    c = client(sess)
    assert ok(c.get("/api/state"))["status"]["can_run"]             # not known yet: nothing has tried to load it
    r = ok(c.post("/api/edit", json={"script": "neuron(0, 3).w_out *= 0", "message": "ablate"}))
    assert r["committed"] and "GB" in r["not_run"] and "tests" not in r
    st = ok(c.get("/api/status"))
    assert st["can_run"] is False and "memory" in st["run_error"] and st["changed"] == 32
    bad = c.post("/api/run", json={"prompt": "t1 t2"})
    assert bad.status_code == 400 and "memory" in bad.json()["error"]
    name = sess.ws.struct.layers[0].mlp["W_out"].name                 # GPT-2's mlp.c_proj, stored [d_mlp, d_model]
    assert ok(scoped_get(c, f"/api/matrix?addr={name}&variant=delta"))["shape"] == [128, 32]
    assert float(sess.ws.read(name, (3,)).abs().max()) == 0
