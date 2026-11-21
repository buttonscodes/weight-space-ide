"""Probes read "as a chat message": the text as one user turn, rendered by the runner's chat_prompt (the same
call /api/chat makes, generation prompt included), on a tiny random in-process model with a stub template.

The stub template wraps the text in letters the character vocabulary has ("zz" before, "yy" after), so the
rendered chat can be probed directly too and the two compared exactly.  No checkpoint is loaded.
"""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import wsp.runner
from wsp import probes as P
from wsp.model import Config, init_params
from wsp.server import create_app
from wsp.session import Session
from wsp.tokenizer import VocabTokenizer
from wsp.workspace import Workspace

CFG = dict(d_vocab=26, n_ctx=16, d_model=24, n_layers=2, n_heads=2, d_mlp=40, name="rnd")


@pytest.fixture()
def sess(tmp_path):
    cfg = Config(**CFG)
    Workspace.create(str(tmp_path / "rnd"), cfg, VocabTokenizer(list("abcdefghijklmnopqrstuvwxyz"), "char"),
                     init_params(cfg, seed=1, std=0.3))
    s = Session(str(tmp_path))
    s.open("rnd")
    return s


def client(sess):
    return TestClient(create_app(sess, "http://127.0.0.1:0"), raise_server_exceptions=False)


@pytest.fixture()
def template(monkeypatch):
    """A model with a chat template: every call to chat_prompt is recorded with its options."""
    calls = []

    def chat_prompt(self, messages, **options):
        calls.append((messages, options))
        return "zz" + messages[-1]["content"] + "yy"
    monkeypatch.setattr(P, "chat_template", lambda ws: (True, "the model's own chat template"))
    monkeypatch.setattr(wsp.runner.Runner, "chat_prompt", chat_prompt)
    return calls


def test_a_chat_template_is_found_from_the_model_files_alone(tmp_path):
    def hf(files):
        d = tmp_path / str(len(list(tmp_path.iterdir())))
        d.mkdir()
        for name, text in files.items():
            (d / name).write_text(text)
        return SimpleNamespace(kind="hf", model_dir=str(d))
    assert P.chat_template(hf({"tokenizer_config.json": json.dumps({"chat_template": "{{ messages }}"})}))[0]
    assert P.chat_template(hf({"chat_template.jinja": "{{ messages }}"}))[0]
    assert P.chat_template(hf({"tokenizer_config.json": json.dumps({"bos_token": "<s>"})})) == (False, "this model ships no chat template")
    assert P.chat_template(hf({"tokenizer_config.json": "{not json"}))[0] is False
    assert P.chat_template(SimpleNamespace(kind="hf", model_dir=str(tmp_path / "missing")))[0] is False
    assert P.chat_template(SimpleNamespace(kind="wsp"))[0] is False
    # a GGUF is read as a chat message with its own template (saved into the workspace's model/ when it was opened)
    assert P.chat_template(SimpleNamespace(kind="gguf")) == (False, "this GGUF carries no chat template (tokenizer.chat_template)")
    gguf = hf({"chat_template.jinja": "{{ messages }}"})
    gguf.kind = "gguf"
    assert P.chat_template(gguf) == (True, "the GGUF's own chat template")


def test_the_template_mask_marks_the_tokens_around_the_text_or_says_it_cannot():
    chars = list
    assert P.template_mask(chars, "zzabcyy", "abc", chars("zzabcyy")) == [True, True, False, False, False, True, True]
    assert P.template_mask(chars, "abc", "abc", chars("abc")) == [False, False, False]
    # A tokenizer that joins "cy" into one token: that token is both the text's and the template's.
    def joined(s):
        out, i = [], 0
        while i < len(s):
            out.append(s[i:i + 2] if s[i:i + 2] == "cy" else s[i])
            i += len(out[-1])
        return out
    assert P.template_mask(joined, "zzabcyy", "abc", joined("zzabcyy")) is None
    assert P.template_mask(chars, "zzyy", "abc", chars("zzyy")) is None      # the text is not in the rendering


def test_each_probe_reads_the_rendered_chat_exactly_as_that_text(sess, template):
    c = client(sess)
    mask = [True, True, False, False, False, True, True]
    opts = {"enable_thinking": False}

    reads = c.post("/api/probe/reads", json={"prompt": "abc", "neurons": ["L1N3", "L0N2"], "variants": ["base", "current"],
                                            "chat": True, "chat_options": opts})
    assert reads.status_code == 200, reads.text
    rb = reads.json()
    plain = c.post("/api/probe/reads", json={"prompt": "zzabcyy", "neurons": ["L1N3", "L0N2"], "variants": ["base", "current"]}).json()
    assert rb["token_strs"] == list("zzabcyy") and rb["chat"] == {"template": mask}
    assert rb["values"] == plain["values"] and "chat" not in plain
    assert template[0] == ([{"role": "user", "content": "abc"}], opts)     # one user turn; the options as given

    # Fires: the position counts tokens of the rendering, so it can be past the end of the text itself.
    fires = c.post("/api/probe/fires", json={"prompt": "abc", "k": 4, "position": 6, "chat": True})
    assert fires.status_code == 200, fires.text
    fb = fires.json()
    assert fb["position"] == 6 and fb["token_str"] == "y" and fb["chat"] == {"template": mask}
    assert fb["neurons"] == c.post("/api/probe/fires", json={"prompt": "zzabcyy", "k": 4, "position": 6}).json()["neurons"]
    bad = c.post("/api/probe/fires", json={"prompt": "abc", "position": 7, "chat": True})
    assert bad.status_code == 400 and "position must be" in bad.json()["error"]

    spec = [{"addr": "L1N3", "mode": "add", "dose": 0.5, "unit": "raw"}]
    nxt = c.post("/api/probe/next", json={"prompt": "abc", "steer": spec, "top": 3, "chat": True})
    assert nxt.status_code == 200, nxt.text
    plain = c.post("/api/probe/next", json={"prompt": "zzabcyy", "steer": spec, "top": 3}).json()
    assert nxt.json()["kl"] == plain["kl"] and nxt.json()["top_steered"] == plain["top_steered"]
    assert nxt.json()["chat"] == {"template": mask}

    con = c.post("/api/probe/contrast", json={"a": ["abc", "ab"], "b": ["zzz"], "k": 2, "chat": True})
    assert con.status_code == 200, con.text
    plain = c.post("/api/probe/contrast", json={"a": ["zzabcyy", "zzabyy"], "b": ["zzzzzyy"], "k": 2}).json()
    assert con.json()["up"] == plain["up"] and con.json()["down"] == plain["down"] and con.json()["chat"] == {"rendered": True}
    assert sess.ws.patches == []


def test_the_token_bound_counts_the_rendered_chat(sess, template, monkeypatch):
    c = client(sess)
    monkeypatch.setattr(wsp.runner.Runner, "chat_prompt", lambda self, m, **o: "z" * 300 + m[-1]["content"])
    r = c.post("/api/probe/fires", json={"prompt": "abc", "chat": True})
    assert r.status_code == 400 and "as a chat message has 303 tokens; at most 256" in r.json()["error"]
    # A rendering whose boundary cannot be told: the result says so (no mask) and is still the rendered text's.
    monkeypatch.setattr(wsp.runner.Runner, "chat_prompt", lambda self, m, **o: "zz" + m[-1]["content"][::-1])
    r = c.post("/api/probe/fires", json={"prompt": "abc", "chat": True}).json()
    assert r["chat"] == {"template": None} and r["token_strs"] == list("zzcba")
    # A runner that renders no template refuses even when the files said there was one.
    monkeypatch.setattr(wsp.runner.Runner, "chat_prompt", lambda self, m, **o: None)
    r = c.post("/api/probe/next", json={"prompt": "abc", "chat": True})
    assert r.status_code == 400 and "needs the model's own chat template: this model has none" in r.json()["error"]


def test_the_chat_option_is_refused_before_any_model_is_loaded(sess, monkeypatch):
    built = []

    def refuse(w):
        built.append(w.name)
        raise AssertionError("the runner was built for a request that should have been refused first")
    monkeypatch.setattr(wsp.runner, "make_runner", refuse)
    c = client(sess)
    assert c.get("/api/probe/chat").json() == {"available": False, "reason": "this built-in model has no chat template"}
    reads = {"prompt": "abc", "neurons": ["L0N1"]}
    for path, body, message in [
        ("/api/probe/reads", {**reads, "chat": True}, "needs the model's own chat template: this built-in model has no chat template"),
        ("/api/probe/fires", {"prompt": "abc", "chat": "yes"}, "chat must be true or false"),
        ("/api/probe/next", {"prompt": "abc", "chat": True, "chat_options": {"temperature": 1}}, "Unsupported chat template options"),
        ("/api/probe/contrast", {"a": ["abc"], "b": ["z"], "chat": True, "chat_options": {"enable_thinking": "no"}}, "Thinking switches must be booleans"),
    ]:
        r = c.post(path, json=body)
        assert r.status_code == 400 and message in r.json()["error"], (path, body, r.text)
    monkeypatch.setattr(P, "chat_template", lambda ws: (True, "stub"))
    r = c.get("/api/probe/chat").json()
    assert r == {"available": True, "reason": "stub"}
    assert built == [] and sess.ws._runner is None


def test_a_bad_position_as_a_chat_message_is_refused_before_any_model_is_loaded(sess, template, monkeypatch):
    """With `chat` the position counts tokens of the rendering, so its range waits for the rendering; its type and
    the token bound do not. The other three probes check every field before the model as well."""
    built = []

    def refuse(w):
        built.append(w.name)
        raise AssertionError("the runner was built for a request that should have been refused first")
    monkeypatch.setattr(wsp.runner, "make_runner", refuse)
    c = client(sess)
    for chat in (False, True):
        for position, message in [("abc", "position must be an integer"), (True, "position must be an integer"),
                                  (1.5, "position must be an integer"), (256, "position must be an integer from -")]:
            r = c.post("/api/probe/fires", json={"prompt": "abc", "position": position, "chat": chat})
            assert r.status_code == 400 and message in r.json()["error"], (chat, position, r.text)
    # As a chat message the bound before rendering is the token limit: -256..255.
    r = c.post("/api/probe/fires", json={"prompt": "abc", "position": -257, "chat": True})
    assert r.status_code == 400 and "position must be an integer from -256 to 255" in r.json()["error"]
    for path, body, message in [
        ("/api/probe/fires", {"prompt": "abc", "k": "many", "chat": True}, "k must be an integer"),
        ("/api/probe/fires", {"prompt": "abc", "sign": "up", "chat": True}, "sign must be"),
        ("/api/probe/reads", {"prompt": "abc", "neurons": ["L0N1"], "variants": ["edited"], "chat": True}, "variants must list"),
        ("/api/probe/next", {"prompt": "abc", "top": 0, "chat": True}, "top must be an integer"),
        ("/api/probe/contrast", {"a": ["abc"], "b": ["z"], "k": 0, "chat": True}, "k must be an integer"),
    ]:
        r = c.post(path, json=body)
        assert r.status_code == 400 and message in r.json()["error"], (path, body, r.text)
    assert built == [] and sess.ws._runner is None
