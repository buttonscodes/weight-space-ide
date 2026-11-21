"""How the Directions feature turns a prompt into token ids (docs/DIRECTIONS.md, "How a prompt becomes token ids").

A tiny random Qwen2 checkpoint is built in the test's own directory, with a byte-level tokenizer laid out as Qwen2.5's
(its normalizer, pre-tokenizer, post-processor and special tokens, a byte vocabulary without merges) and Qwen2.5's own
chat template. Expected ids come from references written here: ``before`` is what the IDE rendered before it rendered
as Heretic does (the template rendered by transformers, the prefix appended, the text tokenized without the tokenizer's
own special tokens); ``heretic`` is Heretic's Model.generate written out. Nothing here calls the renderer to make an
expected value. Section 1 (the Qwen-like tokenizer, where none of the differences applies) was written and passing
before the renderer changed; it pins that the ids there are bit-identical to the IDE's before. Sections 2 and 3 are
the fixtures where a difference applies (a tokenizer that adds a beginning-of-text token, a template that prints the
date, a template that raises on or drops a system message), section 4 the rule for records made before.
"""
import json
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")

from wsp import directions as DR  # noqa: E402
from wsp import zoo  # noqa: E402
from wsp.session import Session  # noqa: E402

# Qwen2.5's chat template (tiny-random/qwen2.5 at 7a6a312, chat_template.jinja), whole.
QWEN25_TEMPLATE = (
    "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n"
    "        {{- messages[0]['content'] }}\n    {%- else %}\n"
    "        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n"
    "    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided "
    "with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n"
    "        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n"
    "    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": "
    "<args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n"
    "        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n"
    "        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n"
    "    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n"
    "    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == "
    "\"assistant\" and not message.tool_calls) %}\n"
    "        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n"
    "    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n"
    "        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n"
    "        {{- '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n")
QWEN_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
              r"\s+(?!\S)|\s+")
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
TEXTS = ["Hi", "Write a haiku about the sea.", "  leading spaces\nand a newline", "Ünïcödé — and digits 12345",
         "<|im_start|> is text here"]
OWN = [{"system": "You answer in French.", "user": "Hello there"}, {"system": "", "user": "Empty system"}]


def byte_vocab():
    from tokenizers.pre_tokenizers import ByteLevel
    return {c: i for i, c in enumerate(sorted(ByteLevel.alphabet()))}


def qwen_like_tokenizer(directory, *, post_processor=None, bos=None):
    """A byte-level tokenizer laid out as Qwen2.5's; ``post_processor`` replaces Qwen's (ByteLevel, adds nothing)."""
    from tokenizers import AddedToken, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
    vocab = byte_vocab()
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[], ignore_merges=False))
    tk.normalizer = normalizers.NFC()
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(__import__("tokenizers").Regex(QWEN_SPLIT), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    tk.post_processor = post_processor or processors.ByteLevel(trim_offsets=False)
    tk.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in SPECIALS + ([bos] if bos else [])])
    tk.save(os.path.join(directory, "tokenizer.json"))
    cfg = {"add_prefix_space": False, "backend": "tokenizers", "bos_token": bos, "clean_up_tokenization_spaces": False,
           "eos_token": "<|im_end|>", "errors": "replace", "extra_special_tokens": ["<|im_start|>", "<|im_end|>"],
           "model_max_length": 131072, "pad_token": "<|endoftext|>", "split_special_tokens": False,
           "tokenizer_class": "Qwen2Tokenizer", "unk_token": None}
    with open(os.path.join(directory, "tokenizer_config.json"), "w") as f:
        json.dump(cfg, f)
    return len(vocab) + len(SPECIALS) + (1 if bos else 0)


def build(directory, *, template=QWEN25_TEMPLATE, **tok):
    os.makedirs(directory, exist_ok=True)
    n_vocab = qwen_like_tokenizer(directory, **tok)
    with open(os.path.join(directory, "chat_template.jinja"), "w") as f:
        f.write(template)
    torch.manual_seed(0)
    m = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(
        vocab_size=n_vocab, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=1024, tie_word_embeddings=False))
    m.save_pretrained(directory, safe_serialization=True)
    return directory


def open_ws(tmp_path, ckpt, name="q"):
    s = Session(str(tmp_path / "ws"))
    os.makedirs(s.root, exist_ok=True)
    s.ws = zoo.open_hf(s.root, ckpt, name=name)
    return s


def before(ckpt, text, system=None, prefix=None):
    """The IDE's rendering before it rendered as Heretic does: the messages as given, the template with its own date,
    the prefix appended, tokenized without the tokenizer's own special tokens."""
    from tokenizers import Tokenizer
    tok = transformers.AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    user, system = (text, system) if isinstance(text, str) else (text["user"], text["system"])
    messages = ([{"role": "system", "content": system}] if system is not None else []) + [{"role": "user", "content": user}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + (prefix or "")
    return Tokenizer.from_file(os.path.join(ckpt, "tokenizer.json")).encode(rendered, add_special_tokens=False).ids


@pytest.fixture(scope="module")
def qwen(tmp_path_factory):
    root = tmp_path_factory.mktemp("qwen")
    ckpt = build(str(root / "ckpt"))
    return ckpt, open_ws(root, ckpt)


# ------------------------------------------------------------------ 1. a model where none of the differences applies
@pytest.mark.parametrize("system", [None, "You are a helpful assistant.", ""])
@pytest.mark.parametrize("prefix", [None, "Sure,", "<think>\n\n</think>\n\n"])
def test_on_a_qwen_tokenizer_the_ids_are_those_the_ide_rendered_before(qwen, system, prefix):
    ckpt, s = qwen
    got = DR.render_ids(s.ws, TEXTS, "good", chat=True, system=system, prefix=prefix, chat_options={})
    assert got == [before(ckpt, t, system, prefix) for t in TEXTS]
    assert got[0][:2] == [257, byte_vocab()["s"]]          # <|im_start|> (256 bytes, then <|endoftext|>), "s": no BOS


def test_on_a_qwen_tokenizer_prompts_with_their_own_system_message_render_as_before(qwen):
    ckpt, s = qwen
    got = DR.render_ids(s.ws, OWN + ["plain"], "good", chat=True, system="global", prefix="Sure,", chat_options={})
    assert got == [before(ckpt, OWN[0], prefix="Sure,"), before(ckpt, OWN[1], prefix="Sure,"),
                   before(ckpt, "plain", "global", "Sure,")]


def test_on_a_qwen_tokenizer_a_measurement_runs_on_the_ids_rendered_before(qwen, monkeypatch):
    ckpt, s = qwen
    seen = []
    real = DR.residuals

    def keep(w, prompts_ids, variant="base", module_io=False):
        seen.extend([list(x) for x in prompts_ids])
        return real(w, prompts_ids, variant, module_io)
    monkeypatch.setattr(DR, "residuals", keep)
    DR.measure(s.ws, "pin", TEXTS[:3], TEXTS[3:], chat=True, system="Be brief.", prefix="Sure,", replace=True)
    assert seen == [before(ckpt, t, "Be brief.", "Sure,") for t in TEXTS]


# ------------------------------------------------------------------ 2. Heretic's rendering, written out
BOS = "<|begin_of_text|>"
DATED = QWEN25_TEMPLATE.replace("'<|im_start|>system\\n' + messages[0]['content']",
                                "'<|im_start|>system\\nToday Date: ' + strftime_now('%d %B %Y') + '\\n\\n' + messages[0]['content']")
RAISES = ("{%- for message in messages %}{%- if message['role'] == 'system' %}"
          "{{- raise_exception('System role not supported') }}{%- endif %}{%- endfor %}\n" + QWEN25_TEMPLATE)
DROPS = "{%- set messages = messages | rejectattr('role', 'equalto', 'system') | list %}\n" + QWEN25_TEMPLATE


def heretic(ckpt, text, system=None, prefix=None, template=None):
    """Heretic's Model.generate for one prompt (heretic/model.py at 64e9991), written out: its system-role probe
    (a sentinel system message must survive; a template that raises has no system role), each system message folded
    into the next user turn when there is none (an empty one dropped), the template rendered with the generation
    prompt, the response prefix appended, the text tokenized by the tokenizer called with its defaults. ``template``
    replaces the checkpoint's for the rendering only (the date written in as a literal, as strftime_fixed_date
    prints it)."""
    tok = transformers.AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    user, system = (text, system) if isinstance(text, str) else (text["user"], text["system"])
    try:
        probe = tok.apply_chat_template([{"role": "system", "content": "HERETIC_SYSTEM_ROLE_PROBE"},
                                         {"role": "user", "content": "Hi"}], tokenize=False, add_generation_prompt=True)
        supported = "HERETIC_SYSTEM_ROLE_PROBE" in probe
    except Exception:
        supported = False
    messages = ([{"role": "system", "content": system}] if system is not None else []) + [{"role": "user", "content": user}]
    if not supported and system is not None:
        messages = [{"role": "user", "content": f"{system}\n\n{user}" if system else user}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       **({"chat_template": template} if template else {}))
    return tok(rendered + (prefix or ""))["input_ids"]


def fixture(tmp_path_factory, label, **kw):
    root = tmp_path_factory.mktemp(label)
    ckpt = build(str(root / "ckpt"), **kw)
    return ckpt, open_ws(root, ckpt)


@pytest.fixture(scope="module")
def bos(tmp_path_factory):
    from tokenizers import processors
    n = len(byte_vocab()) + len(SPECIALS)                    # the beginning-of-text token's id
    post = processors.TemplateProcessing(single=f"{BOS} $A", pair=f"{BOS} $A $B:1", special_tokens=[(BOS, n)])
    return fixture(tmp_path_factory, "bos", template="{{- bos_token }}\n" + QWEN25_TEMPLATE, post_processor=post, bos=BOS)


@pytest.fixture(scope="module")
def dated(tmp_path_factory):
    return fixture(tmp_path_factory, "dated", template=DATED)


@pytest.fixture(scope="module")
def raises(tmp_path_factory):
    return fixture(tmp_path_factory, "raises", template=RAISES)


@pytest.fixture(scope="module")
def drops(tmp_path_factory):
    return fixture(tmp_path_factory, "drops", template=DROPS)


def render(s, texts, system=None, prefix=None):
    return DR.render_ids(s.ws, texts, "good", chat=True, system=system, prefix=prefix, chat_options={})


def test_a_tokenizer_that_adds_a_beginning_of_text_token_gives_two_after_a_template_that_prints_one(bos):
    ckpt, s = bos
    n = len(byte_vocab()) + len(SPECIALS)
    for system in (None, "Be brief.", ""):
        got = render(s, TEXTS, system, "Sure,")
        assert got == [heretic(ckpt, t, system, "Sure,") for t in TEXTS]
        assert all(x[:3] == [n, n, 257] for x in got)                      # BOS, BOS, <|im_start|>
        assert all(b[:2] == [n, 257] for b in (before(ckpt, t, system, "Sure,") for t in TEXTS))   # the fixture bites


def test_a_template_that_prints_the_date_prints_heretics_fixed_date(dated):
    from datetime import datetime
    ckpt, s = dated
    literal = DATED.replace("strftime_now('%d %B %Y')", "'01 January 2026'")
    got = render(s, TEXTS + OWN, "Be brief.", "Sure,")
    assert got == [heretic(ckpt, t, "Be brief.", "Sure,", template=literal) for t in TEXTS + OWN]
    assert "Today Date: 01 January 2026\n\nBe brief." in s.ws.tok.decode(got[0])
    # The Chat tab's own rendering keeps today's date (and so did the IDE's directions before).
    days = {datetime.now().strftime("%d %B %Y")}
    chat = s.ws.runner.chat_prompt([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}])
    days.add(datetime.now().strftime("%d %B %Y"))
    assert any(f"Today Date: {d}\n" in chat for d in days)


def test_the_fixed_date_holds_only_in_the_rendering_thread(dated):
    import threading
    from datetime import datetime
    from wsp import directions_rendering as RD
    ckpt, s = dated
    tok = transformers.AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    msgs = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
    inside, leave, seen = threading.Event(), threading.Event(), {}

    def hold():
        with RD.fixed_date() as read:
            seen["inside"] = tok.apply_chat_template(msgs, tokenize=False)
            inside.set()
            leave.wait(30)
        seen["read"] = read["read"]
    t = threading.Thread(target=hold)
    t.start()
    assert inside.wait(30)
    other = tok.apply_chat_template(msgs, tokenize=False)                  # another thread, while the first is inside
    leave.set()
    t.join(30)
    assert "01 January 2026" in seen["inside"] and seen["read"] is True
    assert "01 January 2026" not in other and datetime.now().strftime("%Y") in other
    assert "01 January 2026" not in tok.apply_chat_template(msgs, tokenize=False)      # and afterwards in this one


@pytest.mark.parametrize("which", ["raises", "drops"])
def test_a_template_without_a_system_role_gets_the_system_message_folded_into_the_user_turn(which, request):
    ckpt, s = request.getfixturevalue(which)
    for system in ("Be brief.", "", None):
        got = render(s, TEXTS + OWN, system, "Sure,")
        assert got == [heretic(ckpt, t, system, "Sure,") for t in TEXTS + OWN], system
    folded = s.ws.tok.decode(render(s, ["Hi"], "Be brief.")[0])
    assert "<|im_start|>user\nBe brief.\n\nHi<|im_end|>" in folded
    assert render(s, ["Hi"], "") == render(s, ["Hi"], None)              # an empty system message is dropped
    if which == "raises":
        with pytest.raises(Exception, match="System role not supported"):
            before(ckpt, "Hi", "Be brief.")
    else:
        assert before(ckpt, "Hi", "Be brief.") != render(s, ["Hi"], "Be brief.")[0]


def test_heretics_system_role_rule_from_first_principles():
    from wsp import directions_rendering as RD
    s, u, a = ({"role": r, "content": c} for r, c in (("system", "S"), ("user", "U"), ("assistant", "A")))
    two = [s, u, a, {"role": "system", "content": "T"}, {"role": "user", "content": "V"}]
    assert RD.apply_system_role(two, True) is two
    assert RD.apply_system_role(two, False) == [{"role": "user", "content": "S\n\nU"}, a, {"role": "user", "content": "T\n\nV"}]
    assert RD.apply_system_role([{"role": "system", "content": ""}, u], False) == [u]
    assert RD.apply_system_role([s, {"role": "system", "content": "T"}, u], False) == [{"role": "user", "content": "S\n\nT\n\nU"}]
    assert RD.apply_system_role([u, s], False) == [u, {"role": "user", "content": "S"}]
    assert two[0] == s and s["content"] == "S"                              # the caller's messages are not changed


# ------------------------------------------------------------------ 3. every Directions path renders as Heretic does
def runner_ids(ws, monkeypatch):
    seen = []
    r = ws.runner
    for name in ("run", "generate", "first_token_logits"):
        original = getattr(r, name)

        def wrapped(ids, *a, _name=name, _original=original, **k):
            x = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            while x and isinstance(x[0], list):
                x = x[0]
            seen.append((_name, [int(t) for t in x]))
            return _original(ids, *a, **k)
        monkeypatch.setattr(r, name, wrapped)
    return seen


def test_measurement_module_capture_scores_and_prefix_detection_all_render_as_heretic(drops, monkeypatch):
    from wsp import directions_prefix as DP
    from wsp import directions_scores as DS
    ckpt, s = drops
    ws, system = s.ws, "Be brief."
    seen = runner_ids(ws, monkeypatch)
    want = [heretic(ckpt, t, system, "Sure,") for t in TEXTS]
    meta = DR.measure(ws, "all", TEXTS[:3], TEXTS[3:], chat=True, system=system, prefix="Sure,", module_io=True, replace=True)
    assert [x for n, x in seen if n == "run"] == want and meta["module_io"] is not None
    assert meta["rendering"] == {"version": 2, "same_as_version_1": False}
    seen.clear()
    body = {"state": "current", "chat": True, "system": system, "prefix": "Sure,", "max_response_length": 2,
            "scores": [{"plugin": "KLDivergence", "prompts": TEXTS[:2]}, {"plugin": "KeywordRate", "prompts": TEXTS[2:]}]}
    DS.score(ws, body)
    assert {tuple(x) for n, x in seen if n == "first_token_logits"} == {tuple(x) for x in want[:2]}
    assert [x for n, x in seen if n == "generate"] == want[2:]
    seen.clear()
    DP.detect(ws, {"prompts": TEXTS, "system": system, "max_response_length": 2, "chain_of_thought_blocks": []})
    assert [x for n, x in seen if n == "generate"][:len(TEXTS)] == [heretic(ckpt, t, system) for t in TEXTS]


def test_a_measurement_records_whether_the_ide_rendered_it_the_same_before(qwen, bos, dated):
    for (ckpt, s), same in ((qwen, True), (bos, False), (dated, False)):
        meta = DR.measure(s.ws, "r", TEXTS[:2], TEXTS[2:], chat=True, system="Be brief.", replace=True)
        assert meta["rendering"] == {"version": 2, "same_as_version_1": same}, ckpt
        assert DR._meta(s.ws, "r")["rendering"] == meta["rendering"]
    meta = DR.measure(qwen[1].ws, "plain", ["abc"], ["xyz"], replace=True)           # not a chat: encoded as before
    assert meta["rendering"] == {"version": 2, "same_as_version_1": True}
    # The dated template prints no date without a system message, so those prompts render as before.
    assert DR.measure(dated[1].ws, "nosys", TEXTS[:2], TEXTS[2:], chat=True, replace=True)["rendering"] == \
        {"version": 2, "same_as_version_1": True}


def test_a_template_that_reads_the_date_never_renders_as_before_even_on_the_fixed_day(dated, monkeypatch):
    """On the fixed day itself both versions print the same date, yet version 1 would print another on any other day."""
    from datetime import datetime
    from wsp import directions_rendering as RD
    ckpt, s = dated
    monkeypatch.setattr(RD, "FIXED_DATE", datetime.now().astimezone())     # local, as version 1's clock reads it
    texts = ["Hi", "Write a haiku."]
    assert render(s, texts, "Be brief.") == [before(ckpt, t, "Be brief.") for t in texts]      # today: the same ids
    meta = DR.measure(s.ws, "today", texts[:1], texts[1:], chat=True, system="Be brief.", replace=True)
    assert meta["rendering"] == {"version": 2, "same_as_version_1": False}


# ------------------------------------------------------------------ 4. records made before version 2
def test_which_renderings_are_compatible():
    from wsp import directions_rendering as RD
    v1, same, other = None, {"version": 2, "same_as_version_1": True}, {"version": 2, "same_as_version_1": False}
    assert RD.compatible(v1, v1) and RD.compatible(same, same) and RD.compatible(other, other) and RD.compatible(same, other)
    assert RD.compatible(v1, same) and RD.compatible(same, v1)
    assert not RD.compatible(v1, other) and not RD.compatible(other, v1)
    assert not RD.compatible(v1, {"version": 2}) and not RD.compatible({"version": 3, "same_as_version_1": True}, v1)
