"""Prompt sets from dataset specifications with per-set system messages, on a real model, against Heretic.

The IDE's own measurement (good and bad, with module inputs captured), scores (KL divergence and refusals) and
response-prefix detection are run from the specifications of a Heretic configuration, and the token ids each prompt
reaches the model with are compared with the ids Heretic generated from (written by scripts/heretic_prompt_ids.py).
Everything goes through the IDE's public functions (directions.measure, directions_scores.score,
directions_prefix.detect); the ids are recorded at the runner. The run's response prefix (heretic ids.json
``response_prefix``, empty when the configuration sets none) is passed to the measurement and the scores; the
prefix detection renders its test prompts without one, as Heretic does. A set whose IDE call fails (a template that
refuses the system role, for example) counts as different and records the error.

Acceptance, written before the run: for every prompt of every set, the same ids in the same order.

    python scripts/heretic_compare_prompt_ids.py <model snapshot dir> <isolated IDE root> <heretic ids.json> <out.json>

It loads the model: run it under python -m wsp.memory_guard, offline, with the Hugging Face cache the Heretic run
used, in an isolated root. Nothing is committed (the answer records that no tensor is touched afterwards).

Fixtures for the rendering differences (docs/DIRECTIONS.md, "How a prompt becomes token ids"), model-free:

    python scripts/heretic_compare_prompt_ids.py make-fixture <kind> <tiny-random/qwen2.5 snapshot> <fresh dir>

copies the snapshot changing only its tokenizer and template files. ``kind``:
    none            nothing changed (the control)
    bos             the tokenizer adds a beginning-of-text token <|begin_of_text|> (a new special token, id 151665,
                    below the model's vocabulary size) and the template prints it (bos_token) first
    date            the template prints the date through strftime_now in its system turn
    system-raises   the template raises on a system message (as Gemma 2's)
    system-drops    the template silently drops system messages (and prints its default system turn)
Bound for the fixtures, written before the first run: ids identical for every prompt of every prompt set, after the
change; before it, each altered fixture must show at least one set that differs (the fixture bites).
"""
import json
import os
import shutil
import sys
from importlib.metadata import version
from pathlib import Path
sys.dont_write_bytecode = True

BOS = "<|begin_of_text|>"
BOS_ID = 151665
SYSTEM_TURN = "'<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n'"
DATED_TURN = "'<|im_start|>system\\nToday Date: ' + strftime_now('%d %B %Y') + '\\n\\n' + messages[0]['content'] + '<|im_end|>\\n'"
RAISES = ("{%- for message in messages %}{%- if message['role'] == 'system' %}"
          "{{- raise_exception('System role not supported') }}{%- endif %}{%- endfor %}\n")
DROPS = "{%- set messages = messages | rejectattr('role', 'equalto', 'system') | list %}\n"
KINDS = ("none", "bos", "date", "system-raises", "system-drops")


def make_fixture(kind, src, out):
    src, out = Path(src), Path(out)
    if kind not in KINDS:
        sys.exit(f"kind must be one of {', '.join(KINDS)}")
    out.mkdir(parents=True, exist_ok=False)
    for f in src.iterdir():
        shutil.copyfile(f, out / f.name)                  # copies, never links: the cache stays untouched
    template = (out / "chat_template.jinja").read_text()
    if kind == "bos":
        tj = json.loads((out / "tokenizer.json").read_text())
        assert all(t["id"] != BOS_ID and t["content"] != BOS for t in tj["added_tokens"])
        tj["added_tokens"].append({"id": BOS_ID, "content": BOS, "single_word": False, "lstrip": False, "rstrip": False,
                                   "normalized": False, "special": True})
        seq = {"Sequence": {"id": "A", "type_id": 0}}
        tj["post_processor"] = {"type": "TemplateProcessing",
                                "single": [{"SpecialToken": {"id": BOS, "type_id": 0}}, seq],
                                "pair": [{"SpecialToken": {"id": BOS, "type_id": 0}}, seq, {"Sequence": {"id": "B", "type_id": 1}}],
                                "special_tokens": {BOS: {"id": BOS, "ids": [BOS_ID], "tokens": [BOS]}}}
        (out / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False))
        tc = json.loads((out / "tokenizer_config.json").read_text())
        tc["bos_token"] = BOS
        (out / "tokenizer_config.json").write_text(json.dumps(tc, indent=2))
        template = "{{- bos_token }}\n" + template
    elif kind == "date":
        assert template.count(SYSTEM_TURN) == 1, "the snapshot's template is not Qwen2.5's"
        template = template.replace(SYSTEM_TURN, DATED_TURN)
    elif kind == "system-raises":
        template = RAISES + template
    elif kind == "system-drops":
        template = DROPS + template
    (out / "chat_template.jinja").write_text(template)
    print("fixture", kind, "->", out)


def compare(snap, root, src, out):
    sys.path.insert(0, os.getcwd())
    import torch
    from wsp import directions as DR
    from wsp import directions_prefix as DP
    from wsp import directions_scores as DS
    from wsp.session import Session

    d = json.loads(src.read_text())
    system, S, prefix = d["system_prompt"], d["specifications"], d.get("response_prefix") or None
    session = Session(str(root))
    ws = session.open(str(snap), log=lambda *a, **k: None)
    r = ws.runner
    seen = []

    def flat(ids):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        while ids and isinstance(ids[0], list):
            assert len(ids) == 1, "one prompt per call"
            ids = ids[0]
        return [int(t) for t in ids]

    def spy(name):
        original = getattr(r, name)

        def wrapped(ids, *a, **k):
            seen.append((name, flat(ids)))
            return original(ids, *a, **k)
        setattr(r, name, wrapped)

    for name in ("run", "generate", "first_token_logits"):
        spy(name)

    def phase(fn):
        seen.clear()
        try:
            value, error = fn(), None
        except Exception as e:                            # the set differs; the error is the finding
            value, error = None, f"{type(e).__name__}: {e}"
        return value, list(seen), error

    def unique(rows):
        got = []
        for x in rows:
            if x not in got:
                got.append(x)
        return got

    res = {"versions": {"torch": torch.__version__, "transformers": version("transformers"), "tokenizers": version("tokenizers")},
           "heretic_versions": d["versions"], "heretic_supports_system_role": d.get("supports_system_role"),
           "response_prefix": prefix, "sets": {}}
    # measure: good (a list of two specifications) and bad (one specification), with the module inputs captured
    meta, calls, error = phase(lambda: DR.measure(ws, "systems", S["good"], S["bad"], chat=True, system=system,
                                                  prefix=prefix, variant="base", replace=True, module_io=True))
    ran = [ids for name, ids in calls if name == "run"]
    n_good = len(d["ids"]["good"])
    res["sets"]["good"] = {"ide": ran[:n_good], "calls": len(ran), "error": error}
    res["sets"]["bad"] = {"ide": ran[n_good:], "calls": len(ran), "error": error}
    if meta is not None:
        res["rendering"] = meta.get("rendering")
        stored = DR.prompts(ws, "systems")
        res["stored"] = {"good_objects": sum(isinstance(p, dict) for p in stored["good"]),
                         "good_texts": sum(isinstance(p, str) for p in stored["good"]),
                         "bad_objects": sum(isinstance(p, dict) for p in stored["bad"]),
                         "records": {k: [[p["kind"], p["first"], p["count"], p["commit"][:8], p["rows_checked"]]
                                         for p in v["parts"]] for k, v in meta["datasets"].items() if k in ("good", "bad")}}
    # scores: KL and refusals, each from its specification
    body = {"state": "current", "chat": True, "system": system, "prefix": prefix, "max_response_length": 10,
            "scores": [{"plugin": "KLDivergence", "prompts": S["kl"]}, {"plugin": "KeywordRate", "prompts": S["refusal"]}]}
    scored, calls, error = phase(lambda: DS.score(ws, body))
    res["sets"]["kl"] = {"ide": unique([ids for name, ids in calls if name == "first_token_logits"]), "calls": len(calls),
                         "error": error}
    res["sets"]["refusal"] = {"ide": unique([ids for name, ids in calls if name == "generate"]), "calls": len(calls),
                              "error": error}
    if scored is not None:
        res["scores"] = [{"name": s["name"], "value": s["value"]} for s in scored["scores"]]
        res["score_records"] = [[[p["kind"], p["first"], p["count"], p["system"]] for p in rec["parts"]]
                                for rec in scored["datasets"]]
    # prefix detection: a list of two specifications, the second with its own system prompt
    found, calls, error = phase(lambda: DP.detect(ws, {"prompts": S["prefix"], "system": system, "max_response_length": 10}))
    gen = [ids for name, ids in calls if name == "generate"]
    res["sets"]["prefix"] = {"ide": gen[:len(d["ids"]["prefix"])], "calls": len(gen), "error": error}
    if found is not None:
        res["prefix"] = {"response_prefix": found["response_prefix"], "found_by": found["found_by"],
                         "system_role": found["steps"]["template"]["system_role"]}
    ok = True
    for k, v in res["sets"].items():
        want = d["ids"][k]
        v["heretic_prompts"] = len(want)
        v["equal"] = v["ide"] == want
        v["prompts_equal"] = sum(a == b for a, b in zip(v["ide"], want))
        v["tokens"] = sum(len(x) for x in want)
        first = next((i for i, (a, b) in enumerate(zip(v["ide"], want)) if a != b), None)
        if first is not None:
            v["first_difference"] = {"prompt": first, "ide_head": v["ide"][first][:12], "heretic_head": want[first][:12],
                                     "ide_tokens": len(v["ide"][first]), "heretic_tokens": len(want[first])}
        ok = ok and v["equal"]
        print(k, "EQUAL" if v["equal"] else "DIFFERENT", v["prompts_equal"], "of", len(want), "prompts,", v["tokens"],
              "tokens; runner calls", v["calls"], *(["error:", v["error"]] if v["error"] else []), flush=True)
        del v["ide"]
    res["ok"] = ok
    res["tensors_touched_after"] = len(ws.touched())
    out.write_text(json.dumps(res, indent=1))
    print("rendering", res.get("rendering"))
    print("scores", res.get("scores"))
    print("prefix", res.get("prefix"))
    print("ok", ok, "tensors touched after:", res["tensors_touched_after"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "make-fixture":
        if len(sys.argv) != 5:
            sys.exit(__doc__)
        make_fixture(*sys.argv[2:])
    elif len(sys.argv) == 5:
        compare(*map(Path, sys.argv[1:]))
    else:
        sys.exit(__doc__)
