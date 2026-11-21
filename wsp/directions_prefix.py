"""Response-prefix detection and thinking blocks, as Heretic detects them (its ``response_prefix`` and
``chain_of_thought_blocks``).

Scores and measurements should start where responses start to differ. Heretic finds that point once, before
anything else: if the model's chat template already closes a reasoning block, nothing is needed; if it opens one,
the block's finalizer is the prefix; otherwise it generates responses to a set of test prompts and takes their
common prefix, closing a reasoning block it starts with. When a block was closed this way it generates once more
with the prefix appended and adds what all responses still share (a newline, for example).

Heretic's whole handling of thinking blocks is this: it closes the block through the response prefix; nothing
strips thinking text from responses. The IDE's second way is the template switch ``chat_options.enable_thinking``.

The result is returned, never stored: pass ``response_prefix`` as ``prefix`` to POST /api/directions/measure and
to POST /api/directions/scores. Nothing here writes a weight.
"""
from __future__ import annotations

import os
import re
import time

from . import directions as DR
from . import directions_scores as DS
from .directions import DirectionError
from .directions_rendering import SENTINEL, supports_system_role  # noqa: F401  (the probe every rendering shares)
from .runtime_events import job_scope

# Heretic's default chain_of_thought_blocks, in its order: most thinking models, gpt-oss, and two suggested by users.
CHAIN_OF_THOUGHT_BLOCKS = [["<think>", "</think>"],
                           ["<|channel|>analysis<|message|>", "<|end|><|start|>assistant<|channel|>final<|message|>"],
                           ["<thought>", "</thought>"], ["[THINK]", "[/THINK]"]]
MAX_BLOCKS = 16
MAX_BLOCK_CHARS = 200
DUMMY_SYSTEM, DUMMY_USER = "You are a helpful assistant.", "This is a dummy prompt."
PREFIX_FIELDS = ("prompts", "system", "chat_options", "variant", "max_response_length", "chain_of_thought_blocks",
                 "print_responses")
VARIANTS = ("base", "current")


class _Stopped(Exception):
    pass


def _blocks(value):
    bad = DirectionError(f"chain_of_thought_blocks must list 0..{MAX_BLOCKS} pairs [initializer, finalizer] of text of "
                         f"1..{MAX_BLOCK_CHARS} characters")
    if not isinstance(value, list) or len(value) > MAX_BLOCKS:
        raise bad
    out = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or \
                not all(isinstance(t, str) and 1 <= len(t) <= MAX_BLOCK_CHARS for t in pair):
            raise bad
        out.append([pair[0], pair[1]])
    return out


def check_prefix(ws, body: dict) -> dict:
    """Everything about a prefix detection that needs no model -> the normalized request."""
    from . import probes
    if not isinstance(body, dict):
        raise DirectionError("the request must be an object")
    unknown = sorted(set(body) - set(PREFIX_FIELDS) - {"stream"}, key=str)
    if unknown:
        raise DirectionError(f"{unknown[0]}: unknown field; the fields are {', '.join(PREFIX_FIELDS)} and stream")
    if "prompts" not in body:
        raise DirectionError("prompts: required")
    prompts = list(DR._prompt_value(body["prompts"], "prompts", "prompts"))
    system = DR._optional_text(body.get("system"), "system")
    options = probes.check_chat_options(body.get("chat_options"))
    variant = body.get("variant", "base")
    if variant not in VARIANTS:
        raise DirectionError("variant must be 'base' or 'current'")
    M = DS._max_response_length(body.get("max_response_length", DS.DEFAULT_RESPONSE_LENGTH))
    blocks = _blocks(body["chain_of_thought_blocks"]) if "chain_of_thought_blocks" in body else \
        [list(b) for b in CHAIN_OF_THOUGHT_BLOCKS]
    show = body.get("print_responses", False)
    if not isinstance(show, bool):
        raise DirectionError("print_responses must be true or false")
    ok, reason = probes.chat_template(ws)
    if not ok:
        raise DirectionError(f"'as a chat message' needs the model's own chat template: {reason}")
    if ws.tok is None:
        raise DirectionError("this model has no tokenizer, so prompts cannot be encoded")
    prompts, record = DR._prompt_set(ws, prompts, "prompts", "prompts", system=system, chat=True)
    return {"prompts": prompts, "system": system, "chat_options": options, "variant": variant,
            "max_response_length": M, "chain_of_thought_blocks": blocks, "print_responses": show, "datasets": record}


def dummy_prompt(ws, options: dict) -> tuple[str, bool]:
    """Heretic's dummy prompt: its fixed system message when the template has a system role, then a fixed user
    turn, rendered with the generation prompt -> (rendering, system role supported)."""
    system_role = supports_system_role(ws, options)
    messages = ([{"role": "system", "content": DUMMY_SYSTEM}] if system_role else []) + \
        [{"role": "user", "content": DUMMY_USER}]
    rendered = ws.runner.chat_prompt(messages, **(options or {}))
    if rendered is None:
        raise DirectionError("'as a chat message' needs the model's own chat template: this model has none")
    return rendered, system_role


def detect_from(dummy: str, blocks, responses) -> dict:
    """The detection itself, from the dummy rendering, the blocks and ``responses(prefix or None) -> texts``."""
    prefix, cot, found, block, steps = None, False, "none", None, {}
    for init, fin in blocks:
        if re.search(re.escape(init) + ".*" + re.escape(fin), dummy, flags=re.DOTALL):
            prefix, cot, found, block = "", True, "template_closed", [init, fin]
            break
        if init in dummy:
            prefix, cot, found, block = fin, True, "template_initializer", [init, fin]
            break
    if prefix is None:
        texts = responses(None)
        prefix = os.path.commonprefix(texts).rstrip(" ")
        steps["common_prefix"] = {"responses": len(texts), "prefix": prefix}
        if prefix:
            found = "common_prefix"
            for init, fin in blocks:
                if prefix.startswith(init):
                    prefix, cot, found, block = init + fin, True, "common_prefix_block", [init, fin]
                    break
    if cot and prefix:
        texts = responses(prefix)
        additional = os.path.commonprefix(texts).rstrip(" ")
        steps["recheck"] = {"responses": len(texts), "additional": additional}
        if additional:
            prefix += additional
    return {"response_prefix": prefix, "found_by": found, "block": block, "steps": steps}


@job_scope("direction_prefix")
def detect(ws, body: dict, *, progress=None, stopped=None) -> dict:
    """Detect the response prefix on ``variant`` (base by default, as Heretic detects before any trial). The test
    prompts are rendered with the request's system message and chat options; responses are decoded with the special
    tokens. ``progress(phase, done, total)`` after every response; ``stopped()`` before every response."""
    started = time.monotonic()
    with ws.lock:
        spec = check_prefix(ws, body)
        DS.runner_row(ws)
        options, M, variant = spec["chat_options"], spec["max_response_length"], spec["variant"]
        r = ws.runner
        ctx = int(getattr(r, "max_ctx", 0) or 0) or int(getattr(ws.struct, "n_ctx", 0) or 0)
        stops = DS.stop_ids(ws)
        total, done, seen = 2 * len(spec["prompts"]), [0], {}
        dummy, system_role = dummy_prompt(ws, options)

        def responses(p):
            phase = "common_prefix" if p is None else "recheck"
            ids = DR.render_ids(ws, spec["prompts"], "prompts", chat=True, system=spec["system"], prefix=p or None,
                                chat_options=options)
            DS.check_context(ids, "prompts", M, ctx, " as a chat message")
            texts = []
            for k, x in enumerate(ids):
                if stopped is not None and stopped():
                    raise _Stopped()
                out = DS.generate_response(ws, x, variant, None, M, stops, "prompts", k + 1)
                texts.append(DS.decode(ws, out, skip_special_tokens=False))
                done[0] += 1
                if progress is not None:
                    progress(phase, done[0], total)
            seen[phase] = texts
            return texts
        try:
            found = detect_from(dummy, spec["chain_of_thought_blocks"], responses)
        except _Stopped:
            return {"stopped": True, "done": done[0], "total": total}
        steps = {"template": {"rendering": dummy, "system_role": system_role}, **found["steps"]}
        if spec["print_responses"]:
            for phase, texts in seen.items():
                steps[phase]["texts"] = texts
        return {"response_prefix": found["response_prefix"], "found_by": found["found_by"], "block": found["block"],
                "steps": steps, "generation": DS.generation_block(ws, M), "datasets": spec["datasets"],
                "seconds": round(time.monotonic() - started, 3), "done": done[0], "total": total}
