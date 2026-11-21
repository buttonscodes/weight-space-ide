"""How the Directions feature turns a prompt into token ids: as Heretic does (``Model.generate`` at 64e9991).

One renderer for every prompt the feature measures or scores (measurements with or without module inputs, scores,
the response-prefix detection, searches, checkpoint scoring, steered scores: all through ``directions.render_ids``).
For a prompt read as a chat message:

    messages     the system message (the request's, or the prompt's own; none when there is neither) and the user
                 turn; when the template has no system role (a sentinel system message does not survive into a
                 rendering, ``supports_system_role``), each system message is folded into the next user turn and an
                 empty one is dropped (Heretic's ``_apply_system_role``)
    text         the model's own template with the generation prompt, the date fixed at 2026-01-01 00:00 UTC
                 (Heretic's ``strftime_fixed_date``), the chat options, then the prefix appended
    ids          the text tokenized as Heretic tokenizes it: the checkpoint's transformers tokenizer with its
                 defaults, so the tokenizer's own special tokens are added (a template that prints a
                 beginning-of-text token the tokenizer also adds gives two, in Heretic as here); without that
                 tokenizer, the vocabulary's encoding with its special tokens

That is rendering version 2 (``VERSION``). Version 1 is how the IDE rendered before (a measurement without a
``rendering`` entry): the messages as given, the template with today's date, the text tokenized without the
tokenizer's own special tokens. It is kept only to tell, when a measurement is made, whether it would have given the
same ids (``same_as_version_1``), which decides whether a reproduction record made before can be re-run from a new
measurement (wsp.directions_record). A prompt that is not read as a chat message is encoded as before, in both
versions. The Chat and Run tabs and the probes render with wsp.probes, unchanged.

Heretic fixes the date by passing ``strftime_now=strftime_fixed_date`` to ``apply_chat_template``. transformers 5.22
turns every template argument into plain data first, so a function arrives as the text "<function>" and a template
that reads the date fails. The date is fixed instead where transformers' own ``strftime_now`` reads it: the clock of
``transformers.utils.chat_template_utils`` answers the fixed date in this thread while a Renderer renders, and the
real time everywhere else (Chat renders in other threads, with today's date as before).
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import datetime, timezone

VERSION = 2
FIXED_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)
SENTINEL = "HERETIC_SYSTEM_ROLE_PROBE"


_CLOCK = threading.local()          # .fixed: this thread renders with the fixed date; .read: a template read it
_INSTALLED = threading.Lock()


def strftime_fixed_date(format: str) -> str:
    """Heretic's ``strftime_fixed_date``: the date templates print, fixed so a rendering does not depend on the day."""
    return FIXED_DATE.strftime(format)


def _install_clock() -> None:
    """Once: transformers' template clock answers FIXED_DATE in a thread inside ``fixed_date`` (else the real time)."""
    try:
        from transformers.utils import chat_template_utils as CT
    except ImportError:                 # no template engine, so no template reads the date
        return
    with _INSTALLED:
        real = CT.datetime
        if getattr(real, "_wsp_fixed_clock", False):
            return

        class Clock(real):
            _wsp_fixed_clock = True

            @classmethod
            def now(cls, tz=None):
                if getattr(_CLOCK, "fixed", False):
                    _CLOCK.read = True
                    return FIXED_DATE
                return real.now(tz)
        CT.datetime = Clock


@contextmanager
def fixed_date():
    """While inside, a template rendered in this thread reads 2026-01-01 00:00 UTC -> a dict whose ``read`` says,
    afterwards, whether a template read the date."""
    _install_clock()
    seen = {"read": False}
    _CLOCK.fixed, _CLOCK.read = True, False
    try:
        yield seen
    finally:
        seen["read"] = bool(getattr(_CLOCK, "read", False))
        _CLOCK.fixed, _CLOCK.read = False, False


def supports_system_role(ws, options: dict) -> bool:
    """Whether the template keeps a system message: a sentinel system message survives into the rendering (Heretic's
    ``chat_template_supports_system_role``: a template that raises has none)."""
    try:
        rendered = ws.runner.chat_prompt([{"role": "system", "content": SENTINEL}, {"role": "user", "content": "Hi"}],
                                         **(options or {}))
    except Exception:
        return False
    return rendered is not None and SENTINEL in rendered


def apply_system_role(messages: list[dict], supported: bool) -> list[dict]:
    """Heretic's ``_apply_system_role``: the messages unchanged when the template has a system role; else each system
    message folded into the next user message (joined by a blank line), an empty one dropped, and a system message
    with no user message after it made a user turn."""
    if supported:
        return messages
    result, pending = [], []
    for message in messages:
        if message["role"] == "system":
            if message["content"]:
                pending.append(message["content"])
            continue
        if pending and message["role"] == "user":
            message = {**message, "content": "\n\n".join([*pending, message["content"]])}
            pending = []
        result.append(message)
    if pending:
        result.append({"role": "user", "content": "\n\n".join(pending)})
    return result


def messages(user: str, system: str | None) -> list[dict]:
    return ([{"role": "system", "content": system}] if system is not None else []) + [{"role": "user", "content": user}]


def _vocabulary_encode(ws, special: bool):
    if ws.tok.to_dict()["type"] == "bpe":
        return lambda t: ws.tok.encode(t, special=special)
    return ws.tok.encode


class Renderer:
    """The prompts of one request as chat messages, rendered with one version. The system-role probe runs once, the
    first time a message has a system message; ``dated`` is set when the template read the date."""

    def __init__(self, ws, options: dict | None, version: int = VERSION):
        if version not in (1, 2):
            raise ValueError("rendering version must be 1 or 2")
        self.ws, self.options, self.version = ws, dict(options or {}), version
        self._supported = None
        self.dated = False
        if version == 1:
            from . import probes
            self.encode = probes.chat_encode(ws)
        else:
            tok = getattr(ws.runner, "hf_tokenizer", lambda: None)()
            if tok is not None:
                self.encode = lambda t: tok([t], return_token_type_ids=False)["input_ids"][0]
            else:
                self.encode = _vocabulary_encode(ws, True)

    def text(self, user: str, system: str | None, prefix: str | None) -> str:
        msgs = messages(user, system)
        if self.version == 1:
            rendered = self.ws.runner.chat_prompt(msgs, **self.options)
        else:
            if any(m["role"] == "system" for m in msgs):
                if self._supported is None:
                    self._supported = supports_system_role(self.ws, self.options)
                msgs = apply_system_role(msgs, self._supported)
            with fixed_date() as seen:
                rendered = self.ws.runner.chat_prompt(msgs, **self.options)
            self.dated = self.dated or seen["read"]
        if rendered is None:
            raise ValueError("'as a chat message' needs the model's own chat template: this model has none")
        return rendered + prefix if prefix else rendered


def compatible(then: dict | None, now: dict | None) -> bool:
    """Whether two measurements' ``rendering`` entries (None: version 1) give the same ids: the same version, or a
    version-1 one and a version-2 one that recorded ``same_as_version_1``."""
    a, b = (then or {}).get("version", 1), (now or {}).get("version", 1)
    if a == b:
        return True
    newer = then if a > b else now
    return {a, b} == {1, 2} and (newer or {}).get("same_as_version_1") is True
