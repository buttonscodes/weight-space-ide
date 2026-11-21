"""Token strings and surface categories for a workspace's vocabulary.

A category is a coarse, purely orthographic label (script, case, digits, punctuation, ...).
It is used to describe what a neuron's token lists have in common when the tokens themselves
are not readable at a glance, and to keep reserved ids out of concept statistics.
"""
from __future__ import annotations

import unicodedata

import numpy as np

CATS = [
    "special",        # control / chat / tool tokens
    "bytefrag",       # partial UTF-8 sequences (not a full character)
    "ws_newline",     # whitespace containing a newline
    "ws_space",       # spaces / tabs only
    "number",         # digits (optionally with . , and a leading space)
    "punct",          # punctuation only
    "symbol",         # math / currency / other symbols only
    "code_mix",       # mixes letters/digits with punctuation or underscores (code-like)
    "lat_lower_ws",   # " word"   lowercase Latin, word-initial
    "lat_lower_cont", # "word"    lowercase Latin, continuation / no leading space
    "lat_cap_ws",     # " Word"
    "lat_cap_cont",   # "Word"
    "lat_upper",      # "WORD" / " WORD"
    "lat_camel",      # "someWord", "HTTPServer"
    "lat_accent",     # Latin with non-ASCII letters
    "alnum_mix",      # letters + digits, no punctuation
    "han", "kana", "hangul", "cyrillic", "arabic", "indic", "thai_sea", "greek", "hebrew",
    "other_script",
    "emoji_misc",     # emoji and pictographs
    "cjk_punct",      # full-width / CJK punctuation
    "unused",         # ids beyond the tokenizer (reserved, never trained on)
]
CAT_INDEX = {c: i for i, c in enumerate(CATS)}


def _script(ch: str) -> str:
    o = ord(ch)
    if o < 0x250 or 0x1E00 <= o < 0x1F00 or 0x250 <= o < 0x370:
        return "latin"
    if 0x370 <= o < 0x400 or 0x1F00 <= o < 0x2000:
        return "greek"
    if 0x400 <= o < 0x530 or 0x2DE0 <= o < 0x2E00 or 0xA640 <= o < 0xA6A0:
        return "cyrillic"
    if 0x590 <= o < 0x600:
        return "hebrew"
    if 0x600 <= o < 0x780 or 0x8A0 <= o < 0x900 or 0xFB50 <= o < 0xFE00 or 0xFE70 <= o < 0xFF00:
        return "arabic"
    if 0x900 <= o < 0xE00:
        return "indic"
    if 0xE00 <= o < 0xF00 or 0x1000 <= o < 0x10A0 or 0x1780 <= o < 0x1800:
        return "thai_sea"
    if 0x1100 <= o < 0x1200 or 0x3130 <= o < 0x3190 or 0xAC00 <= o < 0xD7B0:
        return "hangul"
    if 0x3040 <= o < 0x3100 or 0x31F0 <= o < 0x3200 or 0xFF66 <= o < 0xFFA0:
        return "kana"
    if 0x3400 <= o < 0x4DC0 or 0x4E00 <= o < 0xA000 or 0xF900 <= o < 0xFB00 or 0x20000 <= o < 0x32000:
        return "han"
    return "other"


def categorize(text: str, is_special: bool = False) -> str:
    """One primary surface category for a decoded token."""
    if is_special:
        return "special"
    if "�" in text or text == "":
        return "bytefrag"
    if text.strip() == "":
        return "ws_newline" if ("\n" in text or "\r" in text) else "ws_space"
    lead = text[0] in " \t"
    core = text.strip()
    cats = [unicodedata.category(c) for c in core]
    letters = [c for c, k in zip(core, cats) if k[0] == "L"]
    digits = [c for c, k in zip(core, cats) if k[0] == "N"]
    puncts = [c for c, k in zip(core, cats) if k[0] in "PS"]
    spaces = [c for c, k in zip(core, cats) if k[0] == "Z" or c in "\n\t\r"]
    if not letters and not digits:
        if any(0x1F000 <= ord(c) <= 0x1FAFF or 0x2600 <= ord(c) <= 0x27BF for c in core):
            return "emoji_misc"
        if any(0x3000 <= ord(c) <= 0x303F or 0xFF00 <= ord(c) <= 0xFF65 for c in core):
            return "cjk_punct"
        if puncts and all(k[0] == "S" for k in cats if k[0] in "PS"):
            return "symbol"
        return "punct" if puncts else "other_script"
    if digits and not letters:
        if all(c in ".,:-/% " for c in puncts) and not spaces:
            return "number"
        return "code_mix"
    scripts: dict[str, int] = {}
    for c in letters:
        s = _script(c)
        scripts[s] = scripts.get(s, 0) + 1
    main = max(scripts, key=scripts.get)
    if main != "latin":
        return main if main in CAT_INDEX else "other_script"
    if puncts or spaces or "_" in core:
        return "code_mix"
    if digits:
        return "alnum_mix"
    if any(ord(c) > 0x7F for c in letters):
        return "lat_accent"
    if core.islower():
        return "lat_lower_ws" if lead else "lat_lower_cont"
    if core.isupper() and len(core) > 1:
        return "lat_upper"
    if core[0].isupper() and (len(core) == 1 or core[1:].islower()):
        return "lat_cap_ws" if lead else "lat_cap_cont"
    return "lat_camel"


def token_table(ws) -> dict:
    """{"text": [str] * d_vocab, "cat": int8 array, "n_tok": tokens the tokenizer knows}.  Cached on the workspace."""
    if "atlas_tokens" in ws.keep:
        return ws.keep["atlas_tokens"]
    V = ws.struct.d_vocab
    tok = ws.tok
    n_tok = min(V, tok.vocab_size) if tok is not None else 0
    special: set[int] = set()
    tk = getattr(tok, "tk", None)
    if tk is not None and hasattr(tk, "get_added_tokens_decoder"):
        special = {int(i) for i, t in tk.get_added_tokens_decoder().items() if getattr(t, "special", False)}
    text, cat = [], np.empty(V, dtype=np.int8)
    for i in range(V):
        if tok is None:                                     # no tokenizer: ids only, every row is a real token
            text.append(f"<{i}>")
            cat[i] = CAT_INDEX["other_script"]
        elif i >= n_tok:
            text.append(f"<{i}>")
            cat[i] = CAT_INDEX["unused"]
        else:
            s = tok.token_str(i)
            text.append(s)
            cat[i] = CAT_INDEX[categorize(s, i in special)]
    ws.keep["atlas_tokens"] = {"text": text, "cat": cat, "n_tok": n_tok}
    return ws.keep["atlas_tokens"]
