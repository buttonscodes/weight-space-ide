"""Tokenizers behind one small interface: encode, decode, token strings, vocabulary size.

VocabTokenizer  an explicit symbol list; "char" mode splits text into characters, "word"
                mode into whitespace-separated symbols (used by the algorithmic toy models,
                whose tokens are things like "7", "+", "=").
BPETokenizer    a Hugging Face `tokenizer.json` (GPT-2 / TinyStories).
"""
from __future__ import annotations

import json
import os


class VocabTokenizer:
    def __init__(self, vocab: list[str], mode: str = "char", unk: str | None = None):
        self.vocab = list(vocab)
        self.mode = mode
        self.index = {s: i for i, s in enumerate(self.vocab)}
        self.unk = unk

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        parts = text.split() if self.mode == "word" else list(text)
        out = []
        for s in parts:
            if s in self.index:
                out.append(self.index[s])
            elif self.unk is not None:
                out.append(self.index[self.unk])
            else:
                raise ValueError(f"{s!r} is not in the vocabulary ({self.vocab_size} symbols, mode={self.mode})")
        return out

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        # skip_special_tokens is accepted for the BPETokenizer signature; a symbol list has no special tokens.
        return (" " if self.mode == "word" else "").join(self.vocab[int(i)] for i in ids)

    def token_str(self, i: int) -> str:
        return self.vocab[int(i)]

    def encode_token(self, s: str) -> list[int]:
        return self.encode(s)

    def to_dict(self) -> dict:
        return {"type": "vocab", "mode": self.mode, "vocab": self.vocab, "unk": self.unk}


class BPETokenizer:
    """Any Hugging Face `tokenizer.json`. With special=True prompts get the tokenizer's own special tokens
    (for example a leading BOS), which is what the model saw in training."""

    def __init__(self, tokenizer_json: str, special: bool = False):
        from tokenizers import Tokenizer
        self.json = tokenizer_json
        self.tk = Tokenizer.from_str(tokenizer_json)
        self.special = special
        self._strs: list[str] | None = None

    @property
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str, special: bool | None = None) -> list[int]:
        return self.tk.encode(text, add_special_tokens=self.special if special is None else special).ids

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return self.tk.decode([int(i) for i in ids], skip_special_tokens=skip_special_tokens)

    def token_str(self, i: int) -> str:
        if self._strs is None:
            self._strs = [self.tk.decode([j], skip_special_tokens=False) for j in range(self.vocab_size)]
        i = int(i)
        return self._strs[i] if i < len(self._strs) else f"<{i}>"

    def encode_token(self, s: str) -> list[int]:
        """Tokenise a string that names a token (never adds special tokens)."""
        return self.tk.encode(s, add_special_tokens=False).ids

    def to_dict(self) -> dict:
        return {"type": "bpe"}


def save_tokenizer(tok, directory: str) -> None:
    with open(os.path.join(directory, "tokenizer_meta.json"), "w") as f:
        json.dump(tok.to_dict(), f)
    if isinstance(tok, BPETokenizer):
        with open(os.path.join(directory, "tokenizer.json"), "w") as f:
            f.write(tok.json)


def load_tokenizer(directory: str):
    with open(os.path.join(directory, "tokenizer_meta.json")) as f:
        meta = json.load(f)
    if meta["type"] == "bpe":
        with open(os.path.join(directory, "tokenizer.json")) as f:
            return BPETokenizer(f.read())
    return VocabTokenizer(meta["vocab"], meta.get("mode", "char"), meta.get("unk"))


def show(s: str) -> str:
    """A token string made visible: spaces and newlines as symbols."""
    return s.replace("\n", "⏎").replace(" ", "·").replace("\t", "⇥")
