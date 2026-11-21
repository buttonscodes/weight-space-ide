"""A small GPT whose parameters are stored the way they are *edited*.

Parameters live in a flat dict under TransformerLens-style names, with heads and neurons
as real tensor axes, so every level of the hierarchy is a plain slice:

    embed.W_E               [d_vocab, d_model]        row t        = token t's embedding
    pos_embed.W_pos         [n_ctx,  d_model]         row p        = position p
    blocks.L.ln1.w / .b     [d_model]
    blocks.L.attn.W_Q/K/V   [n_heads, d_model, d_head]   [h]       = what head h reads
    blocks.L.attn.b_Q/K/V   [n_heads, d_head]
    blocks.L.attn.W_O       [n_heads, d_head, d_model]   [h]       = what head h writes
    blocks.L.attn.b_O       [d_model]
    blocks.L.ln2.w / .b     [d_model]
    blocks.L.mlp.W_in       [d_model, d_mlp]          column n     = what neuron n reads
    blocks.L.mlp.b_in       [d_mlp]
    blocks.L.mlp.W_out      [d_mlp, d_model]          row n        = what neuron n writes
    blocks.L.mlp.b_out      [d_model]
    ln_final.w / .b         [d_model]
    unembed.W_U             [d_model, d_vocab]        column t     = token t's output direction
    unembed.b_U             [d_vocab]

The forward pass is a pure function of that dict, so a model with a different set of weights
is just a different dict: no module surgery, and base / edited / candidate models can be run
side by side.  Hook points follow TransformerLens names (blocks.0.attn.hook_pattern, ...).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

Params = dict[str, torch.Tensor]
Hook = Callable[[torch.Tensor, str], "torch.Tensor | None"]


@dataclass
class Config:
    d_vocab: int
    n_ctx: int
    d_model: int
    n_layers: int
    n_heads: int
    d_head: int = 0            # 0 -> d_model // n_heads
    d_mlp: int = 0             # 0 -> 4 * d_model
    act_fn: str = "gelu"       # gelu | gelu_new | relu
    normalization: str = "LN"  # LN | none   ("none" makes hand-written programs exact)
    attn_only: bool = False
    attn_scale: float = 0.0    # 0 -> sqrt(d_head); GPT-Neo uses 1
    ln_eps: float = 1e-5
    name: str = ""
    family: str = "gpt"        # provenance: gpt | gpt2 | gpt-neo | handwritten

    def __post_init__(self):
        if not self.d_head:
            self.d_head = self.d_model // self.n_heads
        if not self.d_mlp and not self.attn_only:
            self.d_mlp = 4 * self.d_model
        if not self.attn_scale:
            self.attn_scale = math.sqrt(self.d_head)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Config":
        return Config(**{k: v for k, v in d.items() if k in Config.__dataclass_fields__})

    @property
    def use_ln(self) -> bool:
        return self.normalization == "LN"


def param_shapes(cfg: Config) -> dict[str, tuple[int, ...]]:
    """Every parameter name and shape, in forward order."""
    s: dict[str, tuple[int, ...]] = {
        "embed.W_E": (cfg.d_vocab, cfg.d_model),
        "pos_embed.W_pos": (cfg.n_ctx, cfg.d_model),
    }
    for l in range(cfg.n_layers):
        b = f"blocks.{l}"
        if cfg.use_ln:
            s[f"{b}.ln1.w"] = (cfg.d_model,)
            s[f"{b}.ln1.b"] = (cfg.d_model,)
        for m in "QKV":
            s[f"{b}.attn.W_{m}"] = (cfg.n_heads, cfg.d_model, cfg.d_head)
            s[f"{b}.attn.b_{m}"] = (cfg.n_heads, cfg.d_head)
        s[f"{b}.attn.W_O"] = (cfg.n_heads, cfg.d_head, cfg.d_model)
        s[f"{b}.attn.b_O"] = (cfg.d_model,)
        if not cfg.attn_only:
            if cfg.use_ln:
                s[f"{b}.ln2.w"] = (cfg.d_model,)
                s[f"{b}.ln2.b"] = (cfg.d_model,)
            s[f"{b}.mlp.W_in"] = (cfg.d_model, cfg.d_mlp)
            s[f"{b}.mlp.b_in"] = (cfg.d_mlp,)
            s[f"{b}.mlp.W_out"] = (cfg.d_mlp, cfg.d_model)
            s[f"{b}.mlp.b_out"] = (cfg.d_model,)
    if cfg.use_ln:
        s["ln_final.w"] = (cfg.d_model,)
        s["ln_final.b"] = (cfg.d_model,)
    s["unembed.W_U"] = (cfg.d_model, cfg.d_vocab)
    s["unembed.b_U"] = (cfg.d_vocab,)
    return s


def init_params(cfg: Config, seed: int = 0, std: float = 0.02, zero: bool = False) -> Params:
    """Random (GPT-2 style) or all-zero parameters. LayerNorm gains start at 1 either way."""
    g = torch.Generator().manual_seed(seed)
    p: Params = {}
    for name, shape in param_shapes(cfg).items():
        leaf = name.rsplit(".", 1)[1]
        if leaf == "w":
            p[name] = torch.ones(shape)
        elif leaf.startswith("b") or zero:
            p[name] = torch.zeros(shape)
        else:
            scale = std
            if leaf in ("W_O", "W_out"):
                scale = std / math.sqrt(2 * cfg.n_layers)
            p[name] = torch.randn(shape, generator=g) * scale
    return p


def _act(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu":
        return F.relu(x)
    if kind == "gelu_new":
        return F.gelu(x, approximate="tanh")
    return F.gelu(x)


def _ln(x, w, b, eps, hook, name):
    x = x - x.mean(-1, keepdim=True)
    scale = hook((x.pow(2).mean(-1, keepdim=True) + eps).sqrt(), f"{name}.hook_scale")
    return hook(x / scale, f"{name}.hook_normalized") * w + b


def forward(params: Params, cfg: Config, tokens: torch.Tensor, hooks: dict[str, Hook] | None = None,
            cache: dict[str, torch.Tensor] | None = None, names: Iterable[str] | None = None,
            per_head: bool = False) -> torch.Tensor:
    """Logits [batch, pos, d_vocab].

    hooks  : {hook_name: fn(tensor, name) -> tensor or None} to intervene on an activation.
    cache  : if given, filled with activations (all of them, or only those in `names`).
    per_head: also expose blocks.L.attn.hook_result [batch, pos, head, d_model] (costly).
    """
    if tokens.dim() == 1:
        tokens = tokens[None]
    T = tokens.shape[1]
    if T > cfg.n_ctx:
        raise ValueError(f"sequence of {T} tokens exceeds the model's context of {cfg.n_ctx}")
    want = set(names) if names is not None else None

    def hook(x, name):
        if hooks and name in hooks:
            out = hooks[name](x, name)
            if out is not None:
                x = out
        if cache is not None and (want is None or name in want):
            cache[name] = x.detach()
        return x

    p = params
    x = hook(p["embed.W_E"][tokens], "hook_embed") + hook(p["pos_embed.W_pos"][:T][None], "hook_pos_embed")
    mask = torch.ones(T, T, dtype=torch.bool, device=tokens.device).triu(1)
    need_result = per_head or bool(hooks and any(k.endswith("attn.hook_result") for k in hooks)) or \
        bool(want and any(k.endswith("attn.hook_result") for k in want))
    for l in range(cfg.n_layers):
        b = f"blocks.{l}"
        x = hook(x, f"{b}.hook_resid_pre")
        h = _ln(x, p[f"{b}.ln1.w"], p[f"{b}.ln1.b"], cfg.ln_eps, hook, f"{b}.ln1") if cfg.use_ln else x
        q = hook(torch.einsum("btd,hde->bthe", h, p[f"{b}.attn.W_Q"]) + p[f"{b}.attn.b_Q"], f"{b}.attn.hook_q")
        k = hook(torch.einsum("btd,hde->bthe", h, p[f"{b}.attn.W_K"]) + p[f"{b}.attn.b_K"], f"{b}.attn.hook_k")
        v = hook(torch.einsum("btd,hde->bthe", h, p[f"{b}.attn.W_V"]) + p[f"{b}.attn.b_V"], f"{b}.attn.hook_v")
        scores = torch.einsum("bqhe,bkhe->bhqk", q, k) / cfg.attn_scale
        scores = hook(scores.masked_fill(mask, float("-inf")), f"{b}.attn.hook_attn_scores")
        pattern = hook(scores.softmax(-1), f"{b}.attn.hook_pattern")
        z = hook(torch.einsum("bhqk,bkhe->bqhe", pattern, v), f"{b}.attn.hook_z")
        if need_result:
            res = hook(torch.einsum("bqhe,hed->bqhd", z, p[f"{b}.attn.W_O"]), f"{b}.attn.hook_result")
            attn_out = res.sum(2) + p[f"{b}.attn.b_O"]
        else:
            attn_out = torch.einsum("bqhe,hed->bqd", z, p[f"{b}.attn.W_O"]) + p[f"{b}.attn.b_O"]
        x = x + hook(attn_out, f"{b}.hook_attn_out")
        if not cfg.attn_only:
            x = hook(x, f"{b}.hook_resid_mid")
            h = _ln(x, p[f"{b}.ln2.w"], p[f"{b}.ln2.b"], cfg.ln_eps, hook, f"{b}.ln2") if cfg.use_ln else x
            pre = hook(h @ p[f"{b}.mlp.W_in"] + p[f"{b}.mlp.b_in"], f"{b}.mlp.hook_pre")
            post = hook(_act(pre, cfg.act_fn), f"{b}.mlp.hook_post")
            x = x + hook(post @ p[f"{b}.mlp.W_out"] + p[f"{b}.mlp.b_out"], f"{b}.hook_mlp_out")
        x = hook(x, f"{b}.hook_resid_post")
    if cfg.use_ln:
        x = _ln(x, p["ln_final.w"], p["ln_final.b"], cfg.ln_eps, hook, "ln_final")
    return x @ p["unembed.W_U"] + p["unembed.b_U"]


def hook_names(cfg: Config) -> list[str]:
    out = ["hook_embed", "hook_pos_embed"]
    for l in range(cfg.n_layers):
        b = f"blocks.{l}"
        out += [f"{b}.hook_resid_pre"]
        if cfg.use_ln:
            out += [f"{b}.ln1.hook_scale", f"{b}.ln1.hook_normalized"]
        out += [f"{b}.attn.hook_q", f"{b}.attn.hook_k", f"{b}.attn.hook_v", f"{b}.attn.hook_attn_scores",
                f"{b}.attn.hook_pattern", f"{b}.attn.hook_z", f"{b}.attn.hook_result", f"{b}.hook_attn_out"]
        if not cfg.attn_only:
            out += [f"{b}.hook_resid_mid"]
            if cfg.use_ln:
                out += [f"{b}.ln2.hook_scale", f"{b}.ln2.hook_normalized"]
            out += [f"{b}.mlp.hook_pre", f"{b}.mlp.hook_post", f"{b}.hook_mlp_out"]
        out += [f"{b}.hook_resid_post"]
    if cfg.use_ln:
        out += ["ln_final.hook_scale", "ln_final.hook_normalized"]
    return out


@torch.no_grad()
def generate(params: Params, cfg: Config, tokens: list[int], max_new_tokens: int = 40, temperature: float = 0.0,
             top_k: int = 0, seed: int | None = None, stop_ids: Iterable[int] = (),
             on_token: Callable[[int], None] | None = None, hooks: dict[str, Hook] | None = None) -> list[int]:
    """Greedy (temperature 0) or sampled continuation. Returns only the new tokens.

    hooks: as in `forward`, applied at every step (steering)."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    stop = set(stop_ids)
    toks = list(tokens)
    out: list[int] = []
    for _ in range(max_new_tokens):
        ctx = torch.tensor(toks[-cfg.n_ctx:])
        logits = forward(params, cfg, ctx, hooks=hooks)[0, -1]
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k:
                kth = logits.topk(min(top_k, logits.numel())).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = int(torch.multinomial(logits.softmax(-1), 1, generator=g))
        toks.append(nxt)
        out.append(nxt)
        if on_token:
            on_token(nxt)
        if nxt in stop:
            break
    return out


def n_params(params: Params) -> int:
    return sum(v.numel() for v in params.values())
