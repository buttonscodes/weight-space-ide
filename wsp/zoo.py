"""Small models to program: trained here, imported, or blank.

    facts             2 layers, d_model 64, word-level. Trained on a synthetic knowledge base
                      ("Alice lives in Paris ."), so there are crisp facts to edit and a crisp
                      locality test (every other fact).  ~70k parameters, trains in about a minute.
    shakespeare-char  3 layers, d_model 96, character-level tiny-shakespeare.  ~0.35M parameters.
    tinystories-1m    roneneldan/TinyStories-1M (GPT-Neo, 8 layers, d_model 64, GPT-2 BPE). Real
                      English at a size where every non-embedding matrix can be shown cell by cell.
    gpt2              openai-community/gpt2 (124M) for scale tests.
    blank-reverse     all-zero weights, no LayerNorm: write a program that reverses a sequence.
    blank-modadd      all-zero weights, no LayerNorm: write (a + b) mod 7.
    blank             an empty model of any shape.

The blank models are the purest form of weight-space programming: there is nothing to train
and nothing to fine-tune, only a test suite to satisfy.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import random
import re
import time
import urllib.request

import torch
import torch.nn.functional as F

from .model import Config, Params, forward, init_params
from .tokenizer import BPETokenizer, VocabTokenizer
from .workspace import Workspace

HF = "https://huggingface.co/{repo}/resolve/main/{file}"
SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _fetch(url: str, path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def train(cfg: Config, sample, steps: int, lr: float = 3e-3, seed: int = 0, wd: float = 0.01,
          log_every: int = 200, loss_mask=None, log=print) -> Params:
    """AdamW on next-token loss. `sample(step)` returns a LongTensor batch [B, T]."""
    torch.manual_seed(seed)
    params = init_params(cfg, seed)
    for v in params.values():
        v.requires_grad_(True)
    decay = [v for k, v in params.items() if v.dim() >= 2]
    rest = [v for k, v in params.items() if v.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.98))
    t0 = time.time()
    for step in range(steps):
        cur = lr * min(1.0, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = cur
        batch = sample(step)
        logits = forward(params, cfg, batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, cfg.d_vocab), batch[:, 1:].reshape(-1), reduction="none")
        if loss_mask is not None:
            m = loss_mask(batch).reshape(-1).float()
            loss = (loss * m).sum() / m.sum().clamp(min=1)
        else:
            loss = loss.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
        opt.step()
        if log and (step % log_every == 0 or step == steps - 1):
            log(f"step {step:5d}  loss {loss.item():.4f}  {time.time() - t0:.0f}s")
    return {k: v.detach().requires_grad_(False) for k, v in params.items()}


# ------------------------------------------------------------------------------ facts
NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Iris", "Jack", "Kate", "Liam",
         "Mia", "Noah", "Olivia", "Paul", "Quinn", "Rosa", "Sam", "Tara", "Uma", "Victor", "Wendy", "Yusuf"]
RELATIONS = {
    "lives in": ["Paris", "Rome", "Tokyo", "Cairo", "Lima", "Oslo", "Delhi", "Perth"],
    "likes": ["tea", "coffee", "juice", "milk", "soda", "water"],
    "plays": ["chess", "tennis", "piano", "guitar", "golf", "cards"],
}


def facts_kb(seed: int = 0) -> list[tuple[str, str, str]]:
    rng = random.Random(seed)
    return [(n, rel, rng.choice(objs)) for n in NAMES for rel, objs in RELATIONS.items()]


def build_facts(directory: str, steps: int = 2500, log=print) -> Workspace:
    kb = facts_kb()
    vocab = ["<bos>", "."] + NAMES + sorted({w for r in RELATIONS for w in r.split()}) + \
        [o for objs in RELATIONS.values() for o in objs]
    tok = VocabTokenizer(vocab, "word")
    cfg = Config(d_vocab=len(vocab), n_ctx=32, d_model=64, n_layers=2, n_heads=4, d_mlp=256, name="facts")
    sents = [tok.encode(f"{n} {r} {o} .") for n, r, o in kb]
    rng = random.Random(1)

    def sample(step):
        rows = []
        for _ in range(64):
            seq = [0]
            while len(seq) < cfg.n_ctx:
                seq += rng.choice(sents)
            rows.append(seq[:cfg.n_ctx])
        return torch.tensor(rows)

    params = train(cfg, sample, steps, lr=3e-3, log=log)
    notes = ("Synthetic knowledge base: 24 people x 3 relations (lives in / likes / plays), word-level tokens.\n"
             "Prompts are space-separated words, e.g. `Alice lives in`. Every sentence ends with ` .`\n")
    ws = Workspace.create(directory, cfg, tok, params, notes)
    cases = [{"type": "next_token", "prompt": f"<bos> {n} {r}", "expect": o, "name": f"{n} {r} {o}"} for n, r, o in kb]
    ws.write_json("tests/facts.json", {"name": "facts", "description": "every fact in the knowledge base",
                                       "cases": cases})
    ws.write_json("kb.json", kb)
    old = next(o for n, r, o in kb if (n, r) == ("Alice", "lives in"))
    new = next(c for c in RELATIONS["lives in"] if c != old)
    moved = [dict(c, expect=new, name=f"Alice lives in {new} (edited)") if c["prompt"] == "<bos> Alice lives in" else c for c in cases]
    ws.write_json("tests/alice-moves.json", {"name": "alice-moves", "cases": moved,
                                             "description": f"Alice now lives in {new}; the other 71 facts are unchanged"})
    ws.write_json("task.json", {
        "title": f"Alice moves to {new}",
        "instructions": f"The model was trained on a small knowledge base in which Alice lives in {old}. Change the weights so "
                        f"that `Alice lives in` is followed by `{new}`, while every other fact (including what Alice likes and "
                        f"plays, and where everyone else lives) stays exactly as it was. Find where the fact is stored before "
                        f"you edit, and prefer a small, explainable change.",
        "scope": ["blocks.0.mlp", "blocks.1.mlp"], "suite": "alice-moves", "preserve": {"max_kl": 0.02}})
    return ws


# ------------------------------------------------------------------------------ shakespeare
def build_shakespeare(directory: str, steps: int = 3000, cache: str = "", log=print) -> Workspace:
    path = _fetch(SHAKESPEARE, os.path.join(cache or directory + "-cache", "tinyshakespeare.txt"))
    text = open(path).read()
    chars = sorted(set(text))
    tok = VocabTokenizer(chars, "char")
    data = torch.tensor(tok.encode(text))
    cfg = Config(d_vocab=len(chars), n_ctx=96, d_model=96, n_layers=3, n_heads=4, d_mlp=384, name="shakespeare-char")
    g = torch.Generator().manual_seed(0)

    def sample(step):
        ix = torch.randint(0, len(data) - cfg.n_ctx - 1, (48,), generator=g)
        return torch.stack([data[i:i + cfg.n_ctx] for i in ix])

    params = train(cfg, sample, steps, lr=2e-3, log=log)
    ws = Workspace.create(directory, cfg, tok, params,
                          "Character-level model of tiny-shakespeare (65 symbols). Prompts are raw text.\n")
    ws.write_json("tests/style.json", {"name": "style", "description": "a few things the trained model does", "cases": [
        {"type": "next_token", "prompt": "ROMEO:\nWhat say you to my s", "expect": ["u", "o", "i", "e", "w", "p", "t", "a", "h"], "top_k": 9,
         "name": "letter after 's' is a plausible letter"},
        {"type": "next_token", "prompt": "First Citizen", "expect": ":", "name": "speaker name is followed by a colon"},
        {"type": "next_token", "prompt": "KING RICHARD III:", "expect": "\n", "name": "newline after the speaker line"},
    ]})
    return ws


# ------------------------------------------------------------------------------ hugging face imports
def _split_heads_linear(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """nn.Linear weight [n_heads*d_head, d_model] -> [n_heads, d_model, d_head]"""
    d_out, d_in = w.shape
    return w.reshape(n_heads, d_out // n_heads, d_in).transpose(1, 2).contiguous()


def convert_gpt_neo(sd: dict, hf: dict, n_ctx: int | None = None) -> tuple[Config, Params]:
    d, L, H = hf["hidden_size"], hf["num_layers"], hf["num_heads"]
    # layers alternate global / local(window_size) attention; within window_size tokens the two are identical
    n_ctx = min(n_ctx or hf["window_size"], hf["window_size"], hf["max_position_embeddings"])
    cfg = Config(d_vocab=hf["vocab_size"], n_ctx=n_ctx, d_model=d, n_layers=L, n_heads=H, d_mlp=hf.get("intermediate_size") or 4 * d,
                 act_fn=hf["activation_function"], attn_scale=1.0, ln_eps=hf["layer_norm_epsilon"], family="gpt-neo")
    g = lambda k: sd["transformer." + k].float()  # noqa: E731
    p: Params = {"embed.W_E": g("wte.weight").clone(), "pos_embed.W_pos": g("wpe.weight")[:n_ctx].clone()}
    for l in range(L):
        a, b = f"h.{l}.attn.attention", f"blocks.{l}"
        p[f"{b}.ln1.w"], p[f"{b}.ln1.b"] = g(f"h.{l}.ln_1.weight"), g(f"h.{l}.ln_1.bias")
        for m, src in (("Q", "q_proj"), ("K", "k_proj"), ("V", "v_proj")):
            p[f"{b}.attn.W_{m}"] = _split_heads_linear(g(f"{a}.{src}.weight"), H)
            p[f"{b}.attn.b_{m}"] = torch.zeros(H, d // H)
        p[f"{b}.attn.W_O"] = g(f"{a}.out_proj.weight").T.reshape(H, d // H, d).contiguous()
        p[f"{b}.attn.b_O"] = g(f"{a}.out_proj.bias")
        p[f"{b}.ln2.w"], p[f"{b}.ln2.b"] = g(f"h.{l}.ln_2.weight"), g(f"h.{l}.ln_2.bias")
        p[f"{b}.mlp.W_in"], p[f"{b}.mlp.b_in"] = g(f"h.{l}.mlp.c_fc.weight").T.contiguous(), g(f"h.{l}.mlp.c_fc.bias")
        p[f"{b}.mlp.W_out"], p[f"{b}.mlp.b_out"] = g(f"h.{l}.mlp.c_proj.weight").T.contiguous(), g(f"h.{l}.mlp.c_proj.bias")
    p["ln_final.w"], p["ln_final.b"] = g("ln_f.weight"), g("ln_f.bias")
    p["unembed.W_U"] = (sd["lm_head.weight"].float() if "lm_head.weight" in sd else g("wte.weight")).T.contiguous()
    p["unembed.b_U"] = torch.zeros(cfg.d_vocab)
    return cfg, {k: v.contiguous() for k, v in p.items()}


def convert_gpt2(sd: dict, hf: dict, n_ctx: int | None = None) -> tuple[Config, Params]:
    d, L, H = hf["n_embd"], hf["n_layer"], hf["n_head"]
    n_ctx = min(n_ctx or hf["n_positions"], hf["n_positions"])
    cfg = Config(d_vocab=hf["vocab_size"], n_ctx=n_ctx, d_model=d, n_layers=L, n_heads=H, d_mlp=4 * d,
                 act_fn="gelu_new", ln_eps=hf["layer_norm_epsilon"], family="gpt2")
    pre = "transformer." if any(k.startswith("transformer.") for k in sd) else ""
    g = lambda k: sd[pre + k].float()  # noqa: E731
    dh = d // H
    p: Params = {"embed.W_E": g("wte.weight").clone(), "pos_embed.W_pos": g("wpe.weight")[:n_ctx].clone()}
    for l in range(L):
        b = f"blocks.{l}"
        p[f"{b}.ln1.w"], p[f"{b}.ln1.b"] = g(f"h.{l}.ln_1.weight"), g(f"h.{l}.ln_1.bias")
        ca_w, ca_b = g(f"h.{l}.attn.c_attn.weight"), g(f"h.{l}.attn.c_attn.bias")   # Conv1D: [d_in, 3d]
        for i, m in enumerate("QKV"):
            p[f"{b}.attn.W_{m}"] = ca_w[:, i * d:(i + 1) * d].reshape(d, H, dh).permute(1, 0, 2).contiguous()
            p[f"{b}.attn.b_{m}"] = ca_b[i * d:(i + 1) * d].reshape(H, dh).contiguous()
        p[f"{b}.attn.W_O"] = g(f"h.{l}.attn.c_proj.weight").reshape(H, dh, d).contiguous()
        p[f"{b}.attn.b_O"] = g(f"h.{l}.attn.c_proj.bias")
        p[f"{b}.ln2.w"], p[f"{b}.ln2.b"] = g(f"h.{l}.ln_2.weight"), g(f"h.{l}.ln_2.bias")
        p[f"{b}.mlp.W_in"], p[f"{b}.mlp.b_in"] = g(f"h.{l}.mlp.c_fc.weight"), g(f"h.{l}.mlp.c_fc.bias")
        p[f"{b}.mlp.W_out"], p[f"{b}.mlp.b_out"] = g(f"h.{l}.mlp.c_proj.weight"), g(f"h.{l}.mlp.c_proj.bias")
    p["ln_final.w"], p["ln_final.b"] = g("ln_f.weight"), g("ln_f.bias")
    p["unembed.W_U"] = g("wte.weight").T.contiguous()
    p["unembed.b_U"] = torch.zeros(cfg.d_vocab)
    return cfg, {k: v.contiguous() for k, v in p.items()}


def build_hf(directory: str, repo: str, name: str, cache: str = "", n_ctx: int | None = None, log=print) -> Workspace:
    """Import a GPT-2 or GPT-Neo checkpoint from the Hugging Face hub."""
    cdir = os.path.join(cache or directory + "-cache", repo.replace("/", "--"))
    hf = json.load(open(_fetch(HF.format(repo=repo, file="config.json"), os.path.join(cdir, "config.json"))))
    tok = BPETokenizer(open(_fetch(HF.format(repo=repo, file="tokenizer.json"), os.path.join(cdir, "tokenizer.json"))).read())
    try:
        from safetensors.torch import load_file
        sd = load_file(_fetch(HF.format(repo=repo, file="model.safetensors"), os.path.join(cdir, "model.safetensors")))
    except Exception:
        log("no model.safetensors; reading pytorch_model.bin (weights only)")
        sd = torch.load(_fetch(HF.format(repo=repo, file="pytorch_model.bin"), os.path.join(cdir, "pytorch_model.bin")),
                        map_location="cpu", weights_only=True)
    cfg, params = (convert_gpt_neo if hf.get("model_type") == "gpt_neo" else convert_gpt2)(sd, hf, n_ctx)
    cfg.name = name
    notes = (f"Imported from {repo}. Token and unembedding matrices were tied in the checkpoint; here they are "
             f"separate tensors (embed.W_E, unembed.W_U), so editing one does not change the other.\n"
             f"Context is {cfg.n_ctx} tokens.\n")
    return Workspace.create(directory, cfg, tok, params, notes)


def build_tinystories(directory: str, cache: str = "", log=print) -> Workspace:
    ws = build_hf(directory, "roneneldan/TinyStories-1M", "tinystories-1m", cache, log=log)
    ws.write_json("tests/stories.json", {"name": "stories", "description": "things the stock model gets right", "cases": [
        {"type": "next_token", "prompt": "Once upon a", "expect": " time", "name": "once upon a time"},
        {"type": "next_token", "prompt": "Once upon a time, there was a little girl named", "expect": [" Lily", " Lucy", " Sue", " Mia", " Sara"], "top_k": 5,
         "name": "a girl's name follows 'named'"},
        {"type": "next_token", "prompt": "One day, a little boy named Tim went to the", "expect": [" park", " store", " zoo", " beach", " forest"], "top_k": 10,
         "name": "a place follows 'went to the'"},
        {"type": "completion", "prompt": "Once upon a time", "max_tokens": 30, "matches": "[A-Za-z]{3,}", "name": "produces words"},
    ]})
    return ws


# ------------------------------------------------------------------------------ blank models (program from scratch)
def build_blank(directory: str, vocab: list[str], n_ctx: int, d_model: int, n_layers: int, n_heads: int,
                d_mlp: int = 0, attn_only: bool = False, act_fn: str = "relu", mode: str = "word", name: str = "blank",
                notes: str = "", d_head: int = 0) -> Workspace:
    tok = VocabTokenizer(vocab, mode)
    cfg = Config(d_vocab=len(vocab), n_ctx=n_ctx, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_head=d_head,
                 d_mlp=d_mlp, attn_only=attn_only, act_fn=act_fn, normalization="none", name=name, family="handwritten")
    return Workspace.create(directory, cfg, tok, init_params(cfg, zero=True), notes)


REVERSE_N = 4
REVERSE_SYMS = ["a", "b", "c", "d", "e"]


def build_blank_reverse(directory: str) -> Workspace:
    """`<bos> x1 x2 x3 x4 | x4 x3 x2 x1`: after the bar, write the sequence backwards."""
    vocab = ["<bos>", "|"] + REVERSE_SYMS
    n = REVERSE_N
    notes = (f"Task: sequences look like `<bos> x1 x2 x3 x4 | x4 x3 x2 x1` with symbols from {REVERSE_SYMS}.\n"
             f"Position 0 is <bos>, positions 1-{n} the input, position {n + 1} the bar, positions {n + 2}-{2 * n + 1} the "
             f"reversed output. The model must predict each output symbol.\n"
             "All weights are zero and there is no LayerNorm, so the forward pass is exactly\n"
             "  x = W_E[tok] + W_pos[pos];  x += attention(x);  logits = x @ W_U + b_U.\n")
    ws = build_blank(directory, vocab, 2 * n + 2, 24, 1, 1, attn_only=True, name="blank-reverse", notes=notes, d_head=12)
    rng = random.Random(0)
    cases = []
    for _ in range(40):
        xs = [rng.choice(REVERSE_SYMS) for _ in range(n)]
        for j in range(n):
            prompt = " ".join(["<bos>"] + xs + ["|"] + xs[::-1][:j])
            cases.append({"type": "next_token", "prompt": prompt, "expect": xs[::-1][j]})
    ws.write_json("tests/reverse.json", {"name": "reverse", "description": "reverse a 4-symbol sequence", "cases": cases})
    ws.write_json("task.json", {
        "title": "Write a sequence reverser",
        "instructions": "The model's weights are all zero. Write weights so that it reverses a 4-symbol sequence: given "
                        "`<bos> x1 x2 x3 x4 |` it must continue `x4 x3 x2 x1`. One attention head is enough: each output "
                        "position should attend to the mirrored input position and copy its symbol to the output.",
        "scope": ["model"], "suite": "reverse", "preserve": None})
    return ws


MODADD_P = 7


def build_blank_modadd(directory: str) -> Workspace:
    """`a + b =` -> (a + b) mod 7."""
    p = MODADD_P
    vocab = [str(i) for i in range(p)] + ["+", "="]
    notes = (f"Task: `a + b =` must be followed by (a + b) mod {p}. Tokens: the digits 0-{p - 1}, `+`, `=`.\n"
             "Positions: 0 = a, 1 = `+`, 2 = b, 3 = `=`. Only the prediction at position 3 is tested.\n"
             "All weights are zero and there is no LayerNorm; the MLP uses ReLU, so the forward pass is exactly\n"
             "  x = W_E[tok] + W_pos[pos];  x += attention(x);  x += relu(x @ W_in + b_in) @ W_out + b_out;  logits = x @ W_U + b_U.\n")
    ws = build_blank(directory, vocab, 4, 32, 1, 2, d_mlp=p * p, name="blank-modadd", notes=notes, d_head=8)
    cases = [{"type": "next_token", "prompt": f"{a} + {b} =", "expect": str((a + b) % p)} for a, b in
             itertools.product(range(p), repeat=2)]
    ws.write_json("tests/modadd.json", {"name": "modadd", "description": f"all {p * p} sums mod {p}", "cases": cases})
    ws.write_json("task.json", {
        "title": f"Write addition mod {p}",
        "instructions": f"The model's weights are all zero. Write weights so that `a + b =` is followed by (a + b) mod {p} "
                        f"for every a and b in 0..{p - 1}. Suggested plan: let the two heads move a and b to the `=` position "
                        f"into separate residual subspaces, then use the {p * p} MLP neurons as a lookup table.",
        "scope": ["model"], "suite": "modadd", "preserve": None})
    return ws


# ------------------------------------------------------------------------------ any checkpoint, in its native layout
def slug(spec: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", spec.rstrip("/").split("/")[-1].lower()).strip("-") or "model"


def _workspace_for(root: str, base: str, source: dict) -> tuple[str, bool]:
    """A workspace name for a checkpoint: the existing workspace over the same source if there is one, else a free name."""
    name, k = base, 1
    while True:
        mp = os.path.join(root, name, "workspace.json")
        if not os.path.exists(mp):
            if not os.path.exists(os.path.join(root, name)):
                return name, False
        else:
            with open(mp) as f:
                m = json.load(f)
            here = {"repo": m.get("repo"), "dir": os.path.normpath(os.path.join(root, name, m.get("model_dir") or ""))}
            if (source.get("repo") and here["repo"] == source["repo"]) or \
                    (source.get("dir") and not m.get("repo") and os.path.abspath(here["dir"]) == os.path.abspath(source["dir"])):
                return name, True
        k += 1
        name = f"{base}-{k}"


def open_hf(root: str, spec: str, name: str | None = None, pull: bool = False, log=print) -> Workspace:
    """A workspace over any safetensors checkpoint, without converting or loading it.

    spec is a Hub repo id ("Qwen/Qwen2.5-0.5B-Instruct", optionally "@revision") or a local directory in
    Hugging Face layout. From the Hub only config, tokenizer and the weight files' headers are fetched;
    tensors are then read on demand. `pull=True` downloads the weights (needed to run the model).
    Opening the same checkpoint again returns the workspace that already exists for it."""
    from . import hub
    spec = spec[3:] if spec.startswith("hf:") else spec
    if os.path.isdir(spec):
        model_dir = os.path.abspath(spec)
        name, exists = (name, os.path.exists(os.path.join(root, name, "workspace.json"))) if name else \
            _workspace_for(root, slug(model_dir), {"dir": model_dir})
        if exists:
            return Workspace.load(os.path.join(root, name))
        return Workspace.create_hf(os.path.join(root, name), name, model_dir,
                                   notes=f"Local checkpoint at {model_dir}, in its native layout.\n")
    repo, _, rev = spec.partition("@")
    name, exists = (name, os.path.exists(os.path.join(root, name, "workspace.json"))) if name else \
        _workspace_for(root, slug(repo), {"repo": repo})
    directory = os.path.join(root, name)
    if exists:
        ws = Workspace.load(directory)
        return globals()["pull"](ws, log) if pull and ws.store.remote else ws
    model_dir = os.path.join(directory, "model")
    try:
        info = hub.resolve(repo, model_dir, rev or "main")
    except FileNotFoundError:
        info = _convert_bin(repo, model_dir, rev or "main", log)
    if pull:
        hub.pull(repo, info["sha"], info["weights"], model_dir, log)
    return Workspace.create_hf(directory, name, model_dir, repo, info["sha"], info["weights"], info.get("urls"),
                               notes=f"{repo} at revision {info['sha'][:12]}, in its native layout.\n")


def _convert_bin(repo: str, model_dir: str, rev: str, log=print) -> dict:
    """Repos that only ship pytorch_model.bin: download it and rewrite it as safetensors (whole file in memory, once)."""
    from safetensors.torch import save_file

    from . import hub
    log(f"{repo} has no safetensors; converting pytorch_model.bin")
    src = hub.fetch(hub.URL.format(repo=repo, rev=rev, file="pytorch_model.bin"), os.path.join(model_dir, "pytorch_model.bin"))
    for f in hub.SMALL:
        try:
            hub.fetch(hub.URL.format(repo=repo, rev=rev, file=f), os.path.join(model_dir, f))
        except Exception:
            pass
    sd = torch.load(src, map_location="cpu", weights_only=True)
    save_file({k: v.clone().contiguous() for k, v in sd.items() if isinstance(v, torch.Tensor)}, os.path.join(model_dir, "model.safetensors"))
    os.remove(src)
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    return {"repo": repo, "sha": rev, "config": config, "weights": ["model.safetensors"], "urls": None}


def pull(ws: Workspace, log=print) -> Workspace:
    """Download the weights of a Hub-backed workspace so that it can be run."""
    from . import hub
    if ws.kind != "hf" or not ws.meta.get("repo"):
        return ws
    hub.pull(ws.meta["repo"], ws.meta["sha"], ws.meta["weights"], ws.model_dir, log)
    return Workspace.load(ws.dir)


BUILDERS = {
    "facts": build_facts,
    "shakespeare-char": build_shakespeare,
    "tinystories-1m": build_tinystories,
    "gpt2": lambda d, cache="", log=print: build_hf(d, "openai-community/gpt2", "gpt2", cache, n_ctx=256, log=log),
    "blank-reverse": build_blank_reverse,
    "blank-modadd": build_blank_modadd,
}
DESCRIPTIONS = {
    "facts": "trained here: 2-layer word-level model of a synthetic knowledge base (crisp facts to edit)",
    "shakespeare-char": "trained here: 3-layer character-level tiny-shakespeare",
    "tinystories-1m": "imported: roneneldan/TinyStories-1M, 8 layers, d_model 64, real English",
    "gpt2": "imported: GPT-2 small, 124M parameters",
    "blank-reverse": "all-zero weights + a test suite: write a sequence reverser by hand",
    "blank-modadd": "all-zero weights + a test suite: write addition mod 7 by hand",
}


def build(name: str, root: str, log=print, **kw) -> Workspace:
    if name not in BUILDERS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(BUILDERS)}")
    directory = os.path.join(root, name)
    if os.path.exists(os.path.join(directory, "config.json")):
        return Workspace.load(directory)
    fn = BUILDERS[name]
    import inspect
    args = inspect.signature(fn).parameters
    if "cache" in args:
        kw.setdefault("cache", os.path.join(root, ".cache"))
    if "log" in args:
        kw.setdefault("log", log)
    return fn(directory, **kw)
