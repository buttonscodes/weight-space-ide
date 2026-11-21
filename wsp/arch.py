"""Structure of a checkpoint: which tensors are the embeddings, the heads, the neurons.

Weights are kept exactly as the checkpoint stores them (native names, native layout). What
differs between architectures is described once, here, as a `Structure`: a list of *parts*,
each saying "these axes of this tensor (or of this slice of a fused tensor) play this role".
Everything above (the hierarchy, head and neuron addresses, cards, chunk plans, portable
weight scripts) is written against roles, so it works for any model that is recognised:

    family            attention weights                              MLP weights
    wsp               W_Q [head, d_model, d_head] ... (this project's own small models)
    gpt2              attn.c_attn [d, 3d] (Q|K|V fused, Conv1D)      mlp.c_fc [d, 4d], mlp.c_proj [4d, d]
    gpt-neo           attn.attention.{q,k,v}_proj, out_proj          mlp.c_fc [4d, d], mlp.c_proj
    gpt-neox/pythia   attention.query_key_value [3d, d] (per-head)   mlp.dense_h_to_4h, dense_4h_to_h
    llama-like        self_attn.{q,k,v,o}_proj, grouped K/V heads    mlp.{gate,up,down}_proj (gated)
                      (Llama, Mistral, Qwen, Gemma, SmolLM, TinyLlama, DeepSeek, Yi, ...)
    phi-3             self_attn.qkv_proj (fused), o_proj             mlp.gate_up_proj (fused), down_proj

Detection is by tensor names and shapes plus config.json, and is per layer: a layer whose
attention is not recognised (for example a recurrent or state-space mixer) still gets its MLP
neurons, and its other tensors stay reachable by name.  A checkpoint that matches nothing is
still fully usable at the tensor level through the name tree (`raw:` addresses).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import runtime_layouts

CANON = {"W_E": ("token", "resid"), "W_pos": ("pos", "resid"), "W_U": ("resid", "token"),
         "W_Q": ("resid", "dhead"), "W_K": ("resid", "dhead"), "W_V": ("resid", "dhead"), "W_O": ("dhead", "resid"),
         "W_in": ("resid", "neuron"), "W_gate": ("resid", "neuron"), "W_up": ("resid", "neuron"),
         "W_out": ("neuron", "resid")}
DOC = {
    "W_E": "token embedding: the row of token t is what t writes into the residual stream at its own position",
    "W_pos": "learned absolute position embedding: the row of position p is added at position p",
    "W_U": "unembedding: the vector of token t is the residual direction that raises the logit of t",
    "b_U": "per-token logit bias",
    "W_Q": "query read: the destination position's residual is projected to each head's query",
    "W_K": "key read: each source position's residual is projected to a key; attention score = query . key (scaled)",
    "W_V": "value read: what a head picks up from the source position it attends to",
    "W_O": "output write: maps each head's mixed value back into the residual stream",
    "b_Q": "query bias", "b_K": "key bias", "b_V": "value bias", "b_O": "attention output bias (added at every position)",
    "W_in": "MLP read: neuron n's pre-activation is x . (its read vector) + bias",
    "W_gate": "gated MLP, gate read: neuron n's activation is act(x . gate_n) * (x . up_n)",
    "W_up": "gated MLP, value read: neuron n's activation is act(x . gate_n) * (x . up_n)",
    "b_in": "MLP neuron biases", "b_gate": "gate biases", "b_up": "value-read biases",
    "W_out": "MLP write: neuron n adds activation * (its write vector) to the residual stream",
    "b_out": "MLP output bias (added at every position)",
    "norm_w": "normalisation gain per residual coordinate", "norm_b": "normalisation bias per residual coordinate",
}


@dataclass
class Part:
    name: str                              # tensor name in the checkpoint
    role: str                              # W_Q, W_out, b_in, norm_w, ...
    roles: tuple                           # role of each axis of tensor[base]: token pos resid head dhead headdh neuron other
    base: tuple = ()                       # basic index selecting this part inside a fused tensor
    head: tuple | None = None              # for a 'headdh' axis: (stride, offset, size, n): head h is offset+h*stride : +size

    def head_index(self, h: int, n_heads: int):
        """Index on the head-bearing axis for head h (heads may share K/V: n < n_heads)."""
        if "head" in self.roles:
            return h
        stride, offset, size, n = self.head
        g = h * n // n_heads
        return slice(offset + g * stride, offset + g * stride + size)


@dataclass
class Layer:
    index: int
    prefix: str = ""                                   # e.g. model.layers.3
    attn: dict = field(default_factory=dict)           # role -> Part   (W_Q b_Q W_K b_K W_V b_V W_O b_O)
    mlp: dict = field(default_factory=dict)            # role -> Part   (W_in | W_gate W_up, b_*, W_out, b_out)
    norms: list = field(default_factory=list)          # Part
    # Tensor names of this layer outside attn / mlp / norms: the unclassified ones, and also the tensors of the
    # mixer / experts / shared parts below, so that addressing, scopes and the name tree reach them as before.
    other: list = field(default_factory=list)
    modules: dict = field(default_factory=dict)        # layer / attn / o_proj / mlp / mlp_out / mixer_out / experts /
                                                       # shared_out module paths (for hooks)
    mixer: dict = field(default_factory=dict)          # role -> Part: a token mixer without heads (linear attention)
    experts: list = field(default_factory=list)        # per routed expert, in index order: role -> Part
    shared: dict = field(default_factory=dict)         # role -> Part: the shared expert

    @property
    def reads(self) -> list[str]:
        return [r for r in ("W_in", "W_gate", "W_up") if r in self.mlp]


@dataclass
class Structure:
    family: str = "generic"
    runner: str = "none"                   # wsp | hf | none
    n_layers: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    d_model: int = 0
    d_head: int = 0
    d_mlp: int = 0
    d_vocab: int = 0
    n_ctx: int = 0
    norm: str = "none"                     # ln | rms | none
    act: str = ""
    embed: Part | None = None
    pos: Part | None = None
    unembed: Part | None = None            # tied models: the embedding tensor again, with role W_U
    unembed_bias: Part | None = None
    tied: bool = False
    final_norm: list = field(default_factory=list)
    layers: list = field(default_factory=list)
    other: list = field(default_factory=list)          # tensor names outside the recognised structure
    parts: dict = field(default_factory=dict)          # tensor name -> [Part]
    modules: dict = field(default_factory=dict)        # final_norm / embed module paths
    notes: list = field(default_factory=list)
    shapes: dict = field(default_factory=dict)         # tensor name -> shape (every tensor in the checkpoint)

    # ------------------------------------------------------------------ building
    def add(self, part: Part) -> Part:
        self.parts.setdefault(part.name, []).append(part)
        return part

    # ------------------------------------------------------------------ queries
    @property
    def recognised(self) -> bool:
        return bool(self.layers) and self.embed is not None

    def has_heads(self, l: int) -> bool:
        return all(r in self.layers[l].attn for r in ("W_Q", "W_K", "W_V", "W_O"))

    def has_neurons(self, l: int) -> bool:
        L = self.layers[l]
        return "W_out" in L.mlp and bool(L.reads)

    def mlp_width(self, l: int) -> int:
        return self._axis_len(self.layers[l].mlp["W_out"], "neuron")

    def _axis_len(self, part: Part, role: str) -> int:
        return self._shape(part)[part.roles.index(role)]

    def _shape(self, part: Part) -> tuple:
        return self.shapes[part.name] if not part.base else _sliced(self.shapes[part.name], part.base)

    def head_slices(self, l: int, h: int) -> list[tuple[str, tuple]]:
        """Every (tensor, index) that belongs to head h: its Q and O, and the K / V it uses."""
        out = []
        for role in ("W_Q", "b_Q", "W_K", "b_K", "W_V", "b_V", "W_O"):
            p = self.layers[l].attn.get(role)
            if p is not None:
                out.append((p.name, self.part_index(p, head=h)))
        return out

    def neuron_slices(self, l: int, n) -> list[tuple[str, tuple]]:
        """Every (tensor, index) that belongs to neuron n (an int, or a slice for a run of neurons)."""
        out = []
        L = self.layers[l]
        for role in ("W_in", "b_in", "W_gate", "b_gate", "W_up", "b_up", "W_out"):
            p = L.mlp.get(role)
            if p is not None:
                out.append((p.name, self.part_index(p, neuron=n)))
        return out

    def part_index(self, part: Part, head: int | None = None, neuron=None) -> tuple:
        """Index into the native tensor for a part, optionally narrowed to one head or neuron(s)."""
        base = list(part.base) + [slice(None)] * (len(self.shapes[part.name]) - len(part.base))
        kept = [d for d, i in enumerate(base) if isinstance(i, slice)]
        for k, role in enumerate(part.roles):
            d = kept[k]
            if head is not None and role in ("head", "headdh"):
                sub = part.head_index(head, self.n_heads)
                base[d] = _compose(base[d], sub, self.shapes[part.name][d])
            if neuron is not None and role == "neuron":
                base[d] = _compose(base[d], neuron, self.shapes[part.name][d])
        while base and base[-1] == slice(None):
            base.pop()
        return tuple(base)

    def view_roles(self, part: Part, head: bool = False, neuron: bool = False) -> list[str]:
        """Axis roles of tensor[part_index(...)] (axes dropped by an int index removed, headdh -> dhead)."""
        out = []
        for role in part.roles:
            if head and role == "head":
                continue
            if neuron and role == "neuron":
                continue
            out.append("dhead" if head and role == "headdh" else role)
        return out

    def axis_roles(self, name: str) -> list[str]:
        """Roles of a native tensor's axes, for labelling views."""
        ps = self.parts.get(name) or []
        nd = len(self.shapes[name])
        if len(ps) == 1 and not ps[0].base:
            return list(ps[0].roles)
        if ps:                                              # fused tensor: axes that every part agrees on
            first = ps[0]
            base = list(first.base) + [slice(None)] * (nd - len(first.base))
            roles = []
            for d in range(nd):
                same = all(_axis_untouched(p, d) for p in ps)
                roles.append(first.roles[_kept_pos(base, d)] if same else "fused")
            return roles
        return [f"axis{d}" for d in range(nd)]

    def doc(self, name: str) -> str:
        ps = self.parts.get(name) or []
        if not ps:
            return ""
        if len(ps) == 1:
            p = ps[0]
            return f"{DOC.get(p.role, p.role)}. Axes: {', '.join(p.roles)}."
        return "fused tensor holding " + "; ".join(f"{p.role} at {_index_str(p.base)}" for p in ps)

    def describe(self) -> str:
        """The architecture in a few lines (what the programmer model is told)."""
        if not self.recognised:
            return ("The checkpoint's architecture was not recognised: tensors are available by name only "
                    "(no head / neuron addresses, no vocabulary readings).")
        gated = self.layers and "W_gate" in self.layers[-1].mlp
        heads = f"{self.n_heads} heads of size {self.d_head}" + (
            f" (keys and values shared in {self.n_kv_heads} groups)" if self.n_kv_heads and self.n_kv_heads != self.n_heads else "")
        mlp = (f"gated MLP of {self.d_mlp} neurons: act(x . gate_n) * (x . up_n) scales neuron n's write vector" if gated
               else f"MLP of {self.d_mlp} neurons: act(x . in_n + b_n) scales neuron n's write vector") if self.d_mlp else "no MLP"
        lines = [f"family {self.family}: {self.n_layers} layers, d_model {self.d_model}, {heads}, {mlp}, vocabulary {self.d_vocab}"
                 + (f", context {self.n_ctx}" if self.n_ctx else "") + f", normalisation {self.norm}"
                 + (", learned absolute positions" if self.pos else ", positions enter inside attention (no position-embedding tensor)")
                 + (", unembedding tied to the embedding (one tensor; an edit changes both)" if self.tied else "") + "."]
        lines += self.notes
        return "\n".join(lines)


def _sliced(shape, base) -> tuple:
    out = []
    for d, n in enumerate(shape):
        i = base[d] if d < len(base) else slice(None)
        if isinstance(i, slice):
            out.append(len(range(*i.indices(n))))
    return tuple(out)


def _compose(outer, inner, n: int):
    """Index `inner` applied to the axis selected by slice `outer` of an axis of length n."""
    r = range(*outer.indices(n))
    if isinstance(inner, slice):
        s = r[inner]
        return slice(s.start, s.stop if s.stop >= 0 else None, s.step) if len(s) else slice(0, 0)
    return r[int(inner)]


def _kept_pos(base, d: int) -> int:
    return sum(isinstance(i, slice) for i in base[:d])


def _axis_untouched(p: Part, d: int) -> bool:
    return d >= len(p.base) or p.base[d] == slice(None)


def _index_str(idx) -> str:
    parts = []
    for i in idx:
        parts.append(":" if i == slice(None) else f"{'' if i.start is None else i.start}:{'' if i.stop is None else i.stop}"
                     if isinstance(i, slice) else str(i))
    return "[" + ", ".join(parts) + "]"


# -------------------------------------------------------------------------------------- this project's own layout
def wsp_structure(cfg, shapes: dict) -> Structure:
    s = Structure(family="wsp", runner="wsp", n_layers=cfg.n_layers, n_heads=cfg.n_heads, n_kv_heads=cfg.n_heads,
                  d_model=cfg.d_model, d_head=cfg.d_head, d_mlp=cfg.d_mlp, d_vocab=cfg.d_vocab, n_ctx=cfg.n_ctx,
                  norm="ln" if cfg.use_ln else "none", act=cfg.act_fn)
    s.shapes = shapes
    s.embed = s.add(Part("embed.W_E", "W_E", ("token", "resid")))
    s.pos = s.add(Part("pos_embed.W_pos", "W_pos", ("pos", "resid")))
    s.unembed = s.add(Part("unembed.W_U", "W_U", ("resid", "token")))
    s.unembed_bias = s.add(Part("unembed.b_U", "b_U", ("token",)))
    if cfg.use_ln:
        s.final_norm = [s.add(Part("ln_final.w", "norm_w", ("resid",))), s.add(Part("ln_final.b", "norm_b", ("resid",)))]
    for l in range(cfg.n_layers):
        b = f"blocks.{l}"
        L = Layer(l, b)
        for m in "QKV":
            L.attn[f"W_{m}"] = s.add(Part(f"{b}.attn.W_{m}", f"W_{m}", ("head", "resid", "dhead")))
            L.attn[f"b_{m}"] = s.add(Part(f"{b}.attn.b_{m}", f"b_{m}", ("head", "dhead")))
        L.attn["W_O"] = s.add(Part(f"{b}.attn.W_O", "W_O", ("head", "dhead", "resid")))
        L.attn["b_O"] = s.add(Part(f"{b}.attn.b_O", "b_O", ("resid",)))
        if not cfg.attn_only:
            L.mlp["W_in"] = s.add(Part(f"{b}.mlp.W_in", "W_in", ("resid", "neuron")))
            L.mlp["b_in"] = s.add(Part(f"{b}.mlp.b_in", "b_in", ("neuron",)))
            L.mlp["W_out"] = s.add(Part(f"{b}.mlp.W_out", "W_out", ("neuron", "resid")))
            L.mlp["b_out"] = s.add(Part(f"{b}.mlp.b_out", "b_out", ("resid",)))
        if cfg.use_ln:
            for ln in ("ln1", "ln2")[:1 if cfg.attn_only else 2]:
                L.norms += [s.add(Part(f"{b}.{ln}.w", "norm_w", ("resid",))), s.add(Part(f"{b}.{ln}.b", "norm_b", ("resid",)))]
        s.layers.append(L)
    return s


# -------------------------------------------------------------------------------------- detection for native checkpoints
_LAYER = re.compile(r"^(?P<pre>(?:.*\.)?(?:layers|h|layer|blocks|block))\.(?P<i>\d+)\.(?P<rest>.+)$")
_ATT = r"(?:self_attn|attn\.attention|attention|attn)"
_ATTN_RULES = [
    (re.compile(rf"^({_ATT})\.q_proj\.(weight|bias)$"), "Q"),
    (re.compile(rf"^({_ATT})\.k_proj\.(weight|bias)$"), "K"),
    (re.compile(rf"^({_ATT})\.v_proj\.(weight|bias)$"), "V"),
    (re.compile(rf"^({_ATT})\.(?:o_proj|out_proj)\.(weight|bias)$"), "O"),
    (re.compile(r"^(attention|self_attn)\.dense\.(weight|bias)$"), "O"),
    (re.compile(r"^(attn)\.c_proj\.(weight|bias)$"), "O"),
    (re.compile(r"^(attn)\.c_attn\.(weight|bias)$"), "QKV:gpt2"),
    (re.compile(r"^(attention)\.query_key_value\.(weight|bias)$"), "QKV:neox"),
    (re.compile(r"^(self_attn)\.qkv_proj\.(weight|bias)$"), "QKV:phi3"),
]
_MLP = r"(?:mlp|feed_forward|ffn)"
_MLP_RULES = [
    (re.compile(rf"^({_MLP})\.gate_proj\.(weight|bias)$"), "gate"),
    (re.compile(rf"^({_MLP})\.up_proj\.(weight|bias)$"), "up"),
    (re.compile(rf"^({_MLP})\.down_proj\.(weight|bias)$"), "out"),
    (re.compile(rf"^({_MLP})\.(?:c_fc|dense_h_to_4h|fc1|fc_in)\.(weight|bias)$"), "in"),
    (re.compile(rf"^({_MLP})\.(?:c_proj|dense_4h_to_h|fc2|fc_out)\.(weight|bias)$"), "out"),
    (re.compile(rf"^({_MLP})\.gate_up_proj\.(weight|bias)$"), "gate_up"),
    (re.compile(rf"^({_MLP})\.w1\.(weight|bias)$"), "gate"),
    (re.compile(rf"^({_MLP})\.w3\.(weight|bias)$"), "up"),
    (re.compile(rf"^({_MLP})\.w2\.(weight|bias)$"), "out"),
]
# A token mixer without attention heads (Qwen3.5's Gated DeltaNet): its output projection.
_MIXER_RULES = [
    (re.compile(r"^(linear_attn)\.out_proj\.(weight|bias)$"), "O"),
]
# Mixture-of-experts layers as checkpoints store them: one tensor per routed expert, and a shared expert.
_EXPERT_RULES = [
    (re.compile(rf"^({_MLP})\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"), "expert"),
]
_SHARED_RULES = [
    (re.compile(rf"^({_MLP})\.(shared_experts?)\.(gate_proj|up_proj|down_proj)\.weight$"), "shared"),
]
_EXPERT_ROLES = {"gate_proj": ("W_gate", ("neuron", "resid")), "up_proj": ("W_up", ("neuron", "resid")),
                 "down_proj": ("W_out", ("resid", "neuron"))}
_STACK_RULES = _ATTN_RULES + _MLP_RULES + _MIXER_RULES + _EXPERT_RULES + _SHARED_RULES
_EMBED = re.compile(r"(?:^|\.)(?:embed_tokens|wte|embed_in|tok_embeddings|word_embeddings)\.weight$")
_POS = re.compile(r"(?:^|\.)(?:wpe|embed_positions|position_embeddings)\.weight$")
_HEAD = re.compile(r"(?:^|\.)(?:lm_head|embed_out)\.(weight|bias)$")
_FNORM = re.compile(r"(?:^|\.)(?:norm|ln_f|final_layer_norm|final_layernorm|norm_f|final_norm)\.(weight|bias)$")


def _cfg(config: dict, *keys, default=None):
    for scope in (config, config.get("text_config") or {}, config.get("llm_config") or {}):
        for k in keys:
            if scope.get(k) is not None:
                return scope[k]
    return default


def detect(shapes: dict[str, tuple], config: dict | None = None) -> Structure:
    """Recognise a decoder language model from its tensor names and shapes (and config.json if there is one)."""
    config = config or {}
    s = Structure(family="generic", runner="hf" if config.get("model_type") or config.get("architectures") else "none")
    s.shapes = shapes
    groups: dict[str, dict[int, dict[str, str]]] = {}
    for name in shapes:
        m = _LAYER.match(name)
        if m:
            groups.setdefault(m.group("pre"), {}).setdefault(int(m.group("i")), {})[m.group("rest")] = name
    best, best_score = None, 0
    for pre, layers in groups.items():
        # Counted over every rule list: a MoE layer whose mixer is linear attention matches none of the dense rules.
        hits = sum(1 for rest in layers.get(min(layers), {}) if any(r.match(rest) for r, _ in _STACK_RULES))
        score = hits * len(layers)
        if re.search(r"vision|visual|audio|image|encoder|vit", pre):
            score *= 0.1
        if score > best_score:
            best, best_score = pre, score
    used: set[str] = set()
    if best is None:
        s.other = list(shapes)
        s.notes.append("no layer stack with recognisable attention or MLP weights was found")
        return s
    root = best.rsplit(".", 1)[0] + "." if "." in best else ""       # e.g. "model." for model.layers

    def pick(rx, two_d=True):
        c = [n for n in shapes if rx.search(n) and n not in used and (not two_d or len(shapes[n]) == 2)]
        c.sort(key=lambda n: (not n.startswith(root), len(n)))
        return c[0] if c else None

    en = pick(_EMBED)
    if en is None:
        s.other = list(shapes)
        s.notes.append("a layer stack was found but no token-embedding tensor")
        return s
    used.add(en)
    s.d_vocab, s.d_model = shapes[en]
    d = s.d_model
    s.embed = s.add(Part(en, "W_E", ("token", "resid")))
    s.modules["embed"] = en.rsplit(".", 1)[0]
    pn = pick(_POS)
    if pn and shapes[pn][1] == d:
        used.add(pn)
        s.pos = s.add(Part(pn, "W_pos", ("pos", "resid")))
        s.n_ctx = shapes[pn][0]
    s.n_ctx = s.n_ctx or int(_cfg(config, "max_position_embeddings", "n_positions", "n_ctx", default=0) or 0)
    heads = [n for n in shapes if _HEAD.search(n)]
    hw = next((n for n in heads if n.endswith(".weight") and shapes[n] == (s.d_vocab, d)), None)
    tie = _cfg(config, "tie_word_embeddings")
    if hw is not None and tie is not True:
        used.add(hw)
        s.unembed = s.add(Part(hw, "W_U", ("token", "resid")))
        s.modules["unembed"] = hw.rsplit(".", 1)[0]
    else:
        s.tied = True
        s.unembed = Part(en, "W_U", ("token", "resid"))
        if hw is not None:
            s.notes.append(f"{hw} is in the checkpoint but the config ties the unembedding to the embedding, so it is not used.")
    hb = next((n for n in heads if n.endswith(".bias") and shapes[n] == (s.d_vocab,)), None)
    if hb:
        used.add(hb)
        s.unembed_bias = s.add(Part(hb, "b_U", ("token",)))
    for n in sorted(shapes):
        if _FNORM.search(n) and n not in used and shapes[n] == (d,) and not _LAYER.match(n) and n.startswith(root):
            used.add(n)
            s.final_norm.append(s.add(Part(n, "norm_w" if n.endswith("weight") else "norm_b", ("resid",))))
            s.modules["final_norm"] = n.rsplit(".", 1)[0]
    s.norm = "ln" if any(p.role == "norm_b" for p in s.final_norm) else "rms" if s.final_norm else "none"

    layers = groups[best]
    s.n_layers = max(layers) + 1
    n_heads = int(_cfg(config, "num_attention_heads", "n_head", "num_heads", default=0) or 0)
    s.act = str(_cfg(config, "hidden_act", "activation_function", "hidden_activation", default="") or "")
    conv1d = False                                         # GPT-2 stores Linear weights as [in, out]
    # The text model's type (a multimodal checkpoint keeps it in text_config), wsp.runtime_layouts.
    hybrid = runtime_layouts.text_model_type(config) in runtime_layouts.HYBRID_TEXT_TYPES
    for l in range(s.n_layers):
        rest = layers.get(l, {})
        L = Layer(l, f"{best}.{l}")
        L.modules["layer"] = L.prefix
        fc = next((nm for r, nm in rest.items() if re.match(rf"^{_MLP}\.c_fc\.weight$", r)), None)
        if fc is not None and "attn.c_attn.weight" in rest:
            conv1d = shapes[fc][0] == d and shapes[fc][1] != d or shapes[rest["attn.c_attn.weight"]] == (d, 3 * d)
        experts: dict[int, dict] = {}
        experts_module = None

        for rest_name, name in rest.items():
            shape = shapes[name]
            if rest_name.startswith('linear_attn.') and hybrid:
                # Recurrent output is traceable/intervenable at the block boundary;
                # it has no ordinary attention probability matrix or Q/K/V head UI.
                L.modules['attn'] = f'{L.prefix}.linear_attn'
            kind = mod = None
            for rx, k in _ATTN_RULES:
                m = rx.match(rest_name)
                if m:
                    kind, mod, wb = k, m.group(1), m.group(2)
                    break
            if kind is not None:
                used.add(name)
                is_w = wb == "weight"
                L.modules["attn"] = f"{L.prefix}.{mod}"
                if kind in ("Q", "K", "V"):
                    roles = (("resid", "headdh") if conv1d else ("headdh", "resid")) if is_w else ("headdh",)
                    L.attn[("W_" if is_w else "b_") + kind] = Part(name, ("W_" if is_w else "b_") + kind, roles)
                elif kind == "O":
                    if is_w:
                        L.attn["W_O"] = Part(name, "W_O", ("headdh", "resid") if conv1d else ("resid", "headdh"))
                        L.modules["o_proj"] = name[:-len(".weight")]
                    else:
                        L.attn["b_O"] = Part(name, "b_O", ("resid",))
                else:
                    L.attn["_fused"] = L.attn.get("_fused", []) + [(kind.split(":")[1], name, is_w)]
                continue
            for rx, k in _MLP_RULES:
                m = rx.match(rest_name)
                if m:
                    kind, mod, wb = k, m.group(1), m.group(2)
                    break
            if kind is not None:
                used.add(name)
                is_w = wb == "weight"
                L.modules["mlp"] = f"{L.prefix}.{mod}"
                if kind == "out":
                    if is_w:
                        L.mlp["W_out"] = Part(name, "W_out", ("neuron", "resid") if conv1d else ("resid", "neuron"))
                        L.modules["mlp_out"] = name[:-len(".weight")]
                    else:
                        L.mlp["b_out"] = Part(name, "b_out", ("resid",))
                elif kind == "gate_up":                     # Phi-3: [2 * d_mlp, d]: first half gate, second half up
                    half = shape[0] // 2
                    if is_w:
                        L.mlp["W_gate"] = Part(name, "W_gate", ("neuron", "resid"), base=(slice(0, half),))
                        L.mlp["W_up"] = Part(name, "W_up", ("neuron", "resid"), base=(slice(half, 2 * half),))
                    else:
                        L.mlp["b_gate"] = Part(name, "b_gate", ("neuron",), base=(slice(0, half),))
                        L.mlp["b_up"] = Part(name, "b_up", ("neuron",), base=(slice(half, 2 * half),))
                else:
                    role = {"in": "in", "gate": "gate", "up": "up"}[kind]
                    if is_w:
                        L.mlp[f"W_{role}"] = Part(name, f"W_{role}", ("resid", "neuron") if conv1d else ("neuron", "resid"))
                    else:
                        L.mlp[f"b_{role}"] = Part(name, f"b_{role}", ("neuron",))
                continue
            # The mixer, routed-expert and shared-expert parts. Their tensors also stay in L.other (see Layer).
            m = next((m for rx, _ in _MIXER_RULES for m in [rx.match(rest_name)] if m), None)
            if m is not None:
                used.add(name)
                L.other.append(name)
                if m.group(2) == "weight":
                    L.mixer["W_O"] = Part(name, "W_O", ("resid", "other"))
                    L.modules["mixer_out"] = name[:-len(".weight")]
                else:
                    L.mixer["b_O"] = Part(name, "b_O", ("resid",))
                continue
            m = next((m for rx, _ in _EXPERT_RULES for m in [rx.match(rest_name)] if m), None)
            if m is not None:
                used.add(name)
                L.other.append(name)
                role, roles = _EXPERT_ROLES[m.group(3)]
                experts.setdefault(int(m.group(2)), {})[role] = Part(name, role, roles)
                experts_module = f"{L.prefix}.{m.group(1)}.experts"
                continue
            m = next((m for rx, _ in _SHARED_RULES for m in [rx.match(rest_name)] if m), None)
            if m is not None:
                used.add(name)
                L.other.append(name)
                role, roles = _EXPERT_ROLES[m.group(3)]
                L.shared[role] = Part(name, role, roles)
                if role == "W_out":
                    L.modules["shared_out"] = name[:-len(".weight")]
                continue
            if shape == (d,) and re.search(r"norm|ln_", rest_name):
                used.add(name)
                L.norms.append(Part(name, "norm_b" if rest_name.endswith("bias") else "norm_w", ("resid",)))
            else:
                L.other.append(name)
                used.add(name)
        if experts:
            if sorted(experts) == list(range(len(experts))):
                L.experts = [experts[e] for e in range(len(experts))]
                L.modules["experts"] = experts_module
            else:
                s.notes.append(f"layer {l}: expert indices are not 0..{len(experts) - 1}; its experts are not recognised")
        s.layers.append(L)

    # head geometry, from the first layer that has attention
    probe = next((L for L in s.layers if "W_Q" in L.attn or "_fused" in L.attn), None)
    d_head = int(_cfg(config, "head_dim", default=0) or 0)
    if probe is not None:
        if "W_Q" in probe.attn:
            qp = probe.attn["W_Q"]
            q_rows = shapes[qp.name][qp.roles.index("headdh")]
        else:
            q_rows = d
        if not n_heads:
            n_heads = max(1, q_rows // (d_head or 64))
        d_head = d_head or q_rows // n_heads
    s.n_heads, s.d_head = n_heads, d_head
    for L in s.layers:
        fused = L.attn.pop("_fused", [])
        for style, name, is_w in fused:
            shape = shapes[name]
            if style == "gpt2":                             # Conv1D [d, 3d]: columns Q | K | V, each head-major
                for i, m in enumerate("QKV"):
                    cols = slice(i * d, (i + 1) * d)
                    L.attn[("W_" if is_w else "b_") + m] = Part(name, ("W_" if is_w else "b_") + m,
                                                               ("resid", "headdh") if is_w else ("headdh",),
                                                               base=(slice(None), cols) if is_w else (cols,))
            elif style == "neox":                           # Linear [3d, d]: per head, rows q | k | v
                for i, m in enumerate("QKV"):
                    L.attn[("W_" if is_w else "b_") + m] = Part(name, ("W_" if is_w else "b_") + m,
                                                               ("headdh", "resid") if is_w else ("headdh",),
                                                               head=(3 * d_head, i * d_head, d_head, n_heads))
            elif style == "phi3":                           # Linear [(H + 2 * H_kv) * d_head, d]: rows Q | K | V
                rows = shape[0]
                n_kv = (rows - n_heads * d_head) // (2 * d_head)
                a = n_heads * d_head
                spans = {"Q": (0, a), "K": (a, a + n_kv * d_head), "V": (a + n_kv * d_head, a + 2 * n_kv * d_head)}
                for m, (r0, r1) in spans.items():
                    L.attn[("W_" if is_w else "b_") + m] = Part(name, ("W_" if is_w else "b_") + m,
                                                               ("headdh", "resid") if is_w else ("headdh",), base=(slice(r0, r1),))
        for role, p in list(L.attn.items()):
            if role in ('W_Q','b_Q') and hybrid:
                p.head = (2*d_head,0,d_head,n_heads)  # per-head query | gate rows
            if "headdh" in p.roles and p.head is None:
                n_here = s._shape(p)[p.roles.index("headdh")] // max(1, d_head)
                p.head = (d_head, 0, d_head, n_here)
            s.add(p)
        for p in list(L.mlp.values()) + L.norms:
            s.add(p)
        for p in list(L.mixer.values()) + [p for ex in L.experts for p in ex.values()] + list(L.shared.values()):
            s.add(p)
    if probe is not None and "W_K" in probe.attn:
        s.n_kv_heads = probe.attn["W_K"].head[3]
    first_mlp = next((l for l in range(s.n_layers) if s.has_neurons(l)), None)
    s.d_mlp = s.mlp_width(first_mlp) if first_mlp is not None else 0
    s.other = [n for n in shapes if n not in used]
    kinds = {"gpt2" if conv1d else None,
             "gated" if any("W_gate" in L.mlp for L in s.layers) else None}
    mt = str(config.get("model_type") or "")
    s.family = mt or ("gpt2" if conv1d else "llama-like" if "gated" in kinds else "decoder")
    mixers = [L.index for L in s.layers if "W_O" in L.mixer]
    if mixers:
        s.notes.append(f"{len(mixers)} layers use a token mixer without attention heads (linear attention); its output "
                       f"projection is the layer's attn.o_proj for directions")
    moe = [L for L in s.layers if L.experts]
    if moe:
        counts = sorted({len(L.experts) for L in moe})
        e = str(counts[0]) if len(counts) == 1 else f"{counts[0]} to {counts[-1]}"
        shared = " and a shared expert" if all("W_out" in L.shared for L in moe) else ""
        s.notes.append(f"{len(moe)} layers have {e} routed experts{shared}; directions treat each expert's down "
                       f"projection as a module of mlp.down_proj")
    unknown_attn = [L.index for L in s.layers if not s.has_heads(L.index)]
    if unknown_attn and len(unknown_attn) < s.n_layers:
        s.notes.append(f"attention heads were not recognised in {len(unknown_attn)} of {s.n_layers} layers "
                       f"(layers {_runs(unknown_attn)}); their tensors are available by name.")
    elif unknown_attn:
        s.notes.append("no attention heads were recognised; MLP neurons and tensors by name are available.")
    if s.other:
        s.notes.append(f"{len(s.other)} tensors lie outside the recognised decoder (for example other towers or heads); "
                       f"they are available by name under raw:.")
    return s


def _runs(xs: list[int]) -> str:
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ", ".join(out)
