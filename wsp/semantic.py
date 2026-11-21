"""Architecture-independent access to the parts of a model.

`Semantic` answers "give me head 3.2's query matrix" or "what does this residual direction
do to the logits" for any recognised architecture, by turning the Structure's parts into lazy
`Ref`s in one canonical layout:

    W_E [token, d_model]       W_U [d_model, token]       W_pos [pos, d_model]
    head:   W_Q, W_K, W_V [d_model, d_head]      W_O [d_head, d_model]
    MLP:    reads [d_model, neuron] (one, or gate + up)      W_out [neuron, d_model]

so analysis code and weight scripts are written once.  Reads stay lazy: a head's W_Q is a few
thousand numbers out of a file of gigabytes, and products with the embedding / unembedding
are streamed when those are too large to hold.
"""
from __future__ import annotations

import torch

from .arch import Part
from .workspace import Ref

READERS = ("W_Q", "W_K", "W_V", "W_in", "W_gate", "W_up", "W_U")


class Semantic:
    def __init__(self, ws):
        self.ws = ws
        self.s = ws.struct

    # ------------------------------------------------------------------ refs
    def ref(self, part: Part, head: int | None = None, neuron=None) -> Ref:
        from .store import norm_index
        s = self.s
        idx = s.part_index(part, head=head, neuron=neuron)
        roles = s.view_roles(part, head=head is not None, neuron=isinstance(neuron, int))
        r = Ref(self.ws, part.name, norm_index(idx, self.ws.shape(part.name)))
        if "resid" in roles and len(roles) >= 2:
            k = roles.index("resid")
            rest = [i for i in range(len(roles)) if i != k]
            perm = [k] + rest if part.role in READERS else rest + [k]
            r = Ref(self.ws, part.name, r.region, tuple(perm))
        return r

    def W_E(self) -> Ref:
        return self.ref(self.s.embed)

    def W_U(self) -> Ref:
        return self.ref(self.s.unembed)

    def W_pos(self) -> Ref | None:
        return self.ref(self.s.pos) if self.s.pos else None

    def embed_row(self, t: int, **kw) -> torch.Tensor:
        return self.W_E().sub(int(t)).read(**kw)

    def unembed_col(self, t: int, **kw) -> torch.Tensor:
        return self.W_U().sub((slice(None), int(t))).read(**kw)

    def head(self, l: int, h: int) -> dict[str, Ref]:
        L = self.s.layers[l]
        out = {}
        for role in ("W_Q", "W_K", "W_V", "W_O", "b_Q", "b_K", "b_V"):
            if role in L.attn:
                out[role[2:] if role.startswith("W_") else role] = self.ref(L.attn[role], head=h)
        return out

    def neuron(self, l: int, n: int) -> dict:
        L = self.s.layers[l]
        reads = []
        for role in L.reads:
            b = L.mlp.get("b_" + role[2:])
            reads.append({"kind": role[2:], "w": self.ref(L.mlp[role], neuron=n), "b": self.ref(b, neuron=n) if b else None})
        return {"reads": reads, "write": self.ref(L.mlp["W_out"], neuron=n)}

    def mlp(self, l: int) -> dict[str, Ref]:
        L = self.s.layers[l]
        return {role: self.ref(p) for role, p in L.mlp.items()}

    def attn(self, l: int) -> dict[str, Ref]:
        return {role: self.ref(p) for role, p in self.s.layers[l].attn.items()}

    def stack(self, l: int, role: str, variant: str = "current", extra=None) -> torch.Tensor:
        """All heads of one attention matrix at once: [n, d_model, d_head] for W_Q / W_K / W_V (n = heads, or
        key-value groups), [n, d_head, d_model] for W_O. The whole tensor, whatever its layout: read in slabs when
        it is above the size a quantized store gives in one read (5120 x 6144 is 31M values)."""
        from .store import norm_index
        part = self.s.layers[l].attn[role]
        t = Ref(self.ws, part.name, norm_index(self.s.part_index(part), self.ws.shape(part.name))).whole(variant, extra)
        roles = list(part.roles)
        if "head" in roles:
            t = t.movedim(roles.index("head"), 0)                      # [n, a, b]
            rest = [r for r in roles if r != "head"]
        else:
            stride, off, size, n = part.head
            t = t.movedim(roles.index("headdh"), 0)                    # [n * ..., d_model]
            t = torch.stack([t[off + g * stride: off + g * stride + size] for g in range(n)])     # [n, d_head, d_model]
            rest = ["dhead", "resid"]
        resid_first = rest[0] == "resid"
        if role == "W_O":
            return t.transpose(1, 2) if resid_first else t
        return t if resid_first else t.transpose(1, 2)

    def writers(self, layers=None) -> list[Ref]:
        """Every view whose last axis is the residual stream it writes to (2-D matrices and 1-D biases)."""
        s = self.s
        out = []
        if layers is None:
            out.append(self.W_E())
            if s.pos:
                out.append(self.W_pos())
        for l in (range(s.n_layers) if layers is None else layers):
            L = s.layers[l]
            if "W_O" in L.attn:
                r = self.ref(L.attn["W_O"])
                out += [r.sub(h) for h in range(r.shape[0])] if len(r.shape) == 3 else [r]
            for role in ("b_O",):
                if role in L.attn:
                    out.append(self.ref(L.attn[role]))
            if "W_out" in L.mlp:
                out.append(self.ref(L.mlp["W_out"]))
            if "b_out" in L.mlp:
                out.append(self.ref(L.mlp["b_out"]))
        return out

    def vocab_bytes(self) -> int:
        """Bytes that would have to be fetched before directions can be read through the vocabulary (0 if on disk)."""
        st = self.ws.store
        names = {p.name for p in (self.s.embed, self.s.unembed) if p is not None}
        return sum(st.size_bytes(n) for n in names if not st.local(n))

    # ------------------------------------------------------------------ meaning of directions
    def final_gain(self, variant: str = "current") -> tuple[torch.Tensor | None, bool]:
        """(gain per coordinate, whether the final normalisation centres) for reading directions as logit effects."""
        w = next((p for p in self.s.final_norm if p.role == "norm_w"), None)
        if w is None:
            return None, self.s.norm == "ln"
        from .atlas.build import norm_offset                    # some families store the gain as (gain - 1)
        return self.ws.read(w.name, (), variant) + norm_offset(self.ws), self.s.norm == "ln"

    def logit_input(self, v: torch.Tensor, variant: str = "current") -> torch.Tensor:
        """Residual direction(s) v [..., d_model] as the unembedding sees them: centred and scaled by the final norm."""
        g, centre = self.final_gain(variant)
        if centre:
            v = v - v.mean(-1, keepdim=True)
        if g is not None:
            v = v * g
        return v

    def logit_effect(self, v: torch.Tensor, variant: str = "current") -> torch.Tensor:
        """Direct effect of residual direction(s) v [..., d_model] on the logits, through the final norm's gain."""
        v = self.logit_input(v, variant)
        return self.W_U().contract(v.T if v.dim() == 2 else v, axis=0, variant=variant).T if v.dim() == 2 \
            else self.W_U().contract(v, axis=0, variant=variant)

    def tied(self) -> bool:
        """Whether the unembedding is the embedding tensor itself (then one pass over it serves both readings)."""
        e, u = self.s.embed, self.s.unembed
        return e is not None and u is not None and e.name == u.name

    def vocab_scores(self, embed: torch.Tensor | None, logit: torch.Tensor | None,
                     variant: str = "current") -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """(embed_scores of the directions in the rows of `embed` [a, d_model] -> [V, a], logit_effect of the rows of
        `logit` [b, d_model] -> [V, b]) in one pass over the embedding matrix and one over the unembedding, or a
        single pass when they are one tensor.

        Each pass over a vocabulary matrix reads and, for a quantized model, decodes all of it (1.27G values for a
        248,320 x 5,120 vocabulary), so readings of several directions are stacked into one contraction."""
        has_e, has_l = embed is not None and len(embed), logit is not None and len(logit)
        adj = self.logit_input(logit.float(), variant) if has_l else None
        if self.tied() and has_l:
            cols = torch.cat([embed.float(), adj]) if has_e else adj
            both = self.embed_scores(cols.T, variant)                               # [V, a + b]
            a = len(embed) if has_e else 0
            return (both[:, :a] if has_e else None), both[:, a:]
        E = self.embed_scores(embed.float().T, variant) if has_e else None
        L = self.W_U().contract(adj.T, axis=0, variant=variant) if has_l else None
        return E, L

    def embed_scores(self, v: torch.Tensor, variant: str = "current") -> torch.Tensor:
        """W_E @ v: how strongly each token's embedding points along v ([d_model] or [d_model, k])."""
        return self.W_E().contract(v, axis=1, variant=variant)

    def pos_scores(self, v: torch.Tensor, variant: str = "current") -> torch.Tensor | None:
        p = self.W_pos()
        return p.contract(v, axis=1, variant=variant) if p is not None else None

    def embed_rows(self, ids: torch.Tensor, variant: str = "current") -> torch.Tensor:
        """Embedding rows for a set of token ids, reading only those rows."""
        E = self.W_E()
        if E.size <= 64_000_000:
            return E.whole(variant)[ids]
        return torch.stack([E.sub(int(i)).read(variant) for i in ids])
