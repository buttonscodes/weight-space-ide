"""Model-free fixtures for selection tests: a tiny random model of this project's own kind, and a
llama-shaped workspace whose store reports a quantized storage type (Q4_K) but holds small float32
arrays in memory.  The fake store answers every read with the values it was built with, the way
the real GGUF store answers with reconstructed values; no file, no decoder and no model is used."""
from __future__ import annotations

import os

import torch

from wsp import arch
from wsp.store import region_index, region_size
from wsp.tokenizer import VocabTokenizer
from wsp.workspace import Workspace

VOCAB = list("abcdefghijklmnop")


MAX_READ = 16_000_000                     # = wsp.gguf_store.MAX_READ (not imported: that module needs the gguf package)
SCHEMES = {"F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"}


class QuantStore:
    """Looks like a GGUF store to the IDE: dtype Q4_K, editable as reconstructed values, immutable base.

    As strict as wsp.gguf_store.GGUFStore where the IDE can tell: a read of more than MAX_READ values is refused,
    only vectors and matrices of a known scheme are float (and readable), names are naturally sorted, and the
    attributes the IDE reads (size_bytes, where, permutations, architecture, ...) exist. `row_reads` counts the
    per-row decode calls a real 2-D read makes (one per row, whatever its width). Values are float32 in memory."""
    remote = False
    architecture = "llama"
    layout_supported = True
    layout_reason = ""
    cache_dir = ""
    row_reads = 0
    row_permutations: dict = {}               # never written: the llama layout has no permuted tensors
    column_permutations: dict = {}

    def __init__(self, tensors: dict[str, torch.Tensor], dtype: str = "Q4_K"):
        self.t = {k: v.float().clone() for k, v in tensors.items()}
        self.kind = dtype
        self.reads: list[tuple[str, int]] = []
        self.files = []

    @property
    def where(self): return dict.fromkeys(self.t)
    def names(self):
        from wsp.store import _natural
        return sorted(self.t, key=_natural)
    def shape(self, name): return tuple(self.t[name].shape)
    def __contains__(self, name): return name in self.t
    def nbytes(self): return sum(v.numel() for v in self.t.values()) // 2
    def size_bytes(self, name): return self.t[name].numel() // 2 if self.dtype(name) != "F32" else self.t[name].numel() * 4
    def total(self): return sum(v.numel() for v in self.t.values())
    def numel(self, name): return self.t[name].numel()
    def is_float(self, name): return self.dtype(name) in SCHEMES and self.t[name].dim() in (1, 2)
    def dtype(self, name): return self.kind if self.t[name].dim() == 2 else "F32"
    def local(self, name): return True
    def fetch(self, name): pass

    def read(self, name, region=None):
        from wsp.store import norm_index
        region = norm_index((), self.shape(name)) if region is None else region
        if region_size(region) > MAX_READ:
            raise ValueError("GGUF read exceeds 16M logical values; use row slabs / a smaller selection")
        if not self.is_float(name):
            raise ValueError(f"{self.dtype(name)} decoding is unsupported; keep this tensor read-only")
        self.reads.append((name, region_size(region)))
        if len(region) == 2:
            self.row_reads += 1 if isinstance(region[0], int) else len(region[0])
        return self.t[name][region_index(region)].clone()

    BLOCKS = {"Q4_0": 32, "Q4_1": 32, "Q5_0": 32, "Q5_1": 32, "Q8_0": 32,
              "Q2_K": 256, "Q3_K": 256, "Q4_K": 256, "Q5_K": 256, "Q6_K": 256}   # = gguf.GGML_QUANT_SIZES[..][0]

    def read_block(self, name):
        """As GGUFStore.read_block: (1, values per packed block) for a quantized matrix, else None."""
        if name not in self.t or self.t[name].dim() != 2 or name in self.column_permutations:
            return None
        block = self.BLOCKS.get(self.dtype(name), 1)
        return (1, block) if block > 1 else None

    def capabilities(self):
        no = dict(supported=False, reason="test fixture: no runner")
        yes = dict(supported=True, reason="test fixture")
        return dict(format="gguf", architecture="llama", tensor_types=[self.kind], representation="reconstructed test values",
                    import_read=yes, logical_edit=yes, generation=no, trace_intervention=no, adapter_export=no, merged_export=no)


def llama_tensors(n_layers=2, d_model=16, d_mlp=24, n_heads=4, n_kv=2, vocab=16, seed=0, wide_layer=None, wide_mlp=0):
    g = torch.Generator().manual_seed(seed)
    dh = d_model // n_heads
    t = {"model.embed_tokens.weight": torch.randn(vocab, d_model, generator=g),
         "model.norm.weight": torch.ones(d_model), "lm_head.weight": torch.randn(vocab, d_model, generator=g)}
    for l in range(n_layers):
        m = wide_mlp if l == wide_layer else d_mlp
        p = f"model.layers.{l}."
        t[p + "input_layernorm.weight"] = torch.ones(d_model)
        t[p + "post_attention_layernorm.weight"] = torch.ones(d_model)
        t[p + "self_attn.q_proj.weight"] = torch.randn(n_heads * dh, d_model, generator=g)
        t[p + "self_attn.k_proj.weight"] = torch.randn(n_kv * dh, d_model, generator=g)
        t[p + "self_attn.v_proj.weight"] = torch.randn(n_kv * dh, d_model, generator=g)
        t[p + "self_attn.o_proj.weight"] = torch.randn(d_model, n_heads * dh, generator=g)
        t[p + "mlp.gate_proj.weight"] = torch.randn(m, d_model, generator=g)
        t[p + "mlp.up_proj.weight"] = torch.randn(m, d_model, generator=g)
        t[p + "mlp.down_proj.weight"] = torch.randn(d_model, m, generator=g)
    return t


def quant_workspace(directory, **kw) -> Workspace:
    tensors = llama_tensors(**kw)
    os.makedirs(os.path.join(directory, "patches"), exist_ok=True)
    os.makedirs(os.path.join(directory, "tests"), exist_ok=True)
    store = QuantStore(tensors)
    shapes = {k: tuple(v.shape) for k, v in tensors.items()}
    cfg = {"model_type": "llama", "num_attention_heads": kw.get("n_heads", 4), "num_key_value_heads": kw.get("n_kv", 2),
           "hidden_act": "silu", "rms_norm_eps": 1e-5}
    struct = arch.detect(shapes, cfg)
    ws = Workspace(str(directory), store, struct, VocabTokenizer(VOCAB), {"kind": "gguf", "name": "quant", "base_sha256": "fixture"})
    ws.rebuild()
    return ws
