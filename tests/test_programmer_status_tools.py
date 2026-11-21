"""The programmer's inspection tools on a model with a large vocabulary: how many passes `inspect card` and
`inspect what=direction` make over the embedding and unembedding matrices, and `inspect what=wiring` when the
graph has no rows.

Reported on a 64-layer GGUF model with a 248,320 x 5,120 vocabulary: a neuron card took 51 s and a direction 27 s,
and wiring for L38N3838 answered only its header line. Each pass over one of those matrices decodes 1.27G values;
the card made four passes (the gate read, the up read and the write each read the embedding again), a direction
two (embedding and unembedding once each).

Model-free. The fake GGUF store of tests/selection_fixtures.py (gated MLP, as on that model), with its single-read
limit and the slab size lowered so that the vocabulary matrices are read in slabs as on the real model, and the tiny
random model of tests/test_atlas_large_reads.py for wiring. No quantized file, decoder or real model is used.
"""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import selection_fixtures as F  # noqa: E402
from test_atlas_large_reads import session  # noqa: E402
from wsp import analysis as an  # noqa: E402
from wsp import arch, tools  # noqa: E402
from wsp import workspace as W  # noqa: E402
from wsp.tokenizer import VocabTokenizer  # noqa: E402
from wsp.workspace import Workspace  # noqa: E402

LIMIT = 100        # the store's single-read limit and the slab size: under the 16 x 16 vocabulary matrices (256)


def quant(directory, tied=False):
    """The fake GGUF workspace of selection_fixtures; tied=True leaves out lm_head, so the unembedding is the
    embedding tensor."""
    tensors = F.llama_tensors()
    if tied:
        del tensors["lm_head.weight"]
    os.makedirs(os.path.join(directory, "patches"), exist_ok=True)
    os.makedirs(os.path.join(directory, "tests"), exist_ok=True)
    cfg = {"model_type": "llama", "num_attention_heads": 4, "num_key_value_heads": 2, "hidden_act": "silu",
           "rms_norm_eps": 1e-5}
    struct = arch.detect({k: tuple(v.shape) for k, v in tensors.items()}, cfg)
    ws = Workspace(str(directory), F.QuantStore(tensors), struct, VocabTokenizer(F.VOCAB),
                   {"kind": "gguf", "name": "quant", "base_sha256": "fixture"})
    ws.rebuild()
    return ws


@pytest.fixture()
def limited(monkeypatch):
    monkeypatch.setattr(F, "MAX_READ", LIMIT)
    monkeypatch.setattr(W, "STREAM_ELEMS", LIMIT)


def per_row_embed_rows(ws, monkeypatch):
    """Semantic.embed_rows reads the rows it needs from an embedding above 64M values and the whole matrix below;
    here the matrix is tiny, so take the large-model branch, as on the reported model (scaled consistently)."""
    E = ws.sem.W_E()
    monkeypatch.setattr(ws.sem, "embed_rows",
                        lambda ids, variant="current": torch.stack([E.sub(int(i)).read(variant) for i in ids]))


def passes(ws, name):
    """Values read from one tensor, as a multiple of its size."""
    n = 1
    for d in ws.store.shape(name):
        n *= d
    return sum(v for t, v in ws.store.reads if t == name) / n


def same(a, b, tol=1e-3):
    """Two _top() readings agree: the same tokens in the same order, scores within tol."""
    for side in ("top", "bottom"):
        assert [x[2] for x in a[side]] == [x[2] for x in b[side]], side
        assert all(abs(x[1] - y[1]) <= tol for x, y in zip(a[side], b[side])), side


def test_the_store_under_test_reads_the_vocabulary_in_slabs(tmp_path, limited):
    ws = quant(tmp_path / "q")
    with pytest.raises(ValueError, match="GGUF read exceeds"):
        ws.store.read(ws.struct.embed.name)


def test_a_neuron_card_reads_each_vocabulary_matrix_once(tmp_path, limited):
    ws = quant(tmp_path / "q")
    E, U = ws.struct.embed.name, ws.struct.unembed.name
    ws.store.reads.clear()
    c = an.neuron_card(ws, 1, 5, k=6)
    assert c["gated"] and [r["kind"] for r in c["reads"]] == ["gate", "up"]
    assert passes(ws, E) == 1 and passes(ws, U) == 1            # was 3 and 1: once per read and once for the write
    assert max(v for _, v in ws.store.reads) <= LIMIT
    sem = ws.sem                                                 # the same readings, one direction at a time
    for r, got in zip(sem.neuron(1, 5)["reads"], c["reads"]):
        same(got["embeddings"], an._top(ws, sem.embed_scores(r["w"].read()), 6))
    w_out = sem.neuron(1, 5)["write"].read()
    same(c["writes"]["promotes"], an._top(ws, sem.logit_effect(w_out), 6))
    same(c["writes"]["embeddings"], an._top(ws, sem.embed_scores(w_out), 6))


def test_a_direction_reads_each_vocabulary_matrix_once(tmp_path, limited):
    ws = quant(tmp_path / "q")
    v = torch.randn(ws.struct.d_model, generator=torch.Generator().manual_seed(3))
    ws.store.reads.clear()
    d = an.direction_tokens(ws, v, 8)
    assert passes(ws, ws.struct.embed.name) == 1 and passes(ws, ws.struct.unembed.name) == 1
    same(d["promotes"], an._top(ws, ws.sem.logit_effect(v), 8))
    same(d["embeddings"], an._top(ws, ws.sem.embed_scores(v), 8))
    assert an.reader_tokens(ws, v, 8)["embeddings"] == d["embeddings"] and "promotes" not in an.reader_tokens(ws, v, 8)


def test_a_head_card_reads_each_vocabulary_matrix_once_besides_its_sampled_rows(tmp_path, limited, monkeypatch):
    ws = quant(tmp_path / "q")
    per_row_embed_rows(ws, monkeypatch)
    sample = len(an._sample_tokens(ws))
    ws.store.reads.clear()
    c = an.head_card(ws, 0, 1, k=5)
    d = ws.struct.d_model
    assert passes(ws, ws.struct.unembed.name) == 1                      # was 4: the OV sample and three components
    assert passes(ws, ws.struct.embed.name) == 1 + sample * d / (16 * d)   # one pass, and the sampled rows
    sem = ws.sem
    h = sem.head(0, 1)
    u, _, vh = torch.linalg.svd(h["V"].read().double() @ h["O"].read().double(), full_matrices=False)
    for i, comp in enumerate(c["ov_svd"]["components"]):
        same(comp["reads"], an._top(ws, sem.embed_scores(u[:, i].float()), 6))
        same(comp["writes"], an._top(ws, sem.logit_effect(vh[i].float()), 6))


def test_a_tied_vocabulary_is_read_once_for_both_readings(tmp_path, limited):
    ws = quant(tmp_path / "t", tied=True)
    assert ws.struct.tied and ws.sem.tied()
    ws.store.reads.clear()
    c = an.neuron_card(ws, 0, 2, k=6)
    assert passes(ws, ws.struct.embed.name) == 1                # gate, up and write: embeddings and logits together
    sem = ws.sem
    w_out = sem.neuron(0, 2)["write"].read()
    same(c["writes"]["promotes"], an._top(ws, sem.logit_effect(w_out), 6))
    same(c["writes"]["embeddings"], an._top(ws, sem.embed_scores(w_out), 6))
    for r, got in zip(sem.neuron(0, 2)["reads"], c["reads"]):
        same(got["embeddings"], an._top(ws, sem.embed_scores(r["w"].read()), 6))


def test_the_inspect_tool_is_registered_as_itself():
    """A helper placed between the tool decorator and the function would have been registered as `inspect`."""
    assert tools.TOOLS["inspect"]["fn"] is tools.inspect


def wiring(s, address):
    return tools.call(s, "inspect", {"address": address, "what": "wiring"})


def test_wiring_with_no_rows_says_why_and_what_to_ask_instead(tmp_path, monkeypatch):
    """A wide model gets a one-layer window; a layer whose attention is not made of heads (a hybrid / linear
    attention layer) then has nothing that writes before its MLP reads, so the graph has no edges."""
    s = session(tmp_path)
    monkeypatch.setattr(an, "default_window", lambda ws, focus=None: [1 if focus is None else focus])
    monkeypatch.setattr(s.ws.struct, "has_heads", lambda l: l != 1)
    for address, unit in (("blocks.1.mlp", "L1N<n>"), ("L1N3", "L1N3")):
        lines = wiring(s, address).splitlines()
        assert lines[0].startswith("composition score") and "(layers 1-1 only" in lines[0]
        assert lines[1] == "note: Layer 1: recurrent/hybrid attention composition is not represented; MLP only"
        assert len(lines) == 3 and lines[2].startswith("no rows: the window is layer 1 alone, and its attention has no "
                                                       "heads the graph scores")
        assert "inspect blocks.0.attn.head.0 what=wiring" in lines[2] and f"atlas action=unit address={unit}" in lines[2]


def test_wiring_with_rows_is_unchanged(tmp_path, monkeypatch):
    s = session(tmp_path)
    monkeypatch.setattr(an, "default_window", lambda ws, focus=None: [1 if focus is None else focus])
    out = wiring(s, "blocks.1.attn.head.0").splitlines()
    assert len(out) > 1 and all(" -> L1MLP." in x and x.startswith("L1H0") for x in out[1:])
    assert not any(x.startswith(("no rows", "note:")) for x in out)


def test_wiring_with_an_empty_window_says_so(tmp_path, monkeypatch):
    s = session(tmp_path)
    monkeypatch.setattr(an, "default_window", lambda ws, focus=None: [])
    assert wiring(s, "blocks.0.mlp") == "no rows: the wiring window has no layers (this model has no layer the graph " \
                                        "can score)."
    assert tools._no_wiring(s.ws, "model", [], {"nodes": []}, None) == "no rows: the wiring window has no layers."


# ---------------------------------------------------------------------------------------------- numerics
# The stacked readings against main's path (one direction at a time, Semantic.embed_scores / logit_effect, then _top),
# on the tiny model's plain store and on the same weights behind a store that refuses large reads, with a patch on both
# vocabulary matrices that makes two embedding rows and two unembedding columns equal (exact ties), and for the base
# and current variants.
TIE = "P('embed.W_E')[3] = P('embed.W_E')[5]\nP('unembed.W_U')[:, 7] = P('unembed.W_U')[:, 9]"


def tied_session(root, monkeypatch, capped):
    from test_atlas_large_reads import cap_reads
    s = session(root)
    assert s.apply(TIE)["ok"]
    if capped:
        cap_reads(s.ws, monkeypatch)
    return s


def per_direction(ws, v, k, variant="current"):
    """main's direction_tokens / reader_tokens, one direction at a time."""
    sem = ws.sem
    return {"promotes": an._top(ws, sem.logit_effect(v, variant), k), "embeddings": an._top(ws, sem.embed_scores(v, variant), k)}


@pytest.mark.parametrize("capped", [False, True])
def test_stacked_readings_equal_one_direction_readings_with_ties_and_a_patch(tmp_path, monkeypatch, capped):
    s = tied_session(tmp_path, monkeypatch, capped)
    ws, sem = s.ws, s.ws.sem
    E3 = sem.W_E().sub(3).read()
    v = E3 + 0.0                                                 # rows 3 and 5 score the same against this direction
    u = sem.W_U().sub((slice(None), 7)).read()                   # columns 7 and 9 are equal: the same logit effect
    dirs = [v, u, torch.randn(ws.struct.d_model, generator=torch.Generator().manual_seed(1))]
    k = 26                                                       # the whole vocabulary: every tie is in the list
    reads, writes = an.vocab_readings(ws, dirs, dirs, k)
    for d, r, w in zip(dirs, reads, writes):
        want = per_direction(ws, d, k)
        assert [x[2] for x in r["embeddings"]["top"]] == [x[2] for x in want["embeddings"]["top"]]
        assert [x[2] for x in w["promotes"]["top"]] == [x[2] for x in want["promotes"]["top"]]
        assert [x[2] for x in w["promotes"]["bottom"]] == [x[2] for x in want["promotes"]["bottom"]]
        same(r["embeddings"], want["embeddings"], 1e-4)
        same(w["promotes"], want["promotes"], 1e-4)
    emb = an.vocab_readings(ws, [v], [], k)[0][0]["embeddings"]["top"]
    scores = {x[2]: x[1] for x in emb}
    assert scores[3] == scores[5]                                # the tie is exact in the stacked reading too
    c = an.neuron_card(ws, 0, 3, k=6)                            # the card reads through the same path
    same(c["writes"]["promotes"], an._top(ws, sem.logit_effect(sem.neuron(0, 3)["write"].read()), 6), 1e-4)


@pytest.mark.parametrize("capped", [False, True])
def test_base_and_current_readings_follow_the_variant(tmp_path, monkeypatch, capped):
    s = tied_session(tmp_path, monkeypatch, capped)
    ws, sem = s.ws, s.ws.sem
    v = torch.randn(ws.struct.d_model, generator=torch.Generator().manual_seed(2))
    for variant in ("base", "current"):
        emb, lgt = sem.vocab_scores(v[None], v[None], variant)
        assert torch.allclose(emb[:, 0], sem.embed_scores(v, variant), atol=1e-5)
        assert torch.allclose(lgt[:, 0], sem.logit_effect(v, variant), atol=1e-5)
    eb, lb = sem.vocab_scores(v[None], v[None], "base")
    ec, lc = sem.vocab_scores(v[None], v[None], "current")
    assert not torch.allclose(eb[3], ec[3]) and torch.allclose(eb[4], ec[4])     # the patch moved row 3, not row 4
    assert not torch.allclose(lb[7], lc[7]) and torch.allclose(lb[8], lc[8])
