"""Tiny random Qwen3.5 checkpoints in the multimodal layout of the published ones; built offline.

``tiny_qwen35(directory, moe=False, seed=0)`` saves a ``Qwen3_5ForConditionalGeneration`` (or, with
``moe``, a ``Qwen3_5MoeForConditionalGeneration`` with 4 routed experts, top 2, expert and shared-expert
width 8) with ``save_pretrained``, so the files carry ``model.language_model.*``, ``model.visual.*`` and
``lm_head.weight`` exactly as transformers writes them, plus a word-level tokenizer and a
``chat_template.jinja``. Its text config: ``hidden_size`` 32, 4 layers, ``full_attention_interval`` 4
(layers 0-2 linear attention, layer 3 full attention), linear attention 2 key / 4 value heads of width 8,
attention 2 heads of 16 (one key-value head), ``attn_output_gate`` true, ``vocab_size`` 40, dense MLP
width 48. Every matrix is scaled by 6 (as ``test_hf.py`` does) so every component matters.

Where the values are this module's own choice (the build spec leaves them open), and why:

- The vision config is the smallest that constructs: depth 1, width 16, 2 heads, MLP width 16, patch 2,
  temporal patch 2, spatial merge 2, 4 position embeddings (a 2 x 2 grid; it must be a square), output
  width 32 (the text model's).
- One key-value head and dense MLP width 48: not given by the spec.
- ``rope_parameters`` with ``partial_rotary_factor`` 0.5 and ``mrope_section`` [2, 1, 1], as
  ``test_hybrid_grouped.py`` uses for the same head width 16 (the rotary half of a head is 8 values, 4 pairs).
- ``attn_output_gate`` is written into the text config as Qwen3.5's published configs carry it; transformers
  5.22.2 has no such field (its Qwen3.5 attention always gates: ``q_proj`` is twice the heads' width) and
  keeps it as an unused attribute.
- The checkpoint has no ``mtp.*`` tensors: ``save_pretrained`` writes none (the model has no MTP head).
"""
from __future__ import annotations

import os

V, D_MODEL, N_LAYERS, EXPERTS, TOP_K, EXPERT_WIDTH = 40, 32, 4, 4, 2, 8


def text_config(moe: bool = False) -> dict:
    cfg = dict(vocab_size=V, hidden_size=D_MODEL, num_hidden_layers=N_LAYERS, full_attention_interval=4,
               linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
               linear_conv_kernel_dim=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
               attn_output_gate=True, max_position_embeddings=64, tie_word_embeddings=False,
               rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                                "mrope_section": [2, 1, 1]})
    if moe:
        cfg.update(num_experts=EXPERTS, num_experts_per_tok=TOP_K, moe_intermediate_size=EXPERT_WIDTH,
                   shared_expert_intermediate_size=EXPERT_WIDTH)
    else:
        cfg.update(intermediate_size=48)
    return cfg


VISION_CONFIG = dict(depth=1, hidden_size=16, intermediate_size=16, num_heads=2, in_channels=3, patch_size=2,
                     temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=D_MODEL, num_position_embeddings=4)

CHAT_TEMPLATE = ("{% for m in messages %}t1 {{ m['content'] }} t2 {% endfor %}"
                 "{% if add_generation_prompt %}t3 {% endif %}")


def tiny_qwen35(directory, moe: bool = False, seed: int = 0) -> str:
    """Build and save the tiny checkpoint in ``directory`` -> the directory (str)."""
    import torch
    import transformers as T
    directory = str(directory)
    if moe:
        config = T.Qwen3_5MoeConfig(text_config=text_config(True), vision_config=dict(VISION_CONFIG), tie_word_embeddings=False)
        cls = T.Qwen3_5MoeForConditionalGeneration
    else:
        config = T.Qwen3_5Config(text_config=text_config(False), vision_config=dict(VISION_CONFIG), tie_word_embeddings=False)
        cls = T.Qwen3_5ForConditionalGeneration
    torch.manual_seed(seed)
    model = cls(config)
    with torch.no_grad():                        # default init is tiny; make every component matter
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                p.mul_(6.0)
            elif "bias" in name:
                p.add_(0.1 * torch.randn_like(p))
    model.save_pretrained(directory, safe_serialization=True)
    from tokenizers import Tokenizer, models, pre_tokenizers
    tk = Tokenizer(models.WordLevel({f"t{i}": i for i in range(V)}, unk_token="t0"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.save(os.path.join(directory, "tokenizer.json"))
    with open(os.path.join(directory, "chat_template.jinja"), "w") as f:
        f.write(CHAT_TEMPLATE)
    return directory
