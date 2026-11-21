"""Synthetic GGUF files for model-free tests, written with the gguf package (no download, no network).

    tiny_qwen2_gguf(path, qtype, seed, chat_template)  a random transformers Qwen2ForCausalLM (2 layers, width 32,
        vocabulary 32) written as llama.cpp's converter lays out a Qwen2 GGUF: every matrix packed as ``qtype``
        (Q8_0 or Q4_0), vectors F32, the architecture metadata the IDE's open reads, and a byte-level BPE
        tokenizer (26 letters, space, newline, one merge, three ChatML control tokens) with, unless
        ``chat_template`` is None, a ``tokenizer.chat_template``.

    tiny_hybrid_gguf(path, qtype, kh, vh, chat_template)  a random transformers Qwen3_5ForCausalLM (4 layers, width
        32; layers 0-2 linear attention, layer 3 full attention; ``kh`` key and ``vh`` value linear heads) written as
        llama.cpp's converter lays out a Qwen3.5 GGUF. The tensor writer is a copy of the hybrid fixture of
        tests/test_hybrid_grouped.py (copied, not imported) with its explicit grouped->tiled head ordering, so the
        IDE's inverse mapping is checked against an independent listing; to it are added the qwen35 metadata the
        IDE's open reads and the same tokenizer as the Qwen2 file.

The values a test compares against are the decoded ones (``gguf.dequantize`` of what was written), never the
float model's: the quantized base is the model.
"""
import gguf
import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3_5ForCausalLM, Qwen3_5TextConfig

LETTERS = list("abcdefghijklmnopqrstuvwxyz")
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
# Byte-level spelling (GPT-2's bytes_to_unicode): a space is "Ġ", a newline "Ċ".
TOKENS = LETTERS + ["Ġ", "Ċ", "ab"] + SPECIALS
MERGES = ["a b"]
ENDOFTEXT, IM_START, IM_END = (TOKENS.index(s) for s in SPECIALS)
CHATML = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
          "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
QTYPES = {"Q8_0": gguf.GGMLQuantizationType.Q8_0, "Q4_0": gguf.GGMLQuantizationType.Q4_0}


def qwen2_config() -> Qwen2Config:
    return Qwen2Config(vocab_size=len(TOKENS), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, rms_norm_eps=1e-6,
                       tie_word_embeddings=False, eos_token_id=IM_END)


def tiny_qwen2_gguf(path, qtype="Q8_0", seed=0, chat_template=CHATML, scale=6.0, add_bos=False):
    """Write the file -> (config, {logical name: decoded float32 tensor}). Every matrix is the random draw times
    ``scale`` (as tests/test_hf.py does, so activations are not vanishingly small); norms and biases as drawn. With
    ``add_bos`` the tokenizer metadata names <|endoftext|> the beginning-of-text token and asks for it
    (tokenizer.ggml.add_bos_token)."""
    cfg = qwen2_config()
    torch.manual_seed(seed)
    ref = Qwen2ForCausalLM(cfg).eval()
    theta = float((getattr(cfg, "rope_parameters", None) or {}).get("rope_theta", 10000.0))
    w = gguf.GGUFWriter(str(path), "qwen2")
    w.add_context_length(cfg.max_position_embeddings)
    w.add_embedding_length(cfg.hidden_size)
    w.add_block_count(cfg.num_hidden_layers)
    w.add_feed_forward_length(cfg.intermediate_size)
    w.add_head_count(cfg.num_attention_heads)
    w.add_head_count_kv(cfg.num_key_value_heads)
    w.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    w.add_rope_freq_base(theta)
    _tokenizer(w, chat_template, add_bos)
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN2, cfg.num_hidden_layers)
    decoded = {}
    for name, value in ref.state_dict().items():
        a = value.detach().numpy().astype(np.float32)
        if value.ndim == 2:
            a = (a * np.float32(scale)).astype(np.float32)
        t =QTYPES[qtype] if value.ndim == 2 else gguf.GGMLQuantizationType.F32
        packed = gguf.quantize(a, t)
        mapped = mapping.get_name(name, try_suffixes=(".weight", ".bias"))
        assert mapped is not None, name
        w.add_tensor(mapped, packed, raw_dtype=t)
        decoded[name] = torch.from_numpy(gguf.dequantize(packed, t).reshape(a.shape).copy())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return cfg, decoded


def _tokenizer(w, chat_template, add_bos=False):
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(TOKENS)
    w.add_token_types([gguf.TokenType.NORMAL] * (len(TOKENS) - len(SPECIALS)) + [gguf.TokenType.CONTROL] * len(SPECIALS))
    w.add_token_merges(MERGES)
    w.add_eos_token_id(IM_END)
    w.add_pad_token_id(ENDOFTEXT)
    if add_bos:
        w.add_bos_token_id(ENDOFTEXT)
        w.add_add_bos_token(True)
    if chat_template is not None:
        w.add_chat_template(chat_template)


def hybrid_config(kh=2, vh=6) -> Qwen3_5TextConfig:
    """tests/test_hybrid_grouped.py's configuration, with the shared tokenizer's vocabulary (also 32) and end id."""
    return Qwen3_5TextConfig(vocab_size=len(TOKENS), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                             num_attention_heads=6, num_key_value_heads=1, head_dim=16, max_position_embeddings=64,
                             linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=kh,
                             linear_num_value_heads=vh, linear_conv_kernel_dim=4, full_attention_interval=4,
                             tie_word_embeddings=False, eos_token_id=IM_END,
                             rope_parameters={'rope_theta': 10000., 'partial_rotary_factor': .5, 'mrope_section': [2, 1, 1]})


def tiny_hybrid_gguf(path, qtype="Q4_0", kh=2, vh=6, seed=713, chat_template=CHATML, scale=6.0):
    """Write the file -> (config, {logical name: decoded float32 tensor in the transformers model's order}). Every
    matrix is the random draw times ``scale``, as in tiny_qwen2_gguf (unscaled, ARA's losses are about 1e-5 and
    L-BFGS stops at its gradient tolerance without moving); ``scale=1`` is test_hybrid_grouped.py's file."""
    cfg = hybrid_config(kh, vh)
    torch.manual_seed(seed)
    ref = Qwen3_5ForCausalLM(cfg).eval()
    writer = gguf.GGUFWriter(str(path), 'qwen35')
    # The metadata transformers' qwen35 config mapping reads (the IDE's open builds the workspace config from it).
    writer.add_context_length(cfg.max_position_embeddings)
    writer.add_embedding_length(cfg.hidden_size)
    writer.add_block_count(cfg.num_hidden_layers)
    writer.add_feed_forward_length(cfg.intermediate_size)
    writer.add_head_count(cfg.num_attention_heads)
    writer.add_head_count_kv(cfg.num_key_value_heads)
    writer.add_key_length(cfg.head_dim)
    writer.add_value_length(cfg.head_dim)
    writer.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    writer.add_full_attention_interval(4)                                   # the config keeps it as layer_types
    writer.add_rope_freq_base(10000.)
    writer.add_rope_dimension_count(cfg.head_dim // 2)                      # partial_rotary_factor .5
    writer.add_rope_dimension_sections([2, 1, 1, 0])
    writer.add_ssm_conv_kernel(cfg.linear_conv_kernel_dim)
    _tokenizer(writer, chat_template)
    # From here to the end of the tensor loop: tests/test_hybrid_grouped.py's writer (matrix type and scale added).
    for key,value in {'ssm.group_count':kh,'ssm.time_step_rank':vh,'ssm.state_size':16,'ssm.inner_size':vh*16}.items():
        writer.add_uint32('qwen35.'+key,value)
    mapper = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35,4)
    state = {}
    leaves = {'A_log':'ssm_a','dt_bias':'ssm_dt.bias','in_proj_qkv.weight':'attn_qkv.weight',
              'in_proj_z.weight':'attn_gate.weight','in_proj_a.weight':'ssm_alpha.weight',
              'in_proj_b.weight':'ssm_beta.weight','conv1d.weight':'ssm_conv1d.weight',
              'norm.weight':'ssm_norm.weight','out_proj.weight':'ssm_out.weight'}
    # Independent explicit head listing, matching pinned upstream grouped->tiled
    # converter, rather than reusing the production inverse mapping.
    def order(dim):
        return [((k*(vh//kh)+v)*dim+d) for v in range(vh//kh) for k in range(kh) for d in range(dim)]
    for name,value in ref.state_dict().items():
        a = value.numpy().copy()
        if name.endswith('.A_log'):
            a = np.linspace(-.4,.5,vh,dtype=np.float32)
        if name.endswith('.dt_bias'):
            a = np.linspace(-.2,.3,vh,dtype=np.float32)
        if value.ndim == 2:
            a = (a * np.float32(scale)).astype(np.float32)
        mapped = mapper.get_name(name,try_suffixes=('.weight','.bias'))
        axis = None; perm = None
        if '.linear_attn.' in name:
            prefix,leaf = name.split('.linear_attn.')
            mapped = prefix.replace('model.layers.','blk.')+'.'+leaves[leaf]
            if leaf in ('in_proj_qkv.weight','conv1d.weight'):
                axis,perm = 0,list(range(2*kh*16))+[2*kh*16+i for i in order(16)]
            elif leaf == 'in_proj_z.weight': axis,perm = 0,order(16)
            elif leaf in ('in_proj_a.weight','in_proj_b.weight','A_log','dt_bias'): axis,perm = 0,order(1)
            elif leaf == 'out_proj.weight': axis,perm = 1,order(16)
        if perm is not None: a = np.take(a,perm,axis=axis)
        if name.endswith('.A_log'): a = -np.exp(a)
        elif name.endswith('norm.weight') and '.linear_attn.norm.' not in name: a = a+1
        elif '.conv1d.' in name: a = a[:,0,:]
        t = QTYPES[qtype] if value.ndim==2 else gguf.GGMLQuantizationType.F32
        packed = gguf.quantize(a,t)
        writer.add_tensor(mapped,packed,raw_dtype=t)
        decoded = gguf.dequantize(packed,t)
        if name.endswith('.A_log'): decoded = np.log(-decoded)
        elif name.endswith('norm.weight') and '.linear_attn.norm.' not in name: decoded = decoded-1
        elif '.conv1d.' in name: decoded = decoded[:,None,:]
        if perm is not None: decoded = np.take(decoded,np.argsort(perm),axis=axis)
        state[name] = torch.from_numpy(decoded.copy())
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    return cfg, state
