"""Pinned cache/mask/recurrent functions with tiny tensors only; no model construction."""

def test_pinned_cache_mask_recurrence_and_convolution_continuation():
    import ast
    import math
    from pathlib import Path
    import torch
    import torch.nn.functional as F
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    torch.manual_seed(42)
    q, k, v = [torch.randn(1, 2, 17, 3) for _ in range(3)]
    full = (q @ k.transpose(-1, -2)) / math.sqrt(3)
    full = full.masked_fill(torch.ones(17, 17, dtype=torch.bool).triu(1), -torch.inf).softmax(-1) @ v
    for cls in [LlamaConfig, Qwen2Config, Qwen3_5TextConfig]:
        kwargs = dict(hidden_size=6, num_attention_heads=2, num_key_value_heads=2,
                      num_hidden_layers=2, head_dim=3)
        if cls is Qwen3_5TextConfig:
            kwargs['layer_types'] = ['linear_attention', 'full_attention']
        config = cls(**kwargs)
        config._attn_implementation = 'eager'
        for block in [1, 4, 8, 16, 17]:
            cache, pieces = DynamicCache(config=config), []
            layer = 1 if cls is Qwen3_5TextConfig else 0
            for start in range(0, 17, block):
                end = min(start + block, 17)
                assert cache.get_seq_length() == start
                mask = create_causal_mask(config=config, inputs_embeds=torch.zeros(1, end-start, 6),
                                          attention_mask=None, past_key_values=cache,
                                          position_ids=torch.arange(start, end)[None])
                kk, vv = cache.update(k[:, :, start:end], v[:, :, start:end], layer)
                scores = (q[:, :, start:end] @ kk.transpose(-1, -2)) / math.sqrt(3)
                pieces.append((scores + mask).softmax(-1) @ vv)
            torch.testing.assert_close(torch.cat(pieces, dim=2), full, atol=1e-6, rtol=1e-5)
        print('PASS pinned causal/cache continuation:', cls.__name__)

    import transformers
    path = Path(transformers.__file__).parent/'models/qwen3_5/modeling_qwen3_5.py'
    tree = ast.parse(path.read_text())
    names = {'l2norm', 'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.decorator_list = []
    scope = {'torch': torch, 'F': F, 'is_torchdynamo_exporting': lambda: False}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<pinned-recurrence>', 'exec'), scope)
    query, key, value = [torch.randn(1, 17, 2, 3) for _ in range(3)]
    args = [query, key, value, -torch.rand(1, 17, 2), torch.rand(1, 17, 2)]
    reference, _ = scope['torch_recurrent_gated_delta_rule'](
        *args, output_final_state=True, use_qk_l2norm_in_kernel=True)
    for block in [1, 4, 8, 16, 17]:
        state, outputs = None, []
        for start in range(0, 17, block):
            end = min(start + block, 17)
            local = [x[:, start:end] for x in args]
            single = end-start == 1 and state is not None
            fn = scope['torch_recurrent_gated_delta_rule' if single else 'torch_chunk_gated_delta_rule']
            options = {} if single else {'chunk_size': min(64, 1 << (end-start-1).bit_length())}
            output, state = fn(*local, initial_state=state, output_final_state=True,
                               use_qk_l2norm_in_kernel=True, **options)
            outputs.append(output)
        torch.testing.assert_close(torch.cat(outputs, dim=1), reference, atol=2e-6, rtol=1e-5)
    print('PASS pinned hybrid recurrent continuation including padded remainder')

    tree = ast.parse((Path(__file__).resolve().parents[1]/'wsp/hybrid_conv.py').read_text())
    names = {'_convolve', 'causal_conv1d_fn', 'causal_conv1d_update'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = {'torch': torch, 'ACT2FN': {'silu': F.silu}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<pure-convolution>', 'exec'), scope)
    x, weight, bias = torch.randn(1, 6, 17), torch.randn(6, 4), torch.randn(6)
    reference = scope['causal_conv1d_fn'](x, weight, bias, 'silu')
    config = Qwen3_5TextConfig(hidden_size=6, num_attention_heads=2, num_key_value_heads=2,
                              num_hidden_layers=2, head_dim=3,
                              layer_types=['linear_attention', 'full_attention'])
    for block in [1, 4, 8, 16, 17]:
        cache, outputs = DynamicCache(config=config), []
        for start in range(0, 17, block):
            end = min(start + block, 17)
            local, state = x[:, :, start:end], cache.get_conv_state(0)
            if state is not None and end-start == 1:
                output = scope['causal_conv1d_update'](local, state, weight, bias, 'silu')
            else:
                joined = cache.update_conv_state(local, 0, conv_kernel_size=4)
                output = scope['causal_conv1d_fn'](joined, weight, bias, 'silu')[:, :, -(end-start):]
            outputs.append(output)
        torch.testing.assert_close(torch.cat(outputs, dim=-1), reference)
    print('PASS project convolution with pinned hybrid cache continuation')
