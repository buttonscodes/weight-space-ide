"""Allocation estimates from scalar metadata only; no runtime imports or weight reads."""
GiB = 1024**3


def estimate(weights, context, hidden, layers, vocab, patch=0, hybrid=None, heads=None,
             *, operation='legacy', kv_heads=None, head_dim=None, attention_layers=None,
             intermediate=None, prefill_chunk=128):
    """Bytes for batch one FP32 activations. Legacy acceptance forecasts stay unchanged.

    Interactive: one bounded prefill query slab against the complete KV history,
    last-token logits only. Trace: full-sequence output and retained per-layer
    activations/attention patterns. Unknown metadata uses conservative fallbacks.
    """
    heads = max(1, int(heads if heads is not None else hidden))
    if operation not in ('legacy', 'interactive', 'forward', 'trace'):
        raise ValueError('Unknown forecast operation')
    if operation == 'legacy':
        parts = dict(weights=weights, runtime=2*GiB, kv=2*layers*context*hidden*4,
                     activations=8*context*max(hidden,vocab)*4, patches=patch*3,
                     temporary=512*1024**2+min(weights,GiB))
        parts['eager_attention'] = 3*heads*context*context*4
        recurrent_layers = layers
    else:
        full = layers if attention_layers is None else int(attention_layers)
        if not 0 <= full <= layers: raise ValueError('Invalid full-attention layer count')
        kv_heads = heads if kv_heads is None else int(kv_heads)
        head_dim = max(1, (hidden+heads-1)//heads) if head_dim is None else int(head_dim)
        if not 0 < kv_heads <= heads or head_dim <= 0: raise ValueError('Invalid attention geometry')
        width = max(hidden, int(intermediate or 4*hidden), heads*head_dim)
        query = min(context, prefill_chunk) if operation == 'interactive' else context
        parts = dict(weights=weights, runtime=2*GiB,
                     kv=2*full*context*kv_heads*head_dim*4,
                     # Includes one active layer's repeated K/V at query-head width.
                     activations=(8*query*width + (2*context*heads*head_dim if full else 0))*4,
                     kv_append=2*context*kv_heads*head_dim*4 if full else 0,
                     logits=(1 if operation == 'interactive' else context)*vocab*4,
                     patches=patch*3, temporary=512*1024**2+min(weights,GiB),
                     eager_attention=3*heads*query*context*4 if full else 0)
        if operation == 'trace':
            parts['retained_activations'] = 8*layers*context*width*4
            parts['retained_attention'] = full*heads*context*context*4
        recurrent_layers = layers-full if hybrid else 0
    if hybrid:
        vh,kd,vd,kh,kernel = hybrid
        parts['recurrent'] = 3*recurrent_layers*vh*kd*vd*4
        parts['convolution'] = recurrent_layers*(2*kh*kd+vh*vd)*kernel*4
        if operation != 'legacy':
            # Pinned gated-delta chunks retain per-token states/triangular work for
            # one layer. Outer prefill bounds query; internal chunk width <=64.
            parts['hybrid_workspace'] = 4*vh*query*(4*(kd+vd)+4*min(64,query))*4
    parts['total'] = sum(parts.values())
    return parts


def gguf_preview(path, context):
    """Reads only GGUF directory/metadata. Never calls dequantize or accesses tensor.data."""
    import gguf
    from pathlib import Path
    reader=gguf.GGUFReader(str(path),mode='r')
    def field(key):
        if key not in reader.fields: raise ValueError(f'Missing forecast metadata: {key}')
        return reader.fields[key].contents()
    arch=field('general.architecture')
    if arch not in ('qwen2','llama','qwen35'):
        raise ValueError(f'No validated forecast/layout for {arch}')
    hidden,layers = int(field(arch+'.embedding_length')),int(field(arch+'.block_count'))
    vocab=len(reader.fields['tokenizer.ggml.tokens'].data)
    hybrid=None
    if arch=='qwen35':
        vh=int(field(arch+'.ssm.time_step_rank'));kh=int(field(arch+'.ssm.group_count'))
        kd=int(field(arch+'.ssm.state_size'));inner=int(field(arch+'.ssm.inner_size'))
        if not 0 < kh <= vh or vh%kh or inner%vh: raise ValueError('Unsupported hybrid head layout')
        hybrid=(vh,kd,inner//vh,kh,int(field(arch+'.ssm.conv_kernel')))
    from .capacity import context_limit
    model_context = int(field(arch+'.context_length')) if arch+'.context_length' in reader.fields else None
    effective = context_limit(context,model_context)
    return dict(architecture=arch,configured_context=context,effective_context=effective,model_context=model_context,parameters=sum(int(t.n_elements) for t in reader.tensors),
                artifact_bytes=Path(path).stat().st_size, tensor_types=sorted({t.tensor_type.name for t in reader.tensors}),
                forecast=estimate(sum(int(t.n_bytes) for t in reader.tensors),effective,hidden,layers,vocab,hybrid=hybrid,heads=int(field(arch+'.attention.head_count')),
                    operation='interactive', kv_heads=int(field(arch+'.attention.head_count_kv')) if arch+'.attention.head_count_kv' in reader.fields else None,
                    head_dim=int(reader.fields[arch+'.attention.key_length'].contents()) if arch+'.attention.key_length' in reader.fields else hidden//int(field(arch+'.attention.head_count')),
                    attention_layers=sum(t.name.endswith('.attn_q.weight') for t in reader.tensors) if hybrid else layers,
                    intermediate=int(field(arch+'.feed_forward_length')) if arch+'.feed_forward_length' in reader.fields else None),
                qualification='GGUF header estimate, no patches, source hash not verified, no inference/layout proof or headroom reservation')
