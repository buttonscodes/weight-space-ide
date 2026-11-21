"""Immutable GGUF base, decoded on demand in an explicitly mapped logical layout.

Decoding reconstructs the quantized checkpoint, NOT the original floating-point
checkpoint. The runtime reads native packed bytes and never calls read for matrices.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch

from .store import Store, _natural, norm_index, region_shape, region_size

MAX_READ = 16_000_000
DECODE_CHUNK = 4_000_000   # values decoded per dequantize call by gather_columns
SCHEMES = {'F32', 'F16', 'BF16', 'Q4_0', 'Q4_1', 'Q5_0', 'Q5_1', 'Q8_0', 'Q2_K', 'Q3_K', 'Q4_K', 'Q5_K', 'Q6_K'}
LAYOUTS = {'qwen2', 'llama', 'qwen35'}


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while block := f.read(8*1024**2):
            h.update(block)
    return h.hexdigest()


def gather_columns(raw, rows, cols, qtype, out):
    """Decode logical columns ``cols`` of rows ``rows`` from packed bytes ``raw`` [rows, bytes per row]
    into ``out`` [len(rows), len(cols)].

    Only the packed blocks that hold a requested column are decoded, each once per row, several
    rows per dequantize call (at most DECODE_CHUNK decoded values per call) rather than one call
    per row. The values are those of a per-row decode: dequantization is block by block."""
    import gguf
    block, width = gguf.GGML_QUANT_SIZES[qtype]
    blocks, inverse = np.unique(cols // block, return_inverse=True)
    within = cols % block
    packed_rows = raw.reshape(raw.shape[0], -1, width)
    step = max(1, DECODE_CHUNK // (len(blocks) * block))
    for s in range(0, len(rows), step):
        chunk = rows[s:s + step]
        packed = packed_rows[np.ix_(chunk, blocks)]                         # [r, blocks, width]
        decoded = gguf.dequantize(packed.reshape(-1, width), qtype).reshape(len(chunk), len(blocks), block)
        out[s:s + len(chunk)] = decoded[:, inverse, within]
    return out


def logical_name(name, architecture):
    if architecture not in LAYOUTS:
        return name
    if architecture == 'qwen35':
        m = re.fullmatch(r'blk\.(\d+)\.ssm_a',name)
        if m:
            return f'model.layers.{m[1]}.linear_attn.A_log'
        m = re.fullmatch(r'blk\.(\d+)\.ssm_dt.bias',name)
        if m:
            return f'model.layers.{m[1]}.linear_attn.dt_bias'
    for src, dst in [('token_embd.', 'model.embed_tokens.'), ('output_norm.', 'model.norm.'), ('output.', 'lm_head.')]:
        if name.startswith(src):
            return dst + name[len(src):]
    m = re.fullmatch(r'blk\.(\d+)\.([^.]+)\.(weight|bias)', name)
    if not m:
        return name
    leaf = {'attn_q': 'self_attn.q_proj', 'attn_k': 'self_attn.k_proj', 'attn_v': 'self_attn.v_proj',
            'attn_output': 'self_attn.o_proj', 'attn_norm': 'input_layernorm', 'ffn_norm': 'post_attention_layernorm',
            'ffn_gate': 'mlp.gate_proj', 'ffn_up': 'mlp.up_proj', 'ffn_down': 'mlp.down_proj',
            'post_attention_norm':'post_attention_layernorm','attn_q_norm':'self_attn.q_norm','attn_k_norm':'self_attn.k_norm',
            'attn_qkv':'linear_attn.in_proj_qkv','attn_gate':'linear_attn.in_proj_z',
            'ssm_alpha':'linear_attn.in_proj_a','ssm_beta':'linear_attn.in_proj_b',
            'ssm_conv1d':'linear_attn.conv1d','ssm_norm':'linear_attn.norm','ssm_out':'linear_attn.out_proj'}.get(m[2])
    return f'model.layers.{m[1]}.{leaf}.{m[3]}' if leaf else name


class GGUFStore(Store):
    remote = False

    def __init__(self, path):
        try:
            import gguf
        except ImportError as exc:
            raise RuntimeError('GGUF import needs optional dependencies: python -m pip install -r requirements-quantized.txt (using the serve interpreter); then restart serve') from exc
        self.path = os.path.abspath(path)
        self.reader = gguf.GGUFReader(self.path, mode='r')
        if self.reader.endianess != gguf.GGUFEndian.LITTLE:
            raise ValueError('Only little-endian GGUF is supported; convert byte order explicitly first')
        self.architecture = self.reader.fields['general.architecture'].contents()
        self.layout_supported = self.architecture in LAYOUTS
        self.layout_reason = ''
        if self.architecture == 'qwen35':
            kh, vh = self.field('qwen35.ssm.group_count'), self.field('qwen35.ssm.time_step_rank')
            if not isinstance(kh,int) or not isinstance(vh,int) or kh <= 0 or vh <= 0 or vh % kh:
                self.layout_supported = False
                self.layout_reason = 'Hybrid value-head count must be a positive multiple of key-head count; raw-name import only'
            if self.field('qwen35.nextn_predict_layers',0):
                self.layout_supported = False
                self.layout_reason = 'Select a text-only GGUF converted with --no-mtp; MTP/vision/speculative components are unsupported'
            if any('hadamard' in k for k in self.reader.fields):
                self.layout_supported = False
                self.layout_reason = 'Hadamard-folded hybrid weights require a separate validated mapping'
        self.where = {}
        self.row_permutations = {}
        self.column_permutations = {}
        self.files = []
        self.cache_dir = ''
        for t in self.reader.tensors:
            name = logical_name(t.name, self.architecture if self.layout_supported else 'unknown')
            if name in self.where:
                raise ValueError(f'GGUF logical tensor collision: {name}')
            self.where[name] = t
        self._names = sorted(self.where, key=_natural)
        if self.architecture == 'llama':
            for name in self._names:
                if '.q_proj.' in name or '.k_proj.' in name:
                    key = 'llama.attention.head_count' + ('_kv' if '.k_proj.' in name else '')
                    heads = self.field(key, self.field('llama.attention.head_count'))
                    rows = self.shape(name)[0]
                    if not heads or rows % (2*heads):
                        raise ValueError(f'Cannot reverse GGUF rotary layout for {name}')
                    self.row_permutations[name] = np.arange(rows).reshape(heads, -1, 2).swapaxes(1, 2).reshape(-1)
        if self.architecture == 'qwen35' and self.layout_supported and kh and vh != kh:
            kd = self.field('qwen35.ssm.state_size')
            inner = self.field('qwen35.ssm.inner_size')
            if not kd or not inner or inner % vh:
                raise ValueError('Missing/inconsistent hybrid head dimensions')
            vd, ratio = inner // vh, vh // kh
            # llama.cpp f477f795 conversion/qwen.py _LinearAttentionVReorderBase:
            # raw is grouped HF heads transposed [K, V/K, D] -> [V/K, K, D].
            # Store logical->raw indices, never permute/reconstruct full weights.
            def inverse(dim):
                return np.argsort(np.arange(vh*dim).reshape(kh,ratio,dim).swapaxes(0,1).reshape(-1))
            value_map, scalar_map = inverse(vd), inverse(1)
            qk = 2*kh*kd
            qkv_map = np.concatenate((np.arange(qk), qk+value_map))
            for name in self._names:
                if '.linear_attn.' not in name:
                    continue
                leaf = name.split('.linear_attn.')[1]
                if leaf in ('in_proj_qkv.weight','conv1d.weight'):
                    self.row_permutations[name] = qkv_map
                elif leaf == 'in_proj_z.weight':
                    self.row_permutations[name] = value_map
                elif leaf in ('in_proj_a.weight','in_proj_b.weight','A_log','dt_bias'):
                    self.row_permutations[name] = scalar_map
                elif leaf == 'out_proj.weight':
                    self.column_permutations[name] = value_map
                if name in self.row_permutations and len(self.row_permutations[name]) != self.shape(name)[0]:
                    raise ValueError(f'Hybrid row mapping shape mismatch: {name}')
                if name in self.column_permutations and len(self.column_permutations[name]) != self.shape(name)[1]:
                    raise ValueError(f'Hybrid column mapping shape mismatch: {name}')

    def field(self, name, default=None):
        f = self.reader.fields.get(name)
        return default if f is None else f.contents()

    def shape(self, name):
        shape = tuple(int(x) for x in reversed(self.where[name].shape))
        return (shape[0],1,shape[1]) if '.linear_attn.conv1d.' in name else shape

    def dtype(self, name):
        return self.where[name].tensor_type.name

    def is_float(self, name):
        # Logical reconstructed weights are editable; packed integers never are.
        shape = self.shape(name)
        return self.dtype(name) in SCHEMES and (len(shape) in (1,2) or
               ('.linear_attn.conv1d.' in name and math.prod(shape)<=65536 and self.dtype(name)=='F32'))

    def nbytes(self):
        return sum(t.n_bytes for t in self.where.values())

    def size_bytes(self, name):
        return int(self.where[name].n_bytes)

    def fetch(self, name):
        pass

    def local(self, name):
        return True

    def native(self, name):
        return self.where[name].data

    def read(self, name, region=None):
        import gguf
        shape = self.shape(name)
        region = norm_index((), shape) if region is None else region
        if region_size(region) > MAX_READ:
            raise ValueError('GGUF read exceeds 16M logical values; use row slabs / a smaller selection '
                             f'({name}: {region_size(region):,} values asked for, limit {MAX_READ:,})')
        if not self.is_float(name):
            raise ValueError(f'{self.dtype(name)} decoding is unsupported; keep this tensor read-only')
        t = self.where[name]
        if '.linear_attn.conv1d.' in name:
            if math.prod(shape) > 65536 or self.dtype(name) != 'F32':
                raise ValueError('Only small F32 depthwise convolutions have a validated logical mapping')
            from .store import region_index
            values = gguf.dequantize(t.data,t.tensor_type).reshape(shape)
            if name in self.row_permutations:
                values = values[self.row_permutations[name]]
            return torch.from_numpy(np.array(values[region_index(region)],dtype=np.float32))
        if len(shape) not in (1, 2):
            raise ValueError('This GGUF decoder currently supports vector and matrix logical reads only')
        if len(shape) == 1:
            values = gguf.dequantize(t.data, t.tensor_type).reshape(shape)
            if name in self.row_permutations:
                values = values[self.row_permutations[name]]
            if self.architecture == 'qwen35' and self.layout_supported:
                if name.endswith('.A_log'):
                    if not (values < 0).all():
                        raise ValueError('Qwen3.5 ssm_a must be strictly negative')
                    values = np.log(-values)
                elif name.endswith('norm.weight') and '.linear_attn.norm.' not in name:
                    values = values-1.0
            idx = region[0]
            return torch.from_numpy(np.array(values[idx if isinstance(idx, int) else list(idx)], dtype=np.float32))
        r, c = region
        rows = np.array([r] if isinstance(r, int) else list(r), dtype=np.int64)
        cols = np.array([c] if isinstance(c, int) else list(c), dtype=np.int64)
        if name in self.row_permutations:
            rows = self.row_permutations[name][rows]
        if name in self.column_permutations:
            cols = self.column_permutations[name][cols]
        result = np.empty((len(rows), len(cols)), dtype=np.float32)
        if not len(cols) or not len(rows):
            return torch.from_numpy(result.reshape(region_shape(region)))
        gather_columns(t.data.view(np.uint8).reshape(shape[0], -1), rows, cols, t.tensor_type, result)
        return torch.from_numpy(result.reshape(region_shape(region)))

    def read_block(self, name):
        """(logical axis, values per packed block along it) for a quantized matrix, else None:
        a read of logical columns decodes whole blocks of this many columns."""
        import gguf
        t = self.where.get(name)
        if t is None or len(self.shape(name)) != 2 or name in self.column_permutations:
            return None
        block = gguf.GGML_QUANT_SIZES[t.tensor_type][0]
        return (1, int(block)) if block > 1 else None

    def config(self):
        a = self.architecture
        if not self.layout_supported:
            return {'model_type': a}
        if a == 'qwen35':
            from transformers.integrations.gguf.gguf_config_mapping import get_gguf_config
            metadata = {k:v.contents() for k,v in self.reader.fields.items()
                        if k.startswith(('general.','qwen35.')) or k in ('tokenizer.ggml.bos_token_id','tokenizer.ggml.eos_token_id','tokenizer.ggml.padding_token_id')}
            metadata['tokenizer.ggml.tokens'] = self.shape('model.embed_tokens.weight')[0]
            config = get_gguf_config(metadata,tuple(t.name for t in self.reader.tensors))
            # Qwen text generation ends on im_end or endoftext; the GGUF has a
            # single eos field while the source generation config lists both.
            # Verify token strings rather than assuming numeric IDs across vocabularies.
            tokens = self.field('tokenizer.ggml.tokens',[])
            stops = [i for i in (config.get('eos_token_id'),config.get('bos_token_id'))
                     if isinstance(i,int) and 0 <= i < len(tokens) and tokens[i] in ('<|im_end|>','<|endoftext|>')]
            if stops:
                config['eos_token_id'] = list(dict.fromkeys(stops))
            return config
        d = self.field(a+'.embedding_length')
        heads = self.field(a+'.attention.head_count')
        return dict(model_type='qwen2' if a == 'qwen2' else 'llama', hidden_size=d,
                    intermediate_size=self.field(a+'.feed_forward_length'),
                    num_hidden_layers=self.field(a+'.block_count'), num_attention_heads=heads,
                    num_key_value_heads=self.field(a+'.attention.head_count_kv', heads),
                    head_dim=d//heads, max_position_embeddings=self.field(a+'.context_length', 2048),
                    rms_norm_eps=self.field(a+'.attention.layer_norm_rms_epsilon', 1e-6),
                    rope_theta=self.field(a+'.rope.freq_base', 10000.0),
                    vocab_size=self.shape('model.embed_tokens.weight')[0],
                    tie_word_embeddings='lm_head.weight' not in self.where,
                    bos_token_id=self.field('tokenizer.ggml.bos_token_id'),
                    eos_token_id=self.field('tokenizer.ggml.eos_token_id'))

    def capabilities(self):
        types = sorted({self.dtype(n) for n in self.names()})
        unsupported = sorted(set(types)-SCHEMES)
        unreadable = [n for n in self.names() if not self.is_float(n)]
        layout = self.layout_supported
        from .packed import availability
        ready, backend_reason = availability()
        runnable = layout and not unreadable and ready
        reason = (backend_reason if layout else
                  f'No validated runtime mapping for {self.architecture}; raw-name import only.')
        if unsupported:
            reason = f'Unsupported tensor types: {unsupported}; no dequantization fallback.'
        if self.layout_reason:
            reason = self.layout_reason
        return dict(format='gguf', architecture=self.architecture, tensor_types=types,
                    representation='reconstructed quantized base + FP32 companion patches',
                    import_read=dict(supported=not unreadable, reason='Bounded logical vector/matrix and validated depthwise reads', unsupported_tensors=unreadable),
                    logical_edit=dict(supported=not unreadable, reason='FP32 patches in displayed logical coordinates',unsupported_tensors=unreadable),
                    generation=dict(supported=runnable, reason=backend_reason if runnable else reason),
                    trace_intervention=dict(supported=runnable, reason='Python layer/head/neuron hooks; no parameter gradients' if runnable else reason),
                    multimodal=dict(supported=False, reason='Text-only backend; separate vision/mmproj is not loaded'),
                    speculative=dict(supported=False, reason='MTP/dflash components are not loaded'),
                    adapter_export=dict(supported=True, reason='WSP companion bundle only; not PEFT/llama.cpp LoRA'),
                    merged_export=dict(supported=runnable,
                        reason='Standard mixed Q8/F16/F32 export or lossy same-scheme, bounded rows; behavioral status UNVALIDATED until explicit guarded gain/drift validation; companion preserves edits'
                        if runnable else 'No validated standalone merge for this layout/backend; export a companion bundle'))


def open_gguf(root, spec, metadata_only=False):
    from .workspace import Workspace
    from .zoo import slug
    path = os.path.abspath(spec)
    key = hashlib.sha256(path.encode()).hexdigest()[:10]
    directory = os.path.join(root, slug(Path(path).stem)+'-'+key)
    if os.path.exists(os.path.join(directory, 'workspace.json')):
        return Workspace.load(directory, metadata_only=metadata_only)
    store = GGUFStore(path)
    os.makedirs(os.path.join(directory, 'patches'))
    os.makedirs(os.path.join(directory, 'model'))
    # The GGUF itself supplies the model config; the tokenizer is reconstructed
    # by Transformers metadata conversion (return_tensors=False), never weights.
    config = store.config()
    model_dir = os.path.join(directory, 'model')
    Path(model_dir, 'config.json').write_text(json.dumps(config, indent=2))
    try:
        from transformers import PreTrainedTokenizerFast
        from transformers.integrations.gguf import convert_gguf_tokenizer, get_gguf_tokenizer
        architecture, token_data, token_config = get_gguf_tokenizer(path)
        tokenizer, _ = convert_gguf_tokenizer(architecture, token_data)
        tokenizer_profile(store,tokenizer,token_config)
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, **token_config).save_pretrained(model_dir)
    except (ImportError, KeyError, ValueError, NotImplementedError) as e:
        tokenizer = None
        Path(model_dir, 'tokenizer-error.txt').write_text(str(e))
    meta = dict(kind='gguf', name=Path(directory).name, gguf_path=path,
                base_sha256=None if metadata_only else file_hash(path), model_dir='model')
    Path(directory, 'workspace.json').write_text(json.dumps(meta, indent=2))
    return Workspace.load(directory, metadata_only=metadata_only)


def tokenizer_profile(store, tokenizer, token_config):
    """Exact source-profile correction where GGUF omits tokenizer normalizer data.

    Qwen3.8-27B source37561816 tokenizer.json specifies NFC; Transformers5.22.2
    GGUF conversion omits it. This is deliberately a named profile, not an
    assumption about all GGUF/Qwen checkpoints. Vocab/merges stay from GGUF.
    """
    if store.architecture == 'qwen35' and store.field('general.name') == 'Qwen3.8-27B':
        if store.field('tokenizer.ggml.pre') != 'qwen35':
            raise ValueError('Qwen3.8-27B requires its verified qwen35 tokenizer profile')
        from tokenizers import normalizers
        tokenizer.normalizer = normalizers.NFC()
        token_config['wsp_tokenizer_profile'] = 'Qwen3.8-27B/source37561816/NFC'
        # The source and converted qwen35 split regex are identical; do not apply
        # the unrelated Mistral regex rewrite suggested by the generic heuristic.
        token_config['fix_mistral_regex'] = False
