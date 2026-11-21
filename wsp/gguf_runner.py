"""Packed ggml base multiplication + independent FP32 logical edit multiplication.

No AutoModel.from_pretrained, model.to(), or full matrix read is used. The HF
skeleton is constructed on meta; its Linear/Embedding modules are replaced before
any parameters are materialized. Norms and biases alone become float32 tensors.
"""
from __future__ import annotations

import gc
import json
import os
import threading
import time

import numpy as np
import torch
from torch import nn

from . import delta as D
from .memory_guard import require_lease, forecast
from .packed import PackedKernel, runtime_kernel
from .capacity import context_limit, DEFAULT_CONTEXT, PREFILL_CHUNK, output_count
from .runner import HFRunner, _locked
from .runtime_events import RuntimeRefusal, job_scope
from .store import norm_index
from .overlay import (GATHER_MIN, DeltaOverlay, _axis, _index, _owned, _put, _spans, _take,  # noqa: F401  (moved; same names)
                      delta_matmul, piece_key)


class PackedLinear(nn.Module):
    def __init__(self, runner, name, bias=None):
        super().__init__()
        self.runner, self.name = runner, name
        self.out_features, self.in_features = runner.ws.shape(name)
        self.bias = bias
        # Each: (list it was built from, piece_key, dtype, device, the pieces, DeltaOverlay). Holding the
        # pieces keeps the ids in the key unique while the overlay lives.
        self._committed = self._extra = None
        register = getattr(runner, '_packed', None)
        if register is not None:
            register.append(self)

    def overlays(self, like=None):
        """The overlays for the variant the runner was set to: committed pieces, then the job's `extra`,
        in the dtype and on the device of `like` (the output they are added to; float32 on the CPU without
        one). Each is rebuilt only when its pieces or strengths change, so alternating base and edited passes
        (drift, candidate tests) and patch toggles elsewhere in the model reuse them."""
        like = torch.empty(0) if like is None else like
        r = self.runner
        committed = r.ws.pieces(self.name)
        # The workspace publishes a new list on every rebuild and never changes one in place, so the same
        # list object needs no key; a script's `extra` list grows in place, so it is always keyed.
        built = False
        if not self._reuse('_committed', committed, committed, like, same_list=True):
            built |= self._build('_committed', committed, committed, like)
        extra = (r._extra or {}).get(self.name)
        pieces = [(p, 1.) for p in extra] if extra else ()
        if not self._reuse('_extra', extra, pieces, like):
            built |= self._build('_extra', extra, pieces, like)
        if built:
            note = getattr(r, '_note_overlays', None)
            if note is not None:
                note()
        return [held[5] for held in (self._committed, self._extra) if held is not None]

    def _reuse(self, slot, source, pieces, like, same_list=False):
        """Keep the held overlay when it is for these pieces (by identity and strength) in this dtype and
        device. A different list with the same pieces, as `Workspace.rebuild` makes for every tensor after
        any toggle, is adopted without rebuilding."""
        held = getattr(self, slot)
        if not pieces or held is None or held[2] != like.dtype or held[3] != like.device:
            return False
        if same_list and held[0] is source:
            return True
        if piece_key(pieces) != held[1]:
            return False
        setattr(self, slot, (source,) + held[1:])
        return True

    def _build(self, slot, source, pieces, like):
        had = getattr(self, slot) is not None
        setattr(self, slot, None)                     # release the old copy before building the new one
        if not pieces:
            return had
        overlay = DeltaOverlay(pieces, self.out_features, self.in_features, like.dtype, like.device)
        setattr(self, slot, (source, piece_key(pieces), like.dtype, like.device, tuple(p for p, _ in pieces), overlay))
        return True

    def release(self, extra=True, committed=False):
        """Drop the job's overlay (`extra`); with `committed`, the committed one too unless its pieces are
        still the active ones, so a deleted or disabled patch's copy does not outlive it."""
        if extra:
            self._extra = None
        held = self._committed
        if committed and held is not None:
            pieces = self.runner.ws.pieces(self.name)
            if not pieces or (held[0] is not pieces and held[1] != piece_key(pieces)):
                self._committed = None

    def overlay_bytes(self):
        """Bytes the overlays hold now on top of the patch pieces themselves."""
        return sum(held[5].nbytes for held in (self._committed, self._extra) if held is not None)

    def forward(self, x):
        r, s = self.runner, self.runner.ws.store
        if getattr(r.ws,'_runtime_client',None):r.ws._runtime_client.layer_checkpoint(r.ws)
        raw = s.native(self.name)
        qtype = s.where[self.name].tensor_type
        # Raw packed row layout is unchanged; undo rotary row permutation on the result.
        native_x = x
        if self.name in s.column_permutations:
            native_x = x[..., np.argsort(s.column_permutations[self.name])]
        out = r.kernel.matmul(raw.view(np.uint8).reshape(self.out_features, -1), qtype,
                              (self.out_features, self.in_features), native_x)
        if self.name in s.row_permutations:
            out = out[..., s.row_permutations[self.name]]
        if r._variant != 'base':                      # the base variant is the packed product alone
            for o in self.overlays(out):
                o.apply(x, out)
        if self.bias is not None:
            out = out + self.bias
        return out


class PackedEmbedding(nn.Module):
    def __init__(self, runner, name):
        super().__init__()
        self.runner, self.name = runner, name
        self.num_embeddings, self.embedding_dim = runner.ws.shape(name)

    def forward(self, ids):
        r = self.runner
        if getattr(r.ws,'_runtime_client',None):r.ws._runtime_client.layer_checkpoint(r.ws)
        unique, inverse = torch.unique(ids.reshape(-1), return_inverse=True)
        rows = []
        for token in unique.tolist():
            region = (token, range(self.embedding_dim))
            row = r.ws.store.read(self.name, region)
            for piece, scale in r.active_pieces(self.name):
                D.add_to(row, region, piece, scale)
            rows.append(row)
        return torch.stack(rows)[inverse].reshape(*ids.shape, self.embedding_dim)


class GGUFRunner(HFRunner):
    def __init__(self, ws):
        # Own partial construction before a lease/native allocation can occur.
        self.ws, self.lock = ws, threading.RLock()
        self.kernel = self.model = None
        self._saved, self._gain, self._pcache = [], {}, {}
        self._named, self._modules = {}, {}
        self._packed = []                         # every PackedLinear, for releasing their overlays
        self._telemetry_stop = threading.Event()
        self._telemetry_thread = None
        try:
            self._initialize(ws)
        except BaseException as original:
            import traceback
            traceback.clear_frames(original.__traceback__)  # release exited constructor locals too
            try:
                self._dispose()
                client = getattr(ws, '_runtime_client', None)
                if client is not None: client.release()
            except BaseException as cleanup:
                # Keep the reservation armed: unknown native/thread cleanup must
                # never be relabeled idle or permit another model construction.
                ws._failed_runner = self
                raise RuntimeError(f'Runner construction failed ({original}); partial cleanup failed ({cleanup}); watchdog remains armed') from original
            raise

    def _initialize(self, ws):
        import importlib.metadata
        if importlib.metadata.version('transformers') != '5.22.2':
            raise RuntimeError('Packed mappings require transformers==5.22.2; install requirements-quantized.txt')
        from transformers import AutoConfig, AutoModelForCausalLM
        from .gguf_store import LAYOUTS, SCHEMES
        if not ws.store.layout_supported:
            raise RuntimeError(f'No validated packed runtime mapping for {ws.store.architecture}')
        require_lease(ws, loading=True)
        self.kernel = PackedKernel()
        for name in ws.names():
            if ws.store.dtype(name) not in SCHEMES or not self.kernel.supports(ws.store.where[name].tensor_type):
                raise RuntimeError(f'No packed kernel for {name} ({ws.store.dtype(name)}); no fallback')
        self.ws, self.lock = ws, threading.RLock()
        self.device, self.dtype = torch.device('cpu'), torch.float32
        managed = getattr(ws, '_runtime_client', None) is not None
        self.max_ctx = context_limit(int(os.environ.get('WSP_MAX_CTX', str(DEFAULT_CONTEXT if managed else 512))), ws.struct.n_ctx) if managed else min(512, int(os.environ.get('WSP_MAX_CTX','512')), ws.struct.n_ctx or 512)
        self._variant, self._extra = 'base', None
        self._saved, self._gain, self._pcache = [], {}, {}
        self._state, self._tok = 'base', None
        self._telemetry_stop = threading.Event()
        self._telemetry_thread = None
        config = AutoConfig.for_model(**ws.hf_config)
        config._attn_implementation = 'eager'
        with torch.device('meta'):
            self.model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32).eval()
        expected = dict(self.model.named_parameters(remove_duplicate=False))
        mapping = {}
        for name, p in expected.items():
            source = name
            if name == 'lm_head.weight' and ws.hf_config.get('tie_word_embeddings'):
                source = 'model.embed_tokens.weight'
            if source not in ws.store or tuple(p.shape) != ws.shape(source):
                raise RuntimeError(f'Unsupported GGUF mapping: {name} {tuple(p.shape)} -> {source}')
            mapping[name] = source
        unexpected = set(ws.names()) - set(mapping.values())
        if unexpected:
            raise RuntimeError(f'Unmapped GGUF tensors: {sorted(unexpected)[:8]}; refusing partial load')
        self.kernel = runtime_kernel(ws.store)
        # Validate the complete plan before any weight read.
        for path, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear):
                source = mapping[path+'.weight']
                bias = nn.Parameter(ws.store.read(mapping[path+'.bias']), requires_grad=False) if module.bias is not None else None
                self.model.set_submodule(path, PackedLinear(self, source, bias))
            elif isinstance(module, nn.Embedding):
                self.model.set_submodule(path, PackedEmbedding(self, mapping[path+'.weight']))
            elif 'RotaryEmbedding' in type(module).__name__:
                with torch.device('cpu'):
                    self.model.set_submodule(path, type(module)(config=config))
        for name, p in list(self.model.named_parameters()):
            if p.device.type != 'meta':
                continue
            source = mapping[name]
            small_conv = name.endswith('.linear_attn.conv1d.weight') and p.numel() <= 65536 and ws.store.dtype(source) == 'F32'
            if p.ndim > 1 and not small_conv:
                raise RuntimeError(f'Packed plan left dense matrix {name}; refusing materialization')
            parent, _, leaf = name.rpartition('.')
            module = self.model.get_submodule(parent)
            setattr(module, leaf, nn.Parameter(ws.store.read(source), requires_grad=False))
        if any(b.device.type == 'meta' for b in self.model.buffers()):
            raise RuntimeError('Unsupported meta buffer in GGUF model; add an explicit small-buffer mapping')
        if ws.hf_config.get('model_type') == 'qwen3_5_text':
            from .hybrid_conv import bind
            for module in self.model.modules():
                if type(module).__name__ == 'Qwen3_5GatedDeltaNet':
                    bind(module)
        self._named = dict(self.model.named_parameters())
        self._modules = dict(self.model.named_modules())
        backend = os.environ.get('WSP_GGML_DEVICE','cpu')
        self.residency = dict(backend='ggml-'+backend, ggml='0.29.0', dtype='float32 activations / packed weights',
                              device=backend, packed_bytes=ws.store.nbytes(),
                              dense_parameter_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters()),
                              forecast=forecast(ws,self.max_ctx,operation='interactive' if managed else 'legacy'),
                              mps_current=None, mps_driver=None, mps_recommended=None, overlay_bytes=0,
                              **(self.kernel.metrics() if hasattr(self.kernel,'metrics') else {}))
        if os.environ.get('WSP_GUARD_TELEMETRY'):
            self._telemetry()
            self._telemetry_thread = threading.Thread(target=self._sample_gpu, daemon=True)
            self._telemetry_thread.start()

    def _sample_gpu(self):
        while not self._telemetry_stop.wait(1):
            self._telemetry()

    def _telemetry(self, active=True):
        path = os.environ.get('WSP_GUARD_TELEMETRY')
        if path:
            from pathlib import Path
            try:
                # Packed execution owns ggml Metal, never a PyTorch MPS device.
                metrics = dict(mps_current=None, mps_driver=None, mps_recommended=None)
                if hasattr(self.kernel,'metrics'):
                    metrics.update(self.kernel.metrics())
                metrics['overlay_bytes'] = self.overlay_bytes()
            except Exception as e:
                metrics = dict(error=str(e))
            target = Path(path+'.gpu.json')
            tmp = target.with_suffix('.tmp')
            # pid: the supervisor counts only telemetry written by its live model worker.
            tmp.write_text(json.dumps(dict(time=time.time(), active=active, backend=os.environ.get('WSP_GGML_DEVICE','cpu'), pid=os.getpid(), **metrics)))
            tmp.replace(target)

    def active_pieces(self, name):
        if self._variant == 'base':
            return ()
        return list(self.ws.pieces(name)) + [(p, 1.) for p in (self._extra or {}).get(name, ())]

    def forward_options(self, tokens):
        if self.ws.hf_config.get('model_type') == 'qwen3_5_text':
            # Upstream's exact chunk algorithm otherwise pads a tiny prompt to64
            # and performs64 Python triangular updates in every recurrent layer.
            n = tokens.shape[-1]
            return dict(chunk_size=min(64,1 << (max(1,n)-1).bit_length()))
        return {}

    def overlay_bytes(self):
        """Bytes every matrix's run-time edit overlays hold now on top of the patch pieces (DeltaOverlay)."""
        return sum(lin.overlay_bytes() for lin in self._packed)

    def _note_overlays(self):
        # What the runner holds: the copies change only when an overlay is built or dropped.
        residency = getattr(self, 'residency', None)
        if residency is not None:
            residency['overlay_bytes'] = self.overlay_bytes()

    def pieces_changed(self):
        """The workspace's patch state changed (`Workspace.rebuild`): drop every committed overlay whose
        pieces are no longer the active ones now, not at the matrix's next forward pass."""
        for lin in self._packed:
            lin.release(extra=False, committed=True)
        self._note_overlays()

    def use(self, variant='current', extra=None):
        self._variant, self._extra = variant, extra
        if not extra:
            # A job's candidate pieces do not outlive it: `use('base')` after a job (the worker's
            # _drop_extra) or any variant without an extra drops every matrix's copy of them.
            for lin in self._packed:
                lin.release(extra=True, committed=False)
            self._note_overlays()
        self._restore()
        self._gain = {}
        # Matrix edits are evaluated as overlays by PackedLinear/PackedEmbedding.
        # Norms and biases are small, and retain exact restore semantics.
        if variant != 'base':
            for name in set(self.ws.touched()) | set(extra or {}):
                if name in self._named:
                    for piece, scale in self.active_pieces(name):
                        self._apply(name, piece, scale)

    @job_scope('forward',runner=True)
    @_locked
    def run(self, ids, variant='current', extra=None, names=(), hooks=None, grad=False):
        t = torch.as_tensor(ids)
        if t.ndim not in (1, 2) or (t.ndim == 2 and t.shape[0] != 1) or not 0 < t.shape[-1] <= self.max_ctx:
            raise ValueError(f'Guarded GGUF runs require batch 1 and 1..{self.max_ctx} tokens')
        if grad:
            raise RuntimeError('Packed GGUF parameter gradients are unsupported; use logical delta edits')
        require_lease(self.ws)
        return super().run(ids, variant, extra, names, hooks, False)

    def _prefill(self, ids):
        """One cache for every chunk; pinned HF derives absolute positions from it."""
        past, logits = None, None
        for start in range(0, len(ids), PREFILL_CHUNK):
            require_lease(self.ws)
            tokens = torch.tensor([list(ids[start:start+PREFILL_CHUNK])], dtype=torch.long, device=self.device)
            with torch.no_grad():
                output = self.model(input_ids=tokens, past_key_values=past, use_cache=True,
                                    logits_to_keep=1, **self.forward_options(tokens))
            past = output.past_key_values
            if past is None:
                raise RuntimeError('Backend failed to provide bounded decode state')
            logits = output.logits[0,-1].float().cpu().clone()
            del output
        return logits, past

    @job_scope('next',runner=True)
    @_locked
    def next_logits(self, ids, variant='current', extra=None, steer=None):
        """Bounded next-token display; full-position logits()/run() retain their semantics."""
        if not 0 < len(ids) <= self.max_ctx:
            raise ValueError(f'Next-token scoring needs 1..{self.max_ctx} prompt tokens')
        require_lease(self.ws)
        self.use(variant, extra)
        with self._steering(steer, variant, extra):
            logits, past = self._prefill(ids)
        del past
        return logits

    def first_token_logits(self, ids, variant='current', extra=None):
        """The next token's raw logits after ``ids`` (wsp.directions_scores): the chunked prefill of next_logits."""
        return self.next_logits(ids, variant, extra)

    # ------------------------------------------------------------------ the workspace's generation config
    # The IDE writes no generation_config.json for a GGUF (the file carries none); one copied into the workspace's
    # model/ (a GGUF paired with its source checkpoint's config) is applied where this loop can reproduce a key
    # exactly. Every other key of wsp.directions_scores.GREEDY_KEYS that the file sets to a value that changes a
    # response is refused by name at generate. Keys outside GREEDY_KEYS (sampling defaults, token ids) are not read.
    GENERATION_INERT = {'repetition_penalty': 1.0, 'no_repeat_ngram_size': 0, 'min_length': 0, 'min_new_tokens': 0}

    @classmethod
    def read_generation_config(cls, model_dir) -> dict:
        """``<model_dir>/generation_config.json`` as this loop applies it -> {'applied': {key: value}, 'refused':
        [(key, value)]}, both in GREEDY_KEYS order; a missing file (or no model_dir) applies nothing."""
        from .directions_scores import GREEDY_KEYS
        path = os.path.join(model_dir, 'generation_config.json') if model_dir else None
        if not path or not os.path.isfile(path):
            return {'applied': {}, 'refused': []}
        try:
            with open(path) as f:
                cfg = json.load(f)
        except (OSError, ValueError) as e:
            raise ValueError(f"the workspace's model/generation_config.json cannot be read: {e}") from None
        if not isinstance(cfg, dict):
            raise ValueError("the workspace's model/generation_config.json is not a JSON object")
        applied, refused = {}, []
        for key in GREEDY_KEYS:
            value = cfg.get(key)
            if value is None or (key in cls.GENERATION_INERT and value == cls.GENERATION_INERT[key]):
                continue
            if key == 'eos_token_id':
                ok = isinstance(value, list) and value and all(type(v) is int for v in value)
                if type(value) is not int and not ok:
                    raise ValueError(f"the generation config's eos_token_id must be a token id or a list of them, not {json.dumps(value)}")
            elif key == 'repetition_penalty':
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                    raise ValueError(f"the generation config's repetition_penalty must be a positive number, not {json.dumps(value)}")
                value = float(value)
            else:
                refused.append((key, value))
                continue
            applied[key] = value
        return {'applied': applied, 'refused': refused}

    def generation_config(self) -> dict:
        """The workspace's generation config as this runner applies it (read_generation_config), read once."""
        if getattr(self, '_generation_config', None) is None:
            self._generation_config = self.read_generation_config(getattr(self.ws, 'model_dir', None))
        return self._generation_config

    @staticmethod
    def repetition_penalty(logits, ids, penalty):
        """transformers' RepetitionPenaltyLogitsProcessor on one row of logits: the logit of every id in ``ids`` (the
        prompt and the response so far) divided by ``penalty`` where it is not negative, multiplied where it is
        -> new logits [V]; the same operations in the same dtype as transformers (2-D path)."""
        scores = torch.as_tensor(logits).reshape(1, -1)
        vocab = scores.shape[-1]
        index = torch.as_tensor(ids, dtype=torch.long).reshape(1, -1).clamp(max=vocab)
        scores = torch.nn.functional.pad(scores, (0, 1))
        score = torch.gather(scores, 1, index)
        score = torch.where(score < 0, score * penalty, score / penalty)
        return scores.scatter(1, index, score)[0, :vocab]

    @job_scope('generate',runner=True)
    @_locked
    def generate(self, ids, variant='current', extra=None, max_new_tokens=40, temperature=0., top_k=0, seed=None, stop_ids=(), on_token=None,
                 steer=None):
        generation = self.generation_config()
        for key, value in generation['refused']:
            raise ValueError(f"the generation config sets {key}={json.dumps(value)}, which the GGUF runner does not apply; "
                             "remove it from the workspace's model/generation_config.json or score on a transformers checkpoint")
        penalty = generation['applied'].get('repetition_penalty')
        managed = getattr(self.ws, '_runtime_client', None) is not None
        limit = int(os.environ.get('WSP_MAX_NEW_TOKENS', '2048' if managed else '64'))
        if not managed: limit = min(64,limit)
        requested_output = max_new_tokens
        max_new_tokens = output_count(max_new_tokens,len(ids),self.max_ctx,maximum=None if managed else limit)
        require_lease(self.ws)
        if not len(ids):
            raise ValueError('Generation requires a nonempty prompt')
        self.use(variant, extra)
        result, past = [], None
        timings = []
        rng = torch.Generator().manual_seed(int(seed or 0))
        eos = self.ws.hf_config.get('eos_token_id')
        stops = set(stop_ids) | ({eos} if isinstance(eos, int) else set(eos or []))
        eos = generation['applied'].get('eos_token_id')
        stops |= {eos} if isinstance(eos, int) else set(eos or [])
        seen = [int(i) for i in ids]          # the repetition penalty's ids: the prompt, then each generated token
        steering = self._steering_handles(steer, variant, extra)  # pre-hooks: each position pushed once, prefill and decode
        try:
            started = time.monotonic()
            logits, past = self._prefill(ids)
            timings.append(time.monotonic()-started)
            for step in range(max_new_tokens):
                require_lease(self.ws)
                if penalty is not None:
                    logits = self.repetition_penalty(logits, seen, penalty)
                if temperature > 0:
                    logits = logits / temperature
                    if top_k:
                        logits[logits < logits.topk(min(top_k, len(logits))).values[-1]] = -torch.inf
                    token = int(torch.multinomial(logits.softmax(-1), 1, generator=rng))
                else:
                    token = int(logits.argmax())
                result.append(token)
                seen.append(token)
                if on_token is not None: on_token(token)  # the model worker streams each sampled token
                if token in stops or step+1 == max_new_tokens:
                    break
                started = time.monotonic()
                tokens = torch.tensor([[token]], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    output = self.model(input_ids=tokens, past_key_values=past, use_cache=True,
                                        logits_to_keep=1, **self.forward_options(tokens))
                past = output.past_key_values
                if past is None:
                    raise RuntimeError('Backend failed to provide bounded decode state')
                logits = output.logits[0,-1].float().cpu().clone()
                del output
                timings.append(time.monotonic()-started)
        except RuntimeRefusal as exc:
            exc.tokens=list(result)
            raise
        finally:
            for handle in steering:
                handle.remove()
            past = None
            self.last_generation = dict(prefill_seconds=timings[0] if timings else None,
                decode_seconds=sum(timings[1:]), decode_steps=max(0,len(timings)-1),
                cache_scope='single generation call; discarded', prompt_tokens=len(ids), prefill_chunk=PREFILL_CHUNK,
                requested_output=requested_output, effective_output=max_new_tokens)
        return result

    def _dispose(self):
        self._telemetry_stop.set()
        if self._telemetry_thread:
            self._telemetry_thread.join(timeout=3)
            if self._telemetry_thread.is_alive():
                raise RuntimeError('Runner telemetry thread did not stop')
        self._restore()
        self._named.clear()
        self._modules.clear()
        self._pcache.clear()
        self._packed.clear()
        self.model = None
        if hasattr(self.kernel,'close'):
            self.kernel.close()
        gc.collect()

    def release(self):
        self._dispose()
        self._telemetry(active=False)
