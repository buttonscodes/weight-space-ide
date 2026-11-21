"""Running a model, base and edited, behind one interface.

    FunctionalRunner   this project's own small models (wsp.model.forward on a parameter dict)
    HFRunner           any checkpoint `transformers` can run (AutoModelForCausalLM)

A runner is created only when something has to be *run* (a completion, a test, a trace).
Viewing, analysing and editing weights never create one, so a model that is too large to run
here can still be programmed here.

Both runners expose the same hook points, whatever the architecture calls its modules:

    blocks.L.hook_resid_pre / hook_resid_post    the residual stream before / after layer L
    blocks.L.hook_attn_out / hook_mlp_out        what each sublayer adds to the stream
    blocks.L.attn.hook_z                         per-head mixed values [batch, pos, head, d_head] (input of the output projection)
    blocks.L.attn.hook_pattern                   attention patterns [batch, head, dst, src]   (read only on HF models)
    blocks.L.mlp.hook_post                       neuron activations (input of the MLP's output projection)

and the same three variants of the weights: "base", "current" (base + enabled patches) and a
candidate (`extra`: uncommitted pieces on top of current).  The HF runner holds ONE copy of
the model and switches variants by adding the patch pieces to the parameters in place and
putting the saved originals back, so the cost of switching is the size of the edit.  A candidate
is applied once and stays applied while the same pieces are passed (the same lists of the same
piece objects), so a run of calls with one candidate pays for it once.
"""
from __future__ import annotations

import contextlib
import os
import re
import threading

import torch

from . import runtime_layouts
from .generation_stop import GenerationStopped
from .store import region_index


# Hook kind (blocks.L.<kind>) -> (the key of the layer's module path in Layer.modules, pre- or post-hook on it).
# The mixer kinds are a linear-attention output projection's input and output, the shared kinds the shared expert's
# down projection's (wsp.directions: COMPONENT_SOURCES).
HOOK_KINDS = {"hook_resid_pre": ("layer", "pre"), "hook_resid_post": ("layer", "post"), "hook_attn_out": ("attn", "post"),
              "hook_mlp_out": ("mlp", "post"), "attn.hook_z": ("o_proj", "pre"), "mlp.hook_post": ("mlp_out", "pre"),
              "mixer.hook_in": ("mixer_out", "pre"), "mixer.hook_out": ("mixer_out", "post"),
              "shared.hook_in": ("shared_out", "pre"), "shared.hook_out": ("shared_out", "post")}
# Kinds whose activations are [tokens, width] (a MoE block flattens batch and positions before its shared expert);
# they are shown to hooks and caches as [1, tokens, width], one prompt at a time.
FLAT_KINDS = frozenset({"shared.hook_in", "shared.hook_out"})


def available_memory() -> int | None:
    """Available/reclaimable bytes on macOS or Linux; unknown is explicit."""
    try:
        from .memory_guard import snapshot
        return snapshot()['available']
    except Exception:
        pass
    return None


# How HFRunner applies patches, by the dtype it runs in. "in_place": added to the parameters (one rounding to the
# run dtype per piece). "side": on the weight of an nn.Linear, a float32 path on the module's output in PEFT's order,
# out = (base_out.float() + x.float() @ delta.T).to(dtype), the weight itself untouched; every other tensor in place.
# WSP_PATCH_APPLY=in_place|side overrides (for comparisons).
APPLY = {torch.float32: "in_place", torch.float64: "in_place", torch.bfloat16: "side", torch.float16: "side"}


# A pattern run on a model loaded with the library's default attention (bfloat16 / float16) switches it to eager for
# that pass (HFRunner._eager_attention); these are the config attributes transformers keeps that choice in.
ATTN_STATE = ("_attn_implementation_internal", "_attn_was_changed")
PATTERNS_NEED_EAGER = ("attention patterns need the eager attention kernel, which this library version cannot switch to "
                       "on a model loaded with its default attention; run in float32 for patterns")


def patch_apply(dtype) -> str:
    """The patch mode of a run in ``dtype``: WSP_PATCH_APPLY when set, else APPLY's entry (in place for a dtype it
    does not list)."""
    mode = os.environ.get("WSP_PATCH_APPLY")
    if not mode:
        return APPLY.get(dtype, "in_place")
    if mode not in ("in_place", "side"):
        raise RuntimeError("WSP_PATCH_APPLY must be in_place or side")
    return mode


def nondefault_apply(runner) -> str | None:
    """The runner's patch mode when it is not its dtype's default in APPLY, else None (also for runners without
    one): run descriptions and keys name it only then, so a run in its dtype's default mode is described as before."""
    mode = getattr(runner, "patch_apply", None)
    if mode is None or mode == APPLY.get(getattr(runner, "dtype", None), "in_place"):
        return None
    return mode


def _side_hook(overlays):
    """A forward hook adding ``overlays`` (wsp.overlay.DeltaOverlay, float32) to an nn.Linear's output in float32
    and handing the sum back in the output's dtype: PEFT's LoRA order (base output, upcast, plus the change)."""
    def hook(module, args, kwargs, out):
        x = args[0] if args else kwargs["input"]
        out32 = out.to(torch.float32, copy=True)
        for overlay in overlays:
            overlay.apply(x, out32)
        return out32.to(out.dtype)
    return hook


def _locked(fn):
    """Runner methods switch weights in place, so one forward pass at a time."""
    from functools import wraps
    @wraps(fn)
    def wrapper(self, *a, **kw):
        with self.lock:
            return fn(self, *a, **kw)
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


class TokenCallback:
    """A transformers ``streamer``: ``on_token`` for each generated token id.

    ``generate`` first puts the prompt, then every new token. It is also passed as
    a stopping criterion: once ``on_token`` raised ``GenerationStopped`` (the person
    stopped the request), ``generate`` ends normally at its next stopping check.
    """
    def __init__(self, on_token):
        self.on_token, self.prompt, self.stopped = on_token, True, False

    def put(self, value):
        if self.prompt:
            self.prompt = False
            return
        for token in value.reshape(-1).tolist():
            try:
                self.on_token(int(token))
            except GenerationStopped:
                self.stopped = True

    def end(self):
        pass

    def __call__(self, input_ids, scores=None, **kwargs):
        return torch.full((input_ids.shape[0],), self.stopped, dtype=torch.bool, device=input_ids.device)


class Runner:
    lock: threading.RLock

    def forward_options(self, tokens):
        return {}

    def logits(self, ids, variant: str = "current", extra=None, tail: int | None = None) -> torch.Tensor:
        """[tokens, vocabulary]; with ``tail``, only that many trailing rows."""
        ids = list(ids)[-self.max_ctx:] if self.max_ctx else list(ids)
        with torch.no_grad():
            out = self.run(ids, variant, extra)[0][0]
        return out[-int(tail):] if tail else out

    def first_token_logits(self, ids, variant: str = "current", extra=None) -> torch.Tensor:
        """The raw logits [vocabulary] of the next token after ``ids`` (wsp.directions_scores)."""
        return self.logits(ids, variant, extra, tail=1)[-1]

    def chat_prompt(self, messages: list[dict], **template_options) -> str | None:
        return None

    def available(self, name: str) -> bool:
        return True

    def release(self) -> None:
        pass


# -------------------------------------------------------------------------------------- this project's own models
class FunctionalRunner(Runner):
    def __init__(self, ws):
        self.ws = ws
        self.cfg = ws.cfg
        self.max_ctx = ws.cfg.n_ctx
        self._params: dict = {}
        self.lock = threading.RLock()

    @_locked
    def params(self, variant: str = "current", extra=None) -> dict:
        ws = self.ws
        key = "base" if variant == "base" else ("current", ws.version)
        if key not in self._params:
            self._params = {k: v for k, v in self._params.items() if k == "base"}
            self._params[key] = {n: ws.read(n, (), "base" if variant == "base" else "current") for n in ws.names()}
        p = self._params[key]
        if extra and variant != "base":
            p = dict(p)
            for n in extra:
                p[n] = ws.read(n, (), "current", extra)
        return p

    @_locked
    def run(self, ids, variant="current", extra=None, names=(), hooks=None, grad=False):
        from .model import forward
        t = torch.as_tensor(ids)
        cache: dict = {}
        want = list(names)
        per_head = any(n.endswith("attn.hook_result") for n in want)
        with torch.enable_grad() if grad else torch.no_grad():
            out = forward(self.params(variant, extra), self.cfg, t, hooks=hooks, cache=cache if want else None,
                          names=want or None, per_head=per_head)
        return out, cache

    @_locked
    def generate(self, ids, variant="current", extra=None, max_new_tokens=40, temperature=0.0, top_k=0, seed=None, stop_ids=(), on_token=None,
                 steer=None):
        from .model import generate
        hooks = None
        if steer is not None:
            from .steering import hooks_for
            hooks = hooks_for(self.ws, steer, variant, extra) or None
        return generate(self.params(variant, extra), self.cfg, list(ids), max_new_tokens=max_new_tokens, temperature=temperature,
                        top_k=top_k, seed=seed, stop_ids=stop_ids, on_token=on_token, hooks=hooks)

    @_locked
    def unembed(self, x: torch.Tensor, variant="current", extra=None) -> torch.Tensor:
        p, cfg = self.params(variant, extra), self.cfg
        if cfg.use_ln:
            x = x - x.mean(-1, keepdim=True)
            x = x / (x.pow(2).mean(-1, keepdim=True) + cfg.ln_eps).sqrt() * p["ln_final.w"] + p["ln_final.b"]
        return x @ p["unembed.W_U"] + p["unembed.b_U"]

    @_locked
    def final_norm(self, x: torch.Tensor, variant="current", extra=None) -> torch.Tensor:
        """The model's final norm applied to residual x [..., d_model] (what the unembedding reads); x itself
        when the model has none."""
        p, cfg = self.params(variant, extra), self.cfg
        if not cfg.use_ln:
            return x
        x = x - x.mean(-1, keepdim=True)
        return x / (x.pow(2).mean(-1, keepdim=True) + cfg.ln_eps).sqrt() * p["ln_final.w"] + p["ln_final.b"]

    @_locked
    def final_gain(self, x: torch.Tensor, variant="current", extra=None):
        """Frozen-scale linearisation of the final norm at residual x: (gain per coordinate, centres?)."""
        cfg = self.cfg
        if not cfg.use_ln:
            return torch.ones(cfg.d_model), False
        xc = x - x.mean(-1, keepdim=True)
        scale = (xc.pow(2).mean(-1, keepdim=True) + cfg.ln_eps).sqrt()
        return self.params(variant, extra)["ln_final.w"] / scale, True


# -------------------------------------------------------------------------------------- anything transformers can run
class HFRunner(Runner):
    def __init__(self, ws):
        try:
            import transformers  # noqa: F401
        except ImportError as e:
            raise RuntimeError("running this model needs the `transformers` package (pip install transformers)") from e
        if ws.store.remote:
            raise RuntimeError(f"the weights of {ws.name} are still on the Hub. Reading and editing work from there; to run the "
                               f"model, download it first:  python -m wsp pull {ws.name}")
        from transformers import AutoModelForCausalLM
        if ws.hf_config.get('quantization_config') or ws.hf_config.get('quantization') or not all(ws.store.is_float(n) for n in ws.names()):
            raise RuntimeError('Quantized safetensors runtime/edit mapping is not implemented. Packed MLX/FP8 weights cannot be treated as ordinary floats.')
        if os.environ.get('WSP_GUARD_LEASE'):
            from .memory_guard import require_lease
            require_lease(ws)
        self.ws = ws
        self.lock = threading.RLock()
        dtype = getattr(torch, os.environ.get("WSP_RUN_DTYPE", "float32"))
        self.patch_apply = patch_apply(dtype)                                            # refused before any load
        self.device = torch.device(os.environ.get("WSP_DEVICE", "cpu"))                  # e.g. cuda, cuda:1, mps
        # the tensors the model loads (not a vision tower or MTP head it skips) + working room
        need = runtime_layouts.loaded_params(ws) * torch.empty((), dtype=dtype).element_size() * 1.15
        have = available_memory()
        if have is None:
            raise RuntimeError('Available memory telemetry is unknown; refusing model load')
        if have is not None and need > have:
            raise RuntimeError(f"running {ws.name} needs about {need / 1e9:.1f} GB of memory as {os.environ.get('WSP_RUN_DTYPE', 'float32')} "
                               f"and {have / 1e9:.1f} GB is available. Its weights can still be viewed and edited; to run it, use a "
                               f"machine with more memory (or WSP_RUN_DTYPE=bfloat16 to halve the need)")
        kw = {"low_cpu_mem_usage": True}
        model = None
        # float32 / float64: eager attention first (it returns attention patterns; in float32 the kernels agree to about
        # 1e-6). bfloat16 / float16: the library's default kernel, as Heretic loads it; eager is several times less
        # accurate there. A pattern run switches to eager for its own forward pass (run).
        eager = ({"dtype": dtype, "attn_implementation": "eager"}, {"torch_dtype": dtype, "attn_implementation": "eager"})
        attempts = (() if dtype in (torch.bfloat16, torch.float16) else eager) + ({"dtype": dtype}, {"torch_dtype": dtype})
        for attempt in attempts:
            try:
                model = AutoModelForCausalLM.from_pretrained(ws.model_dir, **kw, **attempt)
                break
            except (TypeError, ValueError) as e:
                err = e
        if model is None:
            raise RuntimeError(f"transformers could not load {ws.name}: {err}")
        self.model = model.eval()
        if self.device.type != "cpu":
            self.model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dtype = dtype
        # the attention kernel of ordinary runs, as the loaded model reports it ("eager", "sdpa", ...)
        self.attention = getattr(self.model.config, "_attn_implementation", None)
        self.max_ctx = int(ws.struct.n_ctx or 0) or 0
        self._named = dict(self.model.named_parameters())
        self._named.update({k: v for k, v in self.model.named_buffers() if k not in self._named})
        self._modules = dict(self.model.named_modules())
        self._pcache: dict = {}
        self._saved: list = []
        self._side_handles: list = []      # the side path's forward hooks while a variant is applied
        self._overlays: dict = {}          # (name, "committed" | "extra") -> (piece_key, the pieces, DeltaOverlay)
        self._owners = None                # id(parameter) -> [(attribute, module)], built on the first side patch
        self._state = "base"
        self._tok = None
        self._gain: dict = {}

    # ------------------------------------------------------------------ name resolution
    def _find(self, table: dict, name: str):
        if name in table:
            return table[name]
        c = [k for k in table if k.endswith("." + name) or name.endswith("." + k)]
        if len(c) == 1:
            return table[c[0]]
        if c:
            return table[min(c, key=len)]
        return None

    def _resolve(self, name: str):
        """Checkpoint tensor ``name`` -> (the loaded parameter, the index of the tensor in it); wsp.runtime_layouts
        maps the checkpoint's names to the loaded model's (identity for layouts it does not list)."""
        if name not in self._pcache:
            shape = tuple(self.ws.shape(name))
            try:
                runtime, index = runtime_layouts.runtime_name(self.ws.hf_config, name, shape)
            except runtime_layouts.Skipped as e:
                raise KeyError(f"checkpoint tensor {name} cannot be changed at run time: {e.reason}") from None
            p = self._named[runtime] if runtime in self._named else self._find(self._named, runtime)
            if p is None or tuple(p[index].shape) != shape:
                raise KeyError(f"checkpoint tensor {name} has no matching parameter in the loaded model")
            self._pcache[name] = (p, index)
        return self._pcache[name]

    def param(self, name: str) -> torch.Tensor:
        p, index = self._resolve(name)
        return p[index] if index else p

    def module(self, path: str | None):
        return self._find(self._modules, runtime_layouts.module_path(self.ws.hf_config, path)) if path else None

    # ------------------------------------------------------------------ variants: patches applied in place or beside
    def _apply(self, name: str, piece, scale: float) -> None:
        """Add one piece to the parameter in place with one rounding to its dtype: the sum is formed in float32 (or
        the parameter's own wider dtype) and rounded once, so a bfloat16 weight is not rounded twice (the delta,
        then the sum). In float32 this is the same arithmetic as adding the float32 delta directly."""
        p, index = self._resolve(name)
        wide = torch.promote_types(p.dtype, torch.float32)
        with torch.no_grad():
            t = p.data[index]                                 # a view: the checkpoint tensor inside the parameter
            if piece.kind == "sparse":
                ix = tuple(c.to(p.device) for c in piece.coords.T)
                self._saved.append((p, index, ix, t[ix].clone()))
                t[ix] = (t[ix].to(wide) + (scale * piece.values).to(p.device, wide)).to(p.dtype)
            else:
                ix = region_index(piece.region)
                self._saved.append((p, index, ix, t[ix].clone()))
                t[ix] = (t[ix].to(wide) + (scale * piece.dense()).to(p.device, wide)).to(p.dtype)

    def _restore(self) -> None:
        if getattr(self, "_side_handles", None):             # GGUFRunner inherits this and never has side hooks
            for h in self._side_handles:
                h.remove()
            self._side_handles = []
        with torch.no_grad():
            for p, index, ix, old in reversed(self._saved):
                p.data[index][ix] = old
        self._saved = []

    def _side_module(self, name: str):
        """The nn.Linear whose whole weight is checkpoint tensor ``name``, when patches on it take the side path (mode
        "side"); None for anything else: a norm, a bias, an embedding, a slice of a fused parameter (index not ()),
        a Conv1D weight, or a parameter more than one module holds (a tied embedding and output head)."""
        mode = getattr(self, "patch_apply", None) or APPLY.get(getattr(self, "dtype", None), "in_place")
        if mode != "side":
            return None
        p, index = self._resolve(name)
        if index != ():
            return None
        if self._owners is None:
            owners: dict = {}
            for _, module in self.model.named_modules(remove_duplicate=False):
                for attr, q in module._parameters.items():
                    if q is not None:
                        owners.setdefault(id(q), []).append((attr, module))
            self._owners = owners
        held = self._owners.get(id(p), [])
        if len(held) == 1 and held[0][0] == "weight" and isinstance(held[0][1], torch.nn.Linear):
            return held[0][1]
        return None

    def _side_on(self, wanted: dict) -> None:
        """Register the side path's forward hooks: ``wanted`` is {(name, slot): [(piece, strength)]}, slot
        "committed" or "extra". Each overlay is kept while its pieces and strengths are the same (wsp.overlay.piece_key,
        as GGUF's PackedLinear keeps them) and dropped once they are not wanted. Committed pieces are added before
        a candidate's, as the in-place path applies them."""
        from .overlay import DeltaOverlay, piece_key
        for slot in [s for s in self._overlays if s not in wanted]:
            del self._overlays[slot]
        per_module: dict = {}
        for (name, slot), pieces in wanted.items():
            key = piece_key(pieces)
            held = self._overlays.get((name, slot))
            module = self._side_module(name)
            if held is None or held[0] != key:
                self._overlays[(name, slot)] = None           # release the old copy before building the new one
                held = (key, tuple(p for p, _ in pieces),
                        DeltaOverlay(pieces, module.out_features, module.in_features, torch.float32, self.device))
                self._overlays[(name, slot)] = held
            per_module.setdefault(id(module), (module, []))[1].append(held[2])
        for module, overlays in per_module.values():
            self._side_handles.append(module.register_forward_hook(_side_hook(overlays), with_kwargs=True))

    def use(self, variant: str = "current", extra=None) -> None:
        ws = self.ws
        if variant == "base":
            want = "base"
        elif not extra:
            want = ("current", ws.version)
        else:
            # The candidate is identified by its pieces' identities, as GGUFRunner's piece_key does: a grown or
            # different extra has another key. _held keeps the pieces alive, so no other object can take their ids.
            want = ("candidate", ws.version, tuple((name, tuple(id(p) for p in pieces)) for name, pieces in extra.items()))
        if want == self._state:
            return
        self._restore()
        # Nothing is applied now. Until every piece is on, the state is "unknown": if an _apply fails part-way, the
        # next call (whatever it asks for) restores and starts again instead of trusting half-applied weights.
        self._state, self._held = None, ()
        self._gain = {}
        if not hasattr(self, "_overlays"):     # a runner made without __init__ (as some tests do) has no side state
            self._overlays, self._owners, self._side_handles = {}, None, []
        if not extra:
            # A candidate's side-path copies do not outlive it (as GGUFRunner drops a job's overlays).
            for slot in [s for s in self._overlays if s[1] == "extra"]:
                del self._overlays[slot]
        if variant != "base":
            # Linear weights in mode "side" go to the side path (hooks registered here, before run registers its
            # capture hooks, so a captured output includes the patch); everything else is added in place.
            side: dict = {}
            for name in ws.touched():
                if self._side_module(name) is not None:
                    side[(name, "committed")] = list(ws.pieces(name))
                    continue
                for piece, scale in ws.pieces(name):
                    self._apply(name, piece, scale)
            for name, pieces in (extra or {}).items():
                if self._side_module(name) is not None:
                    if pieces:
                        side[(name, "extra")] = [(p, 1.0) for p in pieces]
                    continue
                for piece in pieces:
                    self._apply(name, piece, 1.0)
            if side or self._overlays:
                self._side_on(side)
        self._state = want
        self._held = tuple(p for ps in extra.values() for p in ps) if (variant != "base" and extra) else ()

    # ------------------------------------------------------------------ hooks
    def _hook_target(self, name: str):
        m = re.match(r"^blocks\.(\d+)\.(.+)$", name)
        if not m:
            return None
        L = self.ws.struct.layers[int(m.group(1))]
        kind = m.group(2)
        key = HOOK_KINDS.get(kind)
        if key is None:
            return None
        mod = self.module(L.modules.get(key[0]))
        return (mod, key[1], kind) if mod is not None else None

    def available(self, name: str) -> bool:
        return name.endswith("attn.hook_pattern") or self._hook_target(name) is not None

    def _steering_handles(self, steer, variant, extra) -> list:
        """Register the forward pre-hooks that apply ``steer`` (wsp.steering) -> their handles.

        They act on every call of the model until removed, so a cached decode pushes each new
        position once. Resolving may calibrate (which runs the base model), so the variant is
        selected again afterwards."""
        if steer is None:
            return []
        from .steering import hooks_for, module_pre_hook
        hooks = hooks_for(self.ws, steer, variant, extra)
        if not hooks:
            return []
        self.use(variant, extra)
        handles = []
        try:
            for name, fn in hooks.items():
                tgt = self._hook_target(name)
                if tgt is None or tgt[1] != "pre":
                    raise KeyError(f"hook point {name} is not available for this model")
                heads = self.ws.struct.n_heads if tgt[2] == "attn.hook_z" else None
                handles.append(tgt[0].register_forward_pre_hook(module_pre_hook(fn, name, heads), with_kwargs=True))
        except BaseException:
            for h in handles:
                h.remove()
            raise
        return handles

    def _steering(self, steer, variant, extra):
        from .steering import removing
        return removing(self._steering_handles(steer, variant, extra))

    @contextlib.contextmanager
    def _eager_attention(self):
        """The model on eager attention for one forward pass (only eager returns attention patterns), then back on
        exactly what it had, also after an error. transformers keeps the choice on every config of the model
        (``_attn_implementation_internal``; 5.18 also marks a switched sub-model's config ``_attn_was_changed``):
        those attributes are saved here and put back as they were. A model already on eager is left alone."""
        model = self.model
        cfg = getattr(model, "config", None)
        if cfg is None or getattr(cfg, "_attn_implementation", "eager") == "eager":
            yield
            return
        refused = RuntimeError(PATTERNS_NEED_EAGER)
        if not callable(getattr(model, "set_attn_implementation", None)):
            raise refused
        configs, todo = {}, [cfg] + [getattr(m, "config", None) for m in model.modules()]
        while todo:
            c = todo.pop()
            if c is None or id(c) in configs or not hasattr(c, "_attn_implementation"):
                continue
            configs[id(c)] = (c, {k: c.__dict__[k] for k in ATTN_STATE if k in c.__dict__})
            todo += [getattr(c, k, None) for k in getattr(c, "sub_configs", None) or ()]
        try:
            model.set_attn_implementation("eager")
            if cfg._attn_implementation != "eager":
                raise refused
            yield
        finally:
            for c, saved in configs.values():
                for k in ATTN_STATE:
                    if k in saved:
                        object.__setattr__(c, k, saved[k])
                    elif k in c.__dict__:
                        object.__delattr__(c, k)

    @_locked
    def run(self, ids, variant="current", extra=None, names=(), hooks=None, grad=False):
        self.use(variant, extra)
        t = torch.as_tensor(ids)
        if t.dim() == 1:
            t = t[None]
        t = t.to(self.device)
        hooks = hooks or {}
        cache: dict = {}
        handles = []
        s = self.ws.struct
        want = set(names)
        on_cpu = self.device.type == "cpu"

        def visit(name, x, kind):
            """Show an activation to the caller's hook / cache (on the CPU, as float32 when the model is elsewhere or
            in another precision) and hand back what the hook returned, on the model's device."""
            src = x
            if kind == "attn.hook_z":
                x = x.reshape(*x.shape[:-1], s.n_heads, x.shape[-1] // s.n_heads)
            elif kind in FLAT_KINDS:
                if t.shape[0] != 1:
                    raise ValueError(f"hook point {name} takes one prompt at a time")
                x = x.reshape(1, -1, x.shape[-1])
            if not on_cpu:
                x = x.float().cpu()
            out = hooks[name](x, name) if name in hooks else None
            if name in want:
                cache[name] = (out if out is not None else x).detach().float().cpu()
            if out is None:
                return None
            return out.reshape(src.shape).to(src.device)

        for name in want | set(hooks):
            if name.endswith("attn.hook_pattern"):
                continue
            tgt = self._hook_target(name)
            if tgt is None:
                if name in hooks:
                    raise KeyError(f"hook point {name} is not available for this model")
                continue
            mod, when, kind = tgt
            if when == "pre":
                def pre(module, args, kwargs, name=name, kind=kind):
                    if args:
                        new = visit(name, args[0], kind)
                        return None if new is None else ((new.to(args[0].dtype),) + tuple(args[1:]), kwargs)
                    key = "hidden_states" if "hidden_states" in kwargs else next(iter(kwargs))
                    new = visit(name, kwargs[key], kind)
                    return None if new is None else (args, {**kwargs, key: new.to(kwargs[key].dtype)})
                handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
            else:
                def post(module, args, output, name=name, kind=kind):
                    x = output[0] if isinstance(output, tuple) else output
                    new = visit(name, x, kind)
                    if new is None:
                        return None
                    new = new.to(x.dtype)
                    return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
                handles.append(mod.register_forward_hook(post))
        patterns = any(n.endswith("attn.hook_pattern") for n in want)
        try:
            with torch.enable_grad() if grad else torch.no_grad(), \
                    (self._eager_attention() if patterns else contextlib.nullcontext()):
                out = self.model(input_ids=t, output_attentions=patterns, **self.forward_options(t))
        finally:
            for h in handles:
                h.remove()
        if patterns and getattr(out, "attentions", None):
            indices = list(range(len(out.attentions)))
            if runtime_layouts.text_model_type(self.ws.hf_config) in runtime_layouts.HYBRID_TEXT_TYPES:
                indices = [i for i,kind in enumerate(self.model.config.layer_types) if kind == 'full_attention']
                if len(indices) != len(out.attentions):
                    raise RuntimeError('Hybrid attention trace count does not match full-attention layers')
            for l, a in zip(indices,out.attentions):
                if a is not None and f"blocks.{l}.attn.hook_pattern" in want:
                    cache[f"blocks.{l}.attn.hook_pattern"] = a.detach().float().cpu()
        return out.logits.float().cpu(), cache

    @_locked
    def generate(self, ids, variant="current", extra=None, max_new_tokens=40, temperature=0.0, top_k=0, seed=None, stop_ids=(), on_token=None,
                 steer=None):
        self.use(variant, extra)
        if seed is not None:
            torch.manual_seed(int(seed))
        t = torch.as_tensor(list(ids))[None].to(self.device)
        kw = {"max_new_tokens": int(max_new_tokens), "do_sample": temperature > 0}
        if temperature > 0:
            kw.update(temperature=float(temperature), top_k=int(top_k) or None)
        cfg = self.model.generation_config
        pad = cfg.pad_token_id if cfg.pad_token_id is not None else cfg.eos_token_id
        if isinstance(pad, (list, tuple)):
            pad = pad[0]
        # transformers refuses a streamer with beam search; such a model gets no live text.
        if on_token is not None and (getattr(cfg, "num_beams", 1) or 1) == 1:
            from transformers import StoppingCriteriaList
            kw["streamer"] = TokenCallback(on_token)
            kw["stopping_criteria"] = StoppingCriteriaList([kw["streamer"]])  # ends generate when stop is asked
        with torch.no_grad(), (contextlib.nullcontext() if steer is None else self._steering(steer, variant, extra)):
            out = self.model.generate(input_ids=t, attention_mask=torch.ones_like(t), pad_token_id=pad if pad is not None else 0, **kw)
        return out[0, t.shape[1]:].tolist()

    def release_device_cache(self) -> None:
        """Return the blocks the device's allocator keeps after tensors are freed (mps, cuda) to the system; nothing
        on the CPU. For the end of a measurement with large temporaries (a commit's drift: full-vocabulary logits):
        the allocator would hold them for the life of the process, and a memory limit counts them."""
        kind = getattr(getattr(self, "device", None), "type", "cpu")
        if kind == "mps":
            torch.mps.empty_cache()
        elif kind == "cuda":
            torch.cuda.empty_cache()

    @_locked
    def first_token_logits(self, ids, variant="current", extra=None) -> torch.Tensor:
        """The raw float32 logits of the next token after ``ids``: one forward pass computing the output head at
        the last position only, as transformers' ``generate`` computes its first token (not ``next_logits``:
        wsp.probes dispatches on that name)."""
        self.use(variant, extra)
        t = torch.as_tensor(list(ids))[None].to(self.device)
        with torch.no_grad():
            try:
                out = self.model(input_ids=t, attention_mask=torch.ones_like(t), use_cache=False, logits_to_keep=1)
            except TypeError as e:
                if "logits_to_keep" not in str(e):
                    raise
                out = self.model(input_ids=t, attention_mask=torch.ones_like(t), use_cache=False)
        return out.logits[0, -1].to(torch.float32).cpu()

    def _final(self):
        return self.module(self.ws.struct.modules.get("final_norm")), self.model.get_output_embeddings()

    @_locked
    def unembed(self, x: torch.Tensor, variant="current", extra=None) -> torch.Tensor:
        self.use(variant, extra)
        norm, head = self._final()
        with torch.no_grad():
            x = x.to(self.device, self.dtype)
            return head(norm(x) if norm is not None else x).float().cpu()

    @_locked
    def final_norm(self, x: torch.Tensor, variant="current", extra=None) -> torch.Tensor:
        """The model's final norm module applied to residual x [..., d_model] (what the unembedding reads), as
        float32 on the CPU; x itself when the model has none. GGUFRunner inherits this (its own ``use``)."""
        self.use(variant, extra)
        norm = self.module(self.ws.struct.modules.get("final_norm"))      # the norm alone: no output head needed
        if norm is None:
            return x.float().cpu()
        with torch.no_grad():
            flat = x.reshape(1, -1, x.shape[-1]).to(self.device, self.dtype)
            return norm(flat).float().cpu().reshape(x.shape)

    @_locked
    def final_gain(self, x: torch.Tensor, variant="current", extra=None):
        """Frozen-scale linearisation of the final norm at residual x, found by probing the module (so it does not
        matter how the architecture parameterises its norm): (gain per coordinate, centres?)."""
        self.use(variant, extra)
        norm, _ = self._final()
        d = x.shape[-1]
        if norm is None:
            return torch.ones(d), False
        if "g" not in self._gain:
            with torch.no_grad():
                x0 = torch.ones(d)
                x0[1::2] = -1.0
                if d % 2:
                    x0[-1] = 0.0
                x0 = x0 / x0.pow(2).mean().sqrt()
                f = lambda v: norm(v.to(self.device, self.dtype)[None, None])[0, 0].float().cpu()  # noqa: E731
                g = (f(x0) - f(-x0)) / (2 * x0.where(x0 != 0, torch.ones(())))
                centre = bool(torch.allclose(f(x0 + 3.0), f(x0), atol=1e-2 * float(g.abs().mean() + 1e-6)))
                self._gain = {"g": g, "centre": centre, "eps": float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6)))}
        g, centre, eps = self._gain["g"], self._gain["centre"], self._gain["eps"]
        xc = x - x.mean(-1, keepdim=True) if centre else x
        return g / (xc.pow(2).mean(-1, keepdim=True) + eps).sqrt(), centre

    def hf_tokenizer(self):
        """The checkpoint's own transformers tokenizer (loaded once from its files), or None when it has none."""
        if self._tok is None:
            try:
                from transformers import AutoTokenizer
                self._tok = AutoTokenizer.from_pretrained(self.ws.model_dir, local_files_only=True)
            except Exception:
                self._tok = False
        return None if self._tok is False else self._tok

    def chat_prompt(self, messages: list[dict], **template_options) -> str | None:
        tok = self.hf_tokenizer()
        if tok is None or not getattr(tok, "chat_template", None):
            return None
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_options)

    def release(self) -> None:
        self._restore()
        self.model = None


def make_runner(ws) -> Runner:
    if ws.kind == "wsp":
        return FunctionalRunner(ws)
    from .worker_jobs import uses_worker
    if uses_worker(ws):
        # Supervised worker mode: the model runs in the supervisor-spawned model worker.
        from .remote_runner import RemoteRunner
        return RemoteRunner(ws)
    if ws.kind == 'gguf':
        from .gguf_runner import GGUFRunner
        return GGUFRunner(ws)
    if ws.struct.runner != "hf":
        raise RuntimeError(f"{ws.name} has no config.json that says how to run it; its weights can be viewed and edited, not run")
    return HFRunner(ws)
