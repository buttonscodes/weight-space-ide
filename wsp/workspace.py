"""A workspace: a checkpoint that is read lazily, plus a stack of patches.

    current weights = base + sum over enabled patches of  strength_i * delta_i

The base is never loaded whole. It is a `Store` (memory-mapped safetensors files, or files on
the Hub read by range request) and the deltas are compact pieces (block / low-rank / sparse,
see delta.py).  `read(name, index)` returns one region of one tensor with the patches that
touch it applied; every view, analysis, chunk and edit is built on that call, so the memory
and time an operation needs are those of the region it touches.

Two kinds of workspace share this class:

    wsp   this project's own small models (config.json is a wsp Config; run by wsp.model)
    hf    any safetensors checkpoint in Hugging Face layout, kept in its native names and
          layout; its structure is detected (arch.py) and it is run by `transformers`

On disk:  workspace.json  [base.safetensors | model/ (config, tokenizer, weights if pulled)]
          patches/<id>.json + .safetensors   groups.json   tests/   task.json   notes.md
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
from collections import OrderedDict
from contextlib import nullcontext
from .runtime_events import RuntimeRefusal, edit_preflight, job_scope
from typing import Any, Iterator

import torch
from safetensors.torch import load_file, save_file

from . import arch, delta as D
from .store import Region, Store, checkpoint_files, from_float, norm_index, region_index, region_shape, region_size, write_safetensors

CACHE_BYTES = int(float(os.environ.get("WSP_CACHE_MB", "1500")) * 1e6)
STREAM_ELEMS = 16_000_000          # slab size for anything that walks a large tensor


class ScopeError(ValueError):
    pass


class Patch:
    def __init__(self, pid: str, message: str, author: str, source: dict, deltas: dict[str, list[D.Piece]],
                 enabled: bool = True, strength: float = 1.0, created: float | None = None, info: dict | None = None):
        self.id = pid
        self.message = message
        self.author = author
        self.source = source
        self.deltas = deltas
        self.stored = sum(pc.numel() for ps in deltas.values() for pc in ps)
        self.enabled = enabled
        self.strength = strength
        self.created = created or time.time()
        self.info = info or {}
        self._stats: list[dict] | None = None

    def stats(self) -> list[dict]:
        """Per-tensor summary of the pieces; a patch's pieces never change, so it is worked out once."""
        if self._stats is None:
            self._stats = D.summary(self.deltas)
        return self._stats

    def meta(self, with_stats: bool = True) -> dict:
        m = {"id": self.id, "message": self.message, "author": self.author, "source": self.source,
             "enabled": self.enabled, "strength": self.strength, "created": self.created, "info": self.info}
        if with_stats:
            st = self.stats()
            m["stats"] = st
            m["changed"] = sum(s["changed"] for s in st)
            m["stored"] = sum(s["stored"] for s in st)
        return m


class Ref:
    """A lazy view of a region of one tensor, optionally with its axes reordered to a canonical layout.

    Nothing is read until `read()`; `sub()` narrows the view without reading; `write` and
    `add_lowrank` turn a change of the view into delta pieces in the tensor's native layout."""

    def __init__(self, ws: "Workspace", name: str, region: Region | None = None, perm: tuple | None = None):
        self.ws = ws
        self.name = name
        self.region = norm_index((), ws.shape(name)) if region is None else region
        self.kept = [d for d, r in enumerate(self.region) if isinstance(r, range)]
        self.perm = tuple(range(len(self.kept))) if perm is None else tuple(perm)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(self.region[self.kept[k]]) for k in self.perm)

    @property
    def size(self) -> int:
        return region_size(self.region)

    def index(self) -> tuple:
        return region_index(self.region)

    def address(self) -> str:
        from .addressing import region_str
        return self.name + region_str(self.region, self.ws.shape(self.name))

    def sub(self, idx) -> "Ref":
        """Basic indexing (ints / slices) of the view, without reading."""
        if not isinstance(idx, tuple):
            idx = (idx,)
        if len(idx) > len(self.perm):
            raise IndexError(f"{len(idx)} indices for a view of shape {list(self.shape)}")
        region = list(self.region)
        dropped = []
        for k, i in enumerate(idx):
            d = self.kept[self.perm[k]]
            r = region[d]
            if isinstance(i, slice):
                region[d] = r[i]
            else:
                i = int(i)
                if not -len(r) <= i < len(r):
                    raise IndexError(f"index {i} out of range for an axis of size {len(r)}")
                region[d] = r[i]
                dropped.append(self.perm[k])
        left = [p for p in self.perm if p not in dropped]
        order = sorted(left)
        return Ref(self.ws, self.name, tuple(region), tuple(order.index(p) for p in left))

    def read(self, variant: str = "current", extra: dict | None = None, cache: bool = True) -> torch.Tensor:
        t = self.ws.read(self.name, self.region, variant, extra, cache=cache)
        return t.permute(self.perm) if self.perm != tuple(range(len(self.perm))) else t

    def whole(self, variant: str = "current", extra: dict | None = None) -> torch.Tensor:
        """The whole view in memory, for a caller that needs all of it at once (the atlas reads a layer's matrices).

        Equal to `read`, but a view above STREAM_ELEMS values is read in slabs of at most that many and put
        together here: a quantized store refuses a larger single read, so that nothing reconstructs a large
        tensor by accident. The caller has decided that it needs this one."""
        if self.size <= STREAM_ELEMS or not self.kept:
            return self.read(variant, extra)
        out = None
        for pos, t in self.ws.slabs(self.name, self.region, variant, extra, max_elems=STREAM_ELEMS):
            if out is None:
                out = torch.empty((len(self.region[self.kept[0]]),) + tuple(t.shape[1:]), dtype=t.dtype)
            out[pos.start:pos.stop] = t
        return out.permute(self.perm) if self.perm != tuple(range(len(self.perm))) else out

    def to_native(self, t: torch.Tensor) -> torch.Tensor:
        """A tensor in view layout -> the tensor's own layout."""
        if self.perm == tuple(range(len(self.perm))):
            return t
        inv = [0] * len(self.perm)
        for k, p in enumerate(self.perm):
            inv[p] = k
        return t.permute(inv)

    def piece(self, delta_view: torch.Tensor) -> D.Piece:
        edit_preflight(self.ws,self.size)
        return D.block(self.region, self.to_native(delta_view.expand(self.shape)))

    def lowrank_piece(self, A: torch.Tensor, B: torch.Tensor) -> D.Piece:
        """delta of the (2-D) view = A @ B."""
        if len(self.perm) != 2:
            raise ValueError(f"a low-rank edit needs a 2-D view, this one has shape {list(self.shape)}")
        edit_preflight(self.ws,A.numel()+B.numel())
        A = A[:, None] if A.dim() == 1 else A
        B = B[None, :] if B.dim() == 1 else B
        if self.perm == (0, 1):
            return D.lowrank(self.region, A, B)
        return D.lowrank(self.region, B.T, A.T)

    def contract(self, v: torch.Tensor, axis: int = -1, variant: str = "current", extra: dict | None = None) -> torch.Tensor:
        """Sum over one axis of the view against v ([n] or [n, k]) in bounded memory.

        `W.contract(r)` is W @ r for a 2-D view; with axis=0 it is r @ W."""
        nd = len(self.perm)
        axis %= nd
        if self.size <= STREAM_ELEMS:
            return torch.tensordot(self.read(variant, extra).movedim(axis, -1), v, dims=1)
        first = self.perm.index(0)                 # the view axis that is the first native kept axis (what we stream over)
        rows = self.region[self.kept[0]]
        if len(rows) * 4 >= self.ws.shape(self.name)[self.kept[0]]:
            self.ws.store.fetch(self.name)         # a pass over most of a remote tensor: cache it once instead of re-reading ranges
        outs, acc = [], None
        for a in range(0, len(rows), max(1, STREAM_ELEMS // max(1, self.size // len(rows)))):
            step = max(1, STREAM_ELEMS // max(1, self.size // len(rows)))
            part = Ref(self.ws, self.name, self.region[:self.kept[0]] + (rows[a:a + step],) + self.region[self.kept[0] + 1:], self.perm)
            t = part.read(variant, extra)
            if first == axis:
                x = torch.tensordot(t.movedim(axis, -1), v[a:a + step], dims=1)
                acc = x if acc is None else acc + x
            else:
                outs.append(torch.tensordot(t.movedim(axis, -1), v, dims=1))
        if first == axis:
            return acc
        return torch.cat(outs, dim=first - (1 if axis < first else 0))


class Workspace:
    edit_settings: dict | None = None      # resolved settings values edits read (Session.set_edit_settings, wsp.edit_limit)
    edit_timing: dict | None = None        # the last large edit's size and time, in memory only (edit_limit.record_timing)

    def __init__(self, directory: str, store: Store, struct: arch.Structure, tok, meta: dict, cfg=None, hf_config: dict | None = None):
        self.dir = directory
        self.store = store
        self.struct = struct
        self.tok = tok
        self.meta = meta
        self.cfg = cfg                         # wsp Config for this project's own models, else None
        self.hf_config = hf_config or {}
        if self.kind == 'hf':
            self.store.quantized_config = self.hf_config.get('quantization_config') or self.hf_config.get('quantization')
        self.patches: list[Patch] = []
        self.patch_stored = 0
        self.groups: dict[str, dict] = {}
        self.version = 0
        self.lock = threading.RLock()
        self.listeners: list = []
        self.calib: dict[str, Any] = {}        # caches valid for the current weights version
        self.keep: dict[str, Any] = {}         # caches valid for the base weights (never invalidated)
        self._pieces: dict[str, list[tuple[D.Piece, float]]] = {}
        self._cache: OrderedDict = OrderedDict()
        self._cache_bytes = 0
        self._runner = None
        self._sem = None
        self.run_error: str | None = None      # why the model could not be loaded for running, once that has been tried

    # ------------------------------------------------------------------ description
    @property
    def name(self) -> str:
        return self.meta.get("name") or os.path.basename(self.dir)

    @property
    def kind(self) -> str:
        return self.meta.get("kind", "wsp")

    def names(self) -> list[str]:
        return self.store.names()

    def shape(self, name: str) -> tuple[int, ...]:
        return self.store.shape(name)

    def n_params(self) -> int:
        return self.store.total()

    def editable(self, name: str) -> bool:
        return self.store.is_float(name)

    @property
    def sem(self):
        if self._sem is None:
            from .semantic import Semantic
            self._sem = Semantic(self)
        return self._sem

    @property
    def runner(self):
        with self.lock:
            if getattr(self, '_failed_runner', None) is not None:
                raise RuntimeError(self.run_error or 'Partial runner cleanup failed; watchdog remains armed')
            lost = getattr(self._runner, 'lost', None) is not None
            client = getattr(self, '_runtime_client', None)
            if lost and (client is None or client.operation_id is None or client.operation_id != self._runner.lost_in):
                # A runner whose model worker was lost serves the rest of its own operation
                # (refusing at once); anything later loads afresh through a new admission.
                drop = getattr(client, 'drop_lost', None)
                if drop is not None:
                    drop(self)
                else:
                    self._runner = None
            if self._runner is None:
                from .runner import make_runner
                try:
                    client = getattr(self, '_runtime_client', None)
                    with client.operation(self,'load') if client else nullcontext():
                        self._runner = make_runner(self)
                except RuntimeRefusal:
                    self.run_error = None
                    raise
                except Exception as e:                   # remembered, so the IDE says why instead of retrying on every request
                    self.run_error = f"{type(e).__name__}: {e}" if not isinstance(e, RuntimeError) else str(e)
                    raise RuntimeError(self.run_error) from None
            return self._runner

    @property
    def runner_loaded(self) -> bool:
        # A runner whose model worker was lost holds no model.
        return self.kind == "wsp" or (self._runner is not None and getattr(self._runner, 'lost', None) is None)

    @property
    def can_run(self) -> bool:
        if self.kind == "wsp":
            return True
        if self.kind == "gguf":
            return self.store.capabilities()['generation']['supported'] and self.tok is not None and not self.run_error
        return self.struct.runner == "hf" and all(self.store.is_float(n) for n in self.names()) and not self.store.remote and self.struct.recognised and not self.run_error

    def summary(self) -> dict:
        s = self.struct
        capabilities = self.store.capabilities() if self.kind == 'gguf' else None
        if self.kind == 'hf' and not all(self.store.is_float(n) for n in self.names()):
            no = dict(supported=False,reason='Packed/scaled safetensors need a format-specific decoder and runtime; use a validated GGUF or floating-point checkpoint')
            capabilities = dict(format='quantized-safetensors',representation='storage metadata only; logical decoding unsupported',
                                import_read=dict(supported=True,reason='Headers and storage shapes only'),
                                logical_edit=no,generation=no,trace_intervention=no,adapter_export=no,merged_export=no)
        return {"name": self.name, "dir": self.dir, "kind": self.kind, "family": s.family, "recognised": s.recognised,
                "n_params": self.n_params(), "n_tensors": len(self.names()), "bytes": self.store.nbytes(),
                "remote": self.store.remote, "can_run": self.can_run, "n_patches": len(self.patches), "version": self.version,
                "config": {"n_layers": s.n_layers, "n_heads": s.n_heads, "n_kv_heads": s.n_kv_heads, "d_model": s.d_model,
                           "d_head": s.d_head, "d_mlp": s.d_mlp, "d_vocab": s.d_vocab, "n_ctx": s.n_ctx, "norm": s.norm,
                           "act": s.act, "tied": s.tied, "name": self.name},
                "notes": s.notes, "repo": self.meta.get("repo"), "vocab_kind": self.tok.to_dict()["type"] if self.tok else None,
                "vocab_mode": self.tok.to_dict().get("mode") if self.tok else None,
                "capabilities": capabilities}

    # ------------------------------------------------------------------ construction / persistence
    @staticmethod
    def create(directory: str, cfg, tok, base: dict[str, torch.Tensor], notes: str = "") -> "Workspace":
        """A workspace for one of this project's own small models."""
        from .tokenizer import save_tokenizer
        os.makedirs(os.path.join(directory, "patches"), exist_ok=True)
        os.makedirs(os.path.join(directory, "tests"), exist_ok=True)
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=1)
        save_file({k: v.contiguous() for k, v in base.items()}, os.path.join(directory, "base.safetensors"))
        save_tokenizer(tok, directory)
        with open(os.path.join(directory, "workspace.json"), "w") as f:
            json.dump({"kind": "wsp", "name": cfg.name}, f, indent=1)
        if notes:
            with open(os.path.join(directory, "notes.md"), "w") as f:
                f.write(notes)
        return Workspace.load(directory)

    @staticmethod
    def create_hf(directory: str, name: str, model_dir: str, repo: str | None = None, sha: str | None = None,
                  weights: list[str] | None = None, urls: list[str] | None = None, notes: str = "") -> "Workspace":
        """A workspace over a checkpoint in Hugging Face layout (local files, or files still on the Hub)."""
        os.makedirs(os.path.join(directory, "patches"), exist_ok=True)
        os.makedirs(os.path.join(directory, "tests"), exist_ok=True)
        meta = {"kind": "hf", "name": name, "model_dir": os.path.relpath(model_dir, directory), "repo": repo, "sha": sha,
                "weights": weights, "urls": urls}
        with open(os.path.join(directory, "workspace.json"), "w") as f:
            json.dump(meta, f, indent=1)
        if notes:
            with open(os.path.join(directory, "notes.md"), "w") as f:
                f.write(notes)
        return Workspace.load(directory)

    @staticmethod
    def load(directory: str, *, metadata_only: bool = False) -> "Workspace":
        from .model import Config
        from .tokenizer import BPETokenizer, load_tokenizer
        directory = os.path.abspath(directory)
        mp = os.path.join(directory, "workspace.json")
        meta = json.load(open(mp)) if os.path.exists(mp) else {"kind": "wsp"}
        cache = os.path.join(os.path.dirname(directory), ".cache", "blobs")
        if meta["kind"] == "wsp":
            with open(os.path.join(directory, "config.json")) as f:
                cfg = Config.from_dict(json.load(f))
            meta.setdefault("name", cfg.name)
            store = Store([os.path.join(directory, "base.safetensors")])
            shapes = {n: store.shape(n) for n in store.names()}
            ws = Workspace(directory, store, arch.wsp_structure(cfg, shapes), load_tokenizer(directory), meta, cfg=cfg)
        else:
            model_dir = os.path.normpath(os.path.join(directory, meta["model_dir"]))
            cp = os.path.join(model_dir, "config.json")
            hf_config = json.load(open(cp)) if os.path.exists(cp) else {}
            if meta.get("weights"):
                locs = [os.path.join(model_dir, w) if os.path.exists(os.path.join(model_dir, w)) else u
                        for w, u in zip(meta["weights"], meta.get("urls") or [None] * len(meta["weights"]))]
                if any(x is None for x in locs):
                    raise FileNotFoundError(f"weights of {meta.get('name')} are neither in {model_dir} nor on the Hub")
            else:
                locs = checkpoint_files(model_dir)
            if meta['kind'] == 'gguf':
                from .gguf_store import GGUFStore, file_hash
                store = GGUFStore(meta['gguf_path'])
                if not metadata_only:
                    digest = file_hash(store.path)
                    if meta.get('base_sha256') is None and not os.listdir(os.path.join(directory,'patches')):
                        meta['base_sha256'] = digest
                        Path(os.path.join(directory,'workspace.json')).write_text(json.dumps(meta,indent=2))
                    if digest != meta['base_sha256']:
                        raise ValueError('GGUF base SHA-256 changed; refusing to apply saved patches')
            else:
                store = Store(locs, cache)
            shapes = {n: store.shape(n) for n in store.names()}
            tp = os.path.join(model_dir, "tokenizer.json")
            tok = BPETokenizer(open(tp).read(), special=True) if os.path.exists(tp) else None
            ws = Workspace(directory, store, arch.detect(shapes, hf_config), tok, meta, hf_config=hf_config)
            ws.model_dir = model_dir
            ws._metadata_only = metadata_only
        pdir = os.path.join(directory, "patches")
        for fn in sorted(os.listdir(pdir)) if os.path.isdir(pdir) else []:
            if fn.endswith(".json"):
                ws.patches.append(Workspace._read_patch(pdir, fn, shapes))
        gp = os.path.join(directory, "groups.json")
        if os.path.exists(gp):
            with open(gp) as f:
                ws.groups = json.load(f)
        ws.rebuild()
        return ws

    @staticmethod
    def _read_patch(pdir: str, fn: str, shapes: dict) -> Patch:
        """One saved patch: `<id>.json` plus its `.safetensors` (pieces, or a legacy dense/sparse layout)."""
        with open(os.path.join(pdir, fn)) as f:
            m = json.load(f)
        raw = load_file(os.path.join(pdir, fn[:-5] + ".safetensors"))
        deltas = D.unpack(m["pieces"], raw) if "pieces" in m else _legacy_deltas(raw, shapes)
        return Patch(m["id"], m["message"], m["author"], m["source"], deltas, m["enabled"],
                     m["strength"], m["created"], m.get("info"))

    def _save_patch(self, p: Patch, deltas: bool = True) -> None:
        pdir = os.path.join(self.dir, "patches")
        os.makedirs(pdir, exist_ok=True)
        index, tensors = D.pack(p.deltas)
        with open(os.path.join(pdir, p.id + ".json"), "w") as f:
            json.dump({**p.meta(with_stats=False), "pieces": index}, f, indent=1)
        if deltas:
            save_file(tensors, os.path.join(pdir, p.id + ".safetensors"))

    def read_json(self, rel: str, default=None):
        path = os.path.join(self.dir, rel)
        if not os.path.exists(path):
            return default
        with open(path) as f:
            return json.load(f)

    def write_json(self, rel: str, obj) -> None:
        path = os.path.join(self.dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=1)

    def notes(self) -> str:
        path = os.path.join(self.dir, "notes.md")
        return open(path).read() if os.path.exists(path) else ""

    # ------------------------------------------------------------------ reading weights
    def pieces(self, name: str) -> list[tuple[D.Piece, float]]:
        return self._pieces.get(name, ())

    def piece_index(self) -> dict[str, list[tuple[D.Piece, float]]]:
        """The enabled patch pieces of every tensor, as one consistent set: {name: [(piece, strength)]}.

        rebuild() replaces this dict whole (under the lock) and never changes it in place, so a caller that reads a
        tensor in several slabs can lay the same patches on every slab while other edits are committed meanwhile."""
        return self._pieces

    def read(self, name: str, index=(), variant: str = "current", extra: dict | None = None, cache: bool = True) -> torch.Tensor:
        """One region of one tensor as float32: `base`, `current` (base + enabled patches) or `delta` (current - base).

        `extra` is a {tensor: [pieces]} of uncommitted changes to lay on top (a script in progress, a candidate edit)."""
        shape = self.shape(name)
        region = index if _is_region(index, shape) else norm_index(index, shape)
        whole = all(isinstance(r, range) and len(r) == n and r.step == 1 for r, n in zip(region, shape))
        if variant == "delta":
            out = torch.zeros(region_shape(region))
        elif cache and whole and region_size(region) * 4 <= CACHE_BYTES // 4:
            out = self._whole_base(name).clone()
        else:
            out = self.store.read(name, region)
        if variant != "base":
            for piece, scale in self.pieces(name):
                D.add_to(out, region, piece, scale)
            if extra and name in extra:
                for piece in extra[name]:
                    D.add_to(out, region, piece, 1.0)
        return out

    def _whole_base(self, name: str) -> torch.Tensor:
        """Whole base tensors that have been asked for, least recently used first out (bounded by WSP_CACHE_MB)."""
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        t = self.store.read(name)
        self._cache[name] = t
        self._cache_bytes += t.numel() * 4
        while self._cache_bytes > CACHE_BYTES and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._cache_bytes -= old.numel() * 4
        return t

    def ref(self, name: str, index=()) -> Ref:
        return Ref(self, name, norm_index(index, self.shape(name)))

    def slabs(self, name: str, index=(), variant: str = "current", extra: dict | None = None,
              max_elems: int = STREAM_ELEMS) -> Iterator[tuple[range, torch.Tensor]]:
        """A region in slabs along its first kept axis: (positions within the region, tensor)."""
        region = index if _is_region(index, self.shape(name)) else norm_index(index, self.shape(name))
        kept = [d for d, r in enumerate(region) if isinstance(r, range)]
        if not kept or region_size(region) <= max_elems:
            n = len(region[kept[0]]) if kept else 1
            yield range(0, n), self.read(name, region, variant, extra)
            return
        ax = kept[0]
        rows = region[ax]
        if len(rows) * 4 >= self.shape(name)[ax]:
            self.store.fetch(name)
        step = max(1, max_elems // max(1, region_size(region) // len(rows)))
        for a in range(0, len(rows), step):
            part = rows[a:a + step]
            yield range(a, a + len(part)), self.read(name, region[:ax] + (part,) + region[ax + 1:], variant, extra)

    def changed(self, name: str, index=()) -> tuple[int, float]:
        """(weights that differ from base, squared norm of the difference) in a region, from the pieces alone."""
        ps = self.pieces(name)
        if not ps:
            return 0, 0.0
        region = norm_index(index, self.shape(name))
        if region_size(region) <= 4_000_000:
            d = self.read(name, region, "delta")
            return int((d != 0).sum()), float(d.double().pow(2).sum())
        n, sq = 0, 0.0
        for piece, scale in ps:
            c, s = D.changed_in(piece, region)
            n, sq = n + c, sq + s * scale * scale
        return n, sq

    # ------------------------------------------------------------------ patches
    def rebuild(self) -> None:
        with self.lock:
            idx: dict[str, list] = {}
            for p in self.patches:
                if p.enabled and p.strength:
                    for name, pieces in p.deltas.items():
                        idx.setdefault(name, []).extend((pc, p.strength) for pc in pieces)
            self._pieces = idx
            self.patch_stored = sum(p.stored for p in self.patches)
            self.version += 1
            self.calib.clear()
            # A packed runner's run-time copies of pieces that are no longer active go now (GGUFRunner).
            changed = getattr(getattr(self, '_runner', None), 'pieces_changed', None)
            if callable(changed):
                changed()
            client = getattr(self,'_runtime_client',None)
            if client: client.mutated(self)

    def emit(self, kind: str, **data) -> None:
        for fn in list(self.listeners):
            try:
                fn({"type": kind, "version": self.version, **data})
            except Exception:
                pass

    def group_members(self, name: str) -> list[str]:
        if name not in self.groups:
            raise KeyError(f"no group named {name!r}")
        record = self.groups[name]
        members = record.get("members") if isinstance(record, dict) else None
        if not isinstance(members, list):                 # a hand-edited or old groups.json
            raise ValueError(f"selection {name!r} does not hold a list of members")
        return members

    def resolve(self, addr: str) -> list[tuple[str, tuple]]:
        from .addressing import resolve
        return resolve(self, addr)

    def check_scope(self, deltas: dict[str, list[D.Piece]], scope: list[str] | None) -> None:
        """Every changed weight must lie inside one of the scope's regions. Costs the edit, not the tensors."""
        if not scope or "model" in scope:
            return
        allowed: dict[str, list[Region]] = {}
        for addr in scope:
            for name, idx in self.resolve(addr):
                allowed.setdefault(name, []).append(norm_index(idx, self.shape(name)))
        for name, pieces in deltas.items():
            regions = allowed.get(name, [])
            for p in pieces:
                if not _inside(p, regions, self.shape(name)):
                    raise ScopeError(f"edit touches weights of {name} outside the task scope; the scope is {scope}")

    @job_scope('edit')
    def commit(self, deltas: dict[str, list[D.Piece]], message: str, author: str = "human", source: dict | None = None,
               scope: list[str] | None = None, info: dict | None = None) -> Patch | None:
        """Record deltas (relative to the current weights) as a new patch on top of the stack."""
        with self.lock:
            edit_preflight(self,sum(pc.numel() for ps in deltas.values() for pc in ps))
            clean: dict[str, list[D.Piece]] = {}
            for name, pieces in deltas.items():
                if name not in self.store:
                    raise ValueError(f"unknown tensor {name}")
                if not self.editable(name):
                    raise ValueError(f"{name} has dtype {self.store.dtype(name)} and cannot be edited")
                pieces = [p for p in pieces if not p.is_zero()]
                for p in pieces:
                    if not p.finite():
                        raise ValueError(f"the change to {name} contains NaN or inf")
                if pieces:
                    clean[name] = pieces
            if not clean:
                return None
            self.check_scope(clean, scope)
            if getattr(self, '_runtime_client', None) is not None and getattr(self, '_metadata_only', False):
                self._runtime_client.require(self)  # identity verification; never inference
            pid = f"{len(self.patches) + 1:04d}-{int(time.time()) % 100000:05d}"
            p = Patch(pid, message, author, source or {"kind": "manual"}, clean, info=info)
            self.patches.append(p)
            self._save_patch(p)
            self.rebuild()
            client=getattr(self,'_runtime_client',None)
            if client:client.record_commit(self,p.id)
        self.emit("patch", id=p.id, message=message, author=author)
        return p

    def patch(self, pid: str) -> Patch:
        for p in self.patches:
            if p.id == pid:
                return p
        raise KeyError(f"no patch {pid!r}")

    def set_patch(self, pid: str, enabled: bool | None = None, strength: float | None = None) -> Patch:
        with self.lock:
            p = self.patch(pid)
            if enabled is not None:
                p.enabled = bool(enabled)
            if strength is not None:
                p.strength = float(strength)
            self._save_patch(p, deltas=False)
            self.rebuild()
        self.emit("patch", id=pid)
        return p

    def delete_patch(self, pid: str) -> None:
        with self.lock:
            p = self.patch(pid)
            self.patches.remove(p)
            for ext in (".json", ".safetensors"):
                path = os.path.join(self.dir, "patches", pid + ext)
                if os.path.exists(path):
                    os.remove(path)
            self.rebuild()
        self.emit("patch", id=pid)

    def touched(self) -> list[str]:
        return sorted(self._pieces)

    # ------------------------------------------------------------------ export
    def export_patch(self, path: str) -> dict:
        """Only the tensors that differ from the base, whole, under their checkpoint names and dtypes."""
        if self.kind == 'gguf':
            from .companion import export_bundle
            return export_bundle(self, path)
        names = self.touched()
        items = [(n, self.store.dtype(n), self.shape(n)) for n in names]
        write_safetensors(path, items, lambda n: from_float(self.read(n), self.store.dtype(n)),
                          {"base": self.meta.get("repo") or self.name, "format": "pt"})
        return {"path": path, "tensors": names}

    def export(self, directory: str, log=print) -> list[str]:
        """The edited checkpoint, file for file like the original. One tensor in memory at a time."""
        if self.kind == 'gguf':
            raise ValueError('Use export_patch for a WSP companion bundle, or the explicit guarded quantized_export.py command for supported same-scheme GGUF merges (docs/QUANTIZED.md).')
        os.makedirs(directory, exist_ok=True)
        out = []
        for f in self.store.files:
            fname = os.path.basename(f.location.split("?")[0])
            items = [(n, m["dtype"], tuple(m["shape"])) for n, m in sorted(f.tensors.items(), key=lambda kv: kv[1]["data_offsets"][0])]

            def data(n, f=f):
                if n in self._pieces:
                    return from_float(self.read(n), self.store.dtype(n))
                return f.array(n).tobytes()

            log(f"writing {fname} ({len(items)} tensors)")
            write_safetensors(os.path.join(directory, fname), items, data, {**f.user_meta, "format": "pt"})
            out.append(os.path.join(directory, fname))
        return out

    # ------------------------------------------------------------------ groups
    def _members_or_none(self, name: str) -> list:
        """A group's saved members, or [] for an unknown group or a record that holds no list of them."""
        try:
            return self.group_members(name)
        except (KeyError, ValueError):
            return []

    def set_group(self, name: str, members: list[str], note: str = "") -> None:
        """Save a selection. New members must resolve; members it already had are kept even if they no longer
        do (their page reports them). A member that reaches this selection itself is refused."""
        from .addressing import normalize
        had = set(m for m in self._members_or_none(name) if isinstance(m, str))
        for m in members:
            if m not in had:
                self.resolve(m)
        seen, todo = set(), [normalize(m) for m in members]
        while todo:
            a = todo.pop()
            if not a.startswith("group:") or a in seen:
                continue
            seen.add(a)
            if a[6:] == name:
                raise ValueError(f"A selection cannot contain itself: a member of {name!r} reaches group:{name}.")
            todo += [normalize(m) for m in self._members_or_none(a[6:]) if isinstance(m, str)]
        self.groups[name] = {"members": members, "note": note}
        self.write_json("groups.json", self.groups)
        self.emit("groups")

    def delete_group(self, name: str) -> None:
        self.groups.pop(name, None)
        self.write_json("groups.json", self.groups)
        self.emit("groups")

    # ------------------------------------------------------------------ running
    def encode(self, text: str) -> list[int]:
        ids = self.tok.encode(text)
        if not ids:
            raise ValueError("the prompt has no tokens")
        return ids

    def logits(self, text_or_ids, variant: str = "current", extra: dict | None = None, tail: int | None = None) -> torch.Tensor:
        ids = self.encode(text_or_ids) if isinstance(text_or_ids, str) else list(text_or_ids)
        return self.runner.logits(ids, variant, extra, tail=tail)

    def complete(self, prompt: str, variant: str = "current", extra: dict | None = None, **kw) -> str:
        new = self.runner.generate(self.encode(prompt), variant, extra, **kw)
        return self.tok.decode(new)


def _is_region(index, shape) -> bool:
    """Whether an index is already a region: one int or range per axis, every position inside the tensor and not
    negative. Anything else (a negative int such as `W[4, -17]`, a slice) goes through norm_index, so a position
    is always written one way and matches the pieces that edit it."""
    if not (isinstance(index, tuple) and len(index) == len(shape)):
        return False
    for i, n in zip(index, shape):
        if isinstance(i, bool):
            return False
        if isinstance(i, int):
            if not 0 <= i < n:
                return False
        elif isinstance(i, range):
            if len(i) and not (0 <= min(i[0], i[-1]) and max(i[0], i[-1]) < n):
                return False
        else:
            return False
    return True


def _inside(piece: D.Piece, regions: list[Region], shape: tuple) -> bool:
    """Are all of a piece's changed weights inside the union of `regions`?"""
    if not regions:
        return False
    piece = D.trim(piece)
    if piece.kind == "sparse":
        c = piece.coords[piece.values != 0]
        ok = torch.zeros(c.shape[0], dtype=torch.bool)
        for reg in regions:
            m = torch.ones(c.shape[0], dtype=torch.bool)
            for d, r in enumerate(reg):
                m &= (c[:, d] == r) if isinstance(r, int) else torch.isin(c[:, d], torch.tensor(list(r)))
            ok |= m
        return bool(ok.all())

    def covers(reg):                            # per-axis containment of the piece's region in one scope region
        for p, r in zip(piece.region, reg):
            ps = {p} if isinstance(p, int) else p
            rs = {r} if isinstance(r, int) else r
            if isinstance(ps, range) and isinstance(rs, range) and rs.step == 1 and ps.step == 1:
                if len(ps) and not (rs.start <= ps.start and ps.stop <= rs.stop):
                    return False
            elif not set(ps) <= set(rs):
                return False
        return True

    if any(covers(reg) for reg in regions):
        return True
    if region_size(piece.region) > 16_000_000:
        return False
    mask = torch.zeros(region_shape(piece.region))          # exact check on the piece's own region
    for reg in regions:
        if all(D._hits(p, r) is not None for p, r in zip(piece.region, reg)):
            D.add_to(mask, piece.region, D.Piece("block", reg, values=torch.ones(region_shape(reg))))
    dense = piece.dense()
    return not bool(((dense != 0) & (mask == 0)).any())


def _legacy_deltas(raw: dict, shapes: dict) -> dict[str, list[D.Piece]]:
    """Patches written before deltas were stored as pieces (dense tensors, or ::idx / ::val pairs)."""
    out: dict[str, list[D.Piece]] = {}
    for key, t in raw.items():
        if key.endswith("::idx"):
            name = key[:-5]
            shape = shapes[name]
            coords = torch.stack(torch.unravel_index(t, shape), 1)
            out.setdefault(name, []).append(D.Piece("sparse", norm_index((), shape), values=raw[name + "::val"], coords=coords))
        elif not key.endswith("::val"):
            out.setdefault(key, []).append(D.block(norm_index((), shapes[key]), t))
    return out
