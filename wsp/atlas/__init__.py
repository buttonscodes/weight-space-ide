"""The atlas: a map of every unit of the loaded model, built once and consulted while editing.

    build(ws)                 weights stage (always), wiring (cheap models, or on request),
                              observed stage (by default when the model runs in this process; on
                              request when it runs in the model worker)
    Atlas(ws)                 the built map: status, layers, units, concepts, selections

It lives in <workspace>/atlas/ as plain .npz / .json files, one per layer, and is resumable:
an interrupted build continues where it stopped (the wiring stage from the last finished layer, see wiring.py).  The map describes the BASE weights; `status()`
reports which layers have been edited since, so a card can say that it is out of date.

Everything is computed through the workspace's own views (`ws.sem`, `Ref`, the runner's standard
hook points), never through tensor names, so it works unchanged for every architecture the IDE
recognises and every store it can open.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from . import wiring as wiring_files
from .build import MECH, MECH_DOC, ROLE, ROLE_DOC, build_weights, fetch_bytes, tau
from .wiring import build_wiring, wiring_cost
from .tokens import CATS, token_table
from .observe import OBSERVED_FILES

VERSION = 1
STAGES = ("weights", "wiring", "observed")
AUTO_WIRING_FLOPS = 3e13                    # neuron-partner search is run by default below this cost


def atlas_dir(ws) -> str:
    return os.path.join(ws.dir, "atlas")


# Why a default build (CLI, MCP: no stages given) leaves the observed stage out for a model run in the model worker.
OBSERVED_NOT_BY_DEFAULT = ("not run by default on a model that runs in the model worker, because it loads and runs the "
                           "model; ask for the observed stage explicitly to build it")


def observed_unavailable(ws) -> str | None:
    """Why the observed stage cannot be built for this workspace, in words for the page, or None when it can."""
    if not getattr(ws, "can_run", False):
        return "This model cannot be run here, so the map will be weights-only and firing signs will be predictions."
    return None


def observed_note(ws) -> str | None:
    """What building the observed stage involves, for the page, when the model runs in the model worker."""
    from ..worker_jobs import uses_worker
    if not getattr(ws, "can_run", False) or not uses_worker(ws):
        return None
    from .observe import probe_plan
    plan = probe_plan(ws)
    if plan is None:
        return "This stage loads and runs the model on the probe text, one window at a time."
    n = plan["tokens"]
    return f"This stage loads and runs the model on about {round(n, -2) if n >= 1000 else n:,} tokens of probe text, one window at a time."


Busy = wiring_files.Busy


def build_lock(ws):
    """One atlas build per workspace at a time, across threads and processes (see wiring.build_lock)."""
    return wiring_files.build_lock(atlas_dir(ws))


def build_running(ws) -> str | None:
    """Whether a build of this workspace's atlas runs now, whoever started it: "this IDE", "another process" or
    None.  Never waits and never disturbs the build (see wiring.build_running)."""
    return wiring_files.build_running(atlas_dir(ws))


def running_reason(where: str) -> str:
    return f"a build of this atlas is running{wiring_files.WHERE[where]}"


def _while_running(saved: dict | None, where: str | None) -> dict | None:
    """Saved progress while a build runs is that build's, not something to continue."""
    if saved is None or not where:
        return saved
    return {**saved, "usable": False, "reason": running_reason(where)}


def weights_source(ws, variant: str) -> dict:
    """Which weights a stage describes: the base weights, or the edited ones with exactly these patches (the same in
    every process; see wiring.patch_fingerprint)."""
    out = {"model": ws.name, "variant": variant}
    if variant != "base":
        out["patches"] = wiring_files.patch_fingerprint(ws)
    return out


def recorded_source(ws, meta: dict) -> dict | None:
    """The source recorded with the weights stage; for a base atlas built before sources were recorded, the base."""
    w = ((meta or {}).get("stages") or {}).get("weights") or {}
    if "source" in w:
        return w["source"]
    return weights_source(ws, "base") if w and (meta.get("variant") or "base") == "base" else None


def planned_source(ws, meta: dict, variant: str) -> dict | None:
    """The weights-stage source a wiring build of this variant will run under: a build of the edited weights always
    rebuilds the weights stage first; a base build uses the recorded one."""
    return weights_source(ws, variant) if variant != "base" else recorded_source(ws, meta)


def _read_meta(adir: str) -> dict:
    try:
        with open(os.path.join(adir, "meta.json")) as f:
            m = json.load(f)
        return m if isinstance(m, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_meta(mp: str, meta: dict, ws=None) -> None:
    """Write the atlas record; with ws, also drop the workspace's cached Atlas, so no reader is older than the file."""
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    wiring_files.save_json(mp, meta)
    if ws is not None:
        ws.keep.pop("atlas", None)


def wiring_estimate(ws, variant: str = "base", timing=None, restart: bool = False) -> dict:
    """How long the wiring stage would take here, what it holds in memory and what waits for it (see wiring.estimate).
    Reads no weights; times one step of the search on random matrices once per process."""
    adir = atlas_dir(ws)
    meta = _read_meta(adir)
    est = wiring_files.estimate(ws, meta, adir, variant, timing, restart, source=planned_source(ws, meta, variant))
    est["weights_rebuild"] = _weights_rebuild(meta, variant)
    est["running"] = running = wiring_files.build_running(adir)
    if running:
        est["saved"], est["continuing"] = _while_running(est.get("saved"), running), None
    return est


def _weights_rebuild(meta: dict, variant: str) -> dict | None:
    """A build of the edited weights rebuilds the weights stage first and starts a new atlas record (so the observed
    stage's record goes): how long that took last time, when the last weights stage was built in full."""
    if variant == "base":
        return None
    stages = (meta or {}).get("stages") or {}
    w = stages.get("weights") or {}
    return {"seconds": w.get("seconds") if w.get("full") else None, "drops_observed": "observed" in stages}


def build(ws, stages=None, log=print, force: bool = False, fetch: bool = False, corpus: str | None = None,
          tokens: int | None = None, variant: str = "base", progress=None) -> dict:
    """Build (or continue) the atlas.  stages: any of weights / wiring / observed; default is weights, plus wiring
    when it is cheap, plus observed when the model can be run in this process.  A model that runs in the model
    worker leaves observed out of the default (reported as skipped, nothing is loaded); an explicit observed
    request runs it there, one window per worker job.  variant: "base" (default) or "current" to map the weights with the
    enabled patches applied (the weights stage is always rebuilt from scratch; an interrupted wiring stage continues
    when the patches are unchanged), except that the observed stage alone, not forced, is added to the atlas as it is
    (refused if it maps other weights).  force starts every requested stage over; a forced build (or any build of
    the edited weights but that one) drops an earlier observed stage's record before any stage runs.  progress, if given, receives the
    wiring stage's per-layer progress (layers done, time left).  One build per atlas at a time: a second one, in
    this process or another, is refused at once with `Busy`."""
    with build_lock(ws):
        return _build(ws, stages, log, force, fetch, corpus, tokens, variant, progress)


def _build(ws, stages, log, force, fetch, corpus, tokens, variant, progress) -> dict:
    try:
        return _stages(ws, stages, log, force, fetch, corpus, tokens, variant, progress)
    finally:
        ws.keep.pop("atlas", None)                     # however the build ends, readers start again from the files


def _stages(ws, stages, log, force, fetch, corpus, tokens, variant, progress) -> dict:
    from ..worker_jobs import uses_worker
    restart = force
    skipped = {}
    if stages is None:
        stages = ["weights"] + (["wiring"] if wiring_cost(ws) <= AUTO_WIRING_FLOPS else [])
        if ws.can_run and uses_worker(ws):
            skipped["observed"] = OBSERVED_NOT_BY_DEFAULT
            log(f"  observed stage skipped: {OBSERVED_NOT_BY_DEFAULT}")
        elif ws.can_run:
            stages.append("observed")
    stages = [s for s in STAGES if s in stages]
    # The observed stage alone, not forced: added to the atlas as it is, for the weights it was built from (an edited
    # weights atlas is not rebuilt first, as every other build of edited weights is).
    adding = stages == ["observed"] and not restart
    force = restart or (variant != "base" and not adding)
    adir = atlas_dir(ws)
    mp = os.path.join(adir, "meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) and not force else {"version": VERSION, "stages": {}}
    if adding and "weights" in meta["stages"]:
        built = meta.get("variant", "base")
        if built != variant:
            raise ValueError(f"this atlas maps the {built} weights, so its observed stage is built from the {built} weights; "
                             f"rebuild the atlas to map the {variant} weights")
        if recorded_source(ws, meta) != weights_source(ws, variant):
            raise ValueError("the weights this atlas was built from have changed since (its patches are not the enabled "
                             "ones now), so an observed stage would not describe them; rebuild the atlas")
    need = fetch_bytes(ws)
    if need and not fetch:
        raise ValueError(f"building the atlas of {ws.name} reads its embeddings and every MLP, {need / 1e9:.1f} GB that is "
                         f"still on the Hub. Pass fetch=True (CLI: --fetch) to download those tensors into the cache, or pull the model first.")
    if not adding:
        meta.update(model=ws.name, family=ws.struct.family, variant=variant, version_at_build=ws.version,
                    patches_at_build=[p.id for p in ws.patches if p.enabled] if variant != "base" else [])
    if force:
        # Before any stage rewrites a file: an earlier observed stage described the files this build replaces, so its
        # record goes now (the weights and wiring records are replaced as those stages complete, as before).
        old = _read_meta(adir)
        if "observed" in (old.get("stages") or {}):
            old["stages"].pop("observed")
            _save_meta(mp, old, ws)
    ws.keep.pop("atlas", None)
    for st in stages:
        if st == "weights":
            meta["stages"]["weights"] = {**build_weights(ws, adir, log, variant=variant, force=force), "at": time.time(),
                                         "source": weights_source(ws, variant)}
        elif st == "wiring":
            if "weights" not in meta["stages"]:
                raise ValueError("the wiring stage needs the weights stage first")
            if not force and wiring_files.complete(adir, meta):
                if os.path.isdir(wiring_files.partial_dir(adir)):
                    wiring_files.cleanup(adir)                     # leftovers of a finished build
            else:
                meta["stages"].pop("wiring", None)
                _save_meta(mp, meta, ws)
                t0 = time.time()
                est = wiring_files.estimate(ws, meta, adir, variant, restart=restart, source=recorded_source(ws, meta))
                log(f"  neuron partners (about {wiring_cost(ws) / 2:.1e} multiply-adds; {wiring_files.summary(est)})")
                r = build_wiring(ws, adir, log, variant=variant, weights_source=recorded_source(ws, meta),
                                 restart=restart, progress=progress)
                meta["stages"]["wiring"] = {"seconds": round(time.time() - t0, 1), "at": time.time(), "dtype": r["dtype"],
                                            **({"continued_from": r["resumed_from"]} if r["resumed_from"] is not None else {})}
                _save_meta(mp, meta, ws)
                wiring_files.cleanup(adir)                         # from here on the stage is complete
        elif st == "observed":
            if "weights" not in meta["stages"]:
                raise ValueError("the observed stage needs the weights stage first")
            if not ws.can_run:
                raise ValueError(f"{ws.name} cannot be run here, so the observed stage is not available: " + (ws.run_error or
                                 "its weights are not on disk, or no runner is known for it"))
            from .observe import build_observed

            def before_write():
                # Every window has run; from here an earlier observed stage's files are replaced, so its record goes
                # first: a stage stopped while writing leaves no observed files that a reader trusts.
                if meta["stages"].pop("observed", None) is not None:
                    _save_meta(mp, meta, ws)
                ws.keep.pop("atlas", None)
            meta["stages"]["observed"] = {**build_observed(ws, adir, log, corpus=corpus, tokens=tokens, force=force, variant=variant,
                                                           before_write=before_write), "at": time.time()}
        _save_meta(mp, meta, ws)
    return {**meta, "skipped": skipped} if skipped else meta


class Atlas:
    """Read access to a built atlas."""

    def __init__(self, ws):
        self.ws = ws
        self.dir = atlas_dir(ws)
        self._layers: dict[int, dict] = {}
        self._json: dict[str, object] = {}
        self._table: dict[str, np.ndarray] = {}

    @staticmethod
    def of(ws) -> "Atlas":
        if "atlas" not in ws.keep:
            ws.keep["atlas"] = Atlas(ws)
        return ws.keep["atlas"]

    # ------------------------------------------------------------------ files
    def json(self, name: str, default=None):
        if name in OBSERVED_FILES and not self.has("observed"):
            return default                                                         # not this atlas's (or not finished)
        if name not in self._json:
            p = os.path.join(self.dir, name)
            self._json[name] = json.load(open(p)) if os.path.exists(p) else default
        return self._json[name]

    @property
    def meta(self) -> dict:
        return self.json("meta.json", {}) or {}

    @property
    def exists(self) -> bool:
        return "weights" in (self.meta.get("stages") or {})

    def has(self, stage: str) -> bool:
        if stage == "wiring":
            return wiring_files.complete(self.dir, self.meta)
        return stage in (self.meta.get("stages") or {})

    @property
    def layers(self) -> list[int]:
        return list(self.meta["stages"]["weights"]["layers"]) if self.exists else []

    def need(self) -> None:
        if not self.exists:
            raise ValueError(f"no atlas has been built for {self.ws.name} yet (Atlas tab → Build, or: python -m wsp atlas --model {self.ws.name})")

    def layer(self, l: int) -> dict:
        """All per-neuron arrays of one layer (weights, wiring and observed stages merged)."""
        if l not in self._layers:
            A = dict(np.load(os.path.join(self.dir, f"layer_{l:03d}.npz")))
            for pre in ("wiring", "obs"):
                p = os.path.join(self.dir, f"{pre}_{l:03d}.npz")
                # never a partly built wiring, nor observed files the atlas does not record
                if os.path.exists(p) and self.has("wiring" if pre == "wiring" else "observed"):
                    A.update(np.load(p))
            self._layers[l] = A
        return self._layers[l]

    def tok2c(self) -> np.ndarray:
        if "tok2c" not in self._table:
            self._table["tok2c"] = np.load(os.path.join(self.dir, "token_concept.npy"))
        return self._table["tok2c"]

    def column(self, name: str) -> np.ndarray:
        """One per-neuron column over all mapped layers, concatenated in layer order."""
        if name not in self._table:
            if name == "layer":
                self._table[name] = np.concatenate([np.full(len(self.layer(l)["mech"]), l, dtype=np.int32) for l in self.layers])
            elif name == "idx":
                self._table[name] = np.concatenate([np.arange(len(self.layer(l)["mech"]), dtype=np.int32) for l in self.layers])
            else:
                parts = [self.layer(l).get(name) for l in self.layers]
                if any(p is None for p in parts):
                    raise KeyError(f"the atlas has no column {name!r}" + (" (the observed stage has not been run)" if name.startswith("obs_") else ""))
                self._table[name] = np.concatenate(parts)
        return self._table[name]

    # ------------------------------------------------------------------ status
    def stale_layers(self) -> list[int]:
        """Layers whose MLP weights differ from the ones the atlas was built from."""
        s = self.ws.struct
        if self.meta.get("variant", "base") != "base":
            src = ((self.meta.get("stages") or {}).get("weights") or {}).get("source") or {}
            if "patches" in src:                             # the same in every process (not the per-process version)
                return [] if src["patches"] == wiring_files.patch_fingerprint(self.ws) else list(self.layers)
            return [] if self.meta.get("version_at_build") == self.ws.version else list(self.layers)
        touched = set(self.ws.touched())
        return [l for l in self.layers if any(p.name in touched for p in s.layers[l].mlp.values())]

    def wiring_progress(self, stages: dict, running: str | None = None) -> dict | None:
        """Saved progress of an unfinished wiring stage, and whether a build would continue it (wiring.assess, the
        same decision the build makes), with the reason when it would not.  Never raises on damaged files."""
        try:
            m = wiring_files.read_progress(self.dir)
            if m is None:
                return None
            v = (m.get("identity") or {}).get("variant") if not m.get("unreadable") else None
            v = v if v in ("base", "current") else self.meta.get("variant", "base")
            a = wiring_files.public(wiring_files.assess(self.dir, wiring_files.identity(self.ws, v, planned_source(self.ws, self.meta, v))))
        except Exception as e:                                                     # status must always answer
            a, v = {"usable": False, "reason": f"{wiring_files.UNREADABLE} ({type(e).__name__})", "done": 0, "total": 0,
                    "next_layer": None, "state": None}, self.meta.get("variant", "base")
        a = _while_running(a, running)
        return {**a, "variant": v, "rebuilds_weights": v != "base", "drops_observed": v != "base" and "observed" in stages}

    def status(self) -> dict:
        s = self.ws.struct
        m = self.meta
        cost = wiring_cost(self.ws) if s.recognised else 0.0
        stages = dict(m.get("stages") or {})
        if "wiring" in stages and not self.has("wiring"):
            stages.pop("wiring")                                                   # being rebuilt, or files missing
        running = build_running(self.ws)                                           # whoever started it
        out = {"built": self.exists, "model": self.ws.name, "variant": m.get("variant", "base"), "stages": stages, "version": m.get("version"),
               "can_build": bool(s.recognised and s.unembed is not None and any(s.has_neurons(l) for l in range(s.n_layers))),
               "can_observe": bool(self.ws.can_run), "observed_unavailable": observed_unavailable(self.ws),
               "observed_note": observed_note(self.ws),
               "fetch_bytes": fetch_bytes(self.ws) if s.recognised else 0,
               "wiring_cost": cost, "wiring_auto": cost <= AUTO_WIRING_FLOPS,
               "neurons": int(sum(s.mlp_width(l) for l in range(s.n_layers) if s.has_neurons(l))) if s.recognised else 0,
               "wiring_built": self.has("wiring"), "wiring_progress": self.wiring_progress(stages, running),
               "build_running": running}
        if self.exists:
            out["stale_layers"] = self.stale_layers()
            out["tau"] = tau(s.d_model)
            out["null"] = self.json("null.json")
        return out

    # ------------------------------------------------------------------ reading
    def text(self, ids) -> list[str]:
        t = token_table(self.ws)["text"]
        return [t[int(i)] if 0 <= int(i) < len(t) else "?" for i in ids]

    def concept_name(self, c: int) -> str:
        if c < 0:
            return ""
        nodes = (self.json("concepts.json") or {}).get("nodes") or []
        return " ".join(self.text(nodes[c]["tokens"][:4])) if c < len(nodes) else ""

    def unit(self, l: int, n: int) -> dict:
        """Everything the atlas knows about one neuron, with token ids resolved to strings."""
        self.need()
        A = self.layer(l)
        null = self.json("null.json")
        f = lambda k: float(A[k][n])                                               # noqa: E731

        def sig(prefix, which):
            if f"{prefix}_pos_ids" not in A:
                return None
            thr = null[which]["top1_q999"]
            ends = {}
            for end in ("pos", "neg"):
                ids, z = A[f"{prefix}_{end}_ids"][n], A[f"{prefix}_{end}_z"][n].astype(np.float32)
                ends[end] = [{"id": int(i), "token": t, "z": round(float(v), 2), "outlier": bool(abs(v) > thr)}
                             for i, t, v in zip(ids, self.text(ids), z)]
            return ends

        out = {"layer": l, "neuron": n, "addr": f"blocks.{l}.mlp.neuron.{n}", "mech": MECH[int(A["mech"][n])],
               "role": ROLE[int(A["role"][n])], "sign_w": int(A["sign_w"][n]),
               "trig_E": bool(A["trig_E"][n]) if "trig_E" in A else None, "trig_U": bool(A["trig_U"][n]),
               "geometry": {k: round(f(k), 4) for k in ("cos_gu", "cos_gd", "cos_ud", "g_norm", "u_norm", "d_norm", "bias", "bias_z", "out_vis", "d_top_share")},
               "d_top_dim": int(A["d_top_dim"][n]), "gated": bool(self.ws.struct.layers[l].mlp.get("W_gate")),
               "writes": sig("out", "U"), "gate_tokens": sig("gate", "E"), "value_tokens": sig("up", "E"), "gate_predicted": sig("gateU", "U"),
               "categories": {"writes_pos": CATS[int(A["out_cat_pos"][n])], "writes_neg": CATS[int(A["out_cat_neg"][n])]},
               "concepts": {}, "stale": l in self.stale_layers()}
        for key, label in (("out_pos", "writes_pos"), ("out_neg", "writes_neg"), ("gate", "gate_tokens"), ("gateU", "gate_predicted")):
            if f"c_{key}" in A and int(A[f"c_{key}"][n]) >= 0:
                c = int(A[f"c_{key}"][n])
                out["concepts"][label] = {"id": c, "name": self.concept_name(c), "outliers": int(A[f"n_{key}"][n]), "purity": round(f(f"p_{key}"), 2)}
        if "w_down_gate_cos" in A:
            wires = []
            for d_, k in (("down", "gate"), ("down", "up"), ("up", "gate"), ("up", "up")):
                for j in range(A[f"w_{d_}_{k}_cos"].shape[1]):
                    c, ll, ii = float(A[f"w_{d_}_{k}_cos"][n, j]), int(A[f"w_{d_}_{k}_layer"][n, j]), int(A[f"w_{d_}_{k}_idx"][n, j])
                    if ll >= 0 and abs(c) >= 1.3 * tau(self.ws.struct.d_model):
                        wires.append({"direction": "reads this neuron" if d_ == "down" else "is read by this neuron",
                                      "via": "gate" if k == "gate" else "value", "layer": ll, "neuron": ii, "cos": round(c, 3)})
            out["wiring"] = wires
        if "obs_peak" in A:
            from .observe import unit_observed
            out["observed"] = unit_observed(self, l, n)
        return out

    def layer_table(self) -> dict:
        self.need()
        return self.json("layers.json")

    def concepts(self) -> dict:
        self.need()
        c = self.json("concepts.json")
        nodes = [{**n, "text": self.text(n["tokens"][:12])} for n in c["nodes"]]
        groups = [{**g, "text": self.text(g["tokens"][:8])} for g in c["groups"]]
        return {"nodes": nodes, "groups": groups, "space": c["space"]}

    def concept(self, c: int, top: int = 40) -> dict:
        """One token concept: its tokens and the neurons that have it at an end of a signature."""
        self.need()
        nodes = self.json("concepts.json")["nodes"]
        if not 0 <= c < len(nodes):
            raise ValueError(f"concept {c} does not exist (0..{len(nodes) - 1})")
        out = {**nodes[c], "text": self.text(nodes[c]["tokens"]), "name": self.concept_name(c), "by_layer": {}, "neurons": []}
        L, I = self.column("layer"), self.column("idx")
        for key, label, zc in (("out_pos", "writes +", "out_pos_z"), ("out_neg", "writes −", "out_neg_z"), ("gateU", "opens when predicting", "gateU_pos_z"), ("gate", "opens on", "gate_pos_z")):
            try:
                m = (self.column(f"c_{key}") == c) & (self.column(f"n_{key}") >= 2) & (self.column(f"p_{key}") >= 0.5)
            except KeyError:
                continue
            z = np.abs(self.column(zc)[:, 0].astype(np.float32))
            hit = np.nonzero(m)[0]
            out["by_layer"][label] = {int(l): int(n) for l, n in zip(*np.unique(L[hit], return_counts=True))}
            for i in hit[np.argsort(-z[hit])][:top]:
                out["neurons"].append({"layer": int(L[i]), "neuron": int(I[i]), "end": label, "z": round(float(z[i]), 1)})
        out["neurons"].sort(key=lambda r: -r["z"])
        out["neurons"] = out["neurons"][:top]
        return out

    def find_tokens(self, query: str, limit: int = 30) -> list[dict]:
        tt = token_table(self.ws)
        q = query.strip().lower()
        tc = self.tok2c() if self.exists else None
        out = []
        for i, t in enumerate(tt["text"]):
            if t.strip().lower() == q or (len(q) > 2 and q in t.lower()):
                out.append({"id": i, "token": t, "concept": int(tc[i]) if tc is not None else -1, "exact": t.strip().lower() == q})
        out.sort(key=lambda r: (not r["exact"], len(r["token"])))
        return out[:limit]


from .select import select, selection_script  # noqa: E402,F401
