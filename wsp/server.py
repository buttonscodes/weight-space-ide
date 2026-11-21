"""HTTP API and static files of the IDE.

    python -m wsp serve --model facts --port 8765

Everything the browser shows comes from here; the programmer model's tools arrive through
/api/agent/* (from the stdio MCP proxy) or POST /mcp (MCP over HTTP).  Handlers are plain
functions (run in a thread pool); anything that changes weights goes through Session.apply /
Workspace.commit, which take the workspace lock.
"""
from __future__ import annotations

from .runtime_events import RuntimeRefusal

import asyncio
from contextlib import nullcontext
import json
import hashlib
import os
import queue
import secrets
import threading
import time
import uuid
from urllib.parse import urlparse
import tempfile
from pathlib import Path
from typing import Any

import torch
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import numpy as np

from . import atlas as at
from . import addressing as A
from . import analysis as an
from . import chunks, conversations, harness, hierarchy, mcp, testsuite, tools, native_harness
from . import selections as sel_
from .session import Session
from .agent_auth import AuthManager
from .tokenizer import show
from .token_stream import TextStream, stream_id
from .generation_stop import GenerationStopped, Stops

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


def create_app(sess: Session, url: str) -> FastAPI:
    from contextlib import asynccontextmanager

    from .export_jobs import ExportJobs
    exports = ExportJobs()
    from .inspection import InspectionRequests, InspectionBusy
    inspections = InspectionRequests()
    metadata_reads = InspectionRequests(limit=128, max_pending=32)
    auth = AuthManager(sess)
    from .runtime_config import resolve, backend_status
    from .settings import SettingsStore, SettingsConflict, SettingsOutOfStep, SettingsNeedConfirmation, EDITABLE, preview as memory_preview, validate_interactive, supervised
    from .runtime_config import NEXT_LOAD, SUPERVISOR_LIVE
    active_settings = sess.runtime.settings if sess.runtime is not None else getattr(sess, 'settings', None)
    settings_store = SettingsStore(active_settings or resolve())
    # Edits read the active values; the Largest weight patch is applied as soon as it is saved (runtime_config.SERVER_LIVE).
    if hasattr(sess, 'set_edit_settings'): sess.set_edit_settings(settings_store.active['values'])
    settings_csrf = secrets.token_urlsafe(32)
    # One immutable module graph per server instance; cached older modules cannot mix into it.
    assets = {str(p.relative_to(WEB)): p.read_bytes() for p in sorted(Path(WEB).rglob("*"))
              if p.is_file() and p.suffix in (".js", ".css")}
    asset_version = hashlib.sha256(b"".join(name.encode() + content for name, content in assets.items())).hexdigest()[:20]
    index_html = Path(WEB, "index.html").read_text().replace('src="js/app.js?inspection=1"',
        f'src="/assets/{asset_version}/js/app.js"').replace('href="style.css"', f'href="/assets/{asset_version}/style.css"')

    from .lifecycle import Lifecycle

    def resources():
        yield ('model', sess.unload, lambda: [])
        yield ('exports', exports.shutdown, lambda: [])
        yield ('authentication', auth.shutdown, lambda: [getattr(job, "proc", None) for job in list(auth.jobs.values())])
        for run in list(sess.runs.values()):
            if getattr(run, 'native', False):
                yield ('native agent '+run.id, run.close, lambda run=run: [run.proc])
            elif run.status in ('starting', 'running') or (run.proc is not None and run.proc.poll() is None):
                yield ('agent '+run.id, lambda run=run: harness.stop(run), lambda run=run: [run.proc])

    lifecycle = Lifecycle(resources)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await lifecycle.cleanup()

    app = FastAPI(title="Weight-space programming IDE", lifespan=lifespan)
    app.state.ide_lifecycle = lifecycle
    app.state.exports = exports

    app.state.agent_auth = auth
    app.state.settings_store = settings_store
    origin = urlparse(url)
    allowed_hosts = {origin.hostname}
    if origin.hostname in {"127.0.0.1", "localhost", "::1"}:
        allowed_hosts.update({"127.0.0.1", "localhost", "::1"})

    request_lock = asyncio.Lock()
    stops = app.state.generation_stops = Stops()  # Run and Chat requests that the person can stop, by stream id
    # The direction requests that are running (measure, scores, prefix, dataset fetches): the last event each
    # published, by a key of its own, so a reloaded page or another window lists them (GET /api/directions,
    # "requests"). Touched from request threads: every read and write under the lock.
    direction_requests = app.state.direction_requests = {}
    direction_requests_lock = threading.Lock()

    @app.middleware("http")
    async def protect_auth(request: Request, call_next):
        if lifecycle.stopping.is_set():
            return JSONResponse({"error": "IDE is shutting down"}, status_code=503)
        settings_request = request.url.path.startswith("/api/settings")
        # Every Programmer route (sign-in, start, message, approve, stop, close, the saved conversations and their
        # transcripts, resume) answers only this IDE's own page: the right Host, no foreign Origin, not cross-site,
        # and the per-process token on anything that is not a read.
        harness_request = request.url.path == "/api/harness" or request.url.path.startswith("/api/harness/")
        if harness_request or settings_request:
            if settings_request and (request.client is None or request.client.host not in {'127.0.0.1','::1','testclient'}):
                return JSONResponse({'error':'Settings require the local IDE connection'}, status_code=403)
            try:
                host = urlparse("http://" + request.headers.get("host", ""))
                valid_host = host.hostname in allowed_hosts and host.port == origin.port
                supplied_origin = request.headers.get("origin")
                browser_origin = urlparse(supplied_origin) if supplied_origin else None
                valid_origin = not browser_origin or (browser_origin.scheme == origin.scheme and
                    browser_origin.hostname == host.hostname and browser_origin.port == host.port)
            except ValueError:
                valid_host = valid_origin = False
            if not valid_host or not valid_origin or request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"error": "Authentication requests must come from this IDE."
                                     if settings_request or request.url.path.startswith("/api/harness/auth")
                                     else "Programmer requests must come from this IDE."}, status_code=403,
                                    headers={"Cache-Control": "no-store"})
            token = settings_csrf if settings_request else auth.csrf
            token_header = "x-wsp-settings" if settings_request else "x-wsp-auth"
            if request.method != "GET" and not secrets.compare_digest(request.headers.get(token_header, ""), token):
                return JSONResponse({"error": "Reload Settings before saving." if settings_request else
                                     "Reload the Programmer tab before signing in." if request.url.path.startswith("/api/harness/auth")
                                     else "Reload the Programmer tab: this request did not carry its token."}, status_code=403,
                                    headers={"Cache-Control": "no-store"})
            if settings_request and request.method != 'GET':
                length = request.headers.get('content-length', '')
                if not length.isdigit() or int(length) > 16384:
                    return JSONResponse({'error':'Settings request exceeds 16 KiB or has no bounded content length'}, status_code=413)
        if sess.runtime is not None and (request.method != 'GET' or request.url.path in {'/api/tests','/api/export'}):
            # Local operator API only. A stored profile does not enable remote inference.
            peer = request.client.host if request.client else ''
            supplied = request.headers.get('origin')
            browser = urlparse(supplied) if supplied else None
            if (peer not in {'127.0.0.1','::1','testclient'} or
                (browser and (browser.hostname not in allowed_hosts or browser.port != origin.port)) or
                request.headers.get('sec-fetch-site') == 'cross-site'):
                return JSONResponse({'error':'Runtime actions require the local IDE origin'},status_code=403)
            if request.url.path == '/api/generation/stop' or settings_request:
                # Stop only sets a flag that the running request reads; a settings save or preview has its own
                # locks (SettingsStore, RuntimeClient) and must not wait behind a running Run or Chat request.
                response = await call_next(request)
            else:
                async with request_lock:
                    response = await call_next(request)
        else:
            response = await call_next(request)
        if harness_request or settings_request:
            response.headers["Cache-Control"] = "no-store"
            response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(RuntimeRefusal)
    @app.exception_handler(Exception)
    async def on_error(request: Request, exc: Exception):
        if isinstance(exc,RuntimeRefusal):
            if getattr(exc,'committed',None):return JSONResponse(exc.report,status_code=200)
            return JSONResponse({"error":str(exc),"event":exc.record,"status":"not_run","report":getattr(exc,"report",None)},status_code=409)
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)

    def ws():
        if sess.ws is None:
            raise HTTPException(409, "no workspace is open")
        return sess.ws

    def refuse_during_build(w, what: str) -> None:
        """Routes that take the session lock and then the workspace lock would hold the session lock for the whole
        of an atlas build (hours), so every other route that needs it would hang too: refuse at once instead.  (A
        build that starts after this check is still waited for, as before.)"""
        where = at.build_running(w) if w is not None else None
        if where:
            reason = at.running_reason(where)
            raise HTTPException(409, f"{reason[:1].upper()}{reason[1:]}; {what} waits until it finishes, so try again then.")

    def inspection_identity(w):
        groups = hashlib.sha256(json.dumps(w.groups, sort_keys=True).encode()).hexdigest()[:16]
        return inspections.identity(w) + ":" + groups + ":" + str(getattr(w, "inspection_revision", 0))

    def inspect_read(operation, options, calculate, workspace_id="", version=None, cancellable=False):
        passive = operation == "tree" or (operation == "node" and not options[2] and not options[3])
        if not passive and (not workspace_id or version is None):
            raise HTTPException(428, "Refresh the IDE before calculating: workspace_id and version are required.")
        w = ws()
        identity = inspection_identity(w)
        expected = w.version if version is None else version

        def validate():
            if sess.ws is not w or inspection_identity(w) != identity or (workspace_id and workspace_id != inspection_identity(w)) or w.version != expected:
                raise HTTPException(409, "Workspace changed; refresh before calculating again.")

        validate()
        if operation in ("ablation", "atlas_attribute"):                 # these hold the session lock while they wait
            refuse_during_build(w, "this calculation")
        def compute(cancel=None):
            if operation == "tree" or (operation == "node" and not options[2] and not options[3]):
                validate()
                return calculate(w)  # metadata never queues behind an expensive calculation
            with (sess.lock if operation in ("ablation", "atlas_attribute") else nullcontext()), w.lock:
                validate()  # queued requests must not calculate an obsolete version
                if cancellable:
                    class Stop:
                        def is_set(self): return cancel.is_set() or lifecycle.stopping.is_set()
                    return calculate(w, Stop())
                return calculate(w)
        cache = metadata_reads if passive else inspections
        try:
            result = cache.run((identity, expected, operation, options), compute, cancellable=cancellable)
        except InspectionBusy as exc:
            raise HTTPException(429, str(exc)) from exc
        validate()  # includes cache hits and changes while waiting on another request
        return result

    def status() -> dict:
        """What the top bar shows. Never loads a model: for checkpoints run through transformers, test and drift
        figures appear only when an explicit test/edit has already cached them for this version."""
        w = ws()
        loaded = w.runner_loaded
        suite = sess.task.get("suite")
        t = w.calib.get(f"tests:{suite}:current")
        b = w.keep.get(f"tests:{suite}:base")
        return {"tests": {"suite": t["name"], "passed": t["passed"], "total": t["total"], "base_passed": b["passed"] if b else None} if t else None,
                "drift": w.calib.get("drift", {"kl": None, "top1_agree": None, "n": 0}),
                "workspace_id": inspection_identity(w), "n_patches": len(w.patches), "version": w.version, "can_run": w.can_run, "runner_loaded": loaded,
                "run_error": w.run_error, "suite": sess.task.get("suite"),
                "changed": sum(w.changed(n)[0] for n in w.touched()),
                "stored": sum(p.meta()["stored"] for p in w.patches if p.enabled)}

    def generation_limits(w=None, load=None):
        # load: the load settings an operation is pinned to (RuntimeClient.operation passes them to its size function).
        values = settings_store.active['values']
        packed = w is not None and w.kind == 'gguf'
        if load is not None or supervised(sess.runtime):
            # The loaded model's load settings, else the ones the next load uses.
            load = load if load is not None else sess.runtime.config()
            selected, configured = load['selected'], load['context']
        else:
            selected, configured = backend_status(settings_store.active)['selected'], values['context']
        from .capacity import NATIVE_ROWS, context_limit
        bridge_limit = NATIVE_ROWS[selected]
        context = context_limit(configured, getattr(getattr(w,'struct',None),'n_ctx',None)) if packed else None
        output = values['new_tokens']
        if packed and sess.runtime is None:
            # Preserve the stricter legacy acceptance-harness contract.
            context = min(context, 512)
            output = min(output, 64)
        return dict(default_output=min(output, max(1, context-1)) if context else output,
                    output_limit=None if packed and sess.runtime is not None else output, context_limit=context, kernel_token_rows=bridge_limit,
                    model_context=getattr(getattr(w,'struct',None),'n_ctx',None))

    def settings_view(document=None, rev=None):
        active = None
        if document is None: document, rev, active = settings_store.consistent()
        result = settings_store.describe(document, rev, sess.runtime, active)
        saved = settings_store.resolve(document)
        result['saved_preview'] = memory_preview(saved, sess.ws, sess.runtime, getattr(app.state, 'settings_telemetry', None))
        effective = settings_store.resolve(document, effective=True)
        result['preview'] = result['saved_preview'] if saved['values'] == effective['values'] else memory_preview(
            effective, sess.ws, sess.runtime, lambda: result['saved_preview']['machine'])
        if result['overridden']:
            result['preview']['warnings'].append('Preview retains launch overrides for: '+', '.join(result['overridden'])+'. Change those launch arguments/environment to use the saved values.')
        result['watchdog_check'] = None
        if supervised(sess.runtime):
            draft = settings_store.updated(effective, True)
            if any(draft['values'][k] != settings_store.active['values'][k] for k in SUPERVISOR_LIVE | NEXT_LOAD):
                # Ask the supervisor itself (a dry run, nothing changes): whether it accepts the draft and what the
                # idle watch would do with it now. Saving asks again; this is only the warning before.
                check = result['watchdog_check'] = sess.runtime.check_settings(draft)
                effect = check.get('effect') or {}
                if check.get('accepted') is False:
                    result['preview']['errors'].append('The memory watchdog would refuse these settings: '+str((check.get('event') or {}).get('message')))
                elif check.get('accepted') is None:
                    result['preview']['warnings'].append('The memory watchdog could not check these settings now ('+str(check.get('reason'))+'). Saving asks it again.')
                elif effect.get('confirm'):
                    result['preview']['warnings'].append(effect['message']+' Saving asks you to confirm first.')
                elif effect.get('kind') in ('unload', 'unknown'):
                    result['preview']['warnings'].append(effect['message'])
                elif effect.get('kind') == 'after_operation':
                    result['preview']['info'].append(effect['message'])
        result['csrf'] = settings_csrf
        result['managed'] = sess.runtime is not None
        result['editable'] = sorted(EDITABLE)
        from .settings import capacity_recommendation
        recommendation = capacity_recommendation(saved, sess.ws, lambda: result['saved_preview']['machine'], runtime=sess.runtime, overrides=settings_store.overrides)
        result['interactive_defaults'] = recommendation['changes']
        result['capacity_recommendation'] = recommendation
        result['field_limits'] = dict(context=None, new_tokens=None)
        return result

    @app.get('/api/settings')
    def settings_get():
        return settings_view()

    def settings_candidate(body, confirmable=False):
        # A save may also carry ``confirm``: the token a refusal that needs confirmation gave for this candidate.
        if (not {'revision', 'changes'} <= set(body) <= {'revision', 'changes', *(('confirm',) if confirmable else ())}
                or not isinstance(body['revision'], str) or not isinstance(body.get('confirm', ''), str)):
            raise HTTPException(400, 'Expected revision and changes')
        document, rev = settings_store.read()
        if body['revision'] != rev:
            raise HTTPException(409, 'Settings changed on disk. Reload Settings before continuing.')
        return settings_store.candidate(document, body['changes']), rev

    @app.post('/api/settings/preview')
    def settings_preview(body: dict = Body(...)):
        document, rev = settings_candidate(body)
        return settings_view(document, rev)

    @app.post('/api/settings')
    def settings_save(body: dict = Body(...)):
        document, rev = settings_candidate(body, confirmable=True)
        def validate(candidate):
            result = memory_preview(candidate, sess.ws, sess.runtime, getattr(app.state, 'settings_telemetry', None))
            validate_interactive(candidate, result['machine'].get('total'))
            if result['errors']: raise ValueError('; '.join(result['errors']))
        try:
            saved = settings_store.save(rev, body['changes'], validate, runtime=sess.runtime, confirm=body.get('confirm'))
        except SettingsConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except SettingsNeedConfirmation as exc:
            # The supervisor judged the candidate (one sample, in the same request that would apply it): it would
            # unload the loaded model. Nothing changed; the page asks, then sends the same candidate with ``confirm``.
            raise HTTPException(409, dict(message=str(exc), effect=exc.effect, confirm=exc.token)) from exc
        except RuntimeRefusal as exc:
            # Refused by the memory watchdog, or no answer from it while, asked again, it still runs the previous
            # settings: nothing was written or changed. (Unknown or other settings raise SettingsOutOfStep.)
            status_code = 503 if exc.record.get('code') == 'watchdog_unavailable' else 409
            raise HTTPException(status_code, 'Not saved: '+str(exc)) from exc
        except SettingsOutOfStep as exc:
            if hasattr(sess, 'set_edit_settings'): sess.set_edit_settings(settings_store.active['values'])
            raise HTTPException(500, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(500, f'Not saved: {type(exc).__name__}: {exc}') from exc
        if hasattr(sess, 'set_edit_settings'): sess.set_edit_settings(settings_store.active['values'])
        return {**settings_view(), 'applied': saved['applied']}

    @app.get('/api/runtime')
    def runtime_status():
        result = sess.runtime.status(sess.ws) if sess.runtime is not None else dict(supervised=False, limits=settings_store.active['values'], backend=backend_status(settings_store.active), note='Legacy harness / embedded server; use regular wsp serve for user profiles and interactive supervision')
        # Configuration-only summary: no memory preview, tensor read or runner work. A save in flight holds the
        # store's lock across the supervisor exchange; this status read does not wait for it (consistent()).
        try:
            saved = settings_store.snapshot(sess.runtime)
            keys = ('context','new_tokens')
            summary = {name: {k: saved[name]['values'][k] for k in keys}
                       for name in ('active','saved','next_start')}
            summary.update(profile=saved['profile'], target=saved['target'], supervised=saved['supervised'],
                           loaded=saved['loaded'], pending=saved['pending'], out_of_step=saved['out_of_step'],
                           restart_required=saved['restart_required'], overridden=saved['overridden'])
        except (ValueError, OSError) as exc:
            summary = {'error': str(exc), 'profile': settings_store.profile}
        return {**result, 'generation': generation_limits(sess.ws), 'settings_summary': summary}

    @app.post('/api/runtime/dismiss')
    def runtime_dismiss():
        if sess.runtime is not None:sess.runtime.dismiss()
        return {'ok':True}

    @app.post('/api/runtime/unload')
    def runtime_unload():
        sess.unload()
        return {'ok': True}

    @app.get("/api/inspection-capabilities")
    def inspection_capabilities():
        return {"version": 1, "passive_metadata": True}

    # ------------------------------------------------------------------ state
    @app.get("/api/state")
    def state():
        out = {"workspaces": sess.available(), "open": None, "harnesses": harness.available(), "url": url}
        if sess.ws is not None:
            w = sess.ws
            out["open"] = w.summary()
            out["task"] = sess.task
            out["suites"] = testsuite.list_suites(w)
            out["status"] = status()
            out["groups"] = w.groups
            out["notes"] = w.notes()
            out["levels"] = hierarchy.LEVELS
            out["describe"] = w.struct.describe()
            V = w.struct.d_vocab
            out["vocab"] = [show(w.tok.token_str(i)) for i in range(V)] if w.tok is not None and 0 < V <= 512 else None
        return out

    @app.post("/api/open")
    def open_(body: dict = Body(...)):
        logs: list[str] = []
        with sess.lock:
            native_harness.assert_available(sess)
            auth.assert_no_login()
            refuse_during_build(sess.ws, "opening another model")
            sess.open(body["name"].strip(), log=logs.append, pull=bool(body.get("pull")))
        return {"ok": True, "log": logs, "name": sess.ws.name}

    @app.post("/api/pull")
    def pull_():
        """Download the weights of a Hub-backed workspace so that it can be run."""
        from .zoo import pull
        logs: list[str] = []
        with sess.lock:
            native_harness.assert_available(sess)
            auth.assert_no_login()
            refuse_during_build(sess.ws, "downloading the weights")
            name = ws().name
            pull(ws(), logs.append)
            sess.open(name)
        return {"ok": True, "log": logs, "can_run": sess.ws.can_run}

    @app.get("/api/status")
    def status_():
        return status()

    # ------------------------------------------------------------------ hierarchy and weights
    @app.get("/api/tree")
    def tree(addr: str = "model", workspace_id: str = "", version: int | None = None):
        addr = A.normalize(addr)
        return inspect_read("tree", (addr,), lambda w: {
            "node": hierarchy.node(w, addr, kind=hierarchy.kind_of(w, addr), has_children=True, norm=False),
            "children": hierarchy.children(w, addr)}, workspace_id, version)

    def est_tokens(w, slices) -> dict:
        """chunks.est_slices for each format, worked out once per distinct slice shape (a selection of thousands of
        neurons has three)."""
        from collections import Counter
        shapes = Counter(tuple(A.slice_shape(w.shape(n), i)) for n, i in slices)
        return {f: sum(chunks.est_tensor(s, f) * k for s, k in shapes.items()) for f in ("q", "g3", "svd", "stats")}

    def node_data(w, addr, fetch, analysis, statistics):
        kind = hierarchy.kind_of(w, addr)
        slices = w.resolve(addr)
        count = A.count(w, slices)
        selection = sel_.expand(w, addr)
        extras = sel_.node_extras(w, addr, selection)
        if not any(w.pieces(n) for n in {n for n, _ in slices}):    # nothing here is edited
            changed = 0
        elif addr.startswith("group:") or len(slices) > 64:         # thousands of neurons: count by position
            # the revert plan behind the actions has counted exactly this (each weight of every part once)
            # (left out when the plan refused: then selections.changed counts slice by slice)
            changed = extras["actions"]["changed"] if "changed" in extras["actions"] else sel_.changed(w, addr, selection)[0]
        else:
            changed = sum(w.changed(n, i)[0] for n, i in slices if w.pieces(n))
        out = {"addr": addr, "kind": kind, "level": hierarchy.LEVELS.index(kind), "n_params": count,
               "stats": an.region_stats(w, addr) if statistics else {
                   "current": {"n": count}, "delta": {}, "deferred": True, "changed": changed},
               "n_slices": len(slices),
               "slices": [{"addr": A.slice_str(x), "script": A.to_script(*x), "param": x[0], "dtype": w.store.dtype(x[0]),
                           "editable": w.editable(x[0]),
                           "shape": list(A.slice_shape(w.shape(x[0]), x[1])), "doc": w.struct.doc(x[0])} for x in slices[:64]],
               "est_tokens": est_tokens(w, slices),
               "changed": changed, "edit_waits_for_generation": waits_for_generation(w),
               **extras}
        parts = addr.split(".")
        if kind == "unit" and parts[3] in ("head", "neuron"):
            typ = parts[3]
            out["analysis"] = typ
            L = w.struct.layers[int(parts[1])]
            name = (L.attn["W_O"] if typ == "head" else L.mlp["W_out"]).name
            out["write_slices"] = [x for x in out["slices"] if x["param"] == name]
        elif kind == "tensor" and len(slices) == 1 and w.struct.recognised:
            name, idx = slices[0]
            if an.kept_roles(w, name, idx) == ["resid"]:
                out["analysis"] = "direction"
        if out.get("analysis"):
            need = w.sem.vocab_bytes()
            if need:
                out["needs_fetch"] = need
            if analysis:
                if need and not fetch:
                    raise HTTPException(409, "This calculation needs vocabulary weights from the Hub; explicitly allow fetch.")
                if out["analysis"] == "head":
                    out["head"] = an.head_card(w, int(parts[1]), int(parts[4]))
                elif out["analysis"] == "neuron":
                    out["neuron"] = an.neuron_card(w, int(parts[1]), int(parts[4]))
                else:
                    out["direction"] = an.direction_tokens(w, w.read(name, idx))
        return out

    @app.get("/api/node")
    def node(addr: str, fetch: int = 0, analysis: int = 0, statistics: int = 0,
             workspace_id: str = "", version: int | None = None):
        addr = A.normalize(addr)
        return inspect_read("node", (addr, bool(fetch), bool(analysis), bool(statistics)),
                            lambda w: node_data(w, addr, fetch, analysis, statistics), workspace_id, version)

    @app.get("/api/matrix")
    def matrix(addr: str, variant: str = "current", r0: int = 0, r1: int = 0, c0: int = 0, c1: int = 0,
               max_h: int = 256, max_w: int = 384, pool: str = "mean", sort_rows: str = "", sort_cols: str = "",
               workspace_id: str = "", version: int | None = None):
        addr = A.normalize(addr)
        args = (addr, variant, r0, r1 or None, c0, c1 or None, max_h, max_w, pool, sort_rows, sort_cols)
        return inspect_read("matrix", args, lambda w: an.matrix_view(w, *args), workspace_id, version)

    @app.get("/api/matrix/compare")
    def matrix_compare(addr: str, r0: int = 0, r1: int = 0, c0: int = 0, c1: int = 0, max_h: int = 256, max_w: int = 384,
                       pool: str = "mean", workspace_id: str = "", version: int | None = None):
        """Base, current and their difference for one tensor or slice (the inspector's before/after view)."""
        from .compare import matrix_compare as compare
        addr = A.normalize(addr)
        args = (addr, r0, r1 or None, c0, c1 or None, max_h, max_w, pool)
        return inspect_read("matrix_compare", args, lambda w: compare(w, *args), workspace_id, version)

    @app.post("/api/selection/script")
    def selection_script(body: dict = Body(...)):
        """Contract C2: the weight script for scale / ablate / revert of any selection, built here with no member
        cap; the browser applies it through /api/edit as one patch. A refusal is a plain answer, not an error."""
        w = ws()
        try:
            return sel_.script(w, body.get("addr") or "", body.get("action") or "", body.get("factor"))
        except sel_.Refusal as e:
            return JSONResponse({"error": str(e), "refused": True, **e.counts}, status_code=400)

    @app.get("/api/cell")
    def cell(addr: str, r: int, c: int):
        return an.cell_address(ws(), addr, r, c)

    @app.get("/api/svd")
    def svd(addr: str, variant: str = "current", k: int = 6, workspace_id: str = "", version: int | None = None):
        addr = A.normalize(addr)
        return inspect_read("svd", (addr, variant, k), lambda w: an.svd_view(w, addr, k, variant), workspace_id, version)

    @app.get("/api/wiring")
    async def wiring(request: Request, top: int = 1500, layers: str = "", focus: int = -1, workspace_id: str = "", version: int | None = None):
        w = ws()
        if not w.struct.recognised:
            raise ValueError("the architecture of this checkpoint was not recognised, so there are no components to connect")
        ls = [int(x) for x in layers.split(",") if x.strip()] if layers else an.default_window(w, focus if focus >= 0 else None)
        return await asyncio.to_thread(inspect_read, "wiring", (min(4000,max(1,top)), tuple(ls)),
            lambda w,cancel: an.wiring(w,min(4000,max(1,top)),ls,cancel=cancel,
                progress=lambda value:sess.publish(dict(type='wiring_progress',workspace_id=workspace_id,version=version,**value))),
            workspace_id, version, True)

    @app.post('/api/wiring/cancel')
    def cancel_wiring(body: dict = Body(...)):
        w = ws()
        if body.get('workspace_id') != inspection_identity(w) or body.get('version') != w.version:
            raise HTTPException(409,'Workspace changed; refresh before cancelling')
        layers = body.get('layers','')
        focus = int(body.get('focus',-1))
        ls = [int(x) for x in layers.split(',') if x.strip()] if layers else an.default_window(w,focus if focus>=0 else None)
        key=(inspection_identity(w),w.version,'wiring',(min(4000,max(1,int(body.get('top',1500)))),tuple(ls)))
        return {'cancelled':inspections.cancel(key)}

    @app.post("/api/direction")
    def direction(body: dict = Body(...)):
        from .wscript import Interp
        it = Interp(ws())
        it.run("vdir = " + body["expr"] + " + 0")
        return an.direction_tokens(ws(), torch.as_tensor(it.locals["vdir"]).float(), int(body.get("k", 12)))

    # ------------------------------------------------------------------ the atlas
    def atlas_():
        from .atlas import Atlas
        return Atlas.of(ws())

    @app.get("/api/atlas")
    def atlas_status():
        from .atlas import guide
        A = atlas_()
        out = {**A.status(), "job": sess.atlas_job if sess.atlas_job and sess.atlas_job.get("workspace_id") == inspection_identity(ws()) and sess.atlas_job.get("version") == ws().version else None, "shift": guide.shift_plan(ws()) if ws().struct.recognised else None}
        if A.exists:
            out.update(layers=A.layer_table(), validation=A.json("validation.json"), residual=A.json("residual.json"),
                       heads=A.json("heads.json"), heads_observed=A.json("heads_observed.json"),
                       mech=at.MECH, role=at.ROLE, mech_doc=at.MECH_DOC, role_doc=at.ROLE_DOC)
        return out

    @app.post("/api/atlas/build")
    def atlas_build(body: dict = Body(default={})):
        import threading
        # Claim under the session lock before starting a worker. Never queue builds.
        with sess.lock:
            w = ws()
            identity, version = inspection_identity(w), w.version
            if not body.get("workspace_id") or body.get("version") is None:
                raise HTTPException(428, "Refresh the IDE before building the atlas.")
            if body["workspace_id"] != identity or body["version"] != version:
                raise HTTPException(409, "Workspace changed; refresh before building.")
            stages = body.get("stages", ["weights"])
            if not isinstance(stages, list) or not stages or any(x not in ("weights", "wiring", "observed") for x in stages):
                raise HTTPException(422, "Choose weights, wiring and/or observed stages explicitly.")
            options = {"stages": sorted(set(stages)), "fetch": bool(body.get("fetch")),
                       "force": bool(body.get("force")), "tokens": body.get("tokens") or None,
                       "variant": body.get("variant") or "base"}
            if sess.atlas_job and sess.atlas_job.get("running"):
                if (sess.atlas_job["workspace_id"], sess.atlas_job["version"], sess.atlas_job["options"]) == (identity, version, options):
                    return sess.atlas_job
                raise HTTPException(409, "An atlas build is already running; wait for it to finish.")
            unavailable = at.observed_unavailable(w) if "observed" in stages else None
            if unavailable:
                # Refused here, in the page's words, before a job is claimed (the build itself also refuses).
                raise HTTPException(422, unavailable + " Untick Observed and build again.")
            job = {"running": True, "log": [], "error": None, "model": w.name, "progress": None,
                   "workspace_id": identity, "version": version, "options": options}
            sess.atlas_job = job

        def emit(**event):
            sess.publish({"type": "atlas", "workspace_id": identity, "version": version, **event})

        def log(message):
            if lifecycle.stopping.is_set():
                raise RuntimeError("Server stopping; atlas build interrupted at a progress boundary.")
            job["log"].append(str(message))
            del job["log"][:-200]
            emit(line=str(message))

        def progress(state):
            # Wiring stage, after each layer: layers done of the total, time left from the pace so far.
            job["progress"] = dict(state)
            emit(progress=dict(state))

        def work():
            try:
                # Unload and edits use this same lock. Reject a stale queued snapshot. The atlas build lock comes
                # first and is refused at once, so a build started elsewhere (the MCP tool, the CLI) is reported
                # instead of waited for while holding nothing.
                with at.build_lock(w), w.lock:
                    if sess.ws is not w or w.version != version or inspection_identity(w) != identity or lifecycle.stopping.is_set():
                        raise RuntimeError("Workspace changed or server stopped before the build started.")
                    at.build(w, **options, log=log, progress=progress)
            except at.Busy as e:
                job["error"] = str(e)[:1].upper() + str(e)[1:] + "."
            except RuntimeRefusal as e:                      # admission or a lost worker: its own plain words
                job["error"] = str(e)[:1].upper() + str(e)[1:]
                job["event"] = e.record
            except Exception as e:
                job["error"] = f"{type(e).__name__}: {e}"
            finally:
                job["running"] = False
                emit(done=True, error=job["error"])

        threading.Thread(target=work, daemon=True, name="atlas-build").start()
        return job

    @app.get("/api/atlas/estimate")
    def atlas_estimate(variant: str = "base", restart: bool = False):
        """How long the wiring stage would take here, the memory it holds and what waits for it. Loads no model and
        reads no weights; the first call in a process times one step of the search on random matrices."""
        if variant not in ("base", "current"):
            raise HTTPException(422, "variant is base or current")
        w = ws()
        if not w.struct.recognised or not any(w.struct.has_neurons(l) for l in range(w.struct.n_layers)):
            raise HTTPException(409, "This model has no recognised neurons, so it has no wiring stage.")
        return at.wiring_estimate(w, variant, restart=restart)

    @app.get("/api/atlas/layer")
    def atlas_layer(l: int):
        """Compact per-neuron columns of one layer for the neuron grid."""
        A = atlas_()
        A.need()
        L = A.layer(l)
        cols = {"mech": L["mech"], "role": L["role"], "trig": L.get("trig_E", L["trig_U"] * 0).astype("int8") + 2 * L["trig_U"].astype("int8"),
                "sign": L["obs_peak"] if "obs_peak" in L else L["sign_w"], "concept": np.where(L["c_out_pos"] >= 0, L["c_out_pos"], L["c_out_neg"]),
                "vis": np.round(L["out_vis"].astype("float32"), 2), "cos_ud": np.round(L["cos_ud"].astype("float32"), 3)}
        return {"layer": l, "n": int(len(L["mech"])), "observed": "obs_peak" in L, "stale": l in A.stale_layers(),
                **{k: v.tolist() for k, v in cols.items()}}

    @app.get("/api/atlas/unit")
    def atlas_unit(l: int, n: int):
        return atlas_().unit(l, n)

    @app.get("/api/atlas/concepts")
    def atlas_concepts():
        return atlas_().concepts()

    @app.get("/api/atlas/concept")
    def atlas_concept(id: int):
        A = atlas_()
        c = A.concept(id)
        for r in c["neurons"]:
            L = A.layer(r["layer"])
            r["plus"], r["minus"] = A.text(L["out_pos_ids"][r["neuron"]][:4]), A.text(L["out_neg_ids"][r["neuron"]][:4])
        return c

    @app.get("/api/atlas/tokens")
    def atlas_tokens(q: str):
        return atlas_().find_tokens(q)

    @app.post("/api/atlas/select")
    def atlas_select(body: dict = Body(...)):
        A = atlas_()
        r = at.select(A, body.get("selector") or {})
        rows = []
        for l, i, z in list(zip(r["layer"], r["idx"], r["strength"]))[:int(body.get("show", 60))]:
            L = A.layer(int(l))
            rows.append({"layer": int(l), "neuron": int(i), "z": round(float(z), 1), "mech": at.MECH[int(L["mech"][i])],
                         "plus": A.text(L["out_pos_ids"][i][:4]), "minus": A.text(L["out_neg_ids"][i][:4]),
                         "peak": int(L["obs_peak"][i]) if "obs_peak" in L else None})
        return {"n": r["n"], "by_layer": r["by_layer"], "sign_source": r["sign_source"], "rows": rows,
                "script": at.selection_script(r, float(body.get("scale", 0.0))) if r["n"] else ""}

    @app.post("/api/atlas/attribute")
    def atlas_attribute(body: dict = Body(...)):
        from .atlas import guide
        def calculate(w):
            res = guide.attribute(w, body["prompt"], body["target"], body.get("versus") or None,
                                  body.get("variant", "current"), int(body.get("top", 20)), body.get("selector") or None)
            res["script"] = guide.top_script(res, int(body.get("k", 5)), float(body.get("scale", 0.0)))
            return res
        options = json.dumps({k: v for k, v in body.items() if k not in ("workspace_id", "version")}, sort_keys=True)
        return inspect_read("atlas_attribute", (options,), calculate, body.get("workspace_id", ""), body.get("version"))

    # ------------------------------------------------------------------ running the model
    def runnable(w=None):
        w = w if w is not None else ws()
        if w.can_run:
            try:
                w.runner
            except RuntimeRefusal:
                raise
            except RuntimeError:
                pass
        if not w.can_run:
            raise ValueError("this model cannot be run here: " + (
                w.run_error or (w.store.capabilities()["generation"]["reason"] if w.kind == "gguf" else None) or ("its weights are still on the Hub (use Pull to download them)" if w.store.remote else
                                "it has no configuration that says how to run it")))
        return w

    def _next(w, prompt: str, variant: str, k: int, wire=None):
        if wire is not None and variant == 'current':
            # The steered distribution the edited reply samples from: one forward pass, as without steering.
            from .probes import _next_logits
            lg = _next_logits(w, w.encode(prompt), 'current', wire)
        else:
            lg = w.runner.next_logits(w.encode(prompt), variant) if w.kind == 'gguf' else w.logits(prompt, variant)[-1]
        p = lg.softmax(-1).topk(min(k, lg.numel()))
        return [[show(w.tok.token_str(int(i))), round(float(x), 4), round(float(lg[i]), 3)] for x, i in zip(p.values, p.indices)]

    def stoppable(route):
        """Registers the request's stream id while it runs, so that POST /api/generation/stop finds it."""
        from functools import wraps
        @wraps(route)
        def wrapped(body: dict = Body(...)):
            with stops.open(stream_id(body)):
                return route(body)
        return wrapped

    @app.post("/api/generation/stop")
    def generation_stop(body: dict = Body(...)):
        """Stop a running Run or Chat request. Answers at once; the request itself answers with ``stopped``."""
        stream = stream_id(body)
        if stream is None:
            raise ValueError('stream is required: the id the Run or Chat request was sent with')
        return {"ok": True, "found": stops.stop(stream)}

    @app.post("/api/run")
    @stoppable
    def run(body: dict = Body(...)):
        w = ws()
        prompt = body["prompt"]
        ids = w.encode(prompt)
        from .capacity import output_count
        fitted = {}
        def fit(load=None):
            # Sized for the load settings it runs with: called here, before the model is touched, and again by the
            # operation under the runtime client's lock once they are pinned (a save between cannot mismatch them).
            limits = generation_limits(w, load)
            requested = body.get("max_new_tokens", limits["default_output"])
            n = output_count(requested,len(ids),limits['context_limit'],allow_zero=True,maximum=limits['output_limit'])
            fitted.update(limits=limits, requested=requested, n=n)
            return len(ids)+n
        fit()
        limits, requested, n = fitted['limits'], fitted['requested'], fitted['n']
        steer = _steer_request(w, body, ["current"])    # checked before the model is touched; None without steer
        stream = stream_id(body)
        halt = stops.get(stream)
        runnable(w)
        out = {"generation": dict(requested_output=requested,effective_output=n,prompt_tokens=len(ids),context=limits["context_limit"]), "prompt": prompt, "tokens": [show(w.tok.token_str(i)) for i in ids],
               "join": " " if w.tok.to_dict().get("mode") == "word" else ""}
        out.update(base={"next":[],"status":"not_run"},current={"next":[],"status":"not_run"})
        wire, steering = _generation_steer(w, steer, stream, halt)
        if steering is not None:
            out["steering"] = steering
        client=getattr(w,'_runtime_client',None)
        scope_started=False
        try:
            with w.lock if client else nullcontext(), client.operation(w,'completion',tokens=len(ids)+n,size=fit) if client else nullcontext():
                scope_started=True
                limits, requested, n = fitted['limits'], fitted['requested'], fitted['n']
                out["generation"].update(requested_output=requested,effective_output=n,context=limits["context_limit"])
                # The edited model first: its text is the one being worked on, and it is what an interruption keeps.
                for v in ("current", "base"):
                    if halt and halt.is_set():
                        out['stopped']=True;break   # stopped by the person: this and later variants are not run
                    r={"next":[],"status":"complete"};out[v]=r
                    pushed=wire if v=="current" else None   # the base model is never steered
                    if pushed is not None:r["steered"]=True
                    live=TextStream(sess.publish,stream,v,w.tok.decode,stop=halt) if stream and n else None
                    if live:live.begin()
                    try:
                        r["next"]=_next(w,prompt,v,int(body.get("top_k_show",10)),pushed)
                        if live and halt.is_set():raise GenerationStopped()  # stopped while the prompt was scored: generate nothing
                        if n:r["text"]=w.complete(prompt,v,max_new_tokens=n,temperature=float(body.get("temperature",0)),top_k=int(body.get("top_k",0)),seed=body.get("seed",0),**({'on_token':live} if live else {}),**({'steer':pushed} if pushed is not None else {}))
                    except GenerationStopped:
                        pass                        # handled below like a runner that returned early
                    except RuntimeRefusal as exc:
                        r.update(text=w.tok.decode(exc.tokens),status='interrupted',event=exc.record)
                        out['event']=exc.record;out['interrupted']=True
                        if live:live.end('interrupted',exc.tokens)
                        raise
                    except Exception:
                        if live:live.end('failed')
                        raise
                    if live and halt.is_set():      # also when the runner returned early, e.g. stopped while reading the prompt
                        r.update(text=w.tok.decode(live.ids),status='stopped',tokens=len(live.ids))
                        out['stopped']=True;live.end('stopped');break
                    if live:live.end('complete',text=r['text'])
        except RuntimeRefusal as exc:
            if not scope_started:raise
            out['event']=exc.record;out['interrupted']=True
        return out

    def chat_options(body):
        """The chat template options of a request (Chat, and probes read as a chat message)."""
        from .probes import check_chat_options
        return check_chat_options(body.get('chat_options') or {})

    NAMES = {'current': 'edited', 'base': 'base'}
    FIELDS = {'current': 'messages', 'base': 'base_messages'}

    def chat_conversation(value, name):
        """A conversation of a Chat request, checked before the model: a list of turns, each an object. Anything
        else fails later in every path, so this only makes that failure a clear message. An empty list and any
        role value are accepted, as they always were; what a turn needs beyond that depends on the template
        that renders it (plain_turns for the plain turn template)."""
        if not isinstance(value,list) or not all(isinstance(m,dict) for m in value):
            raise ValueError(f'{name} must be a list of turns, each an object with a role and a content')
        return value

    def plain_template(body):
        """The plain turn template of a Chat request (the request's ``template`` over the defaults)."""
        given = body.get("template") or {}
        if not isinstance(given, dict):
            raise ValueError('template must be an object of turn prefixes (system, user, assistant) and a sep')
        tpl = {"system": "", "user": "User: ", "assistant": "Assistant: ", "sep": "\n", **given}
        if not all(isinstance(tpl[k], str) for k in ("system", "user", "assistant", "sep")):
            raise ValueError('template prefixes (system, user, assistant) and sep must be text')
        return tpl

    def plain_turns(messages, field, reason):
        """The plain turn template joins text: each turn it renders needs a role and text content."""
        for i, m in enumerate(messages):
            if 'role' not in m or not isinstance(m.get('content'), str):
                raise ValueError(f'{field}[{i}]: {reason}, so each turn needs a role and its content as text')

    def chat_request(body):
        """Which models a Chat request runs and the conversation each answers, checked before the model.
        -> {variant: messages}, in execution order (the edited model first).

        ``messages`` is the edited model's conversation and ``base_messages`` the base model's; without
        ``base_messages`` the base model answers ``messages``, as before. A request that runs the base model
        only sends its conversation as ``base_messages`` and no ``messages``."""
        variants = body.get('variants')
        if variants is None:
            variants = ['current', 'base']
        elif (not isinstance(variants,list) or not variants or not all(isinstance(v,str) for v in variants)
              or any(v not in NAMES for v in variants) or len(set(variants)) != len(variants)):
            raise ValueError("variants must list 'current' and/or 'base', each at most once")
        base_given = body.get('base_messages') is not None
        if 'base' not in variants and base_given:
            raise ValueError('base_messages is the base model\'s conversation, and this request runs the edited model only')
        if 'current' not in variants:
            if body.get('messages') is not None:
                raise ValueError('messages is the edited model\'s conversation, and this request runs the base model only; send the base model\'s conversation as base_messages')
            if not base_given:
                raise ValueError('this request runs the base model only; send its conversation as base_messages')
            # Only a steering request is refused: without one, p50 and recalibrate are ignored, as in any request.
            steer = body.get('steer')
            if steer is not None and steer != []:
                raise st.SteeringError('steering applies to the edited model only, and this request runs the base model only')
        out = {}
        if 'current' in variants:
            out['current'] = chat_conversation(body.get('messages'), 'messages')
        if 'base' in variants:
            out['base'] = chat_conversation(body['base_messages'], 'base_messages') if base_given else out['current']
        return out

    @app.post("/api/chat")
    @stoppable
    def chat(body: dict = Body(...)):
        """Chat. Models with a chat template are prompted through it; others through a simple turn template.
        Each model answers its own conversation (``messages``, ``base_messages``); ``variants`` says which run."""
        w = ws()
        limits = generation_limits(w)
        requested = body.get("max_new_tokens", limits["default_output"])
        from .capacity import output_count
        output_count(requested,0,None,maximum=limits["output_limit"])
        options = chat_options(body)
        conversations = chat_request(body)
        if w.kind == 'gguf' and any(not isinstance(m.get('content'),str) for c in conversations.values() for m in c):
            raise ValueError('GGUF execution is text-only; image/video/MTP components are not loaded')
        if body.get('force_template'):
            # The plain turn template is certain to be used, so what it needs is checked here, before the model.
            plain_template(body)
            for v, messages in conversations.items():
                plain_turns(messages, FIELDS[v], 'force_template is set')
        # Checked before the model is touched; None without steer, and always None when the edited model does not run.
        steer = _steer_request(w, body, ["current"]) if 'current' in conversations else None
        # A request with neither new field answers exactly as before; one with them also says what each model was given.
        extended = body.get('variants') is not None or body.get('base_messages') is not None
        stream = stream_id(body)
        halt = stops.get(stream)
        w = runnable()
        # One runner for the prompts, both variants and decoding: its tokenizer still works if its worker is lost.
        runner = w.runner
        def prompt_of(messages, field):
            forced = bool(body.get("force_template"))
            text = None if forced else runner.chat_prompt(messages, **options)
            templated = text is not None
            stop = ""
            if text is None:
                # Known only now, after the model's own template was asked: a clear refusal, not a TypeError.
                tpl = plain_template(body)
                plain_turns(messages, field, 'force_template is set' if forced else 'this model has no chat template')
                text = (tpl["system"] + tpl["sep"]) if tpl["system"] else ""
                for m in messages:
                    text += tpl["user" if m["role"] == "user" else "assistant"] + m["content"] + tpl["sep"]
                text += tpl["assistant"].rstrip(" ") if w.tok.to_dict()["type"] == "bpe" else tpl["assistant"]
                stop = tpl["user"].strip()
            ids = w.tok.encode(text, special=False) if templated else w.encode(text)
            return dict(text=text, templated=templated, stop=stop, ids=ids)
        # Every prompt is rendered and fitted to the context before either reply is generated, so a conversation
        # that no longer fits fails the request before the other model has run. Equal conversations render once.
        prompts = {}
        for v, messages in conversations.items():
            same = next((p for u, p in prompts.items() if conversations[u] == messages), None)
            prompts[v] = same if same is not None else prompt_of(messages, FIELDS[v])
        distinct = len({id(p) for p in prompts.values()}) > 1
        fitted = {}
        def fit(load=None):
            # Sized for the load settings it runs with: called here and again by the operation under the runtime
            # client's lock once they are pinned. -> the tokens of the longest sequence requested.
            limits = generation_limits(w, load)
            requested = body.get("max_new_tokens", limits["default_output"])
            sizes = {}
            for v, p in prompts.items():
                if id(p) in sizes:
                    continue
                try:
                    sizes[id(p)] = output_count(requested,len(p['ids']),limits['context_limit'],maximum=limits['output_limit'])
                except ValueError as exc:
                    if not distinct:
                        raise
                    raise ValueError(f"The {NAMES[v]} model's conversation: {exc}") from None
            for p in prompts.values():
                p['n'] = sizes[id(p)]
            fitted.update(limits=limits, requested=requested)
            return max(len(p['ids'])+p['n'] for p in prompts.values())
        first = prompts[next(iter(prompts))]     # the edited model's when it runs, else the base model's
        gen = lambda p: dict(requested_output=fitted['requested'],effective_output=p['n'],prompt_tokens=len(p['ids']),context=fitted['limits']['context_limit'])
        def described(out):
            out['generation'] = gen(first)
            if extended:
                out.update(prompts={v: p['text'] for v, p in prompts.items()}, generations={v: gen(p) for v, p in prompts.items()})
            return out
        fit()
        out = described({"prompt": first['text'], "chat_template": first['templated']})
        # A model that was not asked for has no status at all; one asked for is 'not_run' until it runs.
        out.update(base='',current='',variants={v:'not_run' for v in ('base','current') if v in prompts})
        wire, steering = _generation_steer(w, steer, stream, halt)
        if steering is not None:
            out['steering'] = steering
        client=getattr(w,'_runtime_client',None)
        def renderer(p):
            def render(new):
                reply=w.tok.decode(new)
                if p['templated'] and getattr(runner,'_tok',None):reply=runner._tok.decode(new,skip_special_tokens=True)
                if p['stop'] and p['stop'] in reply:reply=reply.split(p['stop'])[0]
                return reply.strip('\n')
            return render
        # The variants run one after the other: the scope covers the longest sequence of those requested.
        tokens = max(len(p['ids'])+p['n'] for p in prompts.values())
        scope_started=False
        try:
            with w.lock if client else nullcontext(), client.operation(w,'chat',tokens=tokens,size=fit) if client else nullcontext():
                scope_started=True
                described(out)
                # The edited model first: its reply is the one being worked on, and it is what an interruption keeps.
                for v, p in prompts.items():
                    if halt and halt.is_set():
                        out['stopped']=True;break   # stopped by the person: this and later variants are not run
                    failure=None;ids,n,render=p['ids'],p['n'],renderer(p)
                    live=TextStream(sess.publish,stream,v,render,stop=halt) if stream else None
                    if live:live.begin()
                    try:
                        try:
                            pushed=wire if v=='current' else None   # the base model is never steered
                            new=runner.generate(ids,v,None,max_new_tokens=n,temperature=float(body.get("temperature",.7)),top_k=int(body.get("top_k",40)),seed=body.get("seed",0),**({'on_token':live} if live else {}),**({'steer':pushed} if pushed is not None else {}))
                            out['variants'][v]='complete'
                        except GenerationStopped:
                            new=None                # handled below like a runner that returned early
                        except RuntimeRefusal as exc:
                            new=exc.tokens;failure=exc;out['event']=exc.record;out['interrupted']=True;out['variants'][v]='interrupted'
                        if live and halt.is_set() and not failure:  # also a runner that returned early
                            new=list(live.ids);out['variants'][v]='stopped';out['stopped']=True;out['tokens']={v:len(new)}
                        if live:live.close()        # the reply is rendered once more below, by this thread alone
                        out[v]=render(new)
                    except Exception:
                        if live:live.end('failed')
                        raise
                    if live:live.end(out['variants'][v],new)
                    if failure:raise failure
                    if out['variants'][v]=='stopped':break
        except RuntimeRefusal as exc:
            if not scope_started:raise
            out['event']=exc.record;out['interrupted']=True
        return out

    @app.post("/api/trace")
    def trace(body: dict = Body(...)):
        return an.trace(runnable(), body["prompt"], body.get("variant", "current"), body.get("target") or None, body.get("versus") or None)

    @app.post("/api/patching")
    def patching(body: dict = Body(...)):
        return an.patching(runnable(), body["clean"], body["corrupt"], body.get("target") or None, body.get("versus") or None,
                           body.get("variant", "current"))

    @app.post("/api/ablation")
    def ablation(body: dict = Body(...)):
        suite_name = body.get("suite") or sess.task["suite"]
        variant = body.get("variant", "current")
        suite = testsuite.load_suite(ws(), suite_name)
        suite_key = hashlib.sha256(json.dumps(suite, sort_keys=True).encode()).hexdigest()
        def calculate(w):
            runnable(w)
            pairs = []
            for c in suite["cases"]:
                if c["type"] == "next_token":
                    exp = c["expect"] if isinstance(c["expect"], str) else c["expect"][0]
                    pairs.append((w.encode(c["prompt"]), w.tok.encode_token(exp)[0]))
            if not pairs:
                raise ValueError("the suite has no next_token cases to measure loss on")
            return an.ablation_scan(w, pairs[:64], variant)
        return inspect_read("ablation", (suite_name, suite_key, variant), calculate,
                            body.get("workspace_id", ""), body.get("version"))

    # ------------------------------------------------------------------ steering and probes (wsp.steering, wsp.probes)
    # Steering pushes neurons and heads while the model runs; probes read activations. Neither
    # writes a weight or records a patch. Every input is bounded here, before any model is loaded
    # (only the workspace's structure and tokenizer are used), and again in the job.
    from . import probes as pr, steering as st
    from .worker_jobs import uses_worker
    PROBE_CHARS = 20000

    def _steer_events(w, stream=None):
        # A Run or Chat request's events carry its stream id, so the browser shows them on that request's card.
        tag = {"stream": stream} if stream else {}
        return lambda e: sess.publish({"type": "steering", "workspace": w.name, **tag, **e})

    def _probe_ids(w, text, limit, what="prompt"):
        if not isinstance(text, str) or not text:
            raise ValueError(f"{what} must be nonempty text")
        if len(text) > PROBE_CHARS:
            raise ValueError(f"{what} is longer than {PROBE_CHARS} characters")
        ids = w.encode(text)
        if len(ids) > limit:
            raise ValueError(f"{what} has {len(ids)} tokens; at most {limit}")
        return ids

    def _probe_chat(w, body):
        """Before the model: is the text to be read as a chat message -> the template options, or None."""
        chat = body.get("chat", False)
        if not isinstance(chat, bool):
            raise ValueError("chat must be true or false")
        if not chat:
            return None
        options = chat_options(body)
        ok, reason = pr.chat_template(w)
        if not ok:
            raise ValueError(f"'as a chat message' needs the model's own chat template: {reason}")
        return options

    def _chat_ids(w, text, limit, what, options):
        """After runnable(): the text as one user turn, rendered as /api/chat renders it (generation prompt
        included) and tokenized as Chat does -> (ids, template mask or None)."""
        rendered = pr.render_chat(w, text, options)
        encode = pr.chat_encode(w)
        ids = encode(rendered)
        if len(ids) > limit:
            raise ValueError(f"{what} as a chat message has {len(ids)} tokens; at most {limit}")
        return ids, pr.template_mask(encode, rendered, text, ids)

    def _chat_out(mask):
        return {"chat": {"template": mask}}

    def _steer_request(w, body, variants):
        """The steer fields of a request, checked without the model -> (spec, p50, recalibrate) or None.
        Without steering (no "steer", or an empty one) "p50" and "recalibrate" are ignored: they only shape a push."""
        spec = body.get("steer")
        if spec is None or spec == []:
            return None
        recalibrate = body.get("recalibrate", False)
        if not isinstance(recalibrate, bool):
            raise ValueError("recalibrate must be true or false")
        p50 = st.given_p50(w, body.get("p50"))
        if "current" not in variants:
            raise st.SteeringError("the base model is never steered; steer the current variant")
        st.normalize(w, spec)
        return spec, p50, recalibrate

    def _probe_steer(w, req):
        """After runnable(): prepare the spec (calibrating if needed) and, when the model runs in
        this process, read the column norms here so a slow read publishes progress."""
        if req is None:
            return None, None
        spec, p50, recalibrate = req
        events = _steer_events(w)
        wire, info = st.prepare(w, spec, calibrate=True, on_event=events, p50=p50, recalibrate=recalibrate)
        if not uses_worker(w) and wire.get("p50"):
            st.resolve(w, wire, on_event=events)
        return wire, info["calibration"]

    def _generation_steer(w, req, stream=None, halt=None):
        """Run and Chat, after runnable(), before generating: prepare the steering (calibrating on the base
        model when needed), read the column norms, and say what will be applied -> (wire, report) or (None, None).

        Neither the calibration pass nor the column reads can be cut short; a stop that arrives during them
        is seen by the caller before the first variant starts, so nothing is generated, and the report says
        resolved: false (a stop seen after the reads as well as before them)."""
        if req is None:
            return None, None
        spec, p50, recalibrate = req
        events = _steer_events(w, stream)
        wire, info = st.prepare(w, spec, calibrate=True, on_event=events, p50=p50, recalibrate=recalibrate)
        report = {"variant": "current", "entries": info["entries"], "calibration": info["calibration"]}
        if halt is not None and halt.is_set():
            return wire, {**report, "resolved": False}
        # In this process the reads also fill the column-norm cache generate uses; a runtime worker reads them again.
        d = st.describe(w, wire, info, "current", on_event=events)
        if halt is not None and halt.is_set():                       # stopped while the column norms were read
            return wire, {**report, "resolved": False}
        return wire, {**report, "resolved": True, "entries": d["entries"], "calibration": d["calibration"],
                      **{k: d[k] for k in ("layers", "neurons", "heads", "mlps", "directions", "reads")}}

    def _token_strs(w, ids):
        return [show(w.tok.token_str(int(i))) for i in ids]

    @app.post("/api/steer/resolve")
    def steer_resolve(body: dict = Body(...)):
        w = ws()
        variant = body.get("variant", "current")
        if variant != "current":
            raise st.SteeringError("the base model is never steered; steer the current variant")
        calibrate = body.get("calibrate", True)
        if not isinstance(calibrate, bool):
            raise ValueError("calibrate must be true or false")
        recalibrate = body.get("recalibrate", False)
        if not isinstance(recalibrate, bool):
            raise ValueError("recalibrate must be true or false")
        p50 = st.given_p50(w, body.get("p50"))
        wire, _ = st.normalize(w, body.get("steer"))
        measure = [l for l in st.needs_calibration(wire) if str(l) not in p50]
        if calibrate and measure and (recalibrate or st.calibration(w, run=False)[0] is None):
            runnable(w)
        wire, info = st.prepare(w, body.get("steer"), calibrate=calibrate, on_event=_steer_events(w), p50=p50,
                                recalibrate=recalibrate)
        return st.describe(w, wire, info, variant, on_event=_steer_events(w))

    @app.get("/api/steer/groups")
    def steer_groups():
        """The saved selections as steering targets: how many neurons an entry group:<name> pushes, what it
        leaves out (heads, tensor slices, members that no longer resolve), and the bound per request. From the
        selections alone: no weight is read and the model is not run."""
        w = ws()
        out = []
        for name in sorted(w.groups):
            try:
                pairs, skipped = st.group_neurons(w, name)
            except (KeyError, ValueError, st.SteeringError) as e:
                out.append({"name": name, "addr": f"group:{name}", "neurons": 0, "error": str(e)})
                continue
            out.append({"name": name, "addr": f"group:{name}", "neurons": len(pairs),
                        "layers": sorted({l for l, _ in pairs}), "skipped": skipped or {}})
        return {"groups": out, "max_neurons": st.MAX_NEURONS}

    @app.post("/api/probe/reads")
    def probe_reads(body: dict = Body(...)):
        w = ws()
        ids = _probe_ids(w, body.get("prompt"), pr.MAX_TOKENS)
        variants = body.get("variants") or ["current"]
        if not isinstance(variants, list) or not all(v in pr.VARIANTS for v in variants):
            raise ValueError("variants must list 'base' and/or 'current'")
        addrs = body.get("neurons")
        if not isinstance(addrs, list) or not addrs or len(addrs) > pr.MAX_NEURONS:
            raise ValueError(f"neurons must list 1..{pr.MAX_NEURONS} neuron addresses")
        pairs = []
        for a in addrs:
            if not isinstance(a, str) or len(a) > st.MAX_ADDR:
                raise ValueError("each neuron must be an address such as L3N17")
            kind, *rest = st.target(w, a)
            if kind != "neurons":
                raise ValueError(f"{a} is not a neuron")
            pairs += [list(p) for p in rest[0]]
            if len(pairs) > pr.MAX_NEURONS:
                raise ValueError(f"at most {pr.MAX_NEURONS} neurons can be read at once")
        ids, pairs, variants = pr.check_reads(w, ids, pairs, variants)
        req = _steer_request(w, body, variants)
        chat = _probe_chat(w, body)
        runnable(w)
        if chat is not None:
            ids, mask = _chat_ids(w, body["prompt"], pr.MAX_TOKENS, "the prompt", chat)
        wire, calibration = _probe_steer(w, req)
        out = pr.reads(w, ids, pairs, variants, wire)
        return {**out, "ids": ids, "token_strs": _token_strs(w, ids), "addrs": [f"L{l}N{n}" for l, n in out["neurons"]],
                "calibration": calibration, **(_chat_out(mask) if chat is not None else {})}

    @app.post("/api/probe/fires")
    def probe_fires(body: dict = Body(...)):
        w = ws()
        ids = _probe_ids(w, body.get("prompt"), pr.MAX_TOKENS)
        chat = _probe_chat(w, body)
        # As a chat message the position counts tokens of the rendered chat: before the model it is checked to be
        # an integer within the token bound, and against the rendering's length once that is known.
        position = body.get("position", -1)
        if chat is not None:
            pr.check_position(position, pr.MAX_TOKENS)
            position = -1
        args = pr.check_fires(w, ids, position, body.get("k", 20), body.get("sign", "both"), body.get("variant", "current"))
        runnable(w)
        if chat is not None:
            ids, mask = _chat_ids(w, body["prompt"], pr.MAX_TOKENS, "the prompt", chat)
            args = pr.check_fires(w, ids, body.get("position", -1), *args[2:])
        out = pr.fires(w, *args)
        return {**out, "token_str": show(w.tok.token_str(int(out["token"]))), "token_strs": _token_strs(w, ids),
                **(_chat_out(mask) if chat is not None else {})}

    @app.post("/api/probe/contrast")
    def probe_contrast(body: dict = Body(...)):
        w = ws()
        sets = []
        for label in ("a", "b"):
            texts = body.get(label)
            if not isinstance(texts, list) or not 1 <= len(texts) <= pr.MAX_TEXTS:
                raise ValueError(f"{label} must list 1..{pr.MAX_TEXTS} texts")
            sets.append([_probe_ids(w, t, pr.MAX_TOKENS, f"a text of set {label.upper()}") for t in texts])
        sets, k, variant = pr.check_contrast(w, sets[0], sets[1], body.get("k", 20), body.get("variant", "current"))
        chat = _probe_chat(w, body)
        runnable(w)
        if chat is not None:
            sets = [[_chat_ids(w, t, pr.MAX_TOKENS, f"a text of set {label.upper()}", chat)[0] for t in body[label]]
                    for label in ("a", "b")]
        return {**pr.contrast(w, sets[0], sets[1], k, variant), **({"chat": {"rendered": True}} if chat is not None else {})}

    @app.post("/api/probe/next")
    def probe_next(body: dict = Body(...)):
        w = ws()
        ids = _probe_ids(w, body.get("prompt"), pr.MAX_NEXT_TOKENS)
        ids, variant, top = pr.check_next(w, ids, body.get("variant", "current"), body.get("top", 10))
        req = _steer_request(w, body, [variant])
        chat = _probe_chat(w, body)
        runnable(w)
        if chat is not None:
            ids, mask = _chat_ids(w, body["prompt"], pr.MAX_NEXT_TOKENS, "the prompt", chat)
        wire, calibration = _probe_steer(w, req)
        out = pr.next_token(w, ids, variant, wire, top)
        for key in ("top_unsteered", "top_steered", "rose", "fell"):
            for row in out[key]:
                row["token"] = show(w.tok.token_str(row["id"]))
        return {**out, "calibration": calibration, **(_chat_out(mask) if chat is not None else {})}

    @app.get("/api/probe/chat")
    def probe_chat():
        """Whether probes can read a text as a chat message: from the model's files, without loading it."""
        ok, reason = pr.chat_template(ws())
        return {"available": ok, "reason": reason}

    # ------------------------------------------------------------------ behaviour directions (wsp.directions)
    # A measurement runs the good and bad prompts through the model and saves the two mean last-token residual
    # streams per layer; an ablation is a weight script (POST /api/directions/script, applied through /api/edit).
    # Everything that needs no model is checked first and answers 400 with the reason.
    from . import directions as dr_
    DIRECTION_EVENTS_EVERY = 0.25
    DIRECTION_ENDS = ("measured", "scored", "detected", "fetched", "stopped", "failed")

    def track(key, event):
        """Keep a running direction request's event under ``key`` (a copy); a terminal state removes the entry.
        Called before the event is published, so a page that refetches on an event sees the listing it implies."""
        with direction_requests_lock:
            if event["state"] in DIRECTION_ENDS:
                direction_requests.pop(key, None)
            else:
                direction_requests[key] = dict(event)

    def untrack(key):
        """The request has ended, whatever the way (an answer, a refusal, a stop, an exception)."""
        with direction_requests_lock:
            direction_requests.pop(key, None)

    def running_requests(w):
        """The running direction requests of workspace ``w``, in the order they started, copied under the lock."""
        with direction_requests_lock:
            return [dict(e) for e in direction_requests.values() if e.get("workspace") == w.name]

    def _direction_missing(e):
        return JSONResponse({"error": str(e)}, status_code=404)

    from . import directions_prefix as dp_
    from . import directions_scores as ds_

    from . import directions_search as dsr_
    from . import directions_datasets as dd_
    from . import directions_fields as df_
    from . import directions_sampler as dsm_
    from . import directions_evaluate as de_

    @app.get("/api/directions")
    def directions_list():
        w = ws()
        return {"directions": dr_.entries(w), "limits": {**dr_.LIMITS, "scores": ds_.LIMITS, "search": dsr_.LIMITS,
                                                         "datasets": dd_.LIMITS},
                # the methods' parameter fields for this model (null when its layers are not known)
                "parameters": df_.parameter_fields(w),
                # the last event of every running measure, scores, prefix or dataset-fetch request of this workspace
                "requests": running_requests(w)}

    # ------------------------------------------------------------------ the automatic search (wsp.directions_search)
    # Declared before GET /api/directions/{name}, which would otherwise answer /api/directions/searches. A search runs
    # in a background thread, one per workspace, claimed under the session lock; its progress, each trial and its end
    # are direction_search events. Nothing restarts by itself. Only the trial-apply route commits, through
    # Session.apply.
    def _search_refusal(e):
        return JSONResponse({"error": str(e)}, status_code=404 if isinstance(e, dsr_.SearchMissing) else 409)

    def _search_job(w, name=None):
        """This session's search job when it is for workspace ``w`` (and search ``name``), else None."""
        job = sess.search_job
        if not job or job.get("workspace") != w.name or (name is not None and job.get("search") != name):
            return None
        return dict(job)

    def _refuse_busy(w):
        """R3 when a search runs: this session's job, or the searches lock held in this process or another."""
        job = sess.search_job
        if job and job.get("running"):
            raise dsr_.SearchBusy(f"a search is already running on this workspace in this IDE (search "
                                  f"'{job['search']}'); stop it first")
        with dsr_.search_lock(w):           # taken and dropped at once: SearchBusy names who holds it
            pass

    def _start_search(w, name, stream, n_additional_trials=None, accept_changed_libraries=False):
        """Check what needs no model, make the model runnable, claim the job and start its thread -> the job."""
        import threading
        counts = dsr_.check_run(w, name, n_additional_trials=n_additional_trials,
                                accept_changed_libraries=accept_changed_libraries)
        _refuse_busy(w)
        runnable(w)
        with sess.lock:
            _refuse_busy(w)
            job = {"running": True, "workspace": w.name, "search": name, "state": "preparing", "trial": None,
                   "trials_done": counts["final"], "n_trials": counts["n_trials"], "phase": None, "done": 0,
                   "total": None, "started": time.time(), "reason": None, "error": None, "event": None, "stop": None,
                   "stream": stream}
            sess.search_job = job
        tag = {"stream": stream} if stream else {}
        started = time.monotonic()
        said = [None]
        mirrored = ("trial", "trials_done", "n_trials", "phase", "done", "total", "reason", "error", "event")

        def publish(state, **data):
            job.update(state=state, **{k: data[k] for k in mirrored if k in data})
            sess.publish({"type": "direction_search", "workspace": w.name, "search": name, "state": state,
                          "seconds": round(time.monotonic() - started, 3), **data, **tag})

        def progress(index, phase, done, total):
            now = time.monotonic()
            if said[0] is not None and now - said[0] < DIRECTION_EVENTS_EVERY:
                return
            said[0] = now
            if index is None:
                publish("preparing", phase=phase, done=done, total=total)
            else:
                publish("running", trial=index, trials_done=job["trials_done"], n_trials=job["n_trials"], phase=phase,
                        done=done, total=total)

        def on_trial(kind, **data):
            if data.get("trial_state") in dsr_.FINAL:
                job["trials_done"] += 1
            publish("trial", **data)

        def stopped():
            if lifecycle.stopping.is_set():
                return "the server is stopping"
            if sess.ws is not w or job["stop"] == "closed":      # Session.unload sets "closed"
                return "the workspace was closed"
            if job["stop"] == "now":
                return "stopped by the person"
            return None

        def work():
            end, data = "failed", {}
            try:
                out = dsr_.run(w, name, n_additional_trials=n_additional_trials,
                               accept_changed_libraries=accept_changed_libraries, scope=sess.task.get("scope"),
                               progress=progress, stopped=stopped, after_trial=lambda: job["stop"] == "after_trial",
                               event=on_trial)
                end = out["state"]
                data = {"reason": out["reason"], "trials": out["trials"], "front": out["front"]}
                if end == "failed":
                    data.update(error=out["reason"], event=out.get("event"))
            except RuntimeRefusal as e:                      # admission or a lost worker before the first trial
                data = {"error": str(e), "event": e.record}
            except dr_.DirectionError as e:                  # a refusal before the first trial (SearchError included)
                data = {"error": str(e)}
            except Exception as e:
                data = {"error": f"{type(e).__name__}: {e}"}
            finally:
                job["running"] = False
                publish(end, **data)

        publish("preparing", phase=None, done=0, total=None)
        threading.Thread(target=work, daemon=True, name="direction-search").start()
        return dict(job)

    @app.get("/api/directions/searches")
    def directions_searches():
        w = ws()
        return {"searches": dsr_.entries(w), "job": _search_job(w), "limits": dsr_.LIMITS,
                # the installed Optuna (null when missing) and the version the draws were checked with; no import
                "sampler": dsm_.versions()}

    @app.post("/api/directions/searches")
    def directions_search_create(body: Any = Body(...)):
        """Create a search (checked without a model first) and, unless ``start`` is false, start running it."""
        w = ws()
        spec = dsr_.check_search(w, body)
        stream = stream_id(body)
        try:
            if spec["start"]:
                _refuse_busy(w)
            s = dsr_.create(w, spec)
            job = _start_search(w, s["name"], stream) if spec["start"] else None
        except (dsr_.SearchMissing, dsr_.SearchBusy) as e:
            return _search_refusal(e)
        return dsr_.describe(w, s["name"], job=job)

    @app.get("/api/directions/searches/{name}")
    def directions_search_get(name: str, trials: str | None = None):
        w = ws()
        try:
            return dsr_.describe(w, name, full=trials == "full", job=_search_job(w, name))
        except dsr_.SearchMissing as e:
            return _search_refusal(e)

    @app.get("/api/directions/searches/{name}/trials/{index}")
    def directions_search_trial(name: str, index: int):
        try:
            return dsr_.trial(ws(), name, index)
        except dsr_.SearchMissing as e:
            return _search_refusal(e)

    @app.get("/api/directions/searches/{name}/front")
    def directions_search_front(name: str):
        try:
            return dsr_.front_entries(ws(), name)
        except dsr_.SearchMissing as e:
            return _search_refusal(e)

    @app.post("/api/directions/searches/{name}/run")
    def directions_search_run(name: str, body: Any = Body(default={})):
        """Run more of a search's trials (resume, or n_additional_trials) in the background."""
        w = ws()
        args = dsr_.run_body(body)
        stream = stream_id(body)
        try:
            job = _start_search(w, name, stream, **args)
        except (dsr_.SearchMissing, dsr_.SearchBusy) as e:
            return _search_refusal(e)
        return dsr_.describe(w, name, job=job)

    @app.post("/api/directions/searches/{name}/stop")
    def directions_search_stop(name: str, body: Any = Body(default={})):
        """Stop the running search now (at its next stop point) or after the trial in progress."""
        args = dsr_.stop_body(body)
        w = ws()
        job = sess.search_job
        found = bool(job and job.get("running") and job.get("workspace") == w.name and job.get("search") == name)
        if found and (args["when"] == "now" or job["stop"] is None):
            job["stop"] = args["when"]
        return {"ok": True, "found": found}

    @app.delete("/api/directions/searches/{name}")
    def directions_search_delete(name: str):
        w = ws()
        job = _search_job(w, name)
        try:
            if job and job.get("running"):
                raise dsr_.SearchBusy(f"search '{name}' is running; stop it first")
            dsr_.delete(w, name)
        except (dsr_.SearchMissing, dsr_.SearchBusy) as e:
            return _search_refusal(e)
        return {"ok": True, "deleted": name}

    @app.post("/api/directions/searches/{name}/trials/{index}/apply")
    def directions_search_apply(name: str, index: int, body: Any = Body(default={})):
        """Commit a complete trial through Session.apply (a dry run with dry_run); refused when the edit its script
        makes now differs from the one it scored, unless accept_different_edit."""
        args = dsr_.apply_body(body)
        ws()
        try:
            r = dsr_.apply_trial(sess, name, index, **args)
        except (dsr_.SearchMissing, dsr_.SearchBusy) as e:
            return _search_refusal(e)
        if r.get("ok") and r.get("committed"):
            r["status"] = status()
        return r

    @app.get("/api/directions/{name}")
    def directions_get(name: str, orthogonalize_direction: bool = True):
        try:
            meta, good, bad = dr_.load(ws(), name)
        except dr_.DirectionMissing as e:
            return _direction_missing(e)
        return {**meta, "module_io": meta.get("module_io"), "stats": dr_.stats(good, bad, orthogonalize_direction),
                "means_sha256": dsr_.measurement_identity(ws(), name)["means_sha256"]}

    @app.get("/api/directions/{name}/prompts")
    def directions_prompts(name: str):
        try:
            return dr_.prompts(ws(), name)
        except dr_.DirectionMissing as e:
            return _direction_missing(e)

    @app.delete("/api/directions/{name}")
    def directions_delete(name: str):
        try:
            dr_.delete(ws(), name)
        except dr_.DirectionMissing as e:
            return _direction_missing(e)
        return {"ok": True, "deleted": name}

    @app.post("/api/directions/measure")
    @stoppable
    def directions_measure(body: dict = Body(...)):
        """Measure and save a direction. With a ``stream``, POST /api/generation/stop stops it between chunks
        (nothing is saved). Progress on /api/events: state measuring, then measured, stopped or failed."""
        w = ws()
        args = dr_.measure_body(body)
        stream = stream_id(body)
        halt = stops.get(stream)
        checked = dr_.check_measure(w, **args)  # before the model: bounds, names, chat availability, token counts
        runnable(w)
        tag = {"stream": stream} if stream else {}
        started = time.monotonic()
        said = [None]
        key = uuid.uuid4().hex                  # this request's entry in direction_requests

        def event(state, **data):
            payload = {"type": "directions", "workspace": w.name, "name": args["name"], "state": state,
                       "seconds": round(time.monotonic() - started, 3), **data, **tag}
            track(key, payload)
            sess.publish(payload)

        def progress(done, total):
            now = time.monotonic()
            if said[0] is None or now - said[0] >= DIRECTION_EVENTS_EVERY:
                said[0] = now
                event("measuring", done=done, total=total)
        total = len(checked["good"]) + len(checked["bad"])      # resolved prompts (a specification is one object)
        try:
            progress(0, total)
            try:
                out = dr_.measure(w, **args, progress=progress, stopped=lambda: halt is not None and halt.is_set())
            except Exception as e:
                event("failed", error=f"{type(e).__name__}: {e}")
                raise
            if out.get("stopped"):
                event("stopped", done=out["done"], total=out["total"])
                return {"stopped": True, "done": out["done"], "total": out["total"]}
            event("measured", done=total, total=total)
            return out
        finally:
            untrack(key)

    def _direction_run(kind, states, run, w, stream):
        """Run a score or prefix request with its events (``kind``; states running, done, stopped, failed) at most
        every DIRECTION_EVENTS_EVERY while it runs."""
        tag = {"stream": stream} if stream else {}
        started = time.monotonic()
        said = [None]
        last = {"phase": None, "done": 0, "total": None}
        key = uuid.uuid4().hex                  # this request's entry in direction_requests

        def event(state, **data):
            payload = {"type": kind, "workspace": w.name, "state": state,
                       "seconds": round(time.monotonic() - started, 3), **data, **tag}
            track(key, payload)
            sess.publish(payload)

        def progress(phase, done, total):
            last.update(phase=phase, done=done, total=total)
            now = time.monotonic()
            if said[0] is None or now - said[0] >= DIRECTION_EVENTS_EVERY:
                said[0] = now
                event(states[0], phase=phase, done=done, total=total)
        try:
            event(states[0], phase=None, done=0, total=None)
            try:
                out = run(progress)
            except Exception as e:
                event("failed", error=f"{type(e).__name__}: {e}", phase=last["phase"], done=last["done"],
                      total=last["total"])
                raise
            if out.get("stopped"):
                event("stopped", done=out["done"], total=out["total"])
                return {"stopped": True, "done": out["done"], "total": out["total"]}
            event(states[1], done=out.get("done"), total=out.get("total"))
            return out
        finally:
            untrack(key)

    @app.post("/api/directions/scores")
    @stoppable
    def directions_scores(body: dict = Body(...)):
        """Refusal rate, KL divergence and response length (Heretic's scorers) of the base weights, the current
        weights or an uncommitted candidate script; nothing is committed. With a ``stream``, POST
        /api/generation/stop stops it between prompts. Events on /api/events: type direction_scores, state scoring,
        then scored, stopped or failed. ``steer`` (and ``p50``) steer the current or candidate state as Run does; the
        KL reference stays the unsteered base."""
        w = ws()
        args = ds_.scores_body(body)
        stream = stream_id(body)
        halt = stops.get(stream)
        spec = ds_.check_scores(w, args)        # before the model: bounds, plugins, rendering, tokens, disk, steering
        runnable(w)
        # Calibration (a neuron resid entry) and the column reads run here, outside the scoring operation.
        steering = ds_.prepare_steering(w, spec) if spec["steer"] else None
        stopped = lambda: halt is not None and halt.is_set()  # noqa: E731
        return _direction_run("direction_scores", ("scoring", "scored"),
                              lambda progress: ds_.score(w, args, scope=sess.task.get("scope"), progress=progress,
                                                         stopped=stopped,
                                                         **({} if steering is None else {"steering": steering})),
                              w, stream)

    @app.post("/api/directions/prefix")
    @stoppable
    def directions_prefix(body: dict = Body(...)):
        """Heretic's response-prefix detection (thinking blocks included); the result is returned, not stored: pass
        ``response_prefix`` as ``prefix`` to measure and scores. Events: type direction_prefix, state detecting, then
        detected, stopped or failed."""
        w = ws()
        stream = stream_id(body)
        halt = stops.get(stream)
        dp_.check_prefix(w, body)               # before the model
        runnable(w)
        stopped = lambda: halt is not None and halt.is_set()  # noqa: E731
        return _direction_run("direction_prefix", ("detecting", "detected"),
                              lambda progress: dp_.detect(w, body, progress=progress, stopped=stopped), w, stream)

    @app.post("/api/directions/evaluate")
    @stoppable
    def directions_evaluate(body: dict = Body(...)):
        """Heretic's evaluate_model: a modified checkpoint on disk (or a PEFT LoRA adapter, or another workspace; the
        answer's "input" says which), read (never written) and scored as an uncommitted change on top of the loaded
        model, against the base; "check": true compares the weights only."""
        w = ws()
        args = de_.evaluate_body(body)
        stream = stream_id(body)
        halt = stops.get(stream)
        de_.check_evaluate(w, args)            # before any weight read and before the model
        if not args["check"]:
            runnable(w)
        stopped = lambda: halt is not None and halt.is_set()  # noqa: E731
        return _direction_run("direction_scores", ("scoring", "scored"),
                              lambda progress: de_.evaluate(w, args, progress=progress, stopped=stopped), w, stream)

    @app.post("/api/directions/reproduce")
    def directions_reproduce(body: Any = Body(default={})):
        """Run a Heretic reproduce.json natively (wsp.directions_heretic; docs/DIRECTIONS.md, "Running a Heretic
        reproduce.json"): the file's prompt specifications measured (offline, from the cache) with its settings, then
        its modifier with its parameters through Session.apply; a dry run by default. "check": true answers the plan
        and runs nothing. What cannot be honoured is refused before anything runs (400 RecordError, every reason); 409
        while a search runs."""
        from . import directions_heretic as dh_
        from . import directions_record as rec_
        w = ws()
        args = dh_.heretic_reproduce_body(body)
        plan = dh_.check_heretic_reproduce(w, args["data"], measurement=args["measurement"])   # before the model
        if args["check"]:
            return {"heretic": plan, "ran": None}
        if plan["measurement"]["state"] == "new":
            runnable(w)
        try:
            r = dh_.heretic_reproduce(sess, args["data"], measurement=args["measurement"], dry_run=args["dry_run"],
                                      message=args["message"], enforce_scope=args["enforce_scope"])
        except rec_.RecordBusy as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        if r.get("ok") and r.get("committed"):
            r["status"] = status()
        return r

    @app.post("/api/directions/datasets")
    @stoppable
    def directions_datasets(body: dict = Body(...)):
        """A prompt set (texts, prompt objects, Heretic dataset specifications) resolved from local files and the
        Hugging Face cache: its count, hash, record and first prompts, or, when something is not cached, what a fetch
        would get. Only ``"download": true`` fetches (the one network path of datasets); with a ``stream``, POST
        /api/generation/stop stops it between files. Events (fetches only): type direction_datasets, state fetching,
        then fetched, stopped or failed. No model is needed."""
        w = ws()
        try:
            args = dd_.request_body(body)
        except dd_.DatasetError as e:
            raise dr_.DirectionError(str(e)) from None
        stream = stream_id(body)
        halt = stops.get(stream)
        stopped = lambda: halt is not None and halt.is_set()  # noqa: E731
        if not args["download"]:
            try:
                return dd_.run_request(w, body, stopped=stopped)
            except dd_.DatasetError as e:
                raise dr_.DirectionError(str(e)) from None
        tag = {"stream": stream} if stream else {}
        started = time.monotonic()
        said = [None]
        key = uuid.uuid4().hex                  # this request's entry in direction_requests (fetches only)

        def event(state, **data):
            payload = {"type": "direction_datasets", "workspace": w.name, "state": state,
                       "seconds": round(time.monotonic() - started, 3), **data, **tag}
            track(key, payload)
            sess.publish(payload)

        def progress(e):
            now = time.monotonic()
            if said[0] is None or now - said[0] >= DIRECTION_EVENTS_EVERY:
                said[0] = now
                event("fetching", **{k: e[k] for k in ("phase", "file", "done_bytes", "total_bytes")})
        try:
            try:
                out = dd_.run_request(w, body, progress=progress, stopped=stopped)
            except dd_.DatasetError as e:
                event("failed", error=f"DirectionError: {e}")
                raise dr_.DirectionError(str(e)) from None
            except Exception as e:
                event("failed", error=f"{type(e).__name__}: {e}")
                raise
            if out.get("stopped"):
                event("stopped", done_bytes=out["done_bytes"], total_bytes=out["total_bytes"], fetched=out["fetched"])
            else:
                event("fetched", fetched=out["fetched"], fetched_bytes=out["fetched_bytes"])
            return out
        finally:
            untrack(key)

    @app.post("/api/directions/script")
    def directions_script(body: dict = Body(...)):
        """The weight script for ablation parameters and what it will do (no weight read, no model run); apply
        the script through /api/edit (dry run or commit). ``"method": "ara"`` gives ablate_ara's script and plan;
        "ablation" (the default) ablate_direction's."""
        w = ws()
        return {"script": dr_.script_for(body), "plan": dr_.plan(w, body)}

    # ------------------------------------------------------------------ edits and patches
    @app.post("/api/edit")
    def edit(body: dict = Body(...)):
        request = body.get("request")
        r = sess.apply(body["script"], body.get("message", ""), body.get("author", "human"), bool(body.get("dry_run")),
                       bool(body.get("enforce_scope", False)), request=request if isinstance(request, str) and len(request) <= 64 else None)
        if r.get("ok"):
            r["status"] = status()
        return r

    @app.post("/api/rome")
    def rome(body: dict = Body(...)):
        r = sess.rome(body["prompt"], body["subject"], body["target"], body.get("layer"), body.get("message", ""),
                      "human", bool(body.get("dry_run")), body.get("at", "auto"))
        if r.get("ok"):
            r["status"] = status()
        return r

    def waits_for_generation(w) -> bool:
        """Whether an edit waits behind a running generation: /api/run and /api/chat hold the workspace lock only
        when a runtime worker serves them."""
        return bool(getattr(w, "_runtime_client", None))

    @app.get("/api/patches")
    def patches():
        w = ws()
        return {"patches": [p.meta() for p in w.patches], "edit_waits_for_generation": waits_for_generation(w)}

    @app.get("/api/patches/{pid}/record")
    def patch_record(pid: str, remeasure: bool = False):
        """A patch's reproduction record (wsp.directions_record; docs/DIRECTIONS.md, "The reproduction record") and
        what changed since: the base, the enabled patches, each measurement (the checks that refuse a re-run) and the
        libraries (reported). Never hashes a weight file and writes nothing. ``remeasure``: the measure request that
        makes each measurement again from the record. 404 for no patch; a patch without a record has record null."""
        from . import directions_record as rec_
        w = ws()
        if not any(p.id == pid for p in w.patches):
            return JSONResponse({"error": f"no patch '{pid}'"}, status_code=404)
        return rec_.answer(w, pid, remeasure_too=remeasure)

    @app.get("/api/patches/{pid}/record/heretic")
    def patch_record_heretic(pid: str, adapter: str | None = None, model: str | None = None,
                             model_commit: str | None = None):
        """The patch's reproduction record as Heretic's version-4 reproduce.json, for `heretic --reproduce`
        (docs/DIRECTIONS.md, "The mapping to Heretic's reproduce.json"): a download, or 400 RecordError with every
        reason Heretic could not run it. ``adapter`` names an adapter export holding this patch (its hash goes into
        ``hashes``); ``model`` and ``model_commit`` name the Hub checkpoint when the workspace does not know it."""
        from . import directions_record as rec_
        w = ws()
        query = rec_.heretic_query(adapter, model, model_commit)
        if not any(p.id == pid for p in w.patches):
            return JSONResponse({"error": f"no patch '{pid}'"}, status_code=404)
        rec, _ = rec_.read(w, w.patch(pid))
        if rec is None:
            return JSONResponse({"error": rec_.NO_RECORD.format(pid=pid)}, status_code=404)
        data = rec_.to_heretic(w, rec, pid, **query)
        return Response(json.dumps(data, indent=4), media_type="application/json",
                        headers={"Content-Disposition": 'attachment; filename="reproduce.json"'})

    @app.post("/api/patches/{pid}/reproduce")
    def patch_reproduce(pid: str, body: Any = Body(default={})):
        """Re-run a committed direction edit from its reproduction record through Session.apply (docs/DIRECTIONS.md,
        "Re-running an edit"): a dry run by default; refused when the base, the enabled patches or the measurement
        are not those it was made on; with ``measurement``, the record's calls on a new measurement made from the
        record's prompts. A commit makes a new patch (origin "reproduction"); nothing is switched on or off here.
        404 for no patch or no record, 409 while a search runs, 400 RecordError otherwise."""
        from . import directions_record as rec_
        w = ws()
        args = rec_.reproduce_body(body)
        try:
            r = rec_.reproduce(sess, pid, **args)
        except rec_.RecordMissing as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        except rec_.RecordBusy as e:
            return JSONResponse({"error": str(e)}, status_code=409)
        except dr_.DirectionMissing as e:
            return _direction_missing(e)
        if r.get("ok") and r.get("committed"):
            r["status"] = status()
        return r

    @app.post("/api/patch")
    def patch(body: dict = Body(...)):
        w = ws()
        action = body.get("action")
        pid = body.get("id")
        if action not in ("delete", "set"):
            raise ValueError(f"unknown action {action!r} (delete or set)")
        if not isinstance(pid, str) or not pid:
            raise ValueError("give the id of the patch to change")
        if not any(p.id == pid for p in w.patches):       # already gone (deleted elsewhere, or a second click)
            if action == "set":
                return JSONResponse({"error": f"Patch {pid} no longer exists; the list has been refreshed.", "missing": True,
                                     "status": status(), "patches": [p.meta() for p in w.patches]}, status_code=404)
            return {"ok": True, "missing": True, "status": status(), "patches": [p.meta() for p in w.patches]}
        before = {"enabled": w.patch(pid).enabled, "strength": w.patch(pid).strength}
        if action == "delete":
            w.delete_patch(pid)
        else:
            w.set_patch(pid, body.get("enabled"), body.get("strength"))
        out = {"ok": True, "missing": False, "status": status(), "patches": [p.meta() for p in w.patches]}
        warning = sel_.revert_warning(w, pid, action, before)
        if warning:
            out["warning"] = warning
        return out

    @app.get("/api/export")
    def export():
        """Download the edit: for this project's small models the whole edited checkpoint, otherwise a patch file
        holding only the tensors that changed (same names and dtypes as the checkpoint)."""
        w = ws()
        tmp = tempfile.mkdtemp()
        if w.kind == "wsp":
            path = w.export(tmp, log=lambda *_: None)[0]
            name = f"{w.name}-edited.safetensors"
        else:
            if not w.touched():
                raise ValueError("nothing has been edited yet")
            path = w.export_patch(os.path.join(tmp, "patch.safetensors"))["path"]
            name = f"{w.name}-companion.wsppatch.zip" if w.kind == 'gguf' else f"{w.name}-patch.safetensors"
        return FileResponse(path, filename=name)

    @app.post("/api/gguf/export/preview")
    def gguf_export_preview(body: dict = Body(...)):
        from .gguf_mixed import preview_export
        return preview_export(ws(), body.get('precision', 'F16'))

    @app.post("/api/gguf/export")
    def gguf_export_start(body: dict = Body(...)):
        return exports.start(ws(), body.get('precision', 'F16'))

    @app.get("/api/gguf/export/{job}")
    def gguf_export_status(job: str):
        return exports.get(job)

    @app.post("/api/gguf/export/{job}/cancel")
    def gguf_export_cancel(job: str):
        return exports.cancel(job)

    @app.get("/api/gguf/export/{job}/download")
    def gguf_export_download(job: str):
        state = exports.get(job)
        if state['status'] != 'COMPLETE':
            raise ValueError('Export is not complete; no artifact is available')
        return FileResponse(state['report']['path'], filename=Path(state['report']['path']).name)

    @app.post("/api/export_checkpoint")
    def export_checkpoint(body: dict = Body(...)):
        """Write the full edited checkpoint next to the workspace (streamed, one tensor in memory at a time)."""
        w = ws()
        if body.get("name") == "adapters":           # <workspace>/export/adapters/ holds the adapter exports
            return JSONResponse({"error": "the name 'adapters' is taken by the adapter exports "
                                          "(<workspace>/export/adapters); choose another name"}, status_code=400)
        out = os.path.join(w.dir, "export", body.get("name") or "edited")
        logs: list[str] = []
        files = w.export(out, log=logs.append)
        model_dir = getattr(w, "model_dir", None)
        if model_dir:
            for f in os.listdir(model_dir):
                if f.endswith((".json", ".txt", ".model", ".jinja")):
                    with open(os.path.join(model_dir, f), "rb") as src, open(os.path.join(out, f), "wb") as dst:
                        dst.write(src.read())
        return {"dir": out, "files": files, "log": logs}

    @app.post("/api/export_adapter")
    def export_adapter(body: dict = Body(...)):
        """The patches as a LoRA adapter in PEFT's format, in <workspace>/export/adapters/<name>/ (wsp.adapter_export;
        docs/DIRECTIONS.md, "A LoRA adapter (PEFT)"). `dry_run` answers the same without writing. Never changes a
        weight, a patch or the workspace's meta. A refusal is 400 AdapterError, an unknown patch id 404."""
        from . import adapter_export as ax_
        w = ws()
        try:
            return ax_.request(w, body)
        except ax_.AdapterMissing as e:
            return JSONResponse({"error": f"AdapterError: {e}"}, status_code=404)

    @app.post("/api/groups")
    def groups(body: dict = Body(...)):
        """Save, extend or delete a named selection (contract C2). Members may be given as addresses ("members"),
        as [layer, neuron] pairs or neuron addresses ("neurons", tens of thousands at once), or as an atlas
        selector ("selector", expanded here so the browser never builds the list). mode "add" merges into an
        existing selection; neuron runs are stored as ranges."""
        w = ws()
        raw = str(body.get("name") or "")
        name = raw.strip()
        if body.get("delete") and raw in w.groups:          # a name saved with surrounding spaces is still deletable
            name = raw
        if not name:
            raise ValueError("give the selection a name")
        if body.get("delete"):
            w.delete_group(name)
            return {"groups": w.groups}
        new = list(body.get("members") or [])
        pairs = sel_.parse_neurons(body.get("neurons") or [])
        if body.get("selector"):
            r = at.select(atlas_(), body["selector"])
            pairs += [(int(l), int(i)) for l, i in zip(r["layer"], r["idx"])]
        new += sel_.members_from_neurons(pairs)
        if not new and "members" not in body:
            raise ValueError("nothing to save: the selection has no members")
        old = w.group_members(name) if body.get("mode") == "add" and name in w.groups else []
        members = sel_.merge_members(w, old, new)
        note = body.get("note", w.groups.get(name, {}).get("note", "") if old else "")
        w.set_group(name, members, note)
        return {"groups": w.groups, "name": name, "addr": f"group:{name}", "summary": sel_.summary(w, f"group:{name}")}

    # ------------------------------------------------------------------ tests and task
    @app.get("/api/tests")
    def tests(suite: str = "", variant: str = "current"):
        runnable()
        return sess.tests(suite or None, variant) or {"name": None, "cases": [], "passed": 0, "total": 0}

    @app.post("/api/load")
    def load():
        """Load the model for running (what the first run, test or edit would do anyway)."""
        w = runnable()
        if w.kind == 'gguf':
            sess.tests(variant='base'); sess.tests(); sess.drift()
        return status()

    @app.get("/api/suite")
    def suite_get(name: str):
        return testsuite.load_suite(ws(), name)

    @app.post("/api/suite")
    def suite_set(body: dict = Body(...)):
        with sess.lock:
            w = ws()
            refuse_during_build(w, "saving a test suite")
            with w.lock:
                testsuite.save_suite(w, body["name"], body["suite"])
                w.calib.clear()
                for k in [k for k in w.keep if k.startswith("tests:")]:
                    w.keep.pop(k)
                w.inspection_revision = getattr(w, "inspection_revision", 0) + 1
        return {"ok": True, "suites": testsuite.list_suites(w)}

    @app.post("/api/task")
    def task(body: dict = Body(...)):
        with sess.lock:
            auth.assert_no_login()
            return {"task": sess.set_task(body), "status": status()}

    # ------------------------------------------------------------------ chunks and the programmer
    @app.post("/api/plan")
    def plan(body: dict = Body(...)):
        sess.plan = chunks.plan(ws(), body.get("selection") or sess.task.get("scope") or ["model"],
                                int(body.get("budget", 40_000)), body.get("format", "auto"),
                                int(body.get("total_budget", 700_000)))
        return sess.plan

    @app.get("/api/chunk", response_class=PlainTextResponse)
    def chunk(id: int, variant: str = "current", context: str = "full"):
        if not sess.plan:
            raise ValueError("no plan")
        return chunks.render_chunk(ws(), sess.plan["chunks"][id], variant, context)

    @app.get("/api/render", response_class=PlainTextResponse)
    def render(addr: str, format: str = "q", variant: str = "current", context: str = "full"):
        w = ws()
        est = chunks.est_slices(w, w.resolve(addr), format)
        if est > 400_000:
            raise ValueError(f"about {est} tokens; narrow the address or choose a lossy format")
        return chunks.render(w, [addr], format, variant, context)

    @app.get("/api/brief", response_class=PlainTextResponse)
    def brief(mode: str = "tools"):
        return sess.brief(mode)

    @app.get("/api/agent/tools")
    def agent_tools():
        return {"tools": tools.specs()}

    @app.post("/api/agent/call")
    def agent_call(body: dict = Body(...)):
        text, is_error = mcp.call_tool(sess, body["name"], body.get("arguments") or {})
        return {"text": text, "isError": is_error}

    @app.post("/mcp")
    def mcp_http(msg=Body(...)):
        backend = mcp.LocalBackend(sess)
        if isinstance(msg, list):
            out = [r for r in (mcp.handle(backend, m) for m in msg) if r is not None]
            return JSONResponse(out) if out else JSONResponse(None, status_code=202)
        r = mcp.handle(backend, msg)
        return JSONResponse(r) if r is not None else JSONResponse(None, status_code=202)

    def auth_response(state):
        return {**state, "csrf_token": auth.csrf}

    @app.get("/api/harness/auth")
    def harness_auth(provider: str):
        return auth_response(auth.status(provider))

    @app.post("/api/harness/auth/refresh")
    def harness_auth_refresh(body: dict = Body(...)):
        return auth_response(auth.status(body.get("provider"), refresh=True))

    @app.post("/api/harness/auth/login")
    def harness_auth_login(body: dict = Body(...)):
        with sess.lock:
            native_harness.assert_available(sess)
            return auth_response(auth.login(body.get("provider")))

    @app.post("/api/harness/auth/code")
    def harness_auth_code(body: dict = Body(...)):
        return auth_response(auth.active_attempt(body.get("provider"), body.get("attempt_id")).code(body.get("code")))

    @app.post("/api/harness/auth/cancel")
    def harness_auth_cancel(body: dict = Body(...)):
        return auth_response(auth.active_attempt(body.get("provider"), body.get("attempt_id")).cancel())

    @app.post("/api/harness/prepare")
    def harness_prepare(body: dict = Body(...)):
        with sess.lock:
            native_harness.assert_available(sess)
            p = harness.prepare(sess, url, body.get("mode", "tools"), body.get("extra", ""))
        cmd, _ = harness.command(body.get("harness", "claude"), p, url, body.get("mode", "tools"), body.get("model", ""),
                                 float(body.get("budget") or 0), body.get("custom", ""))
        return {"dir": p["dir"], "mcp_config": p["mcp_config"], "command": [c if len(c) < 300 else "<brief>" for c in cmd],
                "brief_tokens": chunks.est_text(p["brief"])}

    @app.post("/api/harness/start")
    def harness_start(body: dict = Body(...)):
        with sess.lock:
            native_harness.assert_available(sess)
            auth.assert_no_login()
            provider = body.get("harness", "claude")
            billing = "per_call"
            if provider in ("claude", "codex"):
                billing = native_harness.billing_of(provider, auth.ensure_ready(provider).get("method"))
            run = harness.start(sess, url, provider, body.get("mode", "tools"), body.get("model", ""),
                                float(body.get("budget") or 0), body.get("extra", ""), body.get("custom", ""), billing)
            return {**run.meta(), "now": time.time()}

    @app.post("/api/harness/stop")
    def harness_stop(body: dict = Body(...)):
        run = sess.runs[body["id"]]
        harness.stop(run)
        return run.meta()

    @app.post("/api/harness/send")
    def harness_send(body: dict = Body(...)):
        return sess.runs[body["id"]].send(body.get("text", ""))

    @app.post("/api/harness/respond")
    def harness_respond(body: dict = Body(...)):
        return sess.runs[body["id"]].respond(body.get("request_id"), body.get("allow"), body.get("answers"))

    @app.post("/api/harness/close")
    def harness_close(body: dict = Body(...)):
        run = sess.runs[body["id"]]
        if getattr(run, "native", False):
            return run.close()
        harness.stop(run)
        return run.meta()

    @app.get("/api/harness")
    def harness_get(id: str = "", since: int = 0):
        # `now` is the server's clock: the browser times the status line and tool calls from event times with it.
        if not id:
            # `workspace`: the open workspace, as each run's meta names its own (the page shows only this one's runs).
            return {"runs": [r.meta() for r in sess.runs.values()], "log": sess.log[-200:], "now": time.time(),
                    "csrf_token": auth.csrf, "workspace": os.path.realpath(sess.ws.dir) if sess.ws is not None else None}
        r = sess.runs.get(id)
        if r is None:
            return JSONResponse({"error": "Unknown run"}, status_code=404)
        return {**r.meta(), "events": r.events[max(0, since):], "now": time.time()}

    # Saved conversations of the open workspace (wsp/conversations.py). Listing and reading only read files in
    # <workspace>/conversations; neither starts a bridge or contacts a provider.
    def verdict(meta, live):
        refusal = native_harness.resume_refusal(sess, meta, live)
        out = {"ok": refusal is None, "code": refusal[0] if refusal else None, "reason": refusal[1] if refusal else None}
        if refusal and refusal[0] == "gone" and native_harness.resume_refusal(sess, meta, live, retry=True) is None:
            out["retry"] = True                     # "Try again": the provider may have answered wrongly
        if refusal and refusal[0] == "elsewhere" and meta.get("holder") == "pid":
            out["release"] = True                   # no file locks here: "It is not running; mark interrupted", confirmed
        return out

    def open_conversation(cid):
        """(conv, None) or (None, a 404 response). Any damage to the entry is a 404 with the reason."""
        try:
            return conversations.Conversation.open(ws().dir, cid), None
        except Exception as exc:
            return None, JSONResponse({"error": str(exc) or type(exc).__name__}, status_code=404)

    @app.get("/api/harness/conversations")
    def harness_conversations():
        w = ws()
        native_harness.wait_finishing(sess)         # a run between "failed" and retired is never listed half-way
        live = native_harness.live_conversations(sess)
        out = []
        for meta in conversations.listing(w.dir, live):
            row = conversations.summary(meta, meta["id"] in live)
            out.append({**row, "resume": verdict(meta, live)})
        return {"conversations": out, "now": time.time(), "csrf_token": auth.csrf, "locking": conversations.locking_note(w.dir),
                "limits": {"conversations": conversations.KEEP, "events": conversations.MAX_EVENTS,
                           "bytes": conversations.MAX_FILE_BYTES, "event_bytes": conversations.MAX_EVENT_BYTES}}

    @app.get("/api/harness/conversation")
    def harness_conversation(id: str):
        conv, error = open_conversation(id)
        if error:
            return error
        try:
            native_harness.wait_finishing(sess)
            live = native_harness.live_conversations(sess)
            meta = conversations.settle(conv, live)
            events = conv.events()
        except Exception as exc:
            return JSONResponse({"error": str(exc) or type(exc).__name__}, status_code=404)
        finally:
            conv.close()
        return {"meta": {**conversations.summary(meta, id in live), "n_events": len(events),
                         "interrupted": meta.get("interrupted"), "closed_reason": meta.get("closed_reason"),
                         "resumes": meta.get("resumes") or [], "resume_error": meta.get("resume_error"),
                         "task_full": meta.get("task")},
                "events": events, "resume": verdict(meta, live), "now": time.time()}

    @app.post("/api/harness/resume")
    def harness_resume(body: dict = Body(...)):
        with sess.lock:
            conv, error = open_conversation(body.get("conversation"))
            if error:
                return error
            handed = False
            try:
                native_harness.wait_finishing(sess)
                live = native_harness.live_conversations(sess)
                try:
                    meta = conversations.settle(conv, live)
                except Exception as exc:
                    return JSONResponse({"error": f"This saved conversation cannot be read ({exc}).", "refusal": "unreadable"},
                                        status_code=409)
                refusal = native_harness.resume_refusal(sess, meta, live, retry=body.get("retry") is True)
                if refusal:
                    return JSONResponse({"error": refusal[1], "refusal": refusal[0]}, status_code=409)
                auth.assert_no_login()
                state = auth.ensure_ready(meta["provider"])          # the same fresh sign-in check as Start
                billing = native_harness.billing_of(meta["provider"], state.get("method"))
                try:
                    run = harness.resume(sess, url, conv, body.get("text", ""), billing)
                except conversations.Busy as exc:
                    return JSONResponse({"error": str(exc), "refusal": "elsewhere"}, status_code=409)
                except conversations.Unreadable as exc:
                    return JSONResponse({"error": f"This saved conversation cannot be used ({exc}).", "refusal": "unreadable"},
                                        status_code=409)
                handed = True                       # the run owns the record now and closes it when it ends
                return {**run.meta(), "events": list(run.events), "now": time.time()}
            finally:
                if not handed:
                    conv.close()

    @app.post("/api/harness/conversation/release")
    def harness_conversation_release(body: dict = Body(...)):
        """The person says the IDE process recorded as running this conversation is not running it: mark it
        interrupted so it can be resumed. Needs `confirm: true` (the page asks first). Refused while another
        live process holds the conversation's lock (it does have it open)."""
        if body.get("confirm") is not True:
            return JSONResponse({"error": "Confirm that no IDE process is running this conversation."}, status_code=400)
        with sess.lock:
            conv, error = open_conversation(body.get("conversation"))
            if error:
                return error
            try:
                if conv.id in native_harness.live_conversations(sess) or conversations.held_here(conv):
                    return JSONResponse({"error": "This IDE process is running this conversation; close it instead."},
                                        status_code=409)
                try:
                    meta = conversations.mark_interrupted(conv)
                except conversations.Busy as exc:
                    return JSONResponse({"error": str(exc), "refusal": "elsewhere"}, status_code=409)
                except Exception as exc:
                    return JSONResponse({"error": f"This saved conversation cannot be used ({exc})."}, status_code=409)
                live = native_harness.live_conversations(sess)
                return {"conversation": {**conversations.summary(meta, False), "resume": verdict(meta, live)}}
            finally:
                conv.close()

    @app.get("/api/events")
    async def events():
        async def gen():
            q = sess.subscribe()
            try:
                yield "retry: 1500\n\n"
                idle = 0
                while not lifecycle.stopping.is_set():
                    try:
                        ev = q.get_nowait()
                    except queue.Empty:
                        # Cancellable async wait: no thread remains stuck in a
                        # blocking queue.get after disconnect or server shutdown.
                        await asyncio.sleep(.1)
                        idle += 1
                        if idle >= 150:
                            yield ": keep-alive\n\n"
                            idle = 0
                    else:
                        yield f"data: {json.dumps(ev)}\n\n"
                        idle = 0
            finally:
                sess.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get(f"/assets/{asset_version}/{{path:path}}")
    def asset(path: str):
        if path not in assets:
            raise HTTPException(404, "unknown asset")
        return Response(assets[path], media_type="text/javascript" if path.endswith(".js") else "text/css",
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @app.get("/")
    @app.get("/index.html")
    def index():
        return HTMLResponse(index_html, headers={"Cache-Control": "no-store"})

    app.mount("/", StaticFiles(directory=WEB), name="web")
    return app
