"""The IDE's state: the open workspace, the current programming task, and the one path every edit takes.

A *task* is what the programmer model is given:

    {"title": "...", "instructions": "...",       what to achieve, in words
     "scope": ["blocks.1.mlp", "L0H3"],            addresses that may be changed (anything else is rejected)
     "suite": "facts",                             the test suite that defines success
     "preserve": {"max_kl": 0.05} | null}          how far the model may drift on unrelated text

`apply` is the single entry point for edits from people and from the programmer model alike:
run the weight script on a copy, check the scope, run the tests and the drift measurement on
the candidate weights, and only then (unless dry_run) commit a patch.  The report that comes
back is the programmer's feedback signal.
"""
from __future__ import annotations

from .runtime_events import RuntimeRefusal, edit_preflight, job_scope

import os
import queue
import threading
import time

from . import chunks, delta as D, edits, testsuite
from .workspace import ScopeError, Workspace
from .wscript import FUNCTION_DOC, ScriptError, run_script

DEFAULT_TASK = {"title": "", "instructions": "", "scope": ["model"], "suite": None, "preserve": {"max_kl": 0.1}}


def is_workspace(path: str) -> bool:
    """A workspace directory (as opposed to, say, a Hugging Face checkpoint directory, which also has a config.json)."""
    return os.path.exists(os.path.join(path, "workspace.json")) or \
        (os.path.exists(os.path.join(path, "config.json")) and os.path.exists(os.path.join(path, "base.safetensors")))


class Session:
    def __init__(self, root: str):
        self.atlas_job: dict | None = None             # the running / last atlas build (shown by the Atlas tab)
        self.search_job: dict | None = None            # the running / last automatic search (wsp.directions_search)
        self.root = os.path.abspath(root)
        self.ws: Workspace | None = None
        self.task: dict = dict(DEFAULT_TASK)
        self.subscribers: list[queue.Queue] = []
        self.plan: dict | None = None
        self.runs: dict[str, dict] = {}
        self.log: list[dict] = []
        self.lock = threading.RLock()
        self.runtime = None
        self.model_roots = []
        self.edit_settings: dict | None = None         # resolved settings values edits read (wsp.edit_limit)

    # ------------------------------------------------------------------ workspaces
    def available(self) -> list[dict]:
        from .zoo import DESCRIPTIONS
        out = []
        have = set()
        for d in sorted(os.listdir(self.root)) if os.path.isdir(self.root) else []:
            base = os.path.join(self.root, d)
            if is_workspace(base):
                have.add(d)
                desc = DESCRIPTIONS.get(d, "")
                mp = os.path.join(base, "workspace.json")
                if not desc and os.path.exists(mp):
                    import json
                    m = json.load(open(mp))
                    desc = m.get("repo") or m.get("model_dir") or ""
                out.append({"name": d, "built": True, "description": desc})
        out += [{"name": n, "built": False, "description": desc} for n, desc in DESCRIPTIONS.items() if n not in have]
        return out

    def open(self, name: str, log=print, pull: bool = False) -> Workspace:
        """Open a workspace by name, build one of the bundled models, or create one over any checkpoint:
        a Hub repo id ("org/model", "hf:org/model") or a local directory in Hugging Face layout."""
        if name.lower().endswith('.gguf') and not os.path.isfile(name):
            matches = [os.path.join(p,name) for p in self.model_roots if os.path.isfile(os.path.join(p,name))]
            if len(matches) > 1: raise ValueError('Model name matches several roots; use an explicit path')
            if matches: name = matches[0]
        from .zoo import BUILDERS, build, open_hf
        simple = "/" not in name and os.sep not in name and not name.startswith("hf:")
        if name.lower().endswith('.gguf') and os.path.isfile(name):
            from .gguf_store import open_gguf
            ws = open_gguf(self.root, name, metadata_only=self.runtime is not None)
        elif name.endswith('.wsppatch.zip') and os.path.isfile(name):
            from .companion import import_bundle
            ws = import_bundle(self.root, name, metadata_only=self.runtime is not None)
        elif is_workspace(name):
            ws = Workspace.load(name, metadata_only=self.runtime is not None)
        elif simple and is_workspace(os.path.join(self.root, name)):
            ws = Workspace.load(os.path.join(self.root, name), metadata_only=self.runtime is not None)
        elif simple and name in BUILDERS:
            ws = build(name, self.root, log=log)
        elif not simple or os.path.isdir(name):
            ws = open_hf(self.root, name, log=log)       # reuses the workspace if this checkpoint has been opened before
        else:
            raise KeyError(f"no workspace or model named {name!r}")
        if pull and ws.store.remote:
            from .zoo import pull as do_pull
            ws = do_pull(ws, log)
        self.unload()
        if self.runtime is not None and ws.kind == 'gguf':
            ws._runtime_client = self.runtime
            self.runtime.attach(ws)
        ws.listeners.append(self.publish)
        ws.edit_settings = self.edit_settings
        self.ws = ws
        self.plan = None
        self.task = {**DEFAULT_TASK, **(ws.read_json("task.json") or {})}
        if not self.task.get("suite"):
            suites = testsuite.list_suites(ws)
            self.task["suite"] = suites[0]["name"] if suites else None
        self.publish({"type": "workspace", "name": ws.name})
        return ws

    def unload(self):
        """Release the runner before releasing its external reservation (also on model switch)."""
        job = self.search_job                          # a running search stops at its next stop point, before the lock
        if job and job.get("running") and self.ws is not None and job.get("workspace") == self.ws.name:
            job["stop"] = "closed"
        with self.lock:
            if self.ws is not None:
                with self.ws.lock:
                    failed = getattr(self.ws, '_failed_runner', None)
                    if failed is not None:
                        with failed.lock: failed._dispose()
                        del self.ws._failed_runner
                    if self.ws._runner is not None:
                        with self.ws._runner.lock:
                            self.ws._runner.release()
                            self.ws._runner = None
                    self.ws.run_error = None
                    self.ws.calib.clear();self.ws.keep.clear()
                    getattr(self.ws,'_cache',{}).clear();self.ws._cache_bytes=0
            if self.runtime is not None: self.runtime.release()

    def set_task(self, task: dict) -> dict:
        from .native_harness import assert_available
        with self.lock:
            assert_available(self)
            return self._set_task(task)

    def _set_task(self, task: dict) -> dict:
        t = {**DEFAULT_TASK, **{k: v for k, v in task.items() if k in DEFAULT_TASK}}
        for a in t["scope"] or []:
            self.ws.resolve(a)
        if t["suite"]:
            testsuite.load_suite(self.ws, t["suite"])
        self.task = t
        self.ws.write_json("task.json", t)
        self.publish({"type": "task"})
        return t

    # ------------------------------------------------------------------ events
    def publish(self, event: dict) -> None:
        event = {"t": time.time(), **event}
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2000)
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q) -> None:
        if q in self.subscribers:
            self.subscribers.remove(q)

    # ------------------------------------------------------------------ tests (cached per weights version)
    @job_scope('tests')
    def tests(self, suite: str | None = None, variant: str = "current", extra=None) -> dict | None:
        ws = self.ws
        suite = suite or self.task.get("suite")
        if not suite or not ws.can_run:
            return None
        try:
            if extra is not None:
                return testsuite.run_suite(ws, suite, "current", extra)
            key = f"tests:{suite}:{variant}"
            store = ws.calib if variant == "current" else ws.keep
            if key not in store:
                result = testsuite.run_suite(ws, suite, variant)
                if result.get("not_run"):
                    refused=RuntimeRefusal(next(c["event"] for c in result["cases"] if c.get("status")=="not_run"))
                    refused.report=result
                    raise refused
                store[key] = result
            return store[key]
        except RuntimeError:
            if ws.run_error:                         # the model could not be loaded: there are no test results
                return None
            raise

    @job_scope('drift')
    def drift(self, extra=None) -> dict:
        ws = self.ws
        none = {"kl": 0.0, "kl_max": 0.0, "top1_agree": 1.0, "n": 0}
        if not ws.can_run:
            return none
        try:
            if extra is not None:
                return testsuite.drift(ws, "current", extra)
            if not ws.touched():                     # base weights: nothing to measure
                return none
            if "drift" not in ws.calib:
                ws.calib["drift"] = testsuite.drift(ws)
            return ws.calib["drift"]
        except RuntimeError:
            if ws.run_error:
                return none
            raise

    # ------------------------------------------------------------------ the edit path
    @job_scope('_evaluate')
    def _evaluate(self, extra: dict) -> dict:
        ws = self.ws
        if ws.can_run:
            try:
                ws.runner                            # loads the model if this is the first thing that needs to run it
            except RuntimeRefusal:
                raise
            except RuntimeError:
                pass
        if not ws.can_run:
            why = ws.run_error or ("its weights are still on the Hub" if ws.store.remote else "it has no runnable configuration")
            return {"drift": {"kl": 0.0, "top1_agree": 1.0, "n": 0},
                    "not_run": f"the model cannot be run here ({why}), so the edit was not tested"}
        out: dict = {"drift": self.drift(extra)}
        if self.task.get("suite"):
            before, after = self.tests(), self.tests(extra=extra)
            if before.get("not_run") or after.get("not_run"):
                blocked=next(c for report in (before,after) for c in report["cases"] if c.get("status")=="not_run")
                raise RuntimeRefusal(blocked["event"])
            was = {c["name"]: c["pass"] for c in before["cases"]}
            out["tests"] = {"suite": after["name"], "before": [before["passed"], before["total"]],
                            "after": [after["passed"], after["total"]],
                            "fixed": [c["name"] for c in after["cases"] if c["pass"] and not was.get(c["name"], False)],
                            "broken": [c["name"] for c in after["cases"] if not c["pass"] and was.get(c["name"], False)],
                            "summary": testsuite.summarize(after)}
        pres = self.task.get("preserve")
        if pres and pres.get("max_kl") is not None:
            out["drift"]["limit"] = pres["max_kl"]
            out["drift"]["within_limit"] = out["drift"]["kl"] <= pres["max_kl"]
            out["drift"]["enforcement"] = "advisory; exceeding max_kl does not reject a commit"
        return out

    def set_edit_settings(self, values: dict | None) -> None:
        """The settings values edits read now (the Largest weight patch setting applies as soon as it is saved)."""
        self.edit_settings = values
        if self.ws is not None:
            self.ws.edit_settings = values

    @job_scope('edit')
    def commit_deltas(self, deltas: dict, message: str, author: str, source: dict, dry_run: bool = False,
                      enforce_scope: bool = True, output: str = "") -> dict:
        """deltas: {tensor name: [pieces]} relative to the current weights."""
        ws = self.ws
        with ws.lock:
            deltas = {k: [p for p in ps if not p.is_zero()] for k, ps in deltas.items()}
            deltas = {k: ps for k, ps in deltas.items() if ps}
            scope = self.task.get("scope") if enforce_scope else None
            try:
                ws.check_scope(deltas, scope)
            except ScopeError as e:
                return {"ok": False, "error": str(e), "output": output}
            if not deltas:
                return {"ok": True, "committed": False, "changed": [], "output": output,
                        "note": "the script changed no weights"}
            for name, ps in deltas.items():
                if not ws.editable(name):
                    return {"ok": False, "error": f"{name} is not a floating-point tensor and cannot be edited", "output": output}
                if any(not p.finite() for p in ps):
                    return {"ok": False, "error": f"the change to {name} contains NaN or inf", "output": output}
            stored = sum(pc.numel() for ps in deltas.values() for pc in ps)
            from . import edit_limit
            if over := edit_limit.over_setting(ws, stored):     # every edit path: scripts, ROME, agent files
                return {"ok": False, "error": edit_limit.refusal(stored, over), "output": output}
            edit_preflight(ws,stored)
            report = self._evaluate(deltas)
            changed = [{"param": c["param"], "changed": c["changed"], "of": ws.store.numel(c["param"]),
                        "norm": round(c["norm"], 5), "max_abs": round(c["max_abs"], 5), "rank": c["rank"],
                        "kinds": c["kinds"], "stored": c["stored"]} for c in D.summary(deltas)]
            report.update({"ok": True, "committed": False, "changed": changed, "output": output,
                           "n_changed": sum(c["changed"] for c in changed), "stored": sum(c["stored"] for c in changed)})
            if not dry_run:
                p = ws.commit(deltas, message, author, source, scope=None,
                              info={"tests": report.get("tests", {}).get("after"), "drift": report["drift"].get("kl")})
                report["committed"] = True
                report["patch"] = p.id
                client=getattr(ws,'_runtime_client',None)
                if client:client.record_commit(ws,p.id,report)
            self.note(author, "edit" if not dry_run else "dry run", message, report)
            return report

    @job_scope('edit')
    def apply(self, script: str, message: str = "", author: str = "human", dry_run: bool = False,
              enforce_scope: bool = True, request: str | None = None, *, origin: dict | None = None,
              check=None) -> dict:
        """Run a weight script against the current weights and commit the result as a patch.

        `request` (the browser's id for this edit) tags its "edit_progress" events and nothing else. `origin`, when
        given, is recorded in the patch's source as "origin" (where the edit came from, e.g. a search trial).
        `check(deltas)`, when given, sees the script's pieces before anything is evaluated; a message it returns
        refuses the edit with that message."""
        from . import edit_limit
        started = time.monotonic()
        try:
            res = run_script(self.ws, script, request=request)
            edit_limit.record_timing(self.ws, sum(p.numel() for ps in res["deltas"].values() for p in ps),
                                     time.monotonic() - started)
        except ScriptError as e:
            return {"ok": False, "error": str(e)}
        except RuntimeRefusal:
            raise
        except RuntimeError as e:                    # e.g. the script asked for activations of a model that cannot be run
            return {"ok": False, "error": str(e)}
        if check is not None:
            refusal = check(res["deltas"])
            if refusal:
                return {"ok": False, "error": refusal}
        source = {"kind": "wscript", "script": script}
        if origin is not None:
            source["origin"] = origin
        if res.get("reverted"):
            # a revert is a compensating patch: record what it was worked out against (selections.revert_warning)
            from .selections import compensates
            dep = compensates(self.ws, res["deltas"], res["reverted"])
            if dep:
                source["compensates"] = dep
        # The reproduction record (wsp.directions_record) of an edit that read a measurement: prepared before the
        # commit (the weights it was made on, the base, the libraries, the measurements' snapshots), written after
        # it. A dry run writes nothing; a record that cannot be written never undoes or fails the commit.
        calls = res.get("calls") or []
        pending = record_error = None
        if calls and not dry_run:
            from . import directions_record as REC
            try:
                pending = REC.prepare(self.ws, script, calls, res["deltas"], origin)
            except Exception as e:
                record_error = f"{type(e).__name__}: {e}"
        report = self.commit_deltas(res["deltas"], message or _first_line(script), author,
                                    source, dry_run, enforce_scope, res["output"])
        if calls and not dry_run and report.get("committed"):
            if pending is not None:
                try:
                    report["record"] = REC.write(self.ws, report["patch"], pending)
                except Exception as e:
                    record_error = f"{type(e).__name__}: {e}"
            if record_error is not None:
                report["record_error"] = record_error
        return report

    @job_scope('edit')
    def rome(self, prompt: str, subject: str, target: str, layer: int | None = None, message: str = "",
             author: str = "human", dry_run: bool = False, at: str = "auto") -> dict:
        try:
            r = edits.rome(self.ws, prompt, subject, target, layer, at)
        except RuntimeRefusal:
            raise
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        rep = self.commit_deltas(r["deltas"], message or f"ROME: {prompt!r} -> {target!r}", author,
                                 {"kind": "rome", "prompt": prompt, "subject": subject, "target": target, "layer": r["layer"],
                                  "at": r["at"]},
                                 dry_run)
        rep["rome"] = {k: r[k] for k in ("layer", "at", "pos", "success", "p_before", "p_after")}
        return rep

    def note(self, author: str, kind: str, message: str, report: dict | None = None) -> None:
        entry = {"t": time.time(), "author": author, "kind": kind, "message": message}
        if report:
            entry["tests"] = report.get("tests", {}).get("after")
            entry["patch"] = report.get("patch")
        self.log.append(entry)
        self.publish({"type": "log", **entry})

    # ------------------------------------------------------------------ what the programmer is told
    def _atlas_brief(self) -> list[str]:
        try:
            return _atlas_lines(self.ws)
        except Exception as e:                                   # the brief must never fail because of the atlas
            return [f"(the atlas could not be read: {e})"]

    def brief(self, mode: str = "tools") -> str:
        ws, t = self.ws, self.task
        suite = testsuite.load_suite(ws, t["suite"]) if t.get("suite") else None
        cur = self.tests() if suite else None
        scope = t.get("scope") or ["model"]
        n_scope = sum(_count(ws, a) for a in scope)
        lines = [
            "You are the programmer in a weight-space programming session.",
            "",
            "Your job is to change what a neural network does by writing its weights directly. There is no "
            "training loop and no gradient descent here: you read weights (or summaries of them), decide what they "
            "should be, and write them. A test suite tells you whether the behaviour is what was asked for.",
            "",
            "## Task" + (f": {t['title']}" if t.get("title") else ""),
            t.get("instructions") or "(no instructions given)",
            "",
            "## The model",
            f"{ws.name}: {ws.n_params():,} weights in {len(ws.names())} tensors. " + ws.struct.describe(),
            ws.notes().strip(),
            "The model is never loaded whole for reading or editing: every tool reads only the region it is asked for, "
            "so prefer narrow addresses (a head, a neuron, a row) over whole tensors."
            if ws.n_params() > 50_000_000 else "",
            "",
            "## What you may change",
            f"Scope: {', '.join(scope)}  ({n_scope:,} weights; about {int(n_scope * 2.9):,} tokens if read densely in format q)."
            + (" Edits outside the scope are rejected." if scope != ["model"] else ""),
        ]
        pres = t.get("preserve")
        if pres and pres.get("max_kl") is not None and ws.can_run:
            lines.append(f"Keep the model's behaviour on unrelated text close to the original: mean KL from the base model "
                         f"should stay at or below {pres['max_kl']}. This is advisory: `drift` reports violations, "
                         "but exceeding it does not reject the edit. Inspect preservation tests and undo harmful edits.")
        lines += ["", "## Success"]
        if not ws.can_run:
            lines.append("This model cannot be run in this session (weights not on disk, or no runnable config): `run`, "
                         "`trace`, tests and drift are unavailable. Work from the weights, and say in `finish` what should be tested.")
        elif suite:
            lines.append(f"Test suite `{t['suite']}` ({len(suite['cases'])} cases; {suite.get('description', '')}). "
                         f"Currently {cur['passed']}/{cur['total']} pass. Examples:")
            for c in suite["cases"][:6]:
                lines.append("  " + _case_line(c))
        else:
            lines.append("No test suite is attached; use `run` to check behaviour yourself.")
        lines += [
            "",
            "## How to work",
            "All access goes through the `wsp` tools:" if mode == "tools" else
            "You have the `wsp` tools, and you may also use your own file and shell tools on checked-out weights:",
            "- `overview` — model, hierarchy, patches so far, task and test status. Start here.",
            "- `list_nodes` — walk the hierarchy (model > layer > attention / MLP > head / neuron > tensor).",
            "- `inspect` — what a head, neuron or tensor does: vocabulary projections, SVD, composition with other parts.",
            "- `plan_chunks`, `read_chunk`, `read_weights` — the weights themselves, in a token budget you choose.",
            "- `run`, `trace` — run the model (base and edited side by side); see predictions, attention, logit lens, "
            "direct logit attribution.",
            "- `vocab` — token ids and tokenisation. Tests are about single tokens; check how your strings tokenise.",
            "- `edit` — apply a weight script. Use `dry_run: true` to see the test results an edit would give without "
            "committing it. Every committed edit is a patch that can be reverted.",
            "- `rome` — rank-one fact edit of an MLP when the task is 'make prompt X continue with Y'.",
            "- `run_tests`, `patches`, `finish`.",
        ]
        if mode == "files":
            lines.append("- `checkout_files` writes tensors as .npy files into your working directory; change them with "
                         "numpy, then `commit_files` turns the difference into a patch.")
        lines += [
            "",
            "Read before you write: a few tool calls to understand the relevant part of the model are cheaper than a "
            "wrong edit. Prefer edits you can explain (a direction with a meaning, a lookup you constructed) over "
            "trial and error. Work in small committed steps and check the tests after each. When the suite passes "
            "(or you have gone as far as you can), call `finish` with a short account of what you changed and why.",
            "",
            "## The atlas",
            *self._atlas_brief(),
            "",
            "## Weight scripts",
            "A weight script is a restricted Python subset interpreted by the IDE (no imports, no arbitrary attribute "
            "access). Weights are reached through views that work on any architecture (`head(L, h).O`, `neuron(L, n).w_out`, "
            "`mlp(L).W_out`), or by checkpoint name with `P(\"...\")` (see `overview` for names and layouts); assigning to a "
            "view or a slice of it is the edit. Views are lazy: index them to the part you need.",
            "```",
            FUNCTION_DOC.rstrip(),
            "```",
        ]
        return "\n".join(x for x in lines if x is not None)


def _atlas_lines(ws) -> list[str]:
    from .atlas import Atlas
    A = Atlas.of(ws)
    if not A.exists:
        return ["No atlas has been built for this model. If the task is about what the model says or knows, build one first "
                "(tool `atlas`, action=build): editing without it is editing blind."]
    lt = A.layer_table()
    v = A.json("validation.json")
    lines = [f"A map of all {A.meta['stages']['weights']['neurons']:,} MLP neurons has been built from the weights"
             + (f" and checked against a forward pass ({', '.join(c['verdict'] for c in v['checks'])})." if v else " (not yet checked against a forward pass).")]
    lines += [f"- L{s['from']}–L{s['to']}: {s['desc']}" for s in lt["stages"]]
    lines += ["Use the `atlas` tool before choosing what to edit: action=search finds the neurons that carry a word or concept, "
              "action=attribute shows which neurons push a given answer in a given prompt (a few neurons chosen that way usually "
              "beat many chosen by signature), action=unit explains one neuron. In scripts, select(...) returns such neurons, "
              "scale_neurons(L, [...], f) scales them, and shift_tokens(tokens, nats) moves a token's score everywhere.",
              "Scaling a neuron changes how strongly it does what it already does; it cannot make it fire in a context where it is silent."]
    return lines


def _count(ws, addr: str) -> int:
    from . import addressing as A
    try:
        return A.count(ws, ws.resolve(addr))
    except Exception:
        return 0


def _first_line(script: str) -> str:
    for line in script.splitlines():
        line = line.strip()
        if line:
            return line.lstrip("# ")[:100]
    return "edit"


def _case_line(c: dict) -> str:
    if c["type"] == "next_token":
        return f"{c['prompt']!r} -> next token {c['expect']!r}" + (f" (top {c['top_k']})" if c.get("top_k", 1) > 1 else "")
    if c["type"] == "not_token":
        return f"{c['prompt']!r} -> next token must not be {c['forbid']!r}"
    if c["type"] == "logit_diff":
        return f"{c['prompt']!r} -> logit({c['a']!r}) - logit({c['b']!r}) >= {c.get('min', 0)}"
    if c["type"] == "completion":
        cond = ", ".join(f"{k} {c[k]!r}" for k in ("contains", "not_contains", "equals", "matches") if k in c)
        return f"{c['prompt']!r} -> completion {cond}"
    return f"{c['prompt']!r} -> stay within KL {c.get('max_kl', 0.05)} of the base model"
