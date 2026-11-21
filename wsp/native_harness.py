"""Persistent local agent bridge. No model loop or credentials live in this module.

Each conversation is also kept on disk (wsp/conversations.py) so that it can be read, and resumed with the
provider's own session, after a server restart.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import uuid

from . import conversations

BRIDGE = Path(__file__).resolve().parents[1] / "bridge"
ACTIVE = {"starting", "running", "waiting", "idle", "interrupting", "closing"}
FINAL_RUN = {"closed", "failed", "done", "stopped", "interrupted"}
# Provider diagnostics saved with a failed resume: the last line only, short, and with anything that looks like
# a credential replaced (stderr can carry API keys, bearer tokens or account details).
SECRET = re.compile(r"(?i)\bbearer\s+\S+|\bsk-[A-Za-z0-9_\-]{6,}|[A-Za-z0-9+/=_]{32,}")


def redact(text: str, limit: int = 300) -> str:
    lines = [l for l in str(text).splitlines() if l.strip()]
    last = lines[-1].strip() if lines else ""
    last = SECRET.sub("[redacted]", last)
    return last if len(last) <= limit else last[:limit] + "…"

# How a sign-in pays for requests, from the method the provider's own status reports (bridge/auth.mjs).
# A claude.ai login, a Claude Code OAuth token and a ChatGPT login use the subscription; an API key, a
# gateway token, a cloud provider (Bedrock, Vertex, Foundry, GCP) or a method the IDE does not recognise
# may be billed per call, so the dollar cost and the Claude budget stay available for those.
SUBSCRIPTION = {"claude": {"claude.ai", "oauth_token"}, "codex": {"chatgpt"}}


def billing_of(provider: str, method) -> str:
    return "subscription" if method in SUBSCRIPTION.get(provider, ()) else "per_call"


def setup_error(provider: str) -> str | None:
    if not shutil.which(provider):
        return f"Install {provider} and sign in with its CLI first."
    if not shutil.which("node"):
        return "Install Node.js 20 or newer for native agent conversations."
    if provider == "claude" and not (BRIDGE / "node_modules/@anthropic-ai/claude-agent-sdk/package.json").is_file():
        return "Install the Claude bridge dependency: cd weight-space-ide/bridge && npm ci"
    return None


def busy(r) -> bool:
    return (r.status in ACTIVE or (getattr(r, "native", False) and not r._retired.is_set())
            or (getattr(r, "proc", None) is not None and r.proc.poll() is None))


def wait_finishing(sess, timeout: float = 5.0) -> None:
    """Wait (briefly) for runs that have reached a final status to retire, so that a reader never sees the step
    between "failed/closed" and "retired" (the bridge's process group is still being stopped there)."""
    end = time.monotonic() + timeout
    for r in list(sess.runs.values()):
        retired = getattr(r, "_retired", None)
        if getattr(r, "native", False) and retired is not None and r.status in FINAL_RUN and not retired.is_set():
            retired.wait(max(0.0, end - time.monotonic()))


def assert_available(sess) -> None:
    wait_finishing(sess)
    if any(busy(r) for r in sess.runs.values()):
        raise RuntimeError("Close the current programmer conversation before starting another or changing workspace/task.")


def live_conversations(sess) -> set[str]:
    """Ids of the saved conversations a run in this server still runs. A run whose status is final no longer
    counts, even in the moment before its thread has retired."""
    return {r.conversation_id for r in list(sess.runs.values())
            if getattr(r, "conversation_id", None) and busy(r) and r.status not in FINAL_RUN}


def new_conversation(sess, provider, mode, model, run_id, native, **fields):  # -> (Conversation | None, problem | None)
    """A saved record for a new run, or (None, reason) when it cannot be written; the run goes ahead either way."""
    ws = getattr(sess, "ws", None)
    if ws is None or not getattr(ws, "dir", None):
        return None, None
    try:
        return conversations.Conversation.create(ws.dir, live_ids=live_conversations(sess), provider=provider, mode=mode,
                                                 model=model or "", run=run_id, native=native,
                                                 task=dict(getattr(sess, "task", None) or {}), **fields), None
    except Exception as exc:                       # a damaged store never stops Start
        return None, f"This conversation is not being saved ({type(exc).__name__}: {exc}); it will not survive a server restart."


class NativeRun:
    native = True

    def __init__(self, sess, provider, prep, mode, model, budget, billing="per_call", conversation=None, history=None,
                 resume=None, first=None):
        """`conversation` is the saved record events go to; a resume also gives its earlier `history`, the provider
        session id to continue (`resume`) and the `first` message to send instead of the brief."""
        self.id = f"run-{uuid.uuid4().hex[:12]}"
        self.harness = provider
        self.cwd = prep["dir"]
        self.workspace = os.path.realpath(sess.ws.dir)
        self.status = "starting"
        self.events = list(history or [])
        self.conversation = conversation
        self.conversation_id = conversation.id if conversation is not None else None
        self.resume = resume
        # Until the provider confirms the session, a resume changes nothing in the saved record but its owner: its
        # events are held in _unsaved, and if the attempt fails only one resume_failed event is saved.
        self._resume_confirmed = resume is None
        self._unsaved = []
        self._failure_saved = False
        self._failure = None                       # the event that ended an unconfirmed resume
        self._provider_text = None                 # the last provider diagnostic (stderr) of an unconfirmed resume
        self._turn_open = False
        self._save_error = None
        self._discard = False
        self._finished = False
        self.pending = {}
        self.session_id = None
        self.started = time.time()
        self.ended = None
        self.last_result = None
        self.billing = billing
        self.proc = None
        self._sess = sess
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._calls = {}
        self._closing = threading.Event()
        self._ready = threading.Event()
        self._retired = threading.Event()
        self._config = dict(provider=provider, cwd=self.cwd, executable=shutil.which(provider),
                            server=prep["server"], brief=first if first is not None else prep["brief"], mode=mode,
                            model=model, budget=budget, **({"resume": resume} if resume else {}))
        self.cmd = [shutil.which("node"), str(BRIDGE / "index.mjs")]
        self._startup_timer = threading.Timer(45, self._startup_timeout)
        self._startup_timer.daemon = True

    def meta(self):
        with self._lock:
            return dict(id=self.id, harness=self.harness, native=True, status=self.status,
                        started=self.started, ended=self.ended, n_events=len(self.events),
                        session=self.session_id, pending=list(self.pending.values()), last_result=self.last_result,
                        workspace=self.workspace, mode=self._config["mode"], model=self._config["model"],
                        billing=self.billing, budget=self._config["budget"] or None,
                        conversation=self.conversation_id, resumed=bool(self.resume))

    def _save(self, write):
        """Run one write to the saved conversation; a failure is reported once and never stops the run.
        Once the run has finished (and released the record) nothing more is written."""
        if self.conversation is None or self._finished:
            return None
        try:
            write(self.conversation)
        except (OSError, ValueError) as exc:
            if self._save_error is None:
                self._save_error = f"Could not save this conversation ({type(exc).__name__}: {exc}); later events may be missing after a restart."
                return self._save_error
        return None

    def _record(self, status):
        """The saved status. A run ended while a turn was in progress is recorded as interrupted."""
        if status in ("starting", "running", "waiting", "interrupting"):
            self._turn_open = True
        elif status == "idle":
            self._turn_open = False
        if not self._resume_confirmed:
            return None                         # a resume the provider has not confirmed leaves the record as it was
        fields = {"status": status, "session": self.session_id, "billing": self.billing, "run": self.id}
        if status in ("closed", "failed"):
            fields["ended"] = self.ended or time.time()
            if status == "closed" and self._turn_open:
                fields.update(status="interrupted", interrupted="The conversation was closed while a turn was running.")
        return self._save(lambda conv: conv.update(**fields))

    def _attempt_failed(self, event=None):
        """Save the end of an unconfirmed resume: one resume_failed event and the reason in the record. Only an
        answer that the session does not exist (`gone`) marks the conversation as not resumable; anything else
        (a timeout, an error, a different session, an ambiguous check) is a failed attempt that can be retried.
        Called with self._lock held, before the failed status is published."""
        if self._resume_confirmed or self._failure_saved or self.conversation is None:
            return
        self._failure_saved = True
        event = event or self._failure or {"kind": "resume_failed", "t": time.time(),
                                           "text": "The conversation ended before the provider confirmed the session."}
        text = str(event.get("text") or "The resume did not complete.")
        if self._provider_text and self._provider_text not in text:
            text += f" Provider output: {self._provider_text}"
        gone = event.get("kind") == "resume_failed" and event.get("gone") is True
        # `session` is the session this attempt tried to continue (never a different one a provider started): an
        # `init` of that session just before it stays trusted (conversations.session_from_events).
        saved = {"kind": "resume_failed", "i": event.get("i"), "t": event.get("t") or time.time(), "text": text,
                 "gone": gone, "run": self.id, "session": self.resume}
        def write(conv):
            conv.append(saved, sync=True)
            if gone:
                conv.update(resume_error=text, resume_error_at=saved["t"])
            else:
                conv.update(last_resume_failure=text, last_resume_failure_at=saved["t"])
        self._unsaved = []
        self._save(write)

    def _state(self, status):
        with self._lock:
            if self._finished or (self.status in ("closed", "failed") and status != "failed"):
                return
            if status in ("failed", "closed"):
                self.ended = time.time()
                self.pending.clear()
                if self.resume and not self._resume_confirmed:
                    self._attempt_failed()
            # The record first, then the status other threads read: whoever sees the new status finds it saved.
            problem = self._record(status)
            self.status = status
            snapshot = self.meta()
        self._sess.publish({"type": "run", "run": snapshot})
        if problem:
            self._emit({"kind": "raw", "text": problem})

    def _finish(self):
        """The run is over: record its final state, release the saved record, retire, and publish that last.
        Everything after this point is live only."""
        problem = None
        with self._lock:
            if self._finished:
                return
            if self.status not in ("closed", "failed"):
                self.ended = time.time()
                self.pending.clear()
                problem = self._record("closed")
                self.status = "closed"
            if self.resume and not self._resume_confirmed:
                self._attempt_failed()
            self._finished = True
            self._retired.set()
            if self.conversation is not None:
                self.conversation.close()
            snapshot = self.meta()
        self._sess.publish({"type": "run", "run": snapshot})
        if problem:
            self._emit({"kind": "raw", "text": problem})

    def _emit(self, event):
        notice = None
        wrong_session = None
        with self._lock:
            if self._discard:
                return                              # a session that was not the resumed one: nothing more is shown
            event = {**event, "i": len(self.events), "t": time.time()}
            kind = event.get("kind")
            if kind == "init" and self.resume and event.get("session") != self.resume:
                # Never let a fresh session pass for the resumed one: this run is ended, the record keeps its id.
                wrong_session = event.get("session")
                self._discard = True
                event = {"kind": "resume_failed", "i": event["i"], "t": event["t"], "gone": False,
                         "text": f"{LABEL.get(self.harness, self.harness)} started a different session ({wrong_session}) "
                                 f"instead of resuming {self.resume}; it was stopped and closed. {MISMATCH_SENT}"}
                kind = "resume_failed"
            if kind == "init" and self.billing == "subscription" and event.get("api_key_source") not in (None, "none"):
                # The provider says it is using an API key after all: those calls are billed, so show their cost.
                self.billing = "per_call"
                notice = (f"This conversation authenticates with {event['api_key_source']}, so its requests are billed per "
                          "call; no dollar budget was set for it. Close it and sign in with your subscription to avoid that.")
            if self.billing == "subscription":
                event.pop("cost_usd", None)          # subscription runs have no per-call cost to show
            # Retain wire events for browser reconnect; the view coalesces deltas.
            self.events.append(event)
            if self.resume and not self._resume_confirmed:
                if kind == "init":
                    pass                             # saved below, with the events held until now
                elif kind == "raw" and event.get("text"):
                    self._provider_text = redact(event["text"])
                if kind in ("resume_failed", "error") and self._failure is None:
                    self._failure = event
                if kind == "resume_failed":
                    self._attempt_failed(event)
                elif kind != "init":
                    self._unsaved.append(event)      # saved only once the provider confirms the session
            else:
                problem = self._save(lambda conv: conv.append(event, sync=kind in ("init", "user", "result", "error", "resume_failed", "resumed")))
                if problem:
                    notice = notice or problem
            if kind == "init":
                self.session_id = event.get("session")
                self._startup_timer.cancel()
                if self.resume and not self._resume_confirmed:
                    self._resume_confirmed = True
                    held, self._unsaved = self._unsaved, []
                    def confirm(conv):
                        for e in held:
                            conv.append(e, sync=e.get("kind") in ("user", "resumed"))
                        conv.append(event, sync=True)
                        conv.update(status=self.status, session=self.session_id, run=self.id, pid=os.getpid(),
                                    boot=conversations.BOOT, resume_error=None, resume_error_at=None,
                                    last_resume_failure=None, last_resume_failure_at=None)
                    notice = self._save(confirm) or notice
                elif (self.conversation is not None and isinstance(self.session_id, str) and self.session_id
                      and (self.conversation.meta.get("session") != self.session_id
                           or self.conversation.meta.get("billing") != self.billing)):
                    # Saved now (meta.json is synced), not at the next status change, which for a first turn is its
                    # end: a server stop during that turn must leave a record that names its session, or it could
                    # never be resumed. The saved status is not touched; the counts are those of this moment.
                    session, billing = self.session_id, self.billing
                    notice = self._save(lambda conv: conv.update(session=session, billing=billing)) or notice
            elif kind == "prompt":
                self.pending[event["request_id"]] = event
            elif kind == "prompt_resolved":
                self.pending.pop(event["request_id"], None)
            elif kind == "result":
                self.last_result = event.get("ok")
        self._sess.publish({"type": "agent", "run": self.id, "event": event})
        if kind in ("prompt", "prompt_resolved", "init"):
            self._sess.publish({"type": "run", "run": self.meta()})
        if notice:
            self._emit({"kind": "raw", "text": notice})
        if wrong_session is not None:
            self._state("failed")
            threading.Thread(target=self.close, daemon=True).start()

    def start(self):
        self._startup_timer.start()
        threading.Thread(target=self._pump, daemon=True, name=f"agent-{self.id}").start()

    def _startup_timeout(self):
        if not self.session_id and not self._closing.is_set():
            self._emit({"kind": "error", "text": "Agent startup timed out after 45 seconds. Check the CLI login and MCP setup."})
            self._state("failed")
            self.close()

    def _write(self, frame):
        with self._write_lock:
            if not self.proc or self.proc.poll() is not None or not self.proc.stdin:
                raise RuntimeError("Agent connection is closed")
            self.proc.stdin.write(json.dumps(frame) + "\n")
            self.proc.stdin.flush()

    def request(self, method, **params):
        if self._closing.is_set():
            raise RuntimeError("Conversation is closing")
        rid = uuid.uuid4().hex
        call = {"done": threading.Event()}
        with self._lock:
            self._calls[rid] = call
        try:
            self._write({"id": rid, "method": method, "params": params})
            if not call["done"].wait(35):
                # An ambiguous send must never be retried in the same conversation.
                self.close()
                raise RuntimeError("Agent did not acknowledge the command; conversation closed to prevent duplicate sends")
            if call.get("error"):
                raise RuntimeError(call["error"])
            return self.meta()
        finally:
            with self._lock:
                self._calls.pop(rid, None)

    def send(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter a message")
        if len(text) > 100_000:
            raise ValueError("Message exceeds 100,000 characters")
        with self._lock:
            if self.status != "idle" or self.pending:
                raise RuntimeError("Wait for the current turn to finish")
            self.status = "running"  # reserve before another HTTP request can send
        try:
            return self.request("send", text=text)
        except Exception:
            if not self._closing.is_set():
                self._state("idle")
            raise

    def respond(self, request_id, allow, answers=None):
        if not isinstance(allow, bool) or (answers is not None and not isinstance(answers, dict)):
            raise ValueError("Invalid approval response")
        with self._lock:
            prompt = self.pending.get(str(request_id))
            if prompt is None:
                raise ValueError("This request is no longer pending")
            if prompt.get("questions") and allow:
                if not answers or any(not isinstance(answers.get(str(q["id"])), str) or not answers[str(q["id"])].strip()
                                      for q in prompt["questions"]):
                    raise ValueError("Answer every question before sending")
        return self.request("respond", request_id=str(request_id), allow=allow, answers=answers or {})

    def interrupt(self):
        with self._lock:
            if self.status not in ("running", "waiting", "interrupting"):
                return self.meta()
        return self.request("interrupt")

    def _pump(self):
        try:
            self.proc = subprocess.Popen(self.cmd, cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True, bufsize=1, start_new_session=True)
            self._ready.set()
            if self._closing.is_set():
                return
            def stderr():
                for line in self.proc.stderr:
                    # Provider diagnostics stay distinct from protocol data.
                    self._emit({"kind": "raw", "text": line.rstrip()[:4000]})
            reader = threading.Thread(target=stderr, daemon=True)
            reader.start()
            self._write({"method": "start", "params": self._config})
            for line in self.proc.stdout:
                frame = json.loads(line)
                if "event" in frame:
                    self._emit(frame["event"])
                elif "status" in frame:
                    self._state(frame["status"])
                elif "id" in frame:
                    with self._lock:
                        call = self._calls.get(frame["id"])
                        if call:
                            call["error"] = frame.get("error")
                            call["done"].set()
            code = self.proc.wait()
            reader.join(1)                  # the provider's last words arrive before the failure is recorded
            if not self._closing.is_set() and self.status != "failed":
                if self.resume and not self._resume_confirmed:
                    text = (f"Agent bridge exited ({code}) before the provider confirmed the session; the saved "
                            "conversation was not changed, and Resume can be tried again.")
                else:
                    text = f"Agent bridge exited ({code}). Start a new conversation."
                self._emit({"kind": "error", "text": text})
                self._state("failed")
        except Exception as exc:
            if not self._closing.is_set():
                self._emit({"kind": "error", "text": f"{type(exc).__name__}: {exc}"})
                self._state("failed")
        finally:
            self._ready.set()
            self._startup_timer.cancel()
            with self._lock:
                for call in self._calls.values():
                    call["error"] = call.get("error") or "Agent connection closed"
                    call["done"].set()
            self._kill_group()
            self._finish()

    def _kill_group(self):
        if self.proc:
            # Also kill descendants after the bridge exits; its pgid remains stable.
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def close(self):
        self._closing.set()
        self._startup_timer.cancel()
        if self.status != "failed":
            self._state("closing")
        self._ready.wait(2)
        if self.proc and self.proc.poll() is None:
            try:
                self._write({"method": "close"})
                self.proc.wait(timeout=2)
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                pass
        self._kill_group()
        if self._ready.is_set():
            self._finish()                  # the pump has started (or ended): this is the end of the run
        elif self.status != "failed":
            self._state("closed")           # the pump has not started yet; its own end retires the run
        return self.meta()


def start(sess, prep, provider, mode, model, budget, billing="per_call", extra=""):
    error = setup_error(provider)
    if error:
        raise RuntimeError(error)
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("Budget must be a finite nonnegative number")
    if mode not in ("tools", "files"):
        raise ValueError("Access mode must be tools or files")
    if provider == "codex" and budget:
        raise ValueError("Codex has no USD budget cap. Clear the budget before starting.")
    if budget and billing == "subscription":
        raise ValueError("A dollar budget applies only to sign-ins billed per API call; this one uses your subscription. "
                         "Clear the budget before starting.")
    with sess.lock:
        assert_available(sess)
        run = NativeRun(sess, provider, prep, mode, model, budget, billing)
        run.conversation, problem = new_conversation(sess, provider, mode, model, run.id, True, extra=extra,
                                                     billing=billing, budget=budget or None)
        run.conversation_id = run.conversation.id if run.conversation is not None else None
        sess.runs[run.id] = run
        try:
            if problem:
                run._emit({"kind": "raw", "text": problem})
            run.start()
        except BaseException:
            run._startup_timer.cancel()
            sess.runs.pop(run.id, None)
            if run.conversation is not None:
                run.conversation.close()
            raise
        return run


# ------------------------------------------------------------------ resuming a saved conversation
TASK_KEYS = ("instructions", "scope", "suite", "preserve")
# What a resume tells the provider before the session can be confirmed (see bridge/adapters.mjs for Claude).
MISMATCH_SENT = ("The resume message (the task title, the current patch list and your optional message) may have "
                 "been sent to that new session before it was stopped. The resumed session itself was not changed.")
LABEL = {"claude": "Claude Code", "codex": "Codex", "omp": "oh-my-pi", "custom": "custom command"}


def task_changes(old: dict | None, new: dict | None) -> list[str]:
    """The task fields that differ in a way that changes the work: everything but the title."""
    old, new = old or {}, new or {}
    norm = lambda v: json.dumps(v, sort_keys=True, default=str)
    return [k for k in TASK_KEYS if norm(old.get(k)) != norm(new.get(k))]


def resume_refusal(sess, meta: dict, live: set[str], retry: bool = False) -> tuple[str, str] | None:
    """(code, reason) when this saved conversation cannot be resumed now; None when it can. `retry` is the
    person's "Try again" of a conversation recorded as gone."""
    if meta.get("unreadable"):
        return "unreadable", f"This saved conversation cannot be read ({meta['unreadable']})."
    provider = meta.get("provider")
    label = LABEL.get(provider, str(provider))
    if meta.get("id") in live:
        return "open", "This is the conversation running now."
    if meta.get("elsewhere"):
        owner = f" (pid {meta['pid']})" if meta.get("pid") else ""
        if meta.get("holder") == "pid":
            return "elsewhere", (f"The record says another IDE process on this machine{owner} is running this "
                                 "conversation, and this filesystem has no file locks to check that. If that process "
                                 "is not running it, mark it interrupted.")
        return "elsewhere", (f"Another IDE process on this machine{owner} has this conversation open. Close it in "
                             "that IDE window, or stop that process, then resume it here.")
    if not meta.get("native") or provider not in ("claude", "codex"):
        return "legacy", (f"This was a {label} run. Those run once and keep no session the IDE could continue. "
                          "Start a new conversation.")
    if not meta.get("session"):
        return "no_session", (f"No {label} session id was recorded (the provider never confirmed the conversation started), "
                              "so there is nothing to resume. Start a new conversation.")
    if meta.get("resume_error") and not retry:
        return "gone", f"{meta['resume_error']} Start a new conversation, or try again."
    here = os.path.realpath(sess.ws.dir)
    if meta.get("workspace") and meta["workspace"] != here:
        return "workspace", (f"This conversation ran in {meta['workspace']}; the workspace is now at {here}, where {label} "
                             "keeps no session for it. Start a new conversation.")
    changed = task_changes(meta.get("task"), getattr(sess, "task", None))
    if changed:
        return "task", (f"The task changed since this conversation started ({', '.join(changed)}). Resuming would continue "
                        "work on a different task: start a new conversation, or set the task back.")
    error = setup_error(provider)
    if error:
        return "setup", error
    try:
        assert_available(sess)
    except RuntimeError:
        return "busy", "Close the current programmer conversation before resuming another."
    return None


def resume_notice(meta: dict, budget: float = 0.0) -> str:
    text = (f"Resuming the provider's own session {meta.get('session')}. The workspace may have changed since this "
            "conversation last ran (weights, patches, tests); the model is told so and given the current patch list.")
    if budget:
        text += (f" The ${budget:g} budget starts again from zero for this resumed conversation: what the conversation "
                 "spent before the resume is not counted against it.")
    return text


def resume_message(sess, meta: dict, text: str = "", limit: int = 200) -> str:
    """The first message of a resumed conversation: what happened, that the workspace may have changed, the patches."""
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(meta.get("ended") or meta.get("started") or time.time()))
    lines = ["[Weight-space IDE] This conversation is being resumed."]
    if meta.get("status") == "interrupted":
        lines.append(f"It was cut off at {when} while a turn was running. Tool calls that were still running then did not "
                     "complete and their results were never delivered.")
    else:
        lines.append(f"It last ran at {when}.")
    lines.append("The workspace may have changed since then: weights, patches, test results and the atlas may differ from "
                 "what you saw. Re-read before relying on earlier observations (call overview first).")
    changed = task_changes(meta.get("task"), getattr(sess, "task", None))
    title = conversations.first_line(getattr(sess, "task", None))
    lines.append(f"The task is unchanged: {title}." if not changed else f"The task changed ({', '.join(changed)}).")
    patches = list(getattr(sess.ws, "patches", None) or [])
    if not patches:
        lines.append("Current patches: none; the weights are the base model.")
    else:
        lines.append(f"Current patches ({len(patches)}):")
        for p in patches[:limit]:
            lines.append(f"- {p.id} [{'on' if p.enabled else 'off'} x{p.strength:g}] {p.author}: {p.message}")
        if len(patches) > limit:
            lines.append(f"- … and {len(patches) - limit} more (call patches to list them)")
    if isinstance(text, str) and text.strip():
        lines += ["", "Message from the person:", text.strip()]
    return "\n".join(lines)


def resume(sess, prep, conv, text="", billing="per_call"):
    """Continue a saved conversation with the provider's own session. The caller has checked resume_refusal.

    A record saved before the session id was written at `init` may have none: the session its own saved events
    confirm is used (conversations.session_from_events). It is not written to the record here; a confirmed resume
    writes it, an unconfirmed one leaves the record as it was."""
    meta = dict(conv.meta)
    history = conv.events()
    if not meta.get("session") and meta.get("native"):
        meta["session"] = conversations.session_from_events(history)
    if not meta.get("session"):
        raise RuntimeError("No provider session id was recorded for this conversation, so there is nothing to resume.")
    provider = meta["provider"]
    error = setup_error(provider)
    if error:
        raise RuntimeError(error)
    if isinstance(text, str) and len(text) > 100_000:
        raise ValueError("Message exceeds 100,000 characters")
    budget = float(meta.get("budget") or 0) if provider == "claude" and billing == "per_call" else 0.0
    with sess.lock:
        assert_available(sess)
        # Take ownership first: another live IDE process holding the record refuses this (conversations.Busy).
        conv.claim()
        run = None
        try:
            first = resume_message(sess, meta, text)
            run = NativeRun(sess, provider, prep, meta.get("mode") or "tools", meta.get("model") or "", budget, billing,
                            conversation=conv, history=history, resume=meta["session"], first=first)
            conv.update(resumes=[*(meta.get("resumes") or []), {"at": time.time(), "run": run.id}])
            sess.runs[run.id] = run
            run._emit({"kind": "resumed", "session": meta["session"], "text": resume_notice(meta, budget)})
            run.start()
        except BaseException:
            # Until the run's own thread owns the record, a failure here gives it back (nothing else would).
            if run is not None:
                run._startup_timer.cancel()
                sess.runs.pop(run.id, None)
            conv.release()
            raise
        return run
