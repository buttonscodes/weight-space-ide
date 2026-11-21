"""The IDE's real web app and routes on a temporary root, with two tiny random models and a FAKE programmer run.

Used by programmer.browser.test.mjs (opt-in; see docs/PROGRAMMER.md) and chat_toggle.browser.test.mjs. No bridge
process, provider, CLI, login or network call: a run is a NativeRun whose events come from a timed script in this
process, published through the session's real event stream and saved by the real conversation store. Sign-in checks
answer "ready (subscription)". HOME and the caches point into the root, so nothing of the person's own configuration
is read.

    python tests/browser/fake_ide.py --root <empty temporary dir> --port N [--autostart]

Control routes for the browser driver (not part of the IDE): POST /fake/prompt (the next step asks a permission
prompt), /fake/question (asks a question), /fake/end (ends the turn); GET /fake/state.
"""
import argparse
import os
from pathlib import Path
import queue
import sys
import threading
import time
import uuid

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--autostart", action="store_true", help="start a conversation in the first workspace at once")
ap.add_argument("--pace", type=float, default=1.0, help="multiply every delay")
args = ap.parse_args()
if not 41000 <= args.port <= 49000 or args.port in (8765, 43879):
    sys.exit("fake_ide: the port must be in 41000-49000 (never an IDE's usual port)")
if not os.path.isdir(args.root):
    sys.exit("fake_ide: --root must be an existing (temporary) directory")

APP_DIR = Path(__file__).resolve().parents[2]          # weight-space-ide/
home = os.path.join(args.root, "home")
os.makedirs(home, exist_ok=True)
for k in ("HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "HF_HOME"):
    os.environ[k] = home
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.pop("WSP_CONFIG", None)
sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR)

from fastapi import Body  # noqa: E402
from wsp import harness, native_harness as native  # noqa: E402
from wsp.model import Config, init_params  # noqa: E402
from wsp.session import Session  # noqa: E402
from wsp.tokenizer import VocabTokenizer  # noqa: E402
from wsp.workspace import Workspace  # noqa: E402

native.setup_error = lambda provider: None          # "installed"; nothing is ever launched
CONTROL = {"gate": queue.Queue(), "runs": []}


class FakeRun(native.NativeRun):
    """A NativeRun whose bridge is a script in this process: the real _emit/_state/_finish do the rest."""

    def start(self):
        self._ready.set()
        self._inbox = queue.Queue()
        self._interrupt = threading.Event()
        self._startup_timer.cancel()
        self._session = self.resume or "fake-session-" + uuid.uuid4().hex[:8]
        CONTROL["runs"].append(self)
        threading.Thread(target=self._script, daemon=True, name=f"fake-{self.id}").start()

    def request(self, method, **params):
        if self._closing.is_set():
            raise RuntimeError("Conversation is closing")
        if method == "interrupt":
            self._interrupt.set()
        else:
            self._inbox.put((method, params))
        return self.meta()

    def _alive(self):
        return not self._closing.is_set() and not self._finished

    def _sleep(self, s):
        end = time.monotonic() + s * args.pace
        while time.monotonic() < end:
            if not self._alive() or self._interrupt.is_set():
                return False
            time.sleep(0.05)
        return True

    def _stream(self, kind, sid, text, piece=6, dt=0.12):
        for k in range(0, len(text), piece):
            if not self._sleep(dt):
                return False
            self._emit({"kind": kind, "id": sid, "text": text[k:k + piece]})
        return True

    def _tool(self, name, inp, secs, result="ok"):
        tid = "toolu_" + uuid.uuid4().hex[:10]
        self._emit({"kind": "tool_use", "id": tid, "name": name, "input": inp})
        ok = self._sleep(secs)
        if ok:
            self._emit({"kind": "tool_result", "id": tid, "text": result})
        return ok

    def _gate(self):
        try:
            return CONTROL["gate"].get_nowait()
        except queue.Empty:
            return None

    def _ask(self, prompt):
        """A prompt the person must answer: the run waits (status waiting) until /api/harness/respond."""
        rid = str(len(self.events) + 1)
        self._emit({"kind": "prompt", "request_id": rid, **prompt})
        self._state("waiting")
        while self._alive():
            try:
                method, params = self._inbox.get(timeout=0.1)
            except queue.Empty:
                if self._interrupt.is_set():
                    self._emit({"kind": "prompt_resolved", "request_id": rid})
                    return None
                continue
            if method == "respond" and params.get("request_id") == rid:
                self._emit({"kind": "prompt_resolved", "request_id": rid})
                self._state("running")
                return params
        return None

    def _turn(self, text, n):
        self._state("running")
        self._emit({"kind": "user", "text": text})
        self._sleep(1)
        self._emit({"kind": "init", "session": self._session, "model": "claude-fake-model", "n_tools": 17,
                    "mcp": [{"name": "wsp", "status": "connected"}], "api_key_source": "none"})
        msg = f"msg_{n}"
        step = 0
        while self._alive() and not self._interrupt.is_set():
            gate = self._gate()
            if gate == "end":
                break
            if gate == "prompt":
                answer = self._ask({"title": "Bash", "input": {"command": "ls -la agent/", "description": "List the agent directory"}})
                if answer is None:
                    break
                if answer.get("allow"):
                    self._tool("Bash", {"command": "ls -la agent/"}, 3, "BRIEF.md\n.mcp.json")
                else:
                    self._emit({"kind": "tool_result", "id": "denied", "text": "Declined by user.", "error": True})
                continue
            if gate == "question":
                answer = self._ask({"title": "AskUserQuestion", "input": {},
                                    "questions": [{"id": "q1", "question": "Which layer should I edit first?",
                                                   "options": [{"label": "L0", "description": "embedding side"}, {"label": "L1"}]}]})
                if answer is None:
                    break
                continue
            # one round: think, say something, call tools (an IDE call alone, then three issued together)
            self._stream("thinking_delta", f"{msg}:{step}a", f"Round {step}: the neurons in layer {step % 2} that read 'c' "
                         "look like they promote 'd'; I should check the card before editing anything.")
            self._stream("text_delta", f"{msg}:{step}b", f"Looking at layer {step % 2}, round {step}.")
            self._tool("mcp__wsp__overview" if step == 0 else "mcp__wsp__inspect",
                       {} if step == 0 else {"what": "card", "address": f"L{step % 2}N{(step * 7) % 40}"}, 4,
                       "card: reads 'c' · writes 'd' · cos 0.12")
            if not self._alive() or self._interrupt.is_set():
                break
            if step % 3 == 1:
                ids = []
                for k in range(3):
                    tid = "toolu_" + uuid.uuid4().hex[:10]
                    ids.append(tid)
                    self._emit({"kind": "tool_use", "id": tid, "name": "mcp__wsp__read_weights",
                                "input": {"selector": f"blocks.{k % 2}.mlp.W_out[{k}]"}})
                for tid in ids:
                    if not self._sleep(3):
                        break
                    self._emit({"kind": "tool_result", "id": tid, "text": "[0.12, -0.03, 0.4, …]"})
            step += 1
        interrupted = self._interrupt.is_set()
        self._interrupt.clear()
        if self._alive():
            self._emit({"kind": "result", "ok": not interrupted, "text": "Interrupted" if interrupted else "",
                        "turns": step + 1, "input_tokens": 12000 + 900 * step, "output_tokens": 400 + 60 * step})
            self._state("idle")

    def _script(self):
        try:
            self._state("idle")                    # the bridge's adapter started, as ClaudeAdapter.start does
            self._turn(self._config["brief"], 0)
            n = 1
            while self._alive():
                try:
                    method, params = self._inbox.get(timeout=0.2)
                except queue.Empty:
                    continue
                if method == "send":
                    self._turn(params.get("text", ""), n)
                    n += 1
        except Exception as exc:                   # a fault in this script shows up in the log, as a bridge's would
            self._emit({"kind": "error", "text": f"fake script: {type(exc).__name__}: {exc}"})
            self._state("failed")
            self._finish()


native.NativeRun = FakeRun

# -------------------------------------------------------------------------------- workspaces, session, app
tok = VocabTokenizer(list("abcdefghijklmnopqrstuvwxyz"), "char")
for name, seed in (("rnd", 1), ("rnd2", 2)):
    if not os.path.isdir(os.path.join(args.root, name)):
        cfg = Config(d_vocab=26, n_ctx=16, d_model=24, n_layers=2, n_heads=2, d_mlp=40, name=name)
        Workspace.create(os.path.join(args.root, name), cfg, tok, init_params(cfg, seed=seed, std=0.3))
sess = Session(args.root)
sess.open("rnd")
url = f"http://127.0.0.1:{args.port}"

from wsp.server import create_app  # noqa: E402
app = create_app(sess, url)
auth = app.state.agent_auth
READY = {"status": "ready", "method": "claude.ai", "billing": "subscription", "message": "Fake sign-in (browser check)."}
auth.status = lambda provider, refresh=False: {"provider": provider, **READY}
auth.check = lambda provider: {"provider": provider, **READY}
auth.ensure_ready = lambda provider: {"provider": provider, **READY}
auth.login = lambda provider: {"provider": provider, **READY}


@app.post("/fake/{what}")
def fake_control(what: str, body: dict = Body(default={})):
    CONTROL["gate"].put(what)
    return {"ok": True}


@app.get("/fake/state")
def fake_state():
    return {"runs": [{"id": r.id, "status": r.status, "events": len(r.events), "conversation": r.conversation_id,
                      "session": r.session_id} for r in CONTROL["runs"]]}


mine = [r for r in app.router.routes if getattr(r, "path", "").startswith("/fake/")]
app.router.routes[:] = mine + [r for r in app.router.routes if r not in mine]   # before the static mount at "/"

if args.autostart:
    def autostart():
        time.sleep(1.5)
        with sess.lock:
            harness.start(sess, url, "claude", "tools", "", 0.0, "", "", "subscription")
    threading.Thread(target=autostart, daemon=True).start()

from wsp.lifecycle import serve  # noqa: E402
print(f"fake IDE listening on {url}", flush=True)
serve(app, host="127.0.0.1", port=args.port, log_level="warning")
