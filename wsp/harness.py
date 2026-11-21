"""Programmer harnesses: native Claude SDK/Codex conversations; legacy OMP/custom runs.

prepare() writes the shared brief and WSP MCP configuration. Native providers retain
an open connection for follow-up turns and approvals. The underlying harness owns
its agent loop and authentication. See native_harness.py and bridge/.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

from . import native_harness

HARNESSES = {
    "claude": "Claude Code (native conversation)",
    "codex": "Codex (native conversation)",
    "omp": "oh-my-pi (omp -p)",
    "custom": "custom command",
}


def prepare(sess, url: str, mode: str = "tools", extra: str = "") -> dict:
    """Write BRIEF.md and .mcp.json into the workspace's agent directory."""
    d = os.path.join(sess.ws.dir, "agent")
    os.makedirs(d, exist_ok=True)
    brief = sess.brief(mode)
    if extra.strip():
        brief += "\n\n## Additional instructions\n" + extra.strip()
    with open(os.path.join(d, "BRIEF.md"), "w") as f:
        f.write(brief)
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server = {"command": sys.executable, "args": ["-m", "wsp.mcp", "--url", url], "env": {"PYTHONPATH": pkg_root}}
    cfg_path = os.path.join(d, ".mcp.json")
    with open(cfg_path, "w") as f:
        json.dump({"mcpServers": {"wsp": server}}, f, indent=1)
    return {"dir": d, "brief": brief, "mcp_config": cfg_path, "server": server}


def command(harness: str, prep: dict, url: str, mode: str = "tools", model: str = "", budget: float = 0.0,
            custom: str = "") -> tuple[list[str], str | None]:
    """(argv, stdin text) for a headless run."""
    brief = prep["brief"]
    if harness in ("claude", "codex"):
        return [shutil.which("node") or "node", str(native_harness.BRIDGE / "index.mjs")], None
    if harness == "omp":
        cmd = ["omp", "-p", brief]
        if model:
            cmd += ["--model", model]
        return cmd, None
    if harness == "custom":
        if not custom.strip():
            raise ValueError("give a command for the custom harness")
        brief_path = os.path.join(prep["dir"], "BRIEF.md")
        sub = {"brief": brief_path, "mcp_config": prep["mcp_config"], "url": url, "dir": prep["dir"]}
        return [a.format(**sub) for a in shlex.split(custom)], brief
    raise ValueError(f"unknown harness {harness!r}; choose from {list(HARNESSES)}")


def resume(sess, url: str, conv, text: str = "", billing: str = "per_call"):
    """Resume a saved native conversation (see native_harness.resume); BRIEF.md and .mcp.json are written afresh."""
    with sess.lock:
        native_harness.assert_available(sess)
        prep = prepare(sess, url, conv.meta.get("mode") or "tools", conv.meta.get("extra") or "")
        return native_harness.resume(sess, prep, conv, text, billing)


def available() -> list[dict]:
    out = []
    for name, label in HARNESSES.items():
        error = native_harness.setup_error(name) if name in ("claude", "codex") else None
        out.append({"name": name, "label": label, "installed": not error if name in ("claude", "codex") else
                    name == "custom" or shutil.which(name) is not None, "setup_error": error,
                    "native": name in ("claude", "codex")})
    return out


def _clip(s, n: int = 6000) -> str:
    s = s if isinstance(s, str) else json.dumps(s)
    return s if len(s) <= n else s[:n] + f"… [{len(s) - n} more characters]"


def normalize(harness: str, line: str) -> list[dict]:
    """Harness output line -> IDE events: init | text | thinking | tool_use | tool_result | result | raw"""
    line = line.rstrip("\n")
    if not line.strip():
        return []
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        return [{"kind": "raw", "text": _clip(line, 2000)}]
    if not isinstance(ev, dict):
        return [{"kind": "raw", "text": _clip(line, 2000)}]
    t = ev.get("type")
    out: list[dict] = []
    if harness == "claude" or t in ("assistant", "user", "system", "result"):
        if t == "system" and ev.get("subtype") == "init":
            out.append({"kind": "init", "model": ev.get("model"), "session": ev.get("session_id"),
                        "mcp": ev.get("mcp_servers"), "n_tools": len(ev.get("tools") or [])})
        elif t == "assistant":
            for b in (ev.get("message") or {}).get("content") or []:
                if b.get("type") == "text" and b.get("text"):
                    out.append({"kind": "text", "text": b["text"]})
                elif b.get("type") == "thinking" and b.get("thinking"):
                    out.append({"kind": "thinking", "text": _clip(b["thinking"], 3000)})
                elif b.get("type") == "tool_use":
                    out.append({"kind": "tool_use", "id": b.get("id"), "name": (b.get("name") or "").replace("mcp__wsp__", ""),
                                "input": b.get("input")})
        elif t == "user":
            content = (ev.get("message") or {}).get("content")
            for b in content if isinstance(content, list) else []:
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    text = "\n".join(x.get("text", "") for x in c if isinstance(x, dict)) if isinstance(c, list) else str(c)
                    out.append({"kind": "tool_result", "id": b.get("tool_use_id"), "text": _clip(text), "error": bool(b.get("is_error"))})
        elif t == "result":
            u = ev.get("usage") or {}
            out.append({"kind": "result", "ok": not ev.get("is_error"), "text": ev.get("result") or "",
                        "cost_usd": ev.get("total_cost_usd"), "turns": ev.get("num_turns"), "duration_ms": ev.get("duration_ms"),
                        "input_tokens": (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0)
                        + (u.get("cache_creation_input_tokens") or 0), "output_tokens": u.get("output_tokens")})
        return out
    if harness == "codex":
        item = ev.get("item") or {}
        it = item.get("type") or item.get("item_type")
        if t == "item.completed" and it in ("agent_message", "assistant_message"):
            out.append({"kind": "text", "text": item.get("text") or ""})
        elif t == "item.completed" and it == "reasoning":
            out.append({"kind": "thinking", "text": _clip(item.get("text") or "", 3000)})
        elif it == "mcp_tool_call" and t == "item.started":
            out.append({"kind": "tool_use", "id": item.get("id"), "name": item.get("tool"), "input": item.get("arguments")})
        elif it == "mcp_tool_call" and t == "item.completed":
            out.append({"kind": "tool_result", "id": item.get("id"), "text": _clip(item.get("result") or item.get("error") or "")})
        elif t == "turn.completed":
            u = ev.get("usage") or {}
            out.append({"kind": "result", "ok": True, "text": "", "input_tokens": u.get("input_tokens"),
                        "output_tokens": u.get("output_tokens")})
        elif t in ("turn.failed", "error"):
            out.append({"kind": "result", "ok": False, "text": _clip(ev.get("message") or ev.get("error") or line, 2000)})
        if out:
            return out
    return [{"kind": "raw", "text": _clip(line, 2000)}]


class Run:
    def __init__(self, rid: str, harness: str, cmd: list[str], cwd: str):
        self.id = rid
        self.harness = harness
        self.cmd = cmd
        self.cwd = cwd
        self.status = "starting"
        self.events: list[dict] = []
        self.started = time.time()
        self.ended: float | None = None
        self.proc: subprocess.Popen | None = None
        self.patches_before = 0
        self.conversation = None          # the saved record (wsp/conversations.py); legacy runs can be read, not resumed
        self.conversation_id = None
        self.workspace = None             # the workspace it runs in (realpath), as a native run's meta has it

    def meta(self) -> dict:
        return {"id": self.id, "harness": self.harness, "status": self.status, "started": self.started, "ended": self.ended,
                "n_events": len(self.events), "command": " ".join(shlex.quote(c) if len(c) < 200 else "'<brief>'" for c in self.cmd),
                "conversation": self.conversation_id, "workspace": self.workspace}

    def save(self, write) -> None:
        if self.conversation is not None:
            try:
                write(self.conversation)
            except (OSError, ValueError):
                pass                      # a legacy run's record is best effort; the run itself goes on

    def release(self) -> None:
        if self.conversation is not None:
            self.conversation.close()


def start(sess, url: str, harness: str = "claude", mode: str = "tools", model: str = "", budget: float = 0.0,
          extra: str = "", custom: str = "", billing: str = "per_call") -> Run:
    """`billing` ("subscription" or "per_call") is how the native provider's sign-in pays; see native_harness.billing_of."""
    with sess.lock:
        native_harness.assert_available(sess)
        prep = prepare(sess, url, mode, extra)
        if harness in ("claude", "codex"):
            return native_harness.start(sess, prep, harness, mode, model, budget, billing, extra)
        cmd, stdin_text = command(harness, prep, url, mode, model, budget, custom)
        if shutil.which(cmd[0]) is None:
            raise FileNotFoundError(f"`{cmd[0]}` is not installed. The working directory is ready at {prep['dir']}: "
                                    f"run your harness there with BRIEF.md as the prompt (.mcp.json declares the wsp server).")
        run = Run(f"run-{uuid.uuid4().hex[:12]}", harness, cmd, prep["dir"])
        run.patches_before = len(sess.ws.patches)
        run.workspace = os.path.realpath(sess.ws.dir)
        # The saved record keeps the harness kind only: a custom command line can carry tokens or paths, so it is
        # not written to disk (the live run still shows it).
        run.conversation, problem = native_harness.new_conversation(sess, harness, mode, model, run.id, False, extra=extra)
        run.conversation_id = run.conversation.id if run.conversation is not None else None
        sess.runs[run.id] = run

    lock = threading.Lock()

    def emit(ev: dict):
        with lock:
            ev = {"i": len(run.events), "t": time.time(), **ev}
            run.events.append(ev)
            run.save(lambda conv: conv.append(ev, sync=ev.get("kind") == "result"))
        sess.publish({"type": "agent", "run": run.id, "event": ev})

    if problem:
        emit({"kind": "raw", "text": problem})

    def pump():
        try:
            run.proc = subprocess.Popen(cmd, cwd=prep["dir"], stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                        start_new_session=True)
            run.save(lambda conv: conv.update(status="running"))
            run.status = "running"
            sess.publish({"type": "run", "run": run.meta()})
            if stdin_text is not None:
                run.proc.stdin.write(stdin_text)
                run.proc.stdin.close()
            for line in run.proc.stdout:
                for ev in normalize(harness, line):
                    emit(ev)
            code = run.proc.wait()
            final = run.status if run.status == "stopped" else "done" if code == 0 else "failed"
            if code != 0 and final == "failed":
                emit({"kind": "raw", "text": f"harness exited with code {code}"})
        except Exception as e:
            final = "failed"
            emit({"kind": "raw", "text": f"{type(e).__name__}: {e}"})
        # The record gets its final status before the run stops counting as live (busy() reads run.status), so a
        # listing in between never mistakes the finished run for one cut off by a server stop.
        ended = time.time()
        run.save(lambda conv: conv.update(status=final, ended=ended))
        run.release()
        run.ended, run.status = ended, final
        sess.publish({"type": "run", "run": run.meta()})

    try:
        threading.Thread(target=pump, daemon=True).start()
    except BaseException:
        sess.runs.pop(run.id, None)          # the pump never ran: give the record back
        run.release()
        raise
    return run


def stop(run: Run) -> None:
    if getattr(run, "native", False):
        return run.interrupt()
    if run.proc and run.proc.poll() is None:
        run.status = "stopped"
        try:
            os.killpg(os.getpgid(run.proc.pid), signal.SIGINT)
            time.sleep(1.5)
            if run.proc.poll() is None:
                os.killpg(os.getpgid(run.proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
