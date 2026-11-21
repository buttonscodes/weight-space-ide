"""Model Context Protocol server for the programmer model's harness.

    python -m wsp.mcp --url http://127.0.0.1:8765

is a stdio MCP server (newline-delimited JSON-RPC 2.0) that any MCP-capable harness can
launch: Claude Code, Codex CLI, oh-my-pi, Cursor, ...  It holds no model itself; every
tools/list and tools/call is forwarded to the running IDE over HTTP, so the harness and the
person at the IDE work on the same weights and every edit shows up live in the browser.

This file uses only the standard library, so it starts instantly and has nothing to install
in the harness's environment.  The IDE also serves the same protocol over HTTP at POST /mcp
for harnesses that prefer a remote server.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

SERVER_INFO = {"name": "wsp", "title": "Weight-space programming IDE", "version": "0.1.0"}
VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
INSTRUCTIONS = ("Tools for reading, analysing and writing the weights of the model open in the weight-space programming "
                "IDE. Call `overview` first. Edits are weight scripts applied with `edit`; every edit is a revertable patch.")


class HttpBackend:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def _req(self, path: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())

    def tools(self) -> list[dict]:
        return self._req("/api/agent/tools")["tools"]

    def call(self, name: str, args: dict) -> tuple[str, bool]:
        r = self._req("/api/agent/call", {"name": name, "arguments": args})
        return r["text"], bool(r.get("isError"))


class LocalBackend:
    def __init__(self, sess):
        self.sess = sess

    def tools(self) -> list[dict]:
        from .tools import specs
        return specs()

    def call(self, name: str, args: dict) -> tuple[str, bool]:
        return call_tool(self.sess, name, args)


def call_tool(sess, name: str, args: dict) -> tuple[str, bool]:
    """Run a tool; failures come back as text with isError so the model can correct itself."""
    from .tools import call
    from .runtime_events import RuntimeRefusal
    try:
        sess.publish({"type": "tool", "name": name, "arguments": args})
        return call(sess, name, args), False
    except RuntimeRefusal as exc:
        if getattr(exc,'committed',None):return json.dumps(exc.report),False
        return json.dumps(dict(status="not_run",event=exc.record,report=getattr(exc,"report",None))), True
    except TypeError as e:
        return f"bad arguments for {name}: {e}", True
    except Exception as e:
        return f"{type(e).__name__}: {e}", True


def handle(backend, msg: dict) -> dict | None:
    """One JSON-RPC message in, one response out (None for notifications)."""
    mid, method, params = msg.get("id"), msg.get("method", ""), msg.get("params") or {}

    def ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def err(code, message):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}

    if mid is None:                       # notification (initialized, cancelled, ...)
        return None
    try:
        if method == "initialize":
            want = params.get("protocolVersion")
            return ok({"protocolVersion": want if want in VERSIONS else VERSIONS[0],
                       "capabilities": {"tools": {"listChanged": False}}, "serverInfo": SERVER_INFO,
                       "instructions": INSTRUCTIONS})
        if method == "ping":
            return ok({})
        if method == "tools/list":
            return ok({"tools": backend.tools()})
        if method == "tools/call":
            name = params.get("name", "")
            if name not in {t["name"] for t in backend.tools()}:
                return err(-32602, f"Unknown tool: {name}")
            text, is_error = backend.call(name, params.get("arguments") or {})
            return ok({"content": [{"type": "text", "text": text}], "isError": is_error})
        if method in ("resources/list", "prompts/list", "resources/templates/list"):
            return ok({method.split("/")[0]: []} if "templates" not in method else {"resourceTemplates": []})
        return err(-32601, f"Method not found: {method}")
    except urllib.error.URLError as e:
        if method == "tools/call":
            return ok({"content": [{"type": "text", "text": f"The IDE is not reachable ({e}). Is `wsp serve` running?"}],
                       "isError": True})
        return err(-32603, f"IDE not reachable: {e}")
    except Exception as e:
        return err(-32603, f"{type(e).__name__}: {e}")


def serve_stdio(url: str) -> None:
    backend = HttpBackend(url)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}) + "\n")
            sys.stdout.flush()
            continue
        for m in (msg if isinstance(msg, list) else [msg]):
            resp = handle(backend, m)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    import argparse
    import os
    ap = argparse.ArgumentParser(description="stdio MCP server that forwards to a running weight-space programming IDE")
    ap.add_argument("--url", default=os.environ.get("WSP_URL", "http://127.0.0.1:8765"))
    serve_stdio(ap.parse_args().url)
