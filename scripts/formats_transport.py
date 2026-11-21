"""E4: what carrying a search trial's candidate to the model worker costs, measured by a real supervised search on a
GGUF model, with the 27B projection beside it (formats build spec 9.1, 9.2). Talks HTTP only, to the server given.

Run with the interpreter that has Optuna, from the IDE's `weight-space-ide` directory, against a supervised serve
the orchestrator started under the outer memory guard on an isolated root, an explicit config and a port that is not
8765. --print-setup prints that config and the serve command and writes nothing:

    python scripts/formats_transport.py --print-setup --iso <ISO/e4> --gguf <model.gguf> --backend cpu|metal \\
        [--cpu-bridge <bridge>] [--metal-bridge <bridge>] [--port 43881]
    python scripts/formats_transport.py --server http://127.0.0.1:43881 --root <ISO/e4/root> \\
        --model <ISO/e4/models/model.gguf> --guard-log <ISO/e4/<model>/watchdog.jsonl> \\
        --out <ISO/e4/<model>/transport.json> [--generation-config <hf snapshot>/generation_config.json]

The measurement: POST /api/open; (with --generation-config) the file copied into the opened workspace's model/
(a local copy, before any model is loaded); POST /api/directions/measure (64 + 64 prompts from the local datasets as
Heretic's qwen05b ARA config takes them, as a chat with Heretic's system prompt, module_io); POST
/api/directions/searches (ARA, lora_rank 50 and Heretic's other ARA defaults; KeywordRate and KLDivergence on 32
prompts each; max_response_length 30; 6 trials, all 6 random startup trials; seed 12345); GET
/api/directions/searches/e4?trials=full until the search is done; then one SIGINT to the supervisor (one Ctrl-C:
its pid is the guard log's preflight pid, checked to be a `wsp serve` process first) and the guard log's teardown
line must say remaining: [] (the one assertion; the exit status is 1 when it fails).

Written to --out, not asserted: every trial's transport and seconds, the search's transport summary, and
  bytes_per_second          = sum(bytes) / sum(encode + send + worker seconds) over the complete trials
  seconds_per_decoded_token = sum(scoring seconds) / sum(response tokens) (the scoring time includes the KL passes)
  the 27B projection: one ARA trial at rank 50 over 32 layers of the 27B model (o/out_proj [6144, 5120], down_proj
    [17408, 5120]: 50 (6144 + 5120) + 50 (17408 + 5120) values per layer, 4 bytes), with Heretic's default scorer
    sizes (100 + 100 prompts, 100 new tokens): jobs = the measured jobs per scored prompt x 200;
    projected_share = T / (T + S), T = bytes_per_job x jobs / bytes_per_second, S = 100 x 100 tokens / rate, at
    5, 10 and 20 tokens/s (the 27B decode rate here is not measured; S leaves the KL passes out)
  the decision of 9.2: above 0.1 measured median share, build E4b; else E4 is ticked with the measured share.
"""
import argparse
import json
import os
import signal
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SHARE_LIMIT = 0.1                      # 9.2 (and directions_search.TRANSPORT_SHARE_LIMIT)
EXPECTED_SHARE = (0.01, 0.03)          # 9.2: from sizes alone on the 0.5B GGUF; far from it is investigated
FORBIDDEN_PORT = 8765                  # the default IDE port, never used here
DATASETS = {
    "good": {"dataset": "mlabonne/harmless_alpaca", "commit": "02c6a92cfcf11bb0c387334f8146d149d65b587f",
             "split": "train[:64]", "column": "text"},
    "bad": {"dataset": "mlabonne/harmful_behaviors", "commit": "01cead01398926d81f7c52bdb790ee8cf77ebba7",
            "split": "train[:64]", "column": "text"},
    "kl": {"dataset": "mlabonne/harmless_alpaca", "commit": "02c6a92cfcf11bb0c387334f8146d149d65b587f",
           "split": "test[:32]", "column": "text"},
    "refusal": {"dataset": "mlabonne/harmful_behaviors", "commit": "01cead01398926d81f7c52bdb790ee8cf77ebba7",
                "split": "test[:32]", "column": "text"},
}
SEARCH = {"name": "e4", "modifiers": [{"plugin": "ARA", "direction": "e4", "lora_rank": 50}],
          "scorers": [{"plugin": "KeywordRate", "prompts": DATASETS["refusal"], "optimization": "minimize",
                       "print_responses": True},
                      {"plugin": "KLDivergence", "prompts": DATASETS["kl"], "optimization": "minimize"}],
          "max_response_length": 30, "n_trials": 6, "n_startup_trials": 6, "seed": 12345}
SCORED_PROMPTS = 32 + 32
PROJECTION = {"layers": 32, "rank": 50, "o_proj": (6144, 5120), "down_proj": (17408, 5120), "bytes_per_value": 4,
              "prompts": 100 + 100, "generated_prompts": 100, "new_tokens": 100, "rates": (5, 10, 20)}


def parse(argv):
    p = argparse.ArgumentParser(prog="formats_transport.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--print-setup", action="store_true")
    p.add_argument("--iso")
    p.add_argument("--gguf")
    p.add_argument("--backend", choices=("cpu", "metal"))
    p.add_argument("--cpu-bridge")
    p.add_argument("--metal-bridge")
    p.add_argument("--port", type=int, default=43881)
    p.add_argument("--server")
    p.add_argument("--root")
    p.add_argument("--model")
    p.add_argument("--guard-log")
    p.add_argument("--out")
    p.add_argument("--generation-config")
    p.add_argument("--timeout", type=float, default=3600.0, help="seconds for the whole search (default 3600)")
    p.add_argument("--poll", type=float, default=15.0)
    a = p.parse_args(argv)
    if a.print_setup:
        if not (a.iso and a.gguf and a.backend):
            p.error("--print-setup needs --iso, --gguf and --backend")
    elif not all((a.server, a.root, a.model, a.guard_log, a.out)):
        p.error("give --server, --root, --model, --guard-log and --out (or --print-setup)")
    return a


def check_server(url):
    u = urllib.parse.urlparse(url)
    if u.scheme != "http" or u.hostname not in ("127.0.0.1", "localhost", "::1") or u.port is None:
        raise SystemExit(f"{url}: give an http address on this machine with an explicit port")
    if u.port == FORBIDDEN_PORT:
        raise SystemExit(f"port {FORBIDDEN_PORT} is the default IDE port; use the isolated serve's port")
    return url.rstrip("/")


def print_setup(a):
    if a.port == FORBIDDEN_PORT:
        raise SystemExit(f"port {FORBIDDEN_PORT} is the default IDE port")
    iso = Path(a.iso)
    config = {"version": 1, "defaults": {"backend": a.backend, "cpu_bridge": a.cpu_bridge, "metal_bridge": a.metal_bridge,
                                         "model_roots": [str(iso / "models")]}}
    print("# 1. the explicit config, as", iso / "config.json")
    print(json.dumps(config, indent=2))
    print("# 2. the model directory: one symbolic link to the GGUF")
    print(f"mkdir -p {iso / 'models'} && ln -s {Path(a.gguf).resolve()} {iso / 'models' / 'model.gguf'}")
    print("# 3. the supervised serve, under the outer guard (from weight-space-ide/, with the Optuna interpreter)")
    print(f"python -m wsp.memory_guard --log {iso}/<model>/watchdog.jsonl --forecast-gib 12 --timeout 3600 -- "
          f"<python> -m wsp serve --config {iso / 'config.json'} --root {iso / 'root'} --empty --port {a.port}")
    print("# 4. this script, against it")
    print(f"python scripts/formats_transport.py --server http://127.0.0.1:{a.port} --root {iso / 'root'} "
          f"--model {iso / 'models' / 'model.gguf'} --guard-log {iso}/<model>/watchdog.jsonl "
          f"--out {iso}/<model>/transport.json [--generation-config <hf snapshot>/generation_config.json]")
    return 0


class Http:
    def __init__(self, base):
        self.base = base

    def call(self, method, path, body=None, timeout=7200):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data is not None else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")
            try:
                return e.code, json.loads(text)
            except ValueError:
                return e.code, {"error": text[:2000]}


def workspace_model_dir(root, name):
    """The opened workspace's model/ directory under the serve's root (by its workspace.json name)."""
    hits = []
    for meta in Path(root).glob("*/workspace.json"):
        try:
            if name in (json.loads(meta.read_text()).get("name"), meta.parent.name):   # Workspace.name's two sources
                hits.append(meta.parent / "model")
        except ValueError:
            continue
    if not hits:
        every = list(Path(root).glob("*/workspace.json"))
        hits = [every[0].parent / "model"] if len(every) == 1 else []
    if len(hits) != 1:
        raise SystemExit(f"{root}: {len(hits)} workspaces named {name!r}; cannot place the generation config")
    return hits[0]


def derived(trials, summary):
    """The numbers of 9.1 from the full trial records (pure; exercised on synthetic records)."""
    done = [t for t in trials if t.get("state") == "complete" and t.get("transport")]
    tb = sum(t["transport"]["bytes"] for t in done)
    ts = sum(t["transport"]["seconds"] for t in done)
    tokens = sum(r.get("tokens", 0) for t in done for s in (t.get("scores") or []) for r in (s.get("rows") or []))
    scoring = sum((t.get("seconds") or {}).get("scoring") or 0.0 for t in done)
    jobs = [t["transport"]["jobs"] for t in done]
    bps = tb / ts if ts > 0 else None
    per_job = PROJECTION["rank"] * (sum(PROJECTION["o_proj"]) + sum(PROJECTION["down_proj"])) \
        * PROJECTION["layers"] * PROJECTION["bytes_per_value"]
    jobs_per_prompt = statistics.median(jobs) / SCORED_PROMPTS if jobs else None
    proj = None
    if bps and jobs_per_prompt:
        n_jobs = jobs_per_prompt * PROJECTION["prompts"]
        t = per_job * n_jobs / bps
        proj = {"bytes_per_job": per_job, "jobs": n_jobs, "transport_seconds": t,
                "shares": {str(rate): t / (t + PROJECTION["generated_prompts"] * PROJECTION["new_tokens"] / rate)
                           for rate in PROJECTION["rates"]}}
    shares = [t["transport"]["share"] for t in done if t["transport"].get("share") is not None]
    median = (summary or {}).get("median_share") if summary else (statistics.median(shares) if shares else None)
    out = {"complete_trials": len(done), "bytes": tb, "transport_seconds": ts, "bytes_per_second": bps,
           "response_tokens": tokens, "scoring_seconds": scoring,
           "seconds_per_decoded_token": scoring / tokens if tokens else None,
           "jobs_per_trial": jobs, "jobs_per_scored_prompt": jobs_per_prompt, "median_share": median,
           "max_share": max(shares) if shares else None, "projection_27b": proj, "expected_share": EXPECTED_SHARE}
    out["decision"] = decision(median, proj)
    return out


def decision(median, proj, model="the GGUF model"):
    if median is None:
        return {"e4": "no measured share: no complete trial carried a candidate"}
    e4 = ("build E4b (9.3): the measured median share is above 0.1" if median > SHARE_LIMIT else
          f"measured median share {median:.4g} on {model}; candidates keep travelling with each job")
    q8 = bool(proj) and any(v > SHARE_LIMIT for v in proj["shares"].values())
    far = not (EXPECTED_SHARE[0] / 3 <= median <= EXPECTED_SHARE[1] * 3)
    return {"e4": e4, "build_e4b": median > SHARE_LIMIT, "ask_owner_q8": q8,
            "far_from_expected": far}


def stop_supervisor(guard_log, port, wait=180.0):
    """One SIGINT to the serve process the guard started, then the guard's teardown line."""
    import psutil
    lines = [json.loads(x) for x in Path(guard_log).read_text().splitlines() if x.strip()]
    pre = next((x for x in lines if x.get("event") == "preflight"), None)
    if pre is None:
        return {"error": "the guard log has no preflight line"}
    pid = pre["pid"]
    try:
        cmd = psutil.Process(pid).cmdline()
    except psutil.Error as e:
        return {"error": f"pid {pid}: {type(e).__name__}"}
    if "serve" not in cmd or str(port) not in cmd:
        return {"error": f"pid {pid} is not the isolated serve on port {port}: {cmd[:6]}"}
    os.kill(pid, signal.SIGINT)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        rows = [json.loads(x) for x in Path(guard_log).read_text().splitlines() if x.strip()]
        end = next((x for x in reversed(rows) if x.get("event") == "teardown"), None)
        if end is not None:
            return {"pid": pid, "teardown": end}
        time.sleep(1.0)
    return {"pid": pid, "error": f"no teardown line within {wait:.0f} s"}


def main(argv):
    a = parse(argv)
    if a.print_setup:
        return print_setup(a)
    base = check_server(a.server)
    port = urllib.parse.urlparse(base).port
    http = Http(base)
    result = {"server_port": port, "model": Path(a.model).name, "search": SEARCH, "datasets": DATASETS, "steps": {}}
    out = Path(a.out)

    def save():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".part")
        tmp.write_text(json.dumps(result, indent=1, default=str))
        os.replace(tmp, out)

    def step(name, method, path, body=None, timeout=7200):
        t0 = time.monotonic()
        status, ans = http.call(method, path, body, timeout)
        result["steps"][name] = {"status": status, "seconds": time.monotonic() - t0,
                                 **({"error": ans.get("error") or ans.get("detail")} if status != 200 and isinstance(ans, dict) else {})}
        print(f"[transport] {name}: HTTP {status} in {result['steps'][name]['seconds']:.1f} s", flush=True)
        save()
        return status, ans

    ok = True
    try:
        status, ans = step("open", "POST", "/api/open", {"name": a.model}, 600)
        if status != 200:
            raise RuntimeError(f"open failed: {ans}")
        result["workspace"] = ans.get("name")
        if a.generation_config:
            target = workspace_model_dir(a.root, ans.get("name")) / "generation_config.json"
            if target.exists():
                raise SystemExit(f"{target} exists: use a fresh isolated root")
            target.write_bytes(Path(a.generation_config).read_bytes())
            result["generation_config"] = json.loads(target.read_text())
        status, ans = step("measure", "POST", "/api/directions/measure",
                           {"name": "e4", "good": DATASETS["good"], "bad": DATASETS["bad"], "chat": True,
                            "system": "You are a helpful assistant.", "module_io": True})
        if status != 200:
            raise RuntimeError(f"measure failed: {ans}")
        status, ans = step("search", "POST", "/api/directions/searches", SEARCH, 600)
        if status != 200:
            raise RuntimeError(f"search create failed: {ans}")
        deadline = time.monotonic() + a.timeout
        desc = None
        while time.monotonic() < deadline:
            status, desc = http.call("GET", "/api/directions/searches/e4?trials=full", timeout=120)
            if status == 200:
                trials = desc.get("trials") or []
                final = [t for t in trials if t.get("state") in ("complete", "failed")]
                job = desc.get("job") or {}
                print(f"[transport] {len(final)} of {SEARCH['n_trials']} trials final", flush=True)
                if not desc.get("running") and not job.get("running") and len(final) >= SEARCH["n_trials"]:
                    break
                if not desc.get("running") and not job.get("running") and job.get("state") in ("failed", "stopped"):
                    break
            time.sleep(a.poll)
        else:
            result["steps"]["poll"] = {"error": f"the search was not done within {a.timeout:.0f} s"}
        if desc is not None:
            result["trials"] = [{"index": t.get("index"), "state": t.get("state"), "transport": t.get("transport"),
                                 "seconds": t.get("seconds"), "values": t.get("values"),
                                 "tokens": [r.get("tokens") for s in (t.get("scores") or []) for r in (s.get("rows") or [])]}
                                for t in desc.get("trials") or []]
            result["summary_transport"] = desc.get("transport")
            result["derived"] = derived(desc.get("trials") or [], desc.get("transport"))
            result["derived"]["decision"] = decision(result["derived"]["median_share"], result["derived"]["projection_27b"],
                                                     Path(a.model).name)
        save()
    except Exception as e:                                   # the supervisor is stopped whatever happened
        result["error"] = f"{type(e).__name__}: {e}"
        ok = False
        save()
    finally:
        stopped = stop_supervisor(a.guard_log, port)
        result["shutdown"] = stopped
        end = stopped.get("teardown") or {}
        clean = "error" not in stopped and end.get("remaining") == []
        result["checks"] = {"guard teardown remaining: []": {"pass": clean, "remaining": end.get("remaining"),
                                                              "reason": end.get("reason")}}
        print(f"{'PASS' if clean else 'FAIL'} guard teardown remaining: [] {json.dumps(stopped, default=str)[:400]}",
              flush=True)
        result["pass"] = clean and ok
        save()
    d = result.get("derived") or {}
    print(f"median share {d.get('median_share')}  bytes/s {d.get('bytes_per_second')}  27B projection "
          f"{(d.get('projection_27b') or {}).get('shares')}  decision {d.get('decision')}", flush=True)
    print(f"{'ok' if result['pass'] else 'NOT OK'}: {out}", flush=True)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
