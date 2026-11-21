"""C4-T0: the first probe of the near-limit comparison, and its pre-written decision (formats build spec 10.1).

Run from the IDE's `weight-space-ide` directory, offline, each model process under the memory guard:

    python scripts/formats_timing.py --stage ide <model dir> <isolated IDE root> <out.json> \\
        [--dtype bfloat16] [--guard-log <the guard's log of this run>]
    python scripts/formats_timing.py --stage heretic <model dir> <heretic run dir> <out.json> \\
        [--ide <the ide stage's out.json>] [--guard-log <the guard's log of the Heretic run>] [--print-config]
    <command> | python scripts/formats_timing.py --stage stamp > run.log

Stage ide (the IDE's interpreter; WSP_RUN_DTYPE is set from --dtype, default bfloat16, before the runner exists):
opens the model directory in the isolated root and records the IDE's memory forecast; loads the runner; one
measurement of 4 + 4 prompts (the local datasets, Heretic's qwen05b specifications) as a chat with Heretic's system
prompt and module_io; one 8-token greedy generation; one first-token logits pass. Each duration is recorded, with
t_fwd (seconds per measured prompt) and t_tok (seconds per generated token), and, with --guard-log, the peak
process-tree footprint the guard has sampled so far.

Stage heretic prepares and starts nothing. Without <run dir>/run.log it prints the c4-probe config (8 + 8
prompts, 4 per score, max_response_length 8, n_trials 1, dtypes ["bfloat16"], device_map "cpu", batch_size 1;
--print-config prints only the TOML, to be redirected into <run dir>/config.toml) and the command to run Heretic
under the guard with its output stamped by --stage stamp. With run.log it parses it: the wall time (the guard's
teardown elapsed with --guard-log; Heretic's last "Elapsed time"), and from a stamped log Heretic's own t_fwd (its
module I/O passes) and t_tok (its first evaluation: 4 responses of 8 tokens and 4 KL passes). With --ide it applies
the decision rule:

    T = 2 tools x (128 t_fwd + 2 methods x (4 trials x (64 t_fwd + 32 x 30 t_tok))), t_fwd and t_tok the larger of
    the two tools'.  T <= 4 h: CPU with the planned counts.  4 h < T <= 12 h: the same on mps for both tools.
    Above: mps, every count halved (32 + 32 measured, 16 per score, 2 trials).

A plain (unstamped) run.log has no per-stage times: the rule then uses the IDE's t_fwd and t_tok alone and says so.
Stage stamp prefixes every line of its input with the seconds since its start and a tab. No assertion here: the
exit status is 0 when the stage did its work.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HOUR = 3600.0
RULE = {"cpu_up_to_seconds": 4 * HOUR, "mps_up_to_seconds": 12 * HOUR}
PLANNED = {"measured": 128, "methods": 2, "trials": 4, "kl_prompts": 64, "scored_prompts": 32, "tokens": 30, "tools": 2}
PROBE = {"measured_each": 8, "scored": 4, "max_response_length": 8, "prefix_test_each": 4}
IDE_PROBE = {"measured_each": 4, "new_tokens": 8}
DATASETS = {   # Heretic's qwen05b specifications
    "good": ("mlabonne/harmless_alpaca", "02c6a92cfcf11bb0c387334f8146d149d65b587f", "train"),
    "bad": ("mlabonne/harmful_behaviors", "01cead01398926d81f7c52bdb790ee8cf77ebba7", "train"),
    "kl": ("mlabonne/harmless_alpaca", "02c6a92cfcf11bb0c387334f8146d149d65b587f", "test"),
    "refusal": ("mlabonne/harmful_behaviors", "01cead01398926d81f7c52bdb790ee8cf77ebba7", "test"),
}


def spec(kind, n):
    dataset, commit, split = DATASETS[kind]
    return {"dataset": dataset, "commit": commit, "split": f"{split}[:{n}]", "column": "text"}


def probe_config(model_dir):
    def table(name, kind, n):
        d = spec(kind, n)
        return (f"[{name}]\ndataset = \"{d['dataset']}\"\ncommit = \"{d['commit']}\"\nsplit = \"{d['split']}\"\n"
                f"column = \"text\"\n")
    m, s = PROBE["measured_each"], PROBE["scored"]
    return "\n".join([
        "# C4-T0 probe: 8 + 8 prompts, 4 per score, 8 new tokens, one trial, bfloat16 on the CPU, one prompt at a time.",
        f"model = {json.dumps(str(model_dir))}", "", "seed = 12345", "print_debug_information = true", "",
        "batch_size = 1", 'dtypes = ["bfloat16"]', 'device_map = "cpu"', f"max_response_length = {PROBE['max_response_length']}",
        "", "modifiers = [", '    { plugin = "heretic.modifiers.ara.ARA" },', "]", "",
        "n_trials = 1", "n_startup_trials = 1", "",
        'export_strategy = "adapter"', 'checkpoint_action = "restart"', "trial_index = 0", 'model_action = "save"',
        'save_directory = "adapter"', "",
        table("[response_prefix_test_prompts]", "good", PROBE["prefix_test_each"]),
        table("[response_prefix_test_prompts]", "bad", PROBE["prefix_test_each"]),
        table("scorer.KLDivergence.prompts", "kl", s), table("scorer.KeywordRate.prompts", "refusal", s),
        "[modifier.ARA]\nprint_loss = true\n",
        table("modifier.ARA.good_prompts", "good", m), table("modifier.ARA.bad_prompts", "bad", m)])


def total_seconds(t_fwd, t_tok):
    """The planned C4's cost in seconds from per-prompt and per-token times (10.1's formula)."""
    P = PLANNED
    return P["tools"] * (P["measured"] * t_fwd + P["methods"] * (P["trials"] * (
        P["kl_prompts"] * t_fwd + P["scored_prompts"] * P["tokens"] * t_tok)))


def decide(T):
    if T <= RULE["cpu_up_to_seconds"]:
        return {"device": "cpu", "counts": "planned (64 + 64 measured, 32 per score, 4 trials)"}
    if T <= RULE["mps_up_to_seconds"]:
        return {"device": "mps", "counts": "planned (64 + 64 measured, 32 per score, 4 trials)",
                "settings": {"WSP_DEVICE": "mps", "device_map": "mps"}}
    return {"device": "mps", "counts": "halved (32 + 32 measured, 16 per score, 2 trials)",
            "settings": {"WSP_DEVICE": "mps", "device_map": "mps"}}


def parse_heretic(lines):
    """A Heretic run.log as (seconds or None, line) pairs -> its durations (pure; exercised on synthetic logs)."""
    import formats_shared as FS
    out = {"stamped": all(t is not None for t, _ in lines) and bool(lines), "elapsed": None, "trials": 0}
    for _, line in lines:
        if line.strip().startswith("Elapsed time:"):
            out["elapsed"] = FS.parse_duration(line.split(":", 1)[1])
        if line.strip().startswith("Running trial"):
            out["trials"] += 1
    if not out["stamped"]:
        return out
    first = lambda pred, after=0.0: next((t for t, l in lines if t >= after and pred(l.strip())), None)  # noqa: E731
    t_start = lines[0][0]
    t_loaded = first(lambda s: s.startswith("* Transformer model with"))
    t_good = first(lambda s: s.startswith("Obtaining module I/O for good prompts"))
    t_bad = first(lambda s: s.startswith("Obtaining module I/O for bad prompts"))
    t_trial = first(lambda s: s.startswith("Running trial 1"))
    t_eval = first(lambda s: s.startswith("* Evaluating"), t_trial or 0.0)
    t_scores = first(lambda s: s.startswith("* Refusals:") or s.startswith("* KL divergence:"), t_eval or 0.0)
    out["load_seconds"] = (t_loaded - t_start) if t_loaded is not None else None
    m, s, k = PROBE["measured_each"], PROBE["scored"], PROBE["max_response_length"]
    if None not in (t_good, t_bad, t_trial):
        out["module_io_good_seconds"], out["module_io_bad_seconds"] = t_bad - t_good, t_trial - t_bad
        out["t_fwd"] = (t_trial - t_good) / (2 * m)
    if t_eval is not None and t_scores is not None:
        out["evaluation_seconds"] = t_scores - t_eval
        if "t_fwd" in out:     # 4 responses of up to 8 tokens and 4 KL passes; the passes counted at t_fwd
            out["t_tok"] = max(0.0, (t_scores - t_eval) - s * out["t_fwd"]) / (s * k)
    out["log_seconds"] = lines[-1][0] - t_start
    return out


def stage_stamp():
    t0 = time.monotonic()
    for line in sys.stdin:
        sys.stdout.write(f"{time.monotonic() - t0:.3f}\t{line.rstrip(chr(10))}\n")
        sys.stdout.flush()
    return 0


def stage_ide(a):
    import formats_shared as FS
    FS.set_run_dtype(a.dtype)
    import torch  # noqa: F401
    from wsp import directions as DR
    from wsp import memory_guard as MG
    from wsp.session import Session
    model, root, out = Path(a.args[0]), Path(a.args[1]), Path(a.args[2])
    result = {"stage": "ide", "dtype": a.dtype, "seconds": {}}
    t0 = time.monotonic()
    session = Session(str(root))
    ws = session.open(str(model), log=lambda *x, **k: None)
    result["seconds"]["open"] = time.monotonic() - t0
    try:
        result["forecast"] = MG.forecast(ws)
    except Exception as e:                               # the forecast is a record here, not a gate
        result["forecast"] = {"error": f"{type(e).__name__}: {e}"}
    t0 = time.monotonic()
    _ = ws.runner
    result["seconds"]["load"] = time.monotonic() - t0
    result["runner"] = DR.runner_description(ws)
    n = IDE_PROBE["measured_each"]
    t0 = time.monotonic()
    DR.measure(ws, "t0", spec("good", n), spec("bad", n), chat=True, system=FS.HERETIC_SYSTEM_PROMPT, variant="base",
               replace=True, module_io=True)
    result["seconds"]["measure"] = time.monotonic() - t0
    ids = DR.render_ids(ws, ["What is 1+1?"], "probe", chat=True, system=FS.HERETIC_SYSTEM_PROMPT, chat_options={})[0]
    t0 = time.monotonic()
    got = ws.runner.generate(list(ids), "base", None, max_new_tokens=IDE_PROBE["new_tokens"], temperature=0.0)
    result["seconds"]["generate"] = time.monotonic() - t0
    result["generated_tokens"] = len(list(got))
    t0 = time.monotonic()
    ws.runner.first_token_logits(list(ids), "base")
    result["seconds"]["first_token_logits"] = time.monotonic() - t0
    result["prompt_tokens"] = len(ids)
    result["t_fwd"] = result["seconds"]["measure"] / (2 * n)
    result["t_tok"] = result["seconds"]["generate"] / max(1, result["generated_tokens"])
    if a.guard_log:
        g = FS.guard_log(a.guard_log)
        result["guard"] = {k: g[k] for k in ("samples", "peak_footprint_bytes", "peak_rss_bytes")}
        result["guard"]["forecast_reserved_bytes"] = (g["preflight"] or {}).get("forecast")
    FS.write_json(out, result)
    session.unload()
    print(f"ide: t_fwd {result['t_fwd']:.3f} s/prompt, t_tok {result['t_tok']:.3f} s/token -> {out}", flush=True)
    return 0


def stage_heretic(a):
    import formats_shared as FS
    model, run, out = Path(a.args[0]), Path(a.args[1]), Path(a.args[2])
    if a.print_config:
        print(probe_config(model))
        return 0
    log = run / "run.log"
    if not log.is_file():
        print("# C4-T0, Heretic: nothing is prepared or started by this script.")
        print(f"# 1. {run}/config.toml (or: formats_timing.py --stage heretic {model} {run} {out} --print-config > "
              f"{run}/config.toml):")
        print(probe_config(model))
        print("# 2. from weight-space-ide/, offline (HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 "
              "HF_HOME=<isolated hf-home>):")
        print(f"python -m wsp.memory_guard --log {run}/watchdog.jsonl --forecast-gib 16 --timeout 1800 -- /bin/sh -c "
              f"'cd {run} && <heretic venv>/bin/heretic 2>&1 | <ide python> -u {Path(__file__).resolve()} --stage stamp "
              f"> run.log'")
        print(f"# 3. then again: formats_timing.py --stage heretic {model} {run} {out} --ide <ide out.json> "
              f"--guard-log {run}/watchdog.jsonl")
        return 0
    result = {"stage": "heretic", "run": str(run), "probe": PROBE, **parse_heretic(FS.heretic_log(log))}
    if a.guard_log:
        g = FS.guard_log(a.guard_log)
        result["guard"] = {"samples": g["samples"], "peak_footprint_bytes": g["peak_footprint_bytes"],
                           "wall_seconds": (g["teardown"] or {}).get("elapsed"),
                           "teardown_reason": (g["teardown"] or {}).get("reason")}
    if a.ide:
        ide = json.loads(Path(a.ide).read_text())
        tools = {"ide": {"t_fwd": ide["t_fwd"], "t_tok": ide["t_tok"]}}
        if "t_fwd" in result and "t_tok" in result:
            tools["heretic"] = {"t_fwd": result["t_fwd"], "t_tok": result["t_tok"]}
        t_fwd = max(v["t_fwd"] for v in tools.values())
        t_tok = max(v["t_tok"] for v in tools.values())
        T = total_seconds(t_fwd, t_tok)
        result["decision"] = {"tools": tools, "t_fwd": t_fwd, "t_tok": t_tok, "T_seconds": T, "T_hours": T / HOUR,
                              "rule": RULE, **decide(T),
                              "note": None if "heretic" in tools else
                              "run.log is not stamped: Heretic's per-stage times are unknown, the IDE's alone decide"}
    FS.write_json(out, result)
    print(json.dumps({k: result.get(k) for k in ("elapsed", "t_fwd", "t_tok", "decision")}, default=str), flush=True)
    return 0


def main(argv):
    p = argparse.ArgumentParser(prog="formats_timing.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("ide", "heretic", "stamp"), required=True)
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    p.add_argument("--guard-log")
    p.add_argument("--ide")
    p.add_argument("--print-config", action="store_true")
    p.add_argument("args", nargs="*")
    a = p.parse_args(argv)
    if a.stage == "stamp":
        return stage_stamp()
    if len(a.args) != 3:
        p.error(f"--stage {a.stage} needs three paths (see the usage)")
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return stage_ide(a) if a.stage == "ide" else stage_heretic(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
