"""B3's GPU half: how long the ARA fit takes on the CPU at the 4B and 27B matrix shapes, without a model (11.1).

Run with the IDE's interpreter from the IDE's `weight-space-ide` directory, under the memory-capped launcher (no
model, no file read, no network):

    python scripts/formats_fit_timing.py <out.json> [--repeats 3] [--shapes 4096x2560,9216x2560,6144x5120,17408x5120]
        [--c4 <heretic_compare_ara.py out.json> ...] [--seed 0]

A shape D_inxd is the IDE's view V [D_in, d] of one output matrix: 4096x2560 and 9216x2560 are Qwen3.5-4B's
o/out_proj and down_proj, 6144x5120 and 17408x5120 the 27B model's. For each shape and repeat: random float32
V ~ N(0, 1 / D_in) (the scale keeps X V of order one; the spec says only "random"), captures of 64 good and 64 bad
rows X ~ N(0, 1), Y = X V + 0.01 N(0, 1), no bias, and one wsp.directions_ara.fit at Heretic's defaults
(lora_rank 50, 5 steps x 20 iterations, history 10, learning rate 1, preserve_row_magnitudes, and the loss weights
of wsp.directions_ara.DEFAULTS), initial values drawn as the IDE draws them. Written per fit: seconds, loss
evaluations (calls of the loss), seconds per evaluation, peak resident memory of this process during the fit.

Then, from the median seconds per fit t of each shape (when both 27B shapes were timed):

    T27 = 32 x t([6144, 5120]) + 32 x t([17408, 5120])
    (one ARA trial over 32 of the 27B model's 64 layers: 32 attention-output fits and 32 down_proj fits)

and the pre-written rule: T27 <= 30 min on the CPU: B3's GPU fit is not a C5 prerequisite (it stays an
optimisation for long searches); T27 > 30 min: B3 (the fit on MPS, in float32) is the first item of the next slice
and a prerequisite of any 27B search (a single 27B trial can still be fitted on the CPU).

--c4 reads the C4 runs' fit times (heretic_compare_ara.py's "timing": fit_own_seconds and fitted_shapes) and
reports, per run, the measured time against sum(count x t(shape)) of this timing; a factor above 2 either way is
flagged (real captures can stop L-BFGS early). No assertion: the exit status is 0 when the timing ran.
"""
import argparse
import json
import math
import os
import statistics
import sys
import threading
import time
from pathlib import Path

SHAPES = "4096x2560,9216x2560,6144x5120,17408x5120"
ROWS = 64                          # good and bad captures each
NOISE = 0.01
T27_TERMS = (("6144x5120", 32), ("17408x5120", 32))
RULE_SECONDS = 30 * 60
HERETIC = {"lora_rank": 50, "n_optimization_steps": 5, "max_iter": 20, "history_size": 10, "learning_rate": 1.0,
           "preserve_row_magnitudes": True}


def parse_shapes(text):
    out = []
    for s in text.split(","):
        a, sep, b = s.strip().lower().partition("x")
        if not sep or not a.isdigit() or not b.isdigit() or int(a) < 1 or int(b) < 1:
            raise SystemExit(f"--shapes {text}: each shape is <D_in>x<d>, for example 6144x5120")
        out.append((int(a), int(b)))
    return out


class Peak:
    """The largest resident size of this process while it runs (sampled every 20 ms)."""

    def __init__(self):
        import psutil
        self.proc, self.peak, self.stop = psutil.Process(), 0, threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            self.stop.wait(0.02)

    def __enter__(self):
        self.peak = self.proc.memory_info().rss
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)


def time_fit(ARA, torch, d_in, d, seed):
    g = torch.Generator().manual_seed(seed)
    V = torch.randn(d_in, d, generator=g) / math.sqrt(d_in)
    Xg, Xb = torch.randn(ROWS, d_in, generator=g), torch.randn(ROWS, d_in, generator=g)
    Yg = Xg @ V + NOISE * torch.randn(ROWS, d, generator=g)
    Yb = Xb @ V + NOISE * torch.randn(ROWS, d, generator=g)
    p = {**ARA.DEFAULTS, **HERETIC}
    calls = {"n": 0}
    real = ARA.loss

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    with Peak() as peak:
        t0 = time.perf_counter()
        data = ARA.prepare_data(V, torch.zeros(d), Xg, Yg, Xb, Yb)
        t_prep = time.perf_counter() - t0
        A0, B0 = ARA.initial_values(torch.Generator().manual_seed(seed), p["lora_rank"], d_in, d)
        ARA.loss = counted
        try:
            t0 = time.perf_counter()
            _, _, info = ARA.fit(data, A0, B0, p, f"{d_in}x{d}")
            t_fit = time.perf_counter() - t0
        finally:
            ARA.loss = real
    return {"seconds": t_fit, "prepare_seconds": t_prep, "evaluations": calls["n"],
            "seconds_per_evaluation": t_fit / calls["n"] if calls["n"] else None, "peak_rss_bytes": peak.peak,
            "loss_before": info["loss_before"], "loss_after": info["loss_after"]}


def t27(medians):
    """T27 and the rule from the median seconds per shape (pure; None when a 27B shape was not timed)."""
    if not all(k in medians for k, _ in T27_TERMS):
        return {"T27_seconds": None, "reason": "the 27B shapes 6144x5120 and 17408x5120 were not both timed"}
    terms = {k: n * medians[k] for k, n in T27_TERMS}
    total = sum(terms.values())
    verdict = ("B3's GPU fit is not a C5 prerequisite (it stays an optimisation for long searches)"
               if total <= RULE_SECONDS else
               "B3 (the ARA fit on MPS, float32) is the first item of the next slice and a prerequisite of any 27B "
               "search; a single 27B trial can still be fitted on the CPU")
    return {"T27_seconds": total, "T27_minutes": total / 60, "terms_seconds": terms, "rule_seconds": RULE_SECONDS,
            "gpu_fit_needed_before_c5": total > RULE_SECONDS, "verdict": verdict}


def against_c4(medians, path):
    data = json.loads(Path(path).read_text())
    rows = []
    for label, r in (data.get("runs") or {}).items():
        timing = r.get("timing") or {}
        shapes, measured = timing.get("fitted_shapes") or {}, timing.get("fit_own_seconds")
        missing = [s for s in shapes if s not in medians]
        row = {"file": str(path), "run": label, "measured_seconds": measured, "fitted_shapes": shapes,
               "untimed_shapes": missing}
        if measured and shapes and not missing:
            expected = sum(n * medians[s] for s, n in shapes.items())
            factor = measured / expected if expected > 0 else math.inf
            row.update(expected_seconds=expected, factor=factor, above_factor_2=factor > 2 or factor < 0.5)
        rows.append(row)
    return rows


def main(argv):
    p = argparse.ArgumentParser(prog="formats_fit_timing.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--shapes", default=SHAPES)
    p.add_argument("--c4", action="append", default=[])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.repeats < 1:
        p.error("--repeats must be at least 1")
    shapes = parse_shapes(a.shapes)
    sys.path.insert(0, os.getcwd())
    import torch
    from wsp import directions_ara as ARA
    result = {"torch": torch.__version__, "threads": torch.get_num_threads(), "rows": ROWS, "noise": NOISE,
              "settings": {**ARA.DEFAULTS, **HERETIC}, "repeats": a.repeats, "shapes": {}}
    medians = {}
    for d_in, d in shapes:
        key = f"{d_in}x{d}"
        fits = []
        for k in range(a.repeats):
            fits.append(time_fit(ARA, torch, d_in, d, a.seed + k))
            print(f"[fit timing] {key} repeat {k + 1}: {fits[-1]['seconds']:.3f} s, {fits[-1]['evaluations']} "
                  f"evaluations", flush=True)
        medians[key] = statistics.median(f["seconds"] for f in fits)
        result["shapes"][key] = {"V": [d_in, d], "fits": fits, "median_seconds": medians[key],
                                 "median_evaluations": statistics.median(f["evaluations"] for f in fits),
                                 "peak_rss_bytes": max(f["peak_rss_bytes"] for f in fits)}
    result["T27"] = t27(medians)
    result["c4"] = [row for path in a.c4 for row in against_c4(medians, path)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"medians": medians, "T27": result["T27"]}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
