"""Behavioural tests: the specification a weight program is written against.

A suite is a JSON file in <workspace>/tests/:

    {"name": "facts", "description": "...", "cases": [
       {"type": "next_token", "prompt": "Alice lives in", "expect": "Paris"},          # or a list; "top_k": 3
       {"type": "not_token",  "prompt": "Alice lives in", "forbid": "Rome"},
       {"type": "logit_diff", "prompt": "...", "a": " he", "b": " she", "min": 1.0},
       {"type": "completion", "prompt": "Once upon a time", "max_tokens": 20,
        "contains": "dragon" | "not_contains": ... | "equals": ... | "matches": "<regex>"},
       {"type": "preserve",   "prompt": "...", "max_kl": 0.05}                          # stay close to the base model
    ]}

Each case reports pass / fail and a number that says how far it is from flipping (margin in
logits, probability, KL), so a programmer without gradients still gets a direction to move in.
`drift` is the standing locality check: how far the edited model has moved from the base
model on text that the edit was not about.
"""
from __future__ import annotations

import os
import re

import torch
import torch.nn.functional as F

from .analysis import corpus
from .runtime_events import RuntimeRefusal, job_scope
from .tokenizer import show


def list_suites(ws) -> list[dict]:
    d = os.path.join(ws.dir, "tests")
    out = []
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if fn.endswith(".json"):
            s = ws.read_json(f"tests/{fn}")
            out.append({"name": fn[:-5], "description": s.get("description", ""), "n": len(s.get("cases", []))})
    return out


def load_suite(ws, name: str) -> dict:
    s = ws.read_json(f"tests/{name}.json")
    if s is None:
        raise KeyError(f"no test suite {name!r}; available: {[x['name'] for x in list_suites(ws)]}")
    return s


def save_suite(ws, name: str, suite: dict) -> None:
    if not re.fullmatch(r"[\w\-]+", name):
        raise ValueError("suite names may contain letters, digits, - and _")
    suite["name"] = name
    for c in suite.get("cases", []):
        if c.get("type") not in ("next_token", "not_token", "logit_diff", "completion", "preserve"):
            raise ValueError(f"unknown case type {c.get('type')!r}")
    ws.write_json(f"tests/{name}.json", suite)


def _ids(ws, strs) -> list[int]:
    out = []
    for s in ([strs] if isinstance(strs, str) else strs):
        ids = ws.tok.encode_token(s)
        if len(ids) != 1:
            raise ValueError(f"{s!r} is {len(ids)} tokens, tests need single tokens")
        out.append(ids[0])
    return out


@torch.no_grad()
def run_case(ws, case: dict, variant: str = "current", extra=None) -> dict:
    kind = case["type"]
    name = case.get("name") or (case["prompt"] if len(case["prompt"]) < 60 else "…" + case["prompt"][-58:])
    res = {"name": name, "type": kind, "prompt": case["prompt"]}
    try:
        r = ws.runner
        ids = ws.encode(case["prompt"])[-(r.max_ctx or 10 ** 9):]
        if kind == "completion":
            text = ws.tok.decode(r.generate(ids, variant, extra, max_new_tokens=int(case.get("max_tokens", 20))))
            ok = True
            if "contains" in case:
                ok &= case["contains"] in text
            if "not_contains" in case:
                ok &= case["not_contains"] not in text
            if "equals" in case:
                ok &= text.strip() == case["equals"].strip()
            if "matches" in case:
                ok &= re.search(case["matches"], text) is not None
            return {**res, "pass": bool(ok), "got": text, "value": 1.0 if ok else 0.0}
        lg = r.logits(ids, variant, extra, tail=1)[-1]       # one row crosses a model-worker boundary
        top = int(lg.argmax())
        res["got"] = show(ws.tok.token_str(top))
        if kind == "next_token":
            exp = _ids(ws, case["expect"])
            k = int(case.get("top_k", 1))
            ranks = [int((lg >= lg[e]).sum()) - 1 for e in exp]       # ties count against the expected token
            best = exp[min(range(len(exp)), key=lambda i: ranks[i])]
            other = lg.clone()
            other[exp] = float("-inf")
            margin = float(lg[best] - other.max())
            ok = min(ranks) < k
            if "min_prob" in case:
                ok &= float(lg.softmax(-1)[exp].sum()) >= case["min_prob"]
            return {**res, "pass": bool(ok), "expect": [show(ws.tok.token_str(e)) for e in exp], "rank": min(ranks),
                    "prob": round(float(lg.softmax(-1)[exp].sum()), 4), "value": round(margin, 4)}
        if kind == "not_token":
            bad = _ids(ws, case["forbid"])
            k = int(case.get("top_k", 1))
            rank = min(int((lg > lg[b]).sum()) for b in bad)
            other = lg.clone()
            other[bad] = float("-inf")
            return {**res, "pass": rank >= k, "forbid": [show(ws.tok.token_str(b)) for b in bad], "rank": rank,
                    "value": round(float(other.max() - lg[bad].max()), 4)}
        if kind == "logit_diff":
            a, b = _ids(ws, case["a"])[0], _ids(ws, case["b"])[0]
            diff = float(lg[a] - lg[b])
            return {**res, "pass": diff >= float(case.get("min", 0.0)), "value": round(diff, 4)}
        if kind == "preserve":
            base = r.logits(ids, "base", tail=1)[-1]
            kl = float(F.kl_div(lg.log_softmax(-1), base.log_softmax(-1), log_target=True, reduction="sum"))
            return {**res, "pass": kl <= float(case.get("max_kl", 0.05)), "value": round(kl, 5),
                    "base": show(ws.tok.token_str(int(base.argmax())))}
        raise ValueError(f"unknown case type {kind!r}")
    except RuntimeRefusal as e:
        return {**res,"pass":None,"status":"not_run","event":e.record,"value":None}
    except Exception as e:
        return {**res, "pass": False, "error": f"{type(e).__name__}: {e}", "value": 0.0}


def run_suite(ws, suite: dict | str, variant: str = "current", extra=None) -> dict:
    if isinstance(suite, str):
        suite = load_suite(ws, suite)
    cases=[];blocked=None
    for case in suite.get("cases",[]):
        if blocked:
            result=dict(name=case.get("name",case["prompt"]),type=case["type"],prompt=case["prompt"],status="not_run",event=blocked,**{"pass":None,"value":None})
        else:result=run_case(ws,case,variant,extra)
        cases.append(result)
        if result.get("status")=="not_run":blocked=result["event"]
    passed = sum(c["pass"] is True for c in cases)
    not_run=sum(c.get("status")=="not_run" for c in cases)
    return {"name": suite.get("name", ""), "passed": passed, "total": len(cases),
            "score": None if not_run else (passed / len(cases) if cases else 1.0), "not_run":not_run, "cases": cases}


@torch.no_grad()
def drift(ws, variant: str = "current", extra=None, n: int = 24) -> dict:
    """How far the weights have moved from the base model on calibration text: mean KL(base || edited) and top-1 agreement."""
    if ws.kind == 'gguf':
        # Bounded sequential calibration, no batched logits or multi-GB base cache.
        seqs = corpus(ws, n_seq=min(n, 4), length=32)[:min(n, 4)]
        total = agree = 0
        kl_sum = kl_max = 0.
        for ids in seqs:
            a = ws.runner.logits(ids, 'base').log_softmax(-1)
            b = ws.runner.logits(ids, variant, extra).log_softmax(-1)
            kl = F.kl_div(b, a, log_target=True, reduction='none').sum(-1)
            total += len(ids)
            agree += int((a.argmax(-1) == b.argmax(-1)).sum())
            kl_sum += float(kl.sum())
            kl_max = max(kl_max, float(kl.max()))
        return dict(kl=round(kl_sum/max(1, total), 5), kl_max=round(kl_max, 4),
                    top1_agree=round(agree/max(1, total), 4), n=total, calibration='4 sequences x <=32 tokens, batch 1')
    seqs = corpus(ws)[:n]
    if not seqs:
        return {"kl": 0.0, "top1_agree": 1.0, "n": 0}
    L = min(len(q) for q in seqs)
    toks = torch.tensor([q[:L] for q in seqs])
    r = ws.runner
    # Logits are [sequences, length, vocabulary]. With a vocabulary of a quarter of a million that is most of a GiB,
    # and the base's logits, the edited model's, both log-softmaxes and the element-wise KL held at once came to
    # several GiB on every commit. Every operation here is per position, so the model is run on batches of 8
    # sequences as before and the arithmetic is done one sequence at a time, with the same numbers; only the base's
    # log-probabilities are kept whole.
    batches = [range(i, min(i + 8, len(toks))) for i in range(0, len(toks), 8)]

    def largest(x, so_far):                         # NaN stays NaN, as one abs().max() over every logit gives
        m = x.abs().max()
        return m if so_far is None else torch.maximum(so_far, m)

    if "drift_base" not in ws.keep:                 # the base model's log-probabilities on the corpus, computed once
        la, top_a, amax = None, None, None
        for rows in batches:
            a = r.run(toks[rows.start:rows.stop], "base")[0]
            if la is None:
                la = a.new_empty((len(toks), *a.shape[1:]))
                top_a = torch.empty((len(toks), a.shape[1]), dtype=torch.long)
            for k, i in enumerate(rows):
                la[i] = a[k].log_softmax(-1)
                top_a[i] = a[k].argmax(-1)
                amax = largest(a[k], amax)
            del a
        ws.keep["drift_base"] = (la, top_a, float(amax))
    la, top_a, amax = ws.keep["drift_base"]
    if amax == 0:                                   # blank base model: nothing to preserve
        return {"kl": 0.0, "top1_agree": 1.0, "n": 0}
    kl = torch.empty(tuple(top_a.shape), dtype=la.dtype)
    same = torch.empty(tuple(top_a.shape), dtype=torch.bool)
    for rows in batches:
        b = r.run(toks[rows.start:rows.stop], variant, extra)[0]
        for k, i in enumerate(rows):
            kl[i] = F.kl_div(b[k].log_softmax(-1), la[i], log_target=True, reduction="none").sum(-1)
            same[i] = top_a[i] == b[k].argmax(-1)
        del b
    release = getattr(r, "release_device_cache", None)
    if release is not None:                         # a GPU allocator keeps the freed logits; a memory limit counts them
        release()
    return {"kl": round(float(kl.mean()), 5), "kl_max": round(float(kl.max()), 4),
            "top1_agree": round(float(same.float().mean()), 4), "n": int(toks.numel())}


def summarize(result: dict, failures: int = 12) -> str:
    """A suite result as compact text (what the programmer model sees after every edit)."""
    lines = [f"{result['name']}: {result['passed']}/{result['total']} pass"]
    shown = 0
    for c in result["cases"]:
        if c["pass"]:
            continue
        if shown == failures:
            lines.append(f"  … {result['total'] - result['passed'] - shown} more failing")
            break
        shown += 1
        if c.get("status")=="not_run":
            lines.append(f"  NOT RUN {c['name']!r}: {c['event']['message']}")
        elif "error" in c:
            lines.append(f"  FAIL {c['name']!r}: {c['error']}")
        elif c["type"] == "next_token":
            lines.append(f"  FAIL {c['name']!r}: want {c['expect']} got {c['got']!r} (rank {c['rank']}, margin {c['value']})")
        elif c["type"] == "completion":
            lines.append(f"  FAIL {c['name']!r}: got {c['got']!r}")
        else:
            lines.append(f"  FAIL {c['name']!r}: value {c['value']} got {c.get('got')!r}")
    return "\n".join(lines)
