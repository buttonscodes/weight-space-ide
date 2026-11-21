"""Compare what Heretic makes from the IDE's reproduce.json with Heretic's own run and with the IDE's adapter.

The end-to-end check of the mapping to Heretic's reproduce.json (docs/DIRECTIONS.md): the orchestrator runs, per
label, under the memory guard, offline, from an empty directory, in Heretic's environment

    heretic --reproduce <copy of the IDE's ide_reproduce_adapter.json with settings.model_action "save" and
        settings.trial_index 0> --ignore-mismatches true --save-directory <reproduced adapter dir>

(the two added settings answer Heretic's prompts; nothing else of the file is changed), and then, without a model:

    python scripts/heretic_compare_reproduction.py <out.json> \\
        <label>=<reproduced adapter dir>,<Heretic run's adapter dir>,<the label's ide.json> ...

<the label's ide.json> is what scripts/heretic_compare_adapter.py --stage ide wrote (method, row_normalization and
the IDE's adapter directory). Files are read one tensor at a time.

    S-F0  the reproduction is an adapter: adapter_config.json and adapter_model.safetensors, the
          same r, lora_alpha and tensor names as Heretic's own run                               exact
    S-F1  every tensor of the reproduction equals the same tensor of Heretic's own run of that
          configuration: Heretic, given only the IDE's file, makes the adapter Heretic made      bit-equal
          (asserted for row_normalization none and pre and for ARA. For row_normalization full
          Heretic's change is a randomized SVD seeded with settings.seed (modifiers/
          abliteration.py:435), and an ablation made by hand in the IDE has no seed, so its file
          says 0: the largest relative difference of the two products is reported instead)
    S-F2  per module of the reproduction, lora_B @ lora_A * (alpha / r) against the IDE's
          adapter's product (zero when the module is absent from the IDE's adapter; the
          reproduction's must then be exactly 0): ablation relative <= 5e-5 on the tiny fixture,
          <= 1e-3 otherwise (the bounds of heretic_compare_adapter.py's P6); ARA reported (each
          tool fits with its own libraries)

Every assertion prints a PASS or FAIL line with its numbers; <out.json> holds every number and a pass flag per
assertion; the exit status is 1 when one fails. Imports nothing from wsp.
"""
import json
import math
import sys
from pathlib import Path

PREFIX, SUFFIXES = "base_model.model.", (".lora_A.weight", ".lora_B.weight")
BOUND_FIXTURE, BOUND_OTHER = 5e-5, 1e-3


def modules_of(keys):
    return sorted({k[len(PREFIX):-len(s)] for k in keys for s in SUFFIXES if k.startswith(PREFIX) and k.endswith(s)})


def rel(a, b):
    d, n = float((a.double() - b.double()).norm()), float(b.double().norm())
    return d / n if n > 0 else (0.0 if d == 0 else math.inf)


def product(f, cfg, m):
    b, a = f.get_tensor(f"{PREFIX}{m}.lora_B.weight"), f.get_tensor(f"{PREFIX}{m}.lora_A.weight")
    r = cfg.get("rank_pattern", {}).get(m, cfg["r"])
    alpha = cfg.get("alpha_pattern", {}).get(m, cfg["lora_alpha"])
    return (b.double() @ a.double()) * (alpha / r)


def main(argv):
    import torch
    from safetensors import safe_open
    if len(argv) < 2:
        raise SystemExit(__doc__)
    out, results, ok_all = Path(argv[0]), {}, True

    def check(label, name, ok, **numbers):
        nonlocal ok_all
        ok_all &= bool(ok)
        results.setdefault(label, {})[name] = {"pass": bool(ok), **numbers}
        print(f"{'PASS' if ok else 'FAIL'} {label} {name} {json.dumps(numbers, default=str)[:500]}", flush=True)
        return bool(ok)

    for arg in argv[1:]:
        label, _, rest = arg.partition("=")
        parts = rest.split(",")
        if not label or len(parts) != 3:
            raise SystemExit(f"{arg}: a label is <label>=<reproduced adapter dir>,<Heretic run's adapter dir>,<ide.json>")
        rdir, hdir, ide = Path(parts[0]), Path(parts[1]), json.loads(Path(parts[2]).read_text())
        files = [d / "adapter_model.safetensors" for d in (rdir, hdir)]
        if not check(label, "S-F0 the reproduction is an adapter", all(f.is_file() for f in files)
                     and (rdir / "adapter_config.json").is_file(), files=[f.is_file() for f in files]):
            continue
        rcfg, hcfg = (json.loads((d / "adapter_config.json").read_text()) for d in (rdir, hdir))
        icfg = json.loads((Path(ide["adapter_dir"]) / "adapter_config.json").read_text())
        rf, hf = (safe_open(str(f), "pt") for f in files)
        i_f = safe_open(str(Path(ide["adapter_dir"]) / "adapter_model.safetensors"), "pt")
        rkeys, hkeys, ikeys = sorted(rf.keys()), sorted(hf.keys()), set(i_f.keys())
        check(label, "S-F0 the same r, lora_alpha and tensor names as Heretic's own run",
              rkeys == hkeys and (rcfg["r"], rcfg["lora_alpha"]) == (hcfg["r"], hcfg["lora_alpha"]),
              r=[rcfg["r"], hcfg["r"]], lora_alpha=[rcfg["lora_alpha"], hcfg["lora_alpha"]], tensors=[len(rkeys), len(hkeys)])
        ara, norm = ide["method"] == "ara", ide.get("row_normalization")
        seeded = not ara and norm == "full"
        differ = [k for k in rkeys if k in set(hkeys) and not torch.equal(rf.get_tensor(k), hf.get_tensor(k))]
        mods = modules_of(rkeys)
        if seeded:
            worst = max((rel(product(rf, rcfg, m), product(hf, hcfg, m)) for m in mods if m in modules_of(hkeys)
                         and float(product(hf, hcfg, m).norm()) > 0), default=0.0)
            results.setdefault(label, {})["S-F1 (reported) row_normalization full, seed 0 against the run's seed"] = {
                "tensors": len(rkeys), "not_bit_equal": len(differ), "worst_relative_of_products": worst}
            print(f"     {label} S-F1 (reported) row_normalization full: {len(differ)} of {len(rkeys)} tensors differ "
                  f"from the run's (seeded SVD), products within {worst:.3e}", flush=True)
        else:
            check(label, "S-F1 Heretic's reproduction from the IDE's file equals Heretic's own run", not differ
                  and rkeys == hkeys, tensors=len(rkeys), not_bit_equal=differ[:5])
        rows, absent_nonzero = [], []
        for m in mods:
            pr = product(rf, rcfg, m)
            if f"{PREFIX}{m}.lora_A.weight" in ikeys:
                rows.append({"module": m, "relative": rel(product(i_f, icfg, m), pr)})
            elif float(pr.abs().max()) != 0.0:
                absent_nonzero.append(m)
        only_ide = sorted(set(modules_of(ikeys)) - set(mods))
        worst = max((r["relative"] for r in rows), default=0.0)
        if ara:
            results.setdefault(label, {})["S-F2 (reported) ARA products, each tool's own fit"] = {
                "worst_relative": worst, "modules": len(mods), "in_ide": len(rows), "absent_nonzero": absent_nonzero}
            print(f"     {label} S-F2 (reported) ARA: worst relative difference of the products {worst:.3e} over "
                  f"{len(rows)} modules", flush=True)
        else:
            bound = BOUND_FIXTURE if "tiny" in str(hdir) else BOUND_OTHER
            check(label, "S-F2 products per module, Heretic's reproduction against the IDE's adapter",
                  worst <= bound and not absent_nonzero and not only_ide, worst_relative=worst, bound=bound,
                  modules=len(mods), in_ide=len(rows), absent_nonzero=absent_nonzero[:5], only_ide=only_ide[:5])
        results[label]["S-F2 rows"] = rows
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"labels": results, "pass": ok_all}, indent=1, default=str))
    print(f"{'ok' if ok_all else 'NOT OK'}: {out}", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
