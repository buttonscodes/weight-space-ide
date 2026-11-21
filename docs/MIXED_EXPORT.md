# Mixed-precision standalone GGUF

Use **companion** export when you need faithful FP32 patches and undo history with
the exact original base. Standard standalone GGUF bakes edits into weights and loses
undo. Mixed precision reduces additional rounding; it does not restore precision
lost when the base was quantized, and it is not automatically behavior-preserving.

The Export tab offers F16 (default), Q8_0, F32, and explicitly lossy same-scheme
output. Preview reports affected tensors, their original/new types, exact predicted
file bytes and increase without hashing or decoding tensor payloads or loading a
runner. A single changed coordinate can promote its entire tensor. Disabled patches,
zero strengths and zero pieces are excluded; nonzero enabled pieces form the affected
set (cancelling pieces can conservatively promote the same tensor). F32 is never
downgraded; F16 is retained for a Q8 request, and BF16 is retained for Q8 or promoted
to F32 for an F16 request to avoid losing BF16 range.

Untouched tensor payloads are copied byte-for-byte. Changed matrices are decoded,
patched, mapped back to canonical storage and encoded one row at a time, with a
1 MiB float slab limit; rows wider than 262,144 values are refused. Small vectors
and depthwise convolutions stay bounded too. Qwen grouped rows/columns, Llama RoPE
permutations, norms and A_log transforms use the existing independently tested maps.
No full matrix/model dequantization or silent same-scheme fallback occurs.

The exporter rebuilds standard GGUF v3 tensor directory offsets and alignment for
changed lengths. Tokenizer/chat metadata and canonical tensor names/shapes survive.
It removes the optional old `general.file_type` label, which would misdescribe the
mixed file; standard per-tensor types are authoritative. Extra `wsp.export.*` metadata
records policy/base hash/UNVALIDATED provenance only. **No runtime adapter or custom
weight format is needed**; pinned llama.cpp independently loads the tested files.

Export requires the external memory watchdog, but never runs inference. It checks
free disk, source hash before/after streaming, destination/report collisions and
cancellation; a partial temp file is never exposed as the final GGUF. Publication
uses exclusive hard links on the destination filesystem. Failures clean owned temp
files and preserve the source and any competing destination. The original Export
API still downloads the companion; the new Export tab explicitly creates standalone
jobs and supports status/cancellation/download. An export's structural COMPLETE state
is separate from its behavioral UNVALIDATED, VERIFIED or FAILED status.

## Explicit guarded validation

`validate_mixed_export.py` is an operator workflow, not an automatic export callback.
Model-load consent is required separately. Criteria are written to `plan.json` before
candidate creation, with a fixed target and three unrelated prompts:

- target gain at least 90% of the exact overlay gain **and** at least 0.1;
- all tested unrelated top-1 outcomes retained;
- unrelated KL versus original base <= 0.01;
- maximum logit error versus overlay <= 0.05;
- identical chat prompt (token outputs are also recorded).

The metric implementation supports a predeclared negative direction; zero gain uses
an absolute margin-error tolerance instead of division. The included operator fixture
is positive Paris-versus-London reinforcement, not new-knowledge acquisition. FP32
merged multiplication can differ from split-base-plus-delta accumulation; passing
these checks is fixed-suite evidence, never a bit-exact or universal promise.

Small-model adaptive selection is bounded to at most four unique increasing modes,
SAME -> Q8_0 -> F16 -> F32, with explicit total time, per-phase time and disk limits.
Each baseline, export and reload is a distinct external-watchdog-owned process;
remaining=[] is required before the next. A quality rejection writes FAILED and
exits3. Only that clean, documented semantic rejection can advance a small-model
promotion search; runtime/safety/preflight failure stops it. Failed files/evidence
remain, never relabelled verified. If F32 fails, use the companion; thresholds and
edit strength are never silently changed. The old same-scheme acceptance helper
now also exits nonzero when its required target gain fails.

Large models require a fresh exact-artifact supervised permit, enumerated mixed
candidate hashes, and **one preselected candidate**, not automatic promotion search.
The prior large permits remain revoked. Any later request-specific authorization
must retain active monitoring, one resident model, fresh headroom and bounded phases.

From the IDE's directory (the one that holds `wsp/`), with the pinned Python/CPU and optional Metal bridge
variables documented in [QWEN38.md](QWEN38.md):

```sh
# Metadata only: no watchdog, model inference or full-file hashing.
PYTHONPATH=. python scripts/quantized_export.py \
  --workspace /outside/repo/edited-workspace --out /outside/repo/new.gguf \
  --precision F16 --preview

# Explicit authorized small-model search; no >=9B model without a new permit.
PYTHONPATH=. python scripts/validate_mixed_export.py \
  --workspace /outside/repo/edited-workspace --out /outside/repo/fresh-validation \
  --precisions Q8_0,F16,F32 --phase-seconds 180 --total-seconds 1800 --max-disk-gib 8

# Export alone, never inference. Reservation must cover the actual model forecast.
PYTHONPATH=. python -m wsp.memory_guard \
  --log /outside/repo/fresh-export/watchdog.jsonl --forecast-gib 8 --timeout 180 \
  --zero-growth -- python scripts/quantized_export.py \
  --workspace /outside/repo/edited-workspace --out /outside/repo/new.gguf --precision F16
```

To validate an existing UI export against an already collected fixed baseline, stop
any loaded runner first, then explicitly run the candidate phase under a fresh guard:
`validate_mixed_export.py --out /outside/repo/fixed-baseline --phase candidate
--precision F16 --candidate-file /outside/repo/export.gguf`. This writes the exact-hash
`export.gguf.validation.json` receipt; it exits3 on quality failure. UI Refresh reads
a matching receipt and displays FAILED or VERIFIED; HTTP200/file validity alone
leaves UNVALIDATED. Never run reference and candidate models concurrently.

## Small-model evidence

| Fixture / candidate | Overlay gain | Candidate gain | Retention | Max logit error | File increase | Result |
|---|---:|---:|---:|---:|---:|---|
| Qwen3.5-0.8B Q4_K_M, same scheme |0.692943|0.159750|23.05%|0.270057|288 B|FAILED|
| Same fixed edit, edited tensor Q8_0 |0.692943|0.669923|96.68%|0.038941|889,120 B|VERIFIED|
| Qwen2.5-0.5B Q4_0, same scheme |0.312491|0.039613|12.68%|0.141155|288 B|FAILED|
| Same fixed edit, edited tensor F16 |0.312491|0.307285|98.33%|0.002860|6,265,120 B|VERIFIED|

All passing candidates retained 3/3 unrelated top tokens and chat prompt identity;
maximum unrelated KL was below 0.000060. The Qwen2 run used the pinned Metal backend,
so its baseline gain differs from the earlier CPU report; historical evidence was
not rewritten. A first Qwen2 setup run failed on an old companion missing its Jinja
asset; the isolated fixture was repaired from the original GGUF template and the
failure was preserved. No missing chat template was silently treated as equivalence.

Independent portability used CPU-only llama.cpp commit
`f477f795cb071baf5b8af6488378366a72034ac4`, built with Metal/OpenMP/BLAS off, four CPU
threads, context64, four generated tokens, seed0, temperature0 and no warmup/chat.
Original and promoted Qwen2 both produced ` Paris, and it`; original Qwen3.5 produced
` the capital of the`, promoted Q8 produced ` Paris.\nThe`. These raw-completion
checks establish external loading/execution, not equality of CPU/Metal logits.
`gguf_external_smoke.py` runs the external child in the watchdog's owned process tree
and records executable/build hashes, high-water RSS and output/timings.

Browser verification on isolated43879 checked F16/Q8/SAME previews, whole-tensor size
cost, companion option and UNVALIDATED/VERIFIED/FAILED labels. The displayed receipts
were reused only after checking the newly exported file hash exactly matched the
previously guarded candidate; this was not represented as a new inference pass.
The runner stayed unloaded before and after preview/export. An IDE already running on port 8765 was untouched.

Raw evidence: `~/.cache/wsp-quantized-ide/mixed-export/evidence/`.
The machine-readable [mixed-export-validation.json](mixed-export-validation.json)
contains artifact hashes, exact guard outcomes, timings and numerical criteria.

## Qwen3.8-27B forecast and supervised result

The existing three patches affect only `blk.63.ffn_down.weight` [5120,17408], whose
Q4_K payload is50,135,040bytes. Predicted standard mixed output:

| Policy | Promoted tensor | Full file | Increase | Revised64-context forecast |
|---|---:|---:|---:|---:|
| Q8_0 |94,699,520 B|19,448,249,248 B|44,564,768 B|22.805 GiB|
| F16 |178,257,920 B|19,531,807,616 B|128,123,136 B|22.883 GiB|
| F32 |356,515,840 B|19,710,065,536 B|306,381,056 B|23.049 GiB|

F16 was the single large candidate, justified by small F16 retention and
lower rounding than Q8. The forecast is recomputed from actual promoted bytes plus
runtime, KV/recurrent/conv, activations, patch copies and temporary buffers. A23GiB
reservation covers this specific F16 forecast; this numerical match to the old
reservation is recomputed, not assumed.24GiB process stop/26GiB ceiling,25GiB preferred
postforecast headroom and20GiB floor remained unchanged.

The authorized sequence at runtime commit `90d7fd9e062aace05a1ea9b4401ed0758a176af2`
passed original/overlay baseline, streamed F16 export and IDE F16 reload. It then
**stopped on the first external CPU baseline safety failure**. No retries, external
mixed reload or actual27B browser test followed. All permits are revoked; do not
reuse them. Prior same-scheme failure remains a failure.

| Exact same27B edit | Gain | Retention | Max logit error vs overlay | Result |
|---|---:|---:|---:|---|
| Overlay baseline |0.5468559265|100%|0|Reference|
| Historical same-Q4 standalone |0.009325|1.7%|See historical report|FAILED|
| New F16 edited tensor |0.5513896942|100.829%|0.0277271271|VERIFIED on fixed suite|

F16 passed the predeclared required gain0.4921703339, all3unrelated top tokens,
maximum unrelated KL0.000072019 and identical chat prompt. Completion remained
` Paris.\nThe`; chat returned `Hello! How can`. This small positive overshoot is
rounding/accumulation difference, not exact equivalence or improved general quality.
The exact mixed file is19,531,807,616bytes, SHA256
`aac8f481e535318cb7c15069e2aa49ed279a4ac475d09431a87d2f9f57dd98ec`.
Only `blk.63.ffn_down.weight` changed from Q4_K to F16. Maximum reconstructed edited
weight error was0.0001576543; original source hash remained unchanged.

The baseline phase took24.747s, streamed export46.333s and candidate phase36.072s.
Candidate first completion took2.241s (1.785tokens/s including startup, prefill0.415s,
three decode steps0.712s); warm chat took1.343s (2.979tokens/s). These are short64-context,
4-token measurements, not long-context throughput claims. The three IDE phases had
normal pressure, zero swap/compressor growth, minimum63.150GiB available, peak18.191GiB
Metal allocation and2.068GiB sampled process RSS; all exited with remaining=[].

The independent CPU-only llama.cpp **original27B baseline**, before any mixed-file
external run, reached25.300GiB process-tree RSS, exceeding the24GiB stop budget.
That sample also had17,842,176bytes compressor growth, violating the zero-growth rule;
pressure remained normal and swap stayed zero. Physical footprint was7.016GiB, but
that does not override the RSS safety stop. The IDE forecast did not bound this
external runtime's observed RSS. Its allocation cause remains unverified; no budget
increase or fallback was attempted. External27B portability is therefore incomplete.

During that stop, `killpg(SIGKILL)` raised PermissionError after earlier termination
signals. The original log/traceback are preserved without a fabricated teardown row.
Escalated checks of both recorded PID/create-time identities found them already gone;
a second identity/group/scoped-listener check confirmed no owned process or43879
listener. `manual-teardown.json` records this separately. The offline correction now
verifies a live owned group member before group signaling, attempts individual owned
signals after group errors, and always records cleanup errors, unresolved identities
and a failure latch. Tests cover denied group signals, separate child groups, PID reuse,
unknown identity, unexpected cleanup errors and survival of an unrelated sentinel.
No real model was reloaded after this correction.

Actual27B browser acceptance and external mixed portability remain outstanding.
Small-model independent runtime and browser proof above passed. Companion remains the
faithful patch/history path; mixed F16 has only the recorded fixed-suite guarantee.
Vision/MTP/dflash remain separate unsupported features.

Raw large evidence: `mixed-export/evidence/supervised-27b-f16/` under the cache root,
including `plan.json`, `baseline.pt/json`, `F16.gguf`, export/validation sidecars,
per-phase watchdog logs, revoked permits and `external-base/manual-teardown.json`.
The full final offline regression passed270Python tests in21.09s;15bridge tests passed.
The large numerical results precede the offline-only watchdog cleanup correction.

To reproduce a separately authorized supervised phase, use a fresh exact-artifact
permit with recomputed forecast and the fixed `plan.json` (the archived permits are
revoked, and the current sequence is stopped):

```sh
PYTHONPATH=. python -m wsp.memory_guard \
  --log /outside/repo/new-phase/watchdog.jsonl --forecast-gib 23 --timeout 600 \
  --zero-growth --supervised-permit /outside/repo/fresh-active-permit.json -- \
  python scripts/validate_mixed_export.py \
  --out /outside/repo/fixed-plan --phase candidate --precision F16 \
  --candidate-file /outside/repo/F16.gguf
```

Use the pinned Python/bridge environment above. The permit must enumerate the exact
candidate hash and base provenance; create it only after explicit authorization,
never by copying/reusing a revoked file. Export, baseline and reload are separate
phases, with a fresh preflight and verified teardown between each. No large automatic
candidate search is provided. Do not rerun the stopped external path under these limits.
