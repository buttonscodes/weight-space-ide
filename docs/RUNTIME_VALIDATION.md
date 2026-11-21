# Portable runtime validation — 2026-11-14

This work was tested from an isolated checkout. The final integrated regression passed **343 Python tests in 50.62 seconds**, with
both pinned bridges configured. **8 frontend tests** and the unchanged agent
bridge’s **15 Node tests** passed.
JavaScript syntax and `git diff --check` passed. Python fixtures are synthetic;
no full-precision checkpoint or 27B model was loaded for this task.

```sh
# From weight-space-ide, using the same interpreter as serve:
WSP_GGML_BRIDGE=/path/to/libwsp-packed.dylib \
WSP_GGML_METAL_BRIDGE=/path/to/libwsp-metal.dylib python -m pytest tests -q
cd bridge && npm test
```

The exact primary interpreter had gguf 0.19.0, psutil 7.2.2, Uvicorn 0.54.0,
Torch 2.15.0 and Transformers 5.22.2. Tests cover no config, env-only settings,
profiles/CLI precedence, malformed settings, unavailable libraries, simulated
platform/RAM differences, source/model/patch binding, fail-closed telemetry,
expired reservations and real subprocess shutdown with open/closed SSE.
Linux/Windows configuration cases are **simulated**, not native runtime proof.

## Sequential real small-model evidence

Regular `python -m wsp serve --config /path/to/small.json --empty` was used on
isolated port 43879, with a separate workspace/cache. No bespoke acceptance launcher
or permit was needed. Config selected Metal, 64 context / 4 new tokens,
600-second residency, 20 GiB floor / 25 GiB preferred, 16 GiB process budget /
24 GiB ceiling, zero swap/compressor growth and one-second external sampling.
Each process tree was stopped and independently checked before the next server.

| Artifact | Actual tensor parameters | Result | Minimum headroom | Peak RSS / footprint / Metal GiB |
|---|---:|---|---:|---|
| Qwen3.5-0.8B Q4_K_M | 752,393,024 | Browser import, explicit base/current Run, unload | 50.548 GiB | 1.498 / 1.410 / 0.497 |
| Qwen2.5-0.5B-named Q4_0 | 630,167,424 | Explicit API Run, browser Systems window, model switch/unload | 64.816 GiB | 2.200 / 1.970 / 0.400 |

These are completion/setup mechanics checks, not editing-quality or long-context
benchmarks. Qwen3.5 base/current returned ` the capital of the`; Qwen2 returned
` Paris, and it` for `The capital of France is`. Qwen2's paired base/current API
operation took 1.540 seconds. Previously established editing/export evidence is
unchanged and is documented separately in QUANTIZED/QWEN38/MIXED_EXPORT.

Qwen2's Systems analysis read 17,088,512 logical values, largest slab 261,632 values,
with a 133,988,352-byte reduction-memory estimate. An old cached browser module
initially masked the controls; explicit cache refresh exposed the correct UI.
New JS/HTML responses request revalidation. An additional metadata-only browser
probe opened Trace and Tests and verified no bind event/reservation/model load;
Trace and Run tests now require explicit actions.

Both real runs maintained normal pressure and zero swap. Qwen2 logged its exact
bind baseline: zero compressor growth. Qwen3.5 enforced the same policy, but its
log predates serialization of that baseline: compressor maximum was below the
recorded prelaunch snapshot, with a 0.47 MiB rebound from its sample minimum.
Do not treat sample max-minus-min as growth from the phase baseline. The original
summary is preserved and qualified, not overwritten as stronger evidence.

The first setup attempt failed closed before inference because the watchdog logger
received a duplicate timestamp argument. Its log/teardown are preserved. The bug
was fixed and a real subprocess active-sampling regression added before the two
successful runs. Final source changes additionally hardened source-stat checks,
serialized runner construction and removed automatic GGUF tab inference; the final
325-test suite covers those changes. Real acceptance preceded the final commit.

## Local evidence

Raw evidence (not committed): `/private/tmp/wsp-portable-runtime-20261114/`:

- `final-pytest.log`, `browser-run.json/png`, `qwen2-run.json`, `systems.png`,
  `qwen2-wiring.json`, `trace-no-auto.json`, `tests-no-auto.json`;
- `summary-original.json`, qualified `summary.json`, `qwen2-summary.json`,
  `metadata-ui-proof.json` and PID/birth identity records;
- cache watchdog logs `serve-rqjsc5w7` (failed setup), `serve-ygq4353s` (Qwen3.5),
  `serve-ewceiao5` (Qwen2) and `serve-ckf49pam` (metadata-only tabs).

All final owned-process checks passed, return code 0 for successful servers,
remaining=[] and errors=[]. Failed setup teardown is preserved separately.
Port 8765, saved workspaces and a running app were not changed.


## Independent review corrections and integration

Review reproduced two lifecycle defects without models: a delayed first Run used
up its telemetry grace during the earlier metadata reservation, and constructor
errors could leave a phantom runner reservation. A separate runner-start clock now
starts only on the false-to-true transition, retaining the residency clock/baseline.
The real-supervisor/inert-child regression waits 5.3 seconds after metadata bind,
then survives 2.5 seconds of synthetic construction; a repeated bind cannot extend
the original missing-telemetry deadline. The expected stop reaps the owned tree.
Partial-construction tests cover failure before native initialization, after native
initialization, after starting the telemetry thread, and native-close failure.
Successful cleanup releases the reservation and retains an inspectable run error;
unknown cleanup stays armed and blocks reconstruction. No model load followed review.

Integration preserves fingerprinted immutable frontend assets, passive status,
explicit calculations, shared pending work, workspace/version/group/suite scope
checks and bounded cache semantics. Systems Cancel now explicitly cancels shared
work; navigation does not cancel another observer's calculation. Backend regressions
cover sharing, stale cancellation, cancelled-result eviction and explicit retry.
The browser rendered the actual integrated Systems controls/graph using explicitly
synthetic responses: zero requests at mount, one scoped request after Analyze,
one SVG and Complete. Evidence: `integration-ui-before.json`, `integration-ui-after.json`,
`integration-ui.png`, `integration-ui-cleanup.json`; the regular server had an empty
session and no bind event. This is UI integration proof, not another real-model run.
`integrated-pytest.log` and `review-frontend.log` retain final checks.
