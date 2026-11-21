# Capacity-aware interactive runtime

Normal interactive profiles now default to **8,192 total context / 2,048 output**.
The configured context includes every prompt/history/template/response token;
the requested output ceiling is reduced to the remaining context after tokenizing
the entire prompt. No history is truncated. An oversized prompt or zero remaining
response space fails explicitly; Run's zero-output next-token scoring accepts an
exactly full prompt. The model's declared window bounds usable context. There is
no arbitrary application context/output ceiling beyond exact JSON integer
representation; conservative admission and live supervision still apply.
Existing explicit profile/CLI/environment overrides still win.

## Execution and ABI compatibility

The installed ggml 0.29.0/ABI 1 bridges are unchanged. Python partitions matmul token
rows into CPU slabs of at most 2,048 and Metal slabs of at most 512, further bounded
by Metal's 128 MiB input/output formula. Native scratch/allocation checks remain.
Only native error−12 (capacity rejected before computation) retries smaller slabs;
other native failures remain errors. A single unsupported row fails closed. Packed
weights remain packed and mapped; no full weight reconstruction was introduced.

Generation uses 128-token prefill chunks into one pinned Transformers 5.22.2 cache,
then single-token decode. Cache-derived positions and masks continue across chunks,
including hybrid recurrent/convolution state. Each call owns its cache; variants,
edits and failed requests do not share it. Next-token display has a separate bounded
`next_logits` path. Existing full-position logits and trace behavior is preserved.
Packed CPU/Metal telemetry does not initialize PyTorch MPS: ggml Metal reports its
own allocations, while missing/stale/error native telemetry remains fail-closed.

## Forecasts and admission

Interactive estimates count packed bytes, runtime/temporary/patch allowances,
actual KV heads and head width across full-attention layers, recurrent/convolution
state across hybrid layers, one cache-append copy, repeated K/V workspace,
last-token vocabulary logits and bounded-query eager attention. The latter is
`3 × query_heads × min(context,128) × context × 4 bytes`. Metadata omissions use
conservative fallbacks. Estimates assume batch one FP32 activations.

Plain full forwards reserve full-sequence logits and transient full attention at
the actual requested sequence length. They retain all-position logits semantics
for edits/tests/drift without charging for retained trace tensors. Traces with
capture names or hooks additionally reserve activation retention and all
full-attention-layer patterns. Interactive forecasts use effective
`min(configured context, model window)` consistently in Settings and admission.
Operation working forecasts use actual sequence length and end with the job; a
past trace does not permanently increase later chat admission. Nested forwards
remain within the owning job's lease and fixed measured-process cap. Patches are
preflighted before allocation and reconciled to committed storage at finish.
Loaded operations charge newly allocated buffer components against fresh available
memory without crediting resident or file-backed pages. See [runtime policy](RUNTIME.md)
for the fixed per-operation process cap, cooperative abort/unload and the last
resort, which stops only the model worker process and never the server.

The forecast bounds the model's memory, measured as the worker process (the
larger of footprint and RSS) plus server growth since the reservation's first
admission. The server's own pre-model footprint (about 0.9–1.0 GiB, Python +
torch) is not part of the forecast or of an operation's limit: a loaded
operation and an idle model may exceed the forecast by that amount, and a
running cold load may exceed its admitted budget (available − preferred reserve,
which can be above the forecast) by that amount, always within the static caps.
That server footprint is fixed at the reservation's first admission (usually
when the workspace is opened); later server growth counts as model memory. The
static caps (budget, ceiling, total − preferred reserve), the floor and pressure
checks still use whole-application and machine figures.

Settings separately labels the active operation limit and idle static cap,
raw free versus reclaimable available RAM, estimates versus allowed budget, and
model context versus native call size. Saving never restarts or hot-applies policy.

Use **Use capacity-aware defaults** to migrate an old 64 / 4 or24 / 26 GiB profile.
It proposes context/output/lifetime and explicitly sets budget, ceiling and minimum
reservation to Auto; deliberate floor/preferred reserves and unrelated settings
remain. Review, Save, then restart explicitly. The preset uses a bounded binary
search for the largest forecast-feasible native model window, including above
8,192. Cold recommendations use measured available RAM minus preferred reserve.
Resident recommendations use the physical ceiling and any retained launch caps;
these are uncertain upper bounds, not predictions of post-unload availability.
A model that cannot fit even a two-token generation window has no preset.
Known physical impossibility still warns when the active reservation is unknown.
Missing model metadata uses clearly qualified fallback values. No recommendation
guarantees an operation will fit.

## Evidence and limitations

Model-free checks exercise mocked native errors/offsets, the pinned cache/mask and
hybrid recurrence/convolution functions on tiny tensors, fake generation/history,
long-output route behavior, geometry-aware forecasts, high-water reservations,
healthy trace refusal, persistent settings, transport and shutdown.

Installed ABI 1 CPU and Metal bridges were additionally tested with a 17×64 raw
synthetic matrix and 513 / 2,049 / 8,201 activation rows. Binary-exact bounded inputs
matched the dense reference exactly; only kilobytes of synthetic matrix data were
mapped. An initial non-binary ramp test observed a Metal dense-reference relative
difference of7.59e−5 (absolute 0.01656, one of 8,721 values at 513 rows); this is retained
as precision evidence, not hidden by claiming general exact FP32 equivalence.
The chunked ramp output also matches concatenated single-slab native calls exactly,
separating wrapper reconstruction from the native dense-reference discrepancy.
No native bridge was rebuilt or replaced.

A directory/header-only check of the existing Qwen3.8-27B-Q4_K_M artifact at 8,192
context estimated23.893 GiB, including18.061 GiB packed weights and1.000 GiB KV.
With a hypothetical71 GiB available and 25 GiB preferred reserve, Auto budget is
46 GiB. This comparison does not reserve memory or prove inference capacity.
No tensor data, model construction, model load, training or inference was part of
that check. No real-model long-context performance or quality acceptance is claimed.
Legacy acceptance-harness 512-context/64-output limits and explicit permits remain
separate and unchanged.

Scalar forecast results are reused across token lease checks. Workspace/version,
patch storage, configured/model context and operation high-water changes invalidate
the cache. Source identity and live lease checks still run; caching never admits
execution or renews the reservation. Optional GGUF KV-head and feed-forward metadata
use the same conservative defaults as runtime forecasting.

Preset recommendations intentionally stay below the known admission/physical cap
by `max(512 MiB, 5% of cap)`, for both cold and resident searches. The displayed
recommendation includes this buffer and the resulting forecast target. It leaves
slack for availability jitter and modest edits without changing admission limits,
claiming allocated memory, or guaranteeing future fit. An insufficient cap blocks
the preset rather than reducing the buffer. Total-context clamping remains scoped
to GGUF; other runners retain their existing sliding-window generation semantics.
