# Qwen3.8-27B text backend — partial supervised acceptance

The implementation now maps unequal hybrid heads and offers packed Metal execution,
with ephemeral decode caches and the existing FP32 edits/hooks. **Actual 27B original/edit
and exact companion acceptance passed; same-scheme standalone gain preservation FAILED.
A later mixed F16 IDE reload passed strict fixed-suite criteria, but external27B portability
and27B browser acceptance remain incomplete.** Full support is not claimed. Use the
companion plus exact immutable base for faithful edit delivery; same-scheme standalone
GGUF export is lossy. The supervised permit is revoked. Do not launch unattended or
reuse it. An existing listener on the default port 8765 and its workspaces are not test fixtures.

The [mixed-precision follow-up](MIXED_EXPORT.md) records the new F16 result and the
external CPU safety stop. Historical measurements below are preserved unchanged.

## Exact artifact and provenance

- `ggml-org/Qwen3.8-27B-GGUF`, revision `c1869413dece09109e55b7dcc2c77ef382b0d0bd`.
- `Qwen3.8-27B-Q4_K_M.gguf`,19,403,684,480bytes; SHA256
  `d15d4b6de55f662ac40e867702d31afe738d032fa06b829b2b1a21a49d4a48ee`.
-851 tensors,26,895,998,464 stored text parameters; actual tensor types F32/Q4_K/Q8_0.
- PRIMARY source `Qwen/Qwen3.8-27B` revision
  `37561816e66b5b9153b1b64c10968c18c1e6f4a0`; source includes vision/MTP, selected
  converter invocation explicitly uses `--no-mtp`. Separate mmproj/MTP/dflash files
  are not downloaded or loaded. Vision, speculative decoding and MoE are unsupported.
- Source and converter evidence is pinned in the external artifact directory below;
  converter is llama.cpp `f477f795cb071baf5b8af6488378366a72034ac4`,
  [`conversion/qwen.py`](https://github.com/ggml-org/llama.cpp/blob/f477f795cb071baf5b8af6488378366a72034ac4/conversion/qwen.py)
  and [`src/models/qwen35.cpp`](https://github.com/ggml-org/llama.cpp/blob/f477f795cb071baf5b8af6488378366a72034ac4/src/models/qwen35.cpp).

The converter stores value heads in tiled `[V/K,K,D]` order; Transformers uses
`[K,V/K,D]`. Logical reads/runtime/export invert this for QKV value rows, z,
alpha/beta, A_log/dt_bias, convolution channels and output-projection columns.
Synthetic2:6 and16:48 fixtures compare every logical tensor and complete logits
with an independently reconstructed dense upstream model, including recurrent
state, repeated grouped heads, convolution, normalization and gates. Full attention
uses sigmoid output gating; linear attention uses SiLU. Full Q projections interleave
per-head Q/gate; attention traces report actual full-attention layer indices.

An all-meta skeleton verified all851 actual shapes with zero weight payload reads.
This is a mapping check only. No original full-precision27B reference was loaded.

## Tokenizer and chat

The exact source tokenizer uses NFC normalization, omitted by Transformers5.22.2
GGUF conversion. A named Qwen3.8-27B profile restores NFC; vocabulary and merges
remain artifact-derived. Source shared vocabulary/token IDs, decomposed Unicode,
multilingual/code probes and chat prompts match after correction. Initial ASCII-only
comparison is preserved and is insufficient by itself. GGUF pads vocabulary to248320;
source shared vocabulary has248044 entries. Stop IDs248046 and248044 correspond to
`<|im_end|>` and `<|endoftext|>`.

Chat template SHA256 is
`af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375`.
The Chat tab exposes model-default/off/low/medium/xhigh thinking. API `chat_options`
accepts `enable_thinking`, `reasoning_effort`, `preserve_thinking`; unknown options
or multimodal content fail before a runner load. Default source thinking can use
many prompt/output tokens; initial supervised tests explicitly disable it.
Companions retain the tokenizer JSON, configuration and Jinja template bytes.

## Optional Metal build

Use Python/dependency versions from [QUANTIZED.md](QUANTIZED.md), plus Xcode command
line tools, CMake and macOS Metal. Build a separate pinned ggml0.29.0 prefix with:

```sh
cmake -S /path/to/ggml-source -B /outside/repo/ggml-metal-build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_BLAS=OFF \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF \
  -DCMAKE_INSTALL_PREFIX=/outside/repo/ggml-metal
cmake --build /outside/repo/ggml-metal-build -j4
cmake --install /outside/repo/ggml-metal-build
python scripts/build_packed_metal.py --prefix /outside/repo/ggml-metal \
  --metal-lib /outside/repo/ggml-metal/lib/libggml-metal.0.29.0.dylib \
  --out /outside/repo/libwsp-metal.dylib
export WSP_GGML_METAL_BRIDGE=/outside/repo/libwsp-metal.dylib
export WSP_GGML_DEVICE=metal
# WSP_GGML_BRIDGE from QUANTIZED.md is also required for explicit export quantization.
```

Metal borrows the immutable GGUF mmap through upstream no-copy host buffers; it
never constructs dense matrix parameters. Per-operation activation allocations are
capped at128MiB. Activations, deltas, recurrent state and Python hooks remain FP32
on CPU, with packed matrix multiplication on Metal. Unsupported kernels fail;
there is no dense or CPU fallback when Metal is selected. Metal's actual
`MTLDevice.currentAllocatedSize` and PyTorch MPS driver allocation are sampled
alongside process-tree RSS/physical footprint. CPU remains the explicit default.

Upstream macOS depthwise `conv1d` was the measured bottleneck:11.6/14seconds in a
three-token profile versus0.21seconds for packed Metal matmuls. A bounded four-tap
cross-correlation replaces only those two functions in each runner's pinned upstream
forward, without changing global Transformers functions. Dense convolution/state
and whole-model reference tests cover it. Decode KV/recurrent/conv caches live only
inside one locked generation call and are discarded on success/error; variant,
patch, scale, undo, reopen and intervention calls cannot inherit stale state.

## Small validation before supervised27B

Sequential Qwen3.5-0.8B Q4_K_M (752,393,024parameters) acceptance on Apple M3 Max:

| Path | End-to-end generation | Prefill | Seven cached decode steps |
|---|---:|---:|---:|
| CPU cache, upstream convolution |0.68 tokens/s|1.44s|9.97s|
| Metal, upstream convolution |0.66 tokens/s|1.49s|10.33s|
| Metal, bounded convolution |9.95 tokens/s|0.075s|0.414s|

Eight output tokens, five prompt tokens, batch1, seed0; timings include reported
cold/hot distinctions in raw evidence. These are not27B throughput estimates.
Optimized full acceptance completed in97.36s, normal pressure, no swap growth,
all owned processes exited. Companion chat/logits were exact. Rank-one+sparse
factual reinforcement passed5/5 cases, but standalone Q6_K internal-row rounding
lost the target gain (4/5); preservation still passed. This is an explicit quality
loss, not an export failure hidden as equivalence. Companion retains exact edits;
standalone keeps original schemes, loses undo, and must be evaluated separately.

Evidence root on the builder machine:
`~/.cache/wsp-quantized-ide/qwen38/`.
`metadata.json`, `metadata-plan.json`, `tokenizer-corrected-evidence.json`,
`evidence/small-cpu-cache`, `evidence/small-metal-full`, `evidence/profile-metal`,
`evidence/small-metal-conv` preserve provenance, timings, telemetry and teardown.
No model payloads/builds/environments are committed.

Final offline regression:232 Python tests with Metal enabled and15 native-agent bridge
tests. Browser acceptance on43879 passed explicit four-token completion, thinking-off
chat, an internal edit, patch enable/disable and capability messages. GGUF Run now
requires an explicit click on first open and after edits; opening a tab does not
start inference. The final three-process small acceptance also preserves scope,
candidate tests, advisory drift and patch history: exact companion, standalone
target gain0.160 versus overlay0.693, all3 unrelated top tokens retained, but the
standalone Paris top-token case fails (3/4 suite versus4/4 overlay/companion).
[Consolidated evidence](qwen38-validation.json) records memory and teardown; the
final staged runs peaked at2.54GiB RSS/2.27GiB footprint/0.53GiB Metal allocation,
with minimum45.84GiB available, normal pressure and no swap/compressor growth.
The browser-only run recorded240KiB compressor growth under the historical1GiB
material-growth allowance; it is preserved, not reported as zero. Supervised27B
uses zero-growth thresholds and would stop on that increase.

## Supervised decision gate

Default inference remains strictly below9B. Only with explicit, supervised approval
may an operator create a short-lived approval JSON containing:
`supervised:true`, `approval_message`, exact `base_sha256`, `parameters`, future
`expires` (at most one hour), `context`, `new_tokens`, `forecast_gib`, `budget_gib`,
`ceiling_gib`, `timeout`, and explicit `allow_derived_exports:true` if approved.
Do not create that file from these instructions alone.

The initial64-token forecast is24,441,583,616bytes (22.76GiB), including18.06GiB
packed weights,2GiB runtime, KV/activations, recurrent/conv state,1.5GiB temporary
allowance and patch copies. Proposed reservation23GiB, process budget24GiB,
hard ceiling26GiB requires fresh availability of at least48GiB to preserve25GiB
forecast headroom. Runtime floor20GiB and pressure/telemetry checks remain unchanged; supervised
mode tightens swap/compression allowances to zero growth. Budget selection requires explicit gate approval; the small-model
16/24GiB policy remains unchanged. No polling until conditions improve.

After approval, from `weight-space-ide` with the pinned environment/bridge variables:

```sh
PYTHONPATH=. python scripts/quantized_phases.py \
  --model /outside/repo/Qwen3.8-27B-Q4_K_M.gguf \
  --out /outside/repo/fresh-supervised-evidence --permit /outside/repo/operator-approval.json
```

This runs original/edit/export, companion/reopen, then standalone in separate
watchdog-owned processes. Each phase must exit cleanly before the next; any failure
latches and stops the sequence. No retries. First generation is four tokens; a
60-second smoke aborts the longer phase as impractical. Logs retain target/preservation
metrics against original base logits, export loss and exact companion behavior.
Revoking/changing/expiring approval terminates only the owned process group.
Browser acceptance uses a separately approved guarded server on reserved43879,
with a fresh workspace root and stop-file; never open an existing workspace.

Pending before claiming full support: standalone required gain preservation and
actual27B browser acceptance. Short-context performance is measured below; long-context
interactive performance is unvalidated. Vision/MTP/dflash, arbitrary architectures,
PEFT/llama.cpp LoRA export and full-precision equivalence remain unsupported.

## First supervised attempt — preserved preflight refusal

The proposed sequence was explicitly approved.
The initial snapshot had49.604GiB available (26.604GiB after the23GiB reservation),
normal pressure and zero swap. The actual watchdog invocation immediately refused:
`Forecast leaves less than25GiB available headroom`, before `subprocess.Popen`.
For this first attempt, no child was spawned, no27B model loaded, and no throughput,
peak-model-memory or behavioral results were produced. Companion, merged and browser phases
were not started; no retry occurred, and the permit was revoked.

The refusal-time numeric snapshot was not persisted by the old error path; the
earlier launch snapshot and later diagnostic sample are distinctly labelled, not
substituted for it. The audit path now records/latches rejected preflights before
spawn; an offline test proves no child is created. The initial60s smoke deadline
is also enforced by the external watchdog, rather than checked only after return.
Evidence: `qwen38/evidence/supervised-27b/original.stdout.log`, `failure.json`,
`launch-preflight.json` and `revoked-approval.json`. Another attempt requires fresh
explicit authorization; this stopped sequence does not authorize a retry.


## Fresh supervised attempt — original/companion passed, standalone quality failed

One fresh attempt was explicitly authorized; this did
not retry under the revoked first permit. Original/edit/export, saved reopen/companion,
and standalone reload ran sequentially under a new exact-artifact 40-minute permit.
Each phase took a fresh baseline, reserved 23 GiB, enforced 24 GiB budget/26 GiB ceiling,
25 GiB preferred forecast headroom/20 GiB hard floor, normal pressure and zero swap or
compressor growth. Context was capped at 64 total tokens, generation at 4, batch 1,
seed 0; first smoke had an external 60 s deadline, each phase 600 s. All three processes
returned 0 and watchdog teardown recorded `remaining=[]` before the next process.

| Phase | Baseline available | Minimum available | Peak RSS | Peak footprint | Peak Metal/MPS driver | Elapsed |
|---|---:|---:|---:|---:|---:|---:|
| Original/edit/export |84.033 GiB|63.830 GiB|2.562 GiB|2.101 GiB|18.077 GiB|229.17 s|
| Saved reopen/companion |83.621 GiB|63.834 GiB|1.843 GiB|1.763 GiB|18.072 GiB|47.42 s|
| Standalone reload |83.616 GiB|63.807 GiB|2.086 GiB|1.896 GiB|18.072 GiB|37.12 s|

RSS alone does not represent the packed Metal allocation; the watchdog checks the
native Metal/MPS metrics too. Every phase recorded pressure 1, zero swap growth and
zero compressor growth relative to its own fixed baseline. Existing compressor
occupancy was not counted as growth. The original GGUF hash remained unchanged.

Completion from a five-token prompt produced ` Paris.\nThe`; thinking-off 15-token chat produced
`Hello! How can`. The first completion including cold setup took 2.335 s (1.71 tokens/s),
with 0.542 s prefill and 0.742 s for three cached decode steps. Warm edited completion
took 1.295 s (3.09 tokens/s), with 0.375 s prefill and 0.742 s decode; companion and standalone
completion took 1.198 s and 1.219 s. These short bounded measurements support usable
short prompts on this M3 Max, not long-context or general interactive performance.

The internal `model.layers.63.mlp.down_proj.weight` rank-one, sparse and block edits
all changed logits. Scope rejection, candidate tests, advisory drift, patch history,
disable/re-enable, scaling, base-versus-edited restoration, residual/full-attention
traces and a zero-MLP-output intervention passed. This reinforced an already-known
fact; it does not demonstrate acquisition of new knowledge. Saved workspace reopen
and companion import produced **zero added logit error**, identical chat prompt,
completion/chat token IDs, task and patch metadata. Overlay and companion target
and preservation suite passed 4/4, with 3/3 unrelated top tokens retained.

**Standalone required gain preservation FAILED.** The Paris-minus-London target
margin was 5.976147 base,6.523003 overlay and 5.985472 merged. Gain fell from 0.546856
to 0.009325 (98.3% lost), below the explicit 0.1 minimum. Although Paris remained top-1
and the coarse target/preservation suite passed 4/4, that does not satisfy the gain
requirement. Maximum merged-versus-overlay logit error was 0.275414; target KL was
0.008482. All 3 unrelated top tokens remained and maximum unrelated KL versus the
original base was 0.000001340. File creation/reload succeeded, edit equivalence did not.

The streamed export requantized 5120 edited rows into their original Q4_K scheme with
a 1 MiB float slab cap; maximum changed-row weight error 0.003249636 and RMS0.000004775.
Derived SHA256: `52560a5d8f66bbed89ae7b83d5390590e80dab9c4ad394de391fdeb48d205e24`.
The companion preserves FP32 edits and undo; the standalone loses patch history and
can erase useful edits. Neither reconstructs original pre-quantization precision.

The sequence stopped on this behavioral failure. The hold was
confirmed; the fresh permit was moved to `revoked-approval.json`. **No 27B
browser server was started**, no further load/retry is authorized, and all owned
processes are gone. The separate small-model browser evidence remains valid only
for that small model. No existing workspace was changed.

Concrete follow-up: the current UI's Capabilities dialog says merged export
“introduces rounding” and its Export action downloads a companion. It has no
post-export gain-preservation warning/result, and the coarse 4/4 badge can conceal
this loss. The staged script records `retains_minimum_target_gain=false` but returns0;
operators must inspect this semantic result, not infer acceptance from process exit.
Before future automated acceptance, make that result an explicit failure gate and
expose the behavioral export warning. That follow-up and actual 27B browser validation
remain outstanding; this reporting pass changes no runtime code and loads no model.

Raw evidence is `qwen38/evidence/supervised-27b-attempt2/`: `original.json`,
`companion.json`, `merged.json`, `first-smoke.json`, each phase's `watchdog.jsonl`
and stdout log, `expected.pt`, `quality-hold.json`, `revoked-approval.json`,
`edited.wsppatch.zip` and `edited-standalone.gguf`. The committed
[qwen38-validation.json](qwen38-validation.json) includes exact metrics and hashes.

### Reproducing a future supervised phase

This is the exact launch shape used for each phase (`original`, `companion`,
`merged`), from the IDE's directory (the one that holds `wsp/`). **It is a recipe, not
permission to run; a new explicit supervised authorization and fresh permit/evidence
directory are required.**
Set the pinned Python/bridge paths described above, then run one phase at a time:

```sh
PYTHONPATH=. WSP_GGML_DEVICE=metal \
  WSP_GGML_BRIDGE=/outside/repo/libwsp-packed.dylib \
  WSP_GGML_METAL_BRIDGE=/outside/repo/libwsp-metal.dylib \
  /outside/repo/venv/bin/python -m wsp.memory_guard \
  --log /outside/repo/fresh-evidence/original/watchdog.jsonl \
  --forecast-gib 23 --timeout 600 --supervised-permit /outside/repo/fresh-approval.json \
  -- /outside/repo/venv/bin/python scripts/quantized_phases.py \
  --model /outside/repo/Qwen3.8-27B-Q4_K_M.gguf \
  --out /outside/repo/fresh-evidence --phase original
```

A future separately authorized browser phase uses `scripts/quantized_serve.py` under
the same external guard, isolated root, port 43879 and owned stop-file. Open that
listener, inspect reconstructed weights and Capabilities, choose Run/completion
with 4 tokens or Chat/thinking off with 4 tokens, then apply a bounded internal edit
and use the patch enable/strength controls. Re-run explicitly after edits; preserve
screenshots/results and stop the server via its stop-file. This 27B UI workflow has
**not** been validated in the present sequence and must not target port 8765.
