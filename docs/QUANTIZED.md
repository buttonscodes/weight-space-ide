# Quantized checkpoint support

GGUF bases are immutable. Views reconstruct quantized values in logical tensor
coordinates; they do **not** recover the original pre-quantization weights. Edits
remain FP32 sparse, block, or low-rank companion patches. Runtime linear maps use
upstream ggml packed CPU or optional Metal multiplication plus separate FP32 delta multiplication.
There is no full-model dequantization fallback.

A single store read is limited to 16M logical values. A weight script that uses a
larger view whole (`x = mlp(41).W_gate` followed by arithmetic) has it decoded in
row slabs and held as FP32. A new whole read is refused when the whole reads the script still
holds would pass 320M values (`WHOLE_READ_VALUES` in `wsp/wscript.py`: three 17,408 x 5,120 MLP
matrices, never a 248,320 x 5,120 vocabulary matrix). Only those decoded tensors are counted: a
tensor computed from one (`x * 1`, `x.clone()`) is not, and tensor memory has no whole-script
total; memory admission bounds each operation on its own. Past the bound the script is refused with the
`for a in range(...): X = P("name")[a:min(a + R, hi), :]` loop and the largest `R`
that fits. This is a decoded copy of one view, never a write to the base.

Validated paths include small Qwen2, Llama/SmolLM2 and a narrow Qwen3.5 text
configuration, across four GGUF file schemes. See [SUPPORT.md](SUPPORT.md) for the evidence
matrix. Other architectures and quantizations are not automatically end-to-end
supported. The UI's **Capabilities** button explains operation-specific limits.
MLX packed integers, FP8 scaled weights, arbitrary fusions, MoE, vision and
MTP tensors remain unsupported. Unequal hybrid heads and optional Metal execution are covered
by [QWEN38.md](QWEN38.md); actual27B supervised acceptance is still pending.

## Install the optional backend

Use a separate Python 3.12 environment and keep dependencies/artifacts outside the
checkout. Tested versions are pinned in `requirements-quantized.txt`. Building
the native bridge requires CMake, a C++17 compiler, and upstream ggml **0.29.0**
(commit `57e2c2ca2a7cae756b1c4dfafa0cdfef9051d234`). Do not use a second OpenMP
runtime inside the PyTorch process: build ggml with OpenMP disabled.

```sh
python3 -m venv "$HOME/.cache/wsp-quantized/venv"
. "$HOME/.cache/wsp-quantized/venv/bin/activate"
pip install -r requirements-quantized.txt
git clone --depth 1 --branch v0.29.0 https://github.com/ggml-org/ggml /path/to/ggml-source
cmake -S /path/to/ggml-source -B "$HOME/.cache/wsp-quantized/build" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_METAL=OFF \
  -DGGML_BLAS=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF \
  -DCMAKE_INSTALL_PREFIX="$HOME/.cache/wsp-quantized/ggml"
cmake --build "$HOME/.cache/wsp-quantized/build" -j 4
cmake --install "$HOME/.cache/wsp-quantized/build"
python scripts/build_packed.py --prefix "$HOME/.cache/wsp-quantized/ggml" \
  --cpu-lib "$HOME/.cache/wsp-quantized/ggml/lib/libggml-cpu.dylib" \
  --out "$HOME/.cache/wsp-quantized/libwsp-packed.dylib"
export WSP_GGML_BRIDGE="$HOME/.cache/wsp-quantized/libwsp-packed.dylib"
python -m pytest tests -q
```

The native bridge and CPU library use `.so` on Linux; that platform's native
runtime is not yet validated here. The bridge has an ABI/version manifest and an
explicit tensor-type allowlist. It borrows mmap addresses, allocates only output
and activation scratch (native scratch capped at 128 MiB), and never rewrites
packed weights. Native ggml quantized dot products quantize activations too;
their error against a decoded float32 matmul is tested separately from the exact
FP32 overlay algebra.

## Open, edit, run, export

The normal open dialog accepts a local `.gguf` path or `.wsppatch.zip` bundle.
The tokenizer and model config are reconstructed from GGUF metadata. Opening
does not load an inference runner. Logical names match supported HF layouts;
Llama Q/K rotary row permutations are inverted. Unknown layouts retain their raw
names and refuse generation before loading.

For an explicitly authorized small-model run, use the external watchdog:

```sh
python -m wsp.memory_guard --log /path/to/evidence/watchdog.jsonl \
  --forecast-gib 7 --timeout 600 -- python scripts/quantized_serve.py \
  --model /path/to/qwen2.5-0.5b-instruct-q4_0.gguf \
  --root /path/to/isolated-workspaces --port 43879 \
  --stop-file /path/to/evidence/stop
```

Choose a free port and a fresh root/stop-file. Do not point validation at an
existing workspace or running server. The default guard requires forecast
headroom >=25 GiB, stops below 20 GiB available, and limits the process tree to
16 GiB (hard ceiling 24 GiB). It samples macOS available/reclaimable memory,
dispatch pressure, compression, swap and process physical footprint/RSS every
second, with in-process MPS allocation telemetry sampled every second once a
runner is active (CPU execution records zero MPS tensor allocation). Linux uses MemAvailable and PSI. Missing telemetry, warning pressure,
material swap/compressor growth, runaway allocation or timeout stop the owned
process group and leave a failure latch. Inspect and resolve the recorded cause
before another run; never weaken limits to make a model fit.

Generation is batch one, <=512 total tokens and <=64 new tokens. The acceptance
script uses eight. Each base/current call starts fresh; KV/recurrent state is
not reused across patches. This CPU implementation favors correctness and
bounded state over decode-cache speed. A fresh lease is required before every
forward call. Write the stop-file and navigate away from the page to end the
validation server; teardown evidence must report no remaining processes.

Existing weight scripts, patch toggles and strength controls work on reconstructed
logical weights. Low-rank overlays multiply their factors directly; sparse edits
scatter into output coordinates. Norm/bias edits use exact small-tensor restore.
Trace/intervention hooks work at supported layer/head/neuron boundaries;
parameter gradients are unsupported. Drift uses four <=32-token calibration
sequences sequentially, with that reduced sample explicitly labelled.

**Export** downloads a WSP companion bundle containing FP32 patches, tokenizer,
config and a manifest binding it to the exact base SHA-256. Opening that bundle
reproduces the patch stack, including disabled patches and strength. It requires
the original base file; `wsp.companion.import_bundle(root, path, base_path=...)`
can select the same hashed base at a new location. It is not a standalone GGUF,
a PEFT adapter, or a llama.cpp LoRA. The generic export endpoint remains companion-only.
The PEFT adapter export (`POST /api/export_adapter`) refuses a GGUF base: the patches were
computed against the reconstructed quantized weights, and the dequantized model an adapter
would apply to is never built; it is written for floating-point checkpoints only
([DIRECTIONS.md](DIRECTIONS.md#a-lora-adapter-peft)). A llama.cpp LoRA for GGUF is not written.
Because companion export never requantizes, its measured added quantization
error is zero; original GGUF reconstruction error is a separate quantity.
Portability also requires identical tokenizer and chat-template assets. The
manifest hashes `model/chat_template.jinja` and named templates in
`model/additional_chat_templates/*.jinja` (the actual Transformers 5.22.2
`CHAT_TEMPLATE_DIR`; its source comments call this `chat_templates`). The
`model/chat_templates/*.jinja` spelling is also retained if present. Tests compare
rendered prompts and token IDs for default and named templates after import.
Only these bounded paths are allowed; traversal aliases, symlink sources,
duplicate archive entries and checksum mismatches are refused. Existing bundle
size/count limits also apply to template assets. Bundles exported before this
fix omitted Jinja files: re-export the original saved workspace. Exact completion
logits alone did not establish chat portability for those older bundles.

**Directions on a GGUF base.** Behaviour directions (measurement, directional ablation,
ARA, scores) run on a GGUF base as on a checkpoint, and every edit is a float32 patch over
the immutable file. "As a chat message" renders with the GGUF's own
`tokenizer.chat_template`, which the open saved as `model/chat_template.jinja`; a GGUF
without one is refused with `this GGUF carries no chat template (tokenizer.chat_template)`.
The Run tab's turn-prefix chat is unchanged. The IDE writes no `generation_config.json` for
a GGUF. When one is copied into the workspace's `model/` (for example the source
checkpoint's, to compare responses with it), the GGUF generation loop, here and in the
model worker, adds its `eos_token_id` to the stop ids and applies its
`repetition_penalty` as `transformers`' `RepetitionPenaltyLogitsProcessor` does; any other
greedy key with a value that changes a response (`no_repeat_ngram_size`, `min_length`,
`suppress_tokens`, ...) is refused by name before generation starts. The file applies to
every generation on that workspace, and as a `model/*.json` file it travels in the
companion bundle. The measurement sees ggml's activation quantization, the directions are
the quantized model's own, ARA's consistency check compares the decoded weights' product
with the packed product (limit 0.05, unchanged), and the PEFT adapter stays refused:
[DIRECTIONS.md](DIRECTIONS.md#on-a-gguf-base).

## Explicit standalone export and its rounding loss

For Qwen2/Llama, an opt-in same-scheme merge copies the original packed GGUF and
requantizes only edited rows using upstream `ggml_quantize_chunk`. One float32
row/slab is bounded to 1 MiB. It handles sparse/block/low-rank patches, scaling,
disabled patches and Llama rotary row mapping. The base is hashed before/after;
the output requires a new filename. Original metadata is retained and a JSON
sidecar records hashes, patches and measured weight rounding error. Qwen3.5
standalone export is refused before allocation because its norm/state/layout
transforms need a separately validated inverse export mapping.

```sh
python -m wsp.memory_guard --log /path/to/export-evidence/watchdog.jsonl \
  --forecast-gib 7 --timeout 600 -- python scripts/quantized_export.py \
  --workspace /path/to/edited-workspace --out /path/to/new-edited.gguf
```

A standalone GGUF needs no companion to run, but loses undo history and may lose
the edit itself. In the real Qwen Q4_0 internal-layer test, the Paris-vs-London
margin gain fell from 0.235 (overlay) to 0.037 (merged): below the 0.1 improvement
goal. All five target/preservation cases still passed, because Paris was already
top before editing. Do not confuse that result with preserving edit strength.
The same suite was evaluated against **original-base** logits, not against the
exported model's own base. Maximum logit difference versus overlay was 0.109;
changed-row maximum weight error was 0.000349. Companion reload was exact.

`preserve.max_kl` is **advisory**: exceeding it reports `within_limit=false` but
still commits. Scope violations and invalid weights reject; KL violations do
not. Inspect the report and undo harmful edits. Acceptance's large city-token
edit is deliberately destructive plumbing validation, not successful localized
knowledge editing. A separate bounded rank-one reinforcement plus sparse probe
measures target and unrelated prompts; it demonstrates reinforcement of an
existing fact, not acquisition of a new one. Tiny edits can change internal
activations yet disappear at the next packed activation quantization step.

## Hybrid text mapping

The tested Qwen3.5 GGUF embeds `Qwen/Qwen3.5-0.8B` as its source. The observed
source revision is `2fc06364715b967f1860aea9cf38778875588b17` (873,438,784
parameters including vision), while the tested text GGUF has 752,393,024 values.
No source reference weights or vision projector were loaded. The quantizer's
exact source revision is not embedded, so it cannot be attested from metadata
alone. GGUF revision/hash are pinned in the support matrix.

Mapping follows installed Transformers 5.22.2 `gguf_config_mapping.py` and
`gguf_conversion_mapping.py`: zero-centred norms subtract one, `ssm_a` maps to
`log(-value)`, depthwise convolutions gain a singleton axis, and projections keep
logical names. This release accepts equal linear key/value head counts and no
MTP tensors; other variants retain raw names and refuse generation. All dense
linear maps stay packed; only small F32 depthwise convolutions/norms/biases are
materialized. Tiny hybrid full-model logits are checked against independently
reconstructed dense weights. Base→edited generation→base restoration and
save/reopen/companion reload were verified on the real hybrid checkpoint.

The CPU reference convolution and gated-delta activation implementations are
slow (about 0.66–0.68 tokens/s here). They do not dequantize model weights. There
is no retained KV/recurrent cache between calls; this implementation prioritizes
bounded correctness over interactive hybrid-generation speed.

## Reproduce acceptance

Download only a verified, authorized model strictly below 9B total parameters.
Phase 1 used `Qwen/Qwen2.5-0.5B-Instruct-GGUF` revision
`9217f5db79a29953eb74d5343926648285ec7e67`, file
`qwen2.5-0.5b-instruct-q4_0.gguf`. This file has 630,167,424 stored logical
parameters (GGUF has a separate output matrix), checked from all tensor shapes.

```sh
python -m wsp.memory_guard --log /path/to/acceptance/watchdog.jsonl \
  --forecast-gib 7 --timeout 600 -- python scripts/quantized_e2e.py \
  --model /path/to/qwen2.5-0.5b-instruct-q4_0.gguf \
  --repo Qwen/Qwen2.5-0.5B-Instruct-GGUF \
  --revision 9217f5db79a29953eb74d5343926648285ec7e67 \
  --out /path/to/acceptance
```

Add `--behavior --merged` for bounded internal low-rank/sparse probes and
standalone export/reload comparisons on supported Qwen2/Llama fixtures.

This exercises actual IDE API import/edit/run/toggle/export, then sequential
release/reopen/reload with complete logits comparison and base immutability.
Evidence JSON records versions, device/dtype, parameter counts, tensor types,
hash, timing, residency and exact behavior. The deliberately strong city-token
edit is an interoperability test and has large measured drift; it is not a claim
of a high-quality factual edit. Browser acceptance is separately recorded.

## Backend investigation

[Current Transformers documentation](https://huggingface.co/docs/transformers/v5.22.2/quantization/gguf)
describes a packed MPS route and a full-dequantization fallback. Installed
5.22.2 source in `quantizers/quantizer_gguf.py` confirms that fallback. The
`ggml-org/ggml-quantization` Hub API and README returned HTTP 401 in this test
environment; this is environment-specific evidence, not a general availability
claim. The IDE's explicit ggml CPU bridge never calls that loader.

Pressure parsing was checked against Apple XNU commit
`f6217f891ac0bb64f3d375211650a4c1ff8ca1ea`:
[dispatch-level sysctl](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/kern/kern_memorystatus_notify.c)
and [pageout counters](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/vm/vm_pageout.c).
`kern.memorystatus_vm_pressure_level` is 1/2/4 (normal/warn/critical).
`vm.memory_pressure` is pageout activity, not that level. A setup-only failed
attempt exposed the distinction; its log is preserved and is not an inference pass.

### Available-memory correction

The guard now estimates headroom as `vm_stat Pages free + File-backed pages`.
It requires and records speculative and purgeable counters but gives neither
additional credit. This deliberately conservative estimate excludes anonymous
inactive pages and avoids assuming purgeable/file-backed disjointness. File-backed
reclamation can require writeback; pressure, swap and compression checks remain
independent stop conditions, not a guarantee that every byte is immediately free.

Apple system_cmds commit `408bba7453608006b89772db185defbac8fe2fd0`,
[`vm_stat/vm_stat.c`](https://github.com/apple-oss-distributions/system_cmds/blob/408bba7453608006b89772db185defbac8fe2fd0/vm_stat/vm_stat.c),
already prints free as `free_count - speculative_count`. XNU commit
`f6217f891ac0bb64f3d375211650a4c1ff8ca1ea`,
[`host.c`](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/kern/host.c)
and [`vm_statistics.h`](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/mach/vm_statistics.h),
define Mach free as including speculative and external pages as file-backed.
Do not subtract speculative again from the CLI free counter or add it separately.
Missing/malformed/duplicate critical counters and impossible totals fail closed.

The original matrix used psutil's free+inactive estimate and overstated headroom.
Its raw logs and historical values remain preserved; `docs/SUPPORT.md` and
`quantized-validation.json.review_corrections` qualify them with conservative
recorded free+file-backed proxies and a new smoke under the corrected estimator.
Thresholds remain unchanged.
