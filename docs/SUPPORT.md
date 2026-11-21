# Tested quantized support

Validated on macOS arm64 with 128 GiB unified memory, ggml 0.29.0 CPU,
Transformers 5.22.2, PyTorch 2.15.0, gguf 0.19.0, NumPy 2.5.4 and psutil 7.2.2.
The original small-model matrix used sequential batch-one tests, context <=512,
generation eight tokens and seed zero under the external watchdog. A later explicitly
supervised27B sequence used64total context/4new tokens; its partial acceptance and
standalone behavioral failure are reported separately in [QWEN38.md](QWEN38.md).
[Machine-readable evidence](quantized-validation.json) contains exact revisions,
hashes, tensor types, timings, memory summaries and teardown outcomes.

| Checkpoint / file scheme | Stored logical parameters | Actual tensor types | Import/edit/run/restore/save/companion | Trace/intervention | Standalone merge |
|---|---:|---|---|---|---|
| Qwen2.5-0.5B-Instruct Q4_0 | 630,167,424 | F32, Q4_0, Q8_0 | Passed API and browser | Passed | Passed reload, **reduced edit strength** |
| SmolLM2-135M-Instruct Q4_K_M | 134,515,008 | F32, Q4_K, Q5_0, Q6_K, Q8_0 | Passed API | Passed | Not tested on this file |
| SmolLM2-360M-Instruct Q5_K_M | 361,821,120 | F32, Q5_1, Q5_K, Q6_K, Q8_0 | Passed API | Passed | Not tested on this file |
| SmolLM2-135M-Instruct Q8_0 | 134,515,008 | F32, Q8_0 | Passed API, real internal rank-one/sparse overlays | Passed | Passed reload; rounding measured |
| Qwen3.5-0.8B text Q4_K_M | 752,393,024 | F32, Q4_K, Q5_K, Q6_K, Q8_0 | Passed API, CPU and Metal | Residual/MLP/full-attention hooks passed | Passed Metal reload; target loss depends on edit magnitude |
| Qwen3.8-27B text Q4_K_M | 26,895,998,464 | F32, Q4_K, Q8_0 | Passed supervised API, edits/restore/save and exact companion; **27B browser incomplete** | Actual residual/full-attention/MLP intervention passed; tiny16:48 dense reference | File/reload passed; **gain preservation FAILED**,0.546856 to0.009325 |

File scheme names do not describe every tensor. The first Qwen file has a
separate Q8_0 output matrix; its stored count differs from the tied HF model.
Qwen3.5 is text-only: the original multimodal source has 873,438,784 parameters,
verified from source metadata, and was never loaded. Its source repo is embedded
in GGUF; the quantizer's precise source revision is not embedded. The observed
source revision/config and mapping qualifications are in [the guide](QUANTIZED.md).

See [Qwen3.8 work and Metal validation](QWEN38.md) for the new grouped-head mapping,
cache, convolution timings, exact27B results and revoked supervised gate. Faithful
27B edit delivery uses a companion plus its exact immutable base; full support is not claimed.

See [mixed-precision export](MIXED_EXPORT.md) for Qwen2 F16 and Qwen3.5 Q8 strict
gain-retention passes, independent llama.cpp portability and explicit validation status.
Actual27B F16 IDE reload passed strict fixed-suite criteria (100.829% gain retention);
external27B CPU baseline exceeded its memory budget, so external mixed and27B browser
acceptance remain incomplete. See the mixed guide for the preserved safety stop.

## Behavioral and export limits

The large output-row edit is an **intentionally destructive plumbing test**:
it causes repeated London, with KL 17.82 for the first Qwen and poor locality.
`within_limit=false` with `committed=true` is intentional existing behavior:
`preserve.max_kl` is advisory, now explicitly labelled and tested.

Separate bounded internal-layer rank-one reinforcement and sparse probes passed
on Qwen2.5 and SmolLM2-135M Q8. On SmolLM2, Paris-vs-London margin improved by
0.1175 from the rank-one patch; combined patches passed 5/5 target/preservation
cases, retained all unrelated top tokens and had held-out KL 0.00014. The fact
was already known. This is not a demonstrated localized new-knowledge edit.
Both patch kinds survived companion reload with zero added logit error.

The Qwen **Q4_0 internal matrix** was explicitly merged and reloaded with the
same suite, compared to original-base logits. Paris-vs-London margins were
7.1907 base, 7.4260 overlay and 7.2274 merged. All five cases passed, but the
merged gain of 0.0366 **failed the >=0.1 reinforcement goal**, retaining only
about 16% of the overlay gain. Maximum last-token logit error versus overlay
was 0.1091; maximum changed-row weight error was 0.0003495. Companion patches
remain the faithful edit representation. Standalone export loses undo and can
round away a useful change; success means valid reload, not edit equivalence.

The separate Q8 output-row merge had max weight error 0.00946 and max logit error
0.3364; its top token remained unchanged. All original GGUF hashes were unchanged.
Standalone export uses upstream quantization on <=1 MiB float slabs, preserves
untouched packed bytes, and retains original metadata with a provenance sidecar.

## Timing and memory

| Run | First logits (s) | Base generation tokens/s | Peak RSS / footprint (GiB) | Original minimum estimate, overstated (GiB) | Recorded free + file-backed minimum proxy (GiB) |
|---|---:|---:|---:|---:|---:|
| qwen2-export | 0.70 | 14.89 | 4.23 / 3.10 | 66.69 | 48.48 |
| smol135-q4km | 0.74 | 18.93 | 0.99 / 0.71 | 67.48 | 48.38 |
| smol360-q5km | 0.60 | 13.69 | 1.34 / 0.72 | 67.16 | 48.09 |
| smol135-q8-final | 0.68 | 21.98 | 1.48 / 1.04 | 66.78 | 48.42 |
| qwen35-q4km | 2.15 | 0.68 | 3.61 / 2.41 | 65.94 | 47.89 |

Review found the original macOS estimator counted anonymous inactive pages as
available. The original numbers and logs are preserved above, **not valid
reclaimable-headroom claims**. Retrospective proxies use recorded psutil free
(already excluding speculative) plus file-backed counters; separate sampling
times mean these are qualifications, not retroactively corrected measurements.
Phase 1's equivalent proxy minimum is 48.11 GiB. All remain above the unchanged
20 GiB stop floor. See [the source-backed correction](QUANTIZED.md#available-memory-correction).

After correction, `review-corrected-smol135` repeated full API mechanics on the
same 134,515,008-parameter Q4_K_M artifact: 12.36 seconds, baseline 50.99 GiB,
minimum 50.15 GiB using one `vm_stat` sample, peak RSS/footprint 1.01/0.73 GiB,
20.30 tokens/s base generation, no swap/compression growth, normal pressure,
return code zero and no remaining owned processes. Forecast was 6 GiB with
the same 25 GiB preferred headroom, 20 GiB floor and 16 GiB budget.

Every successful run ended return code zero with no remaining owned processes,
normal pressure and zero swap growth. Peak MPS tensor allocation was zero (CPU
backend); driver allocation was 475,136 bytes. Qwen3.5's reference CPU hybrid
activation kernels are slow enough to limit interactive use. Dense matrix
parameters are absent; Qwen3.5 materializes only 1.99 MB of small F32 norms,
biases and depthwise convolutions, alongside 521.56 MB packed tensor bytes.

## Offline and browser evidence

The full regression suite covers floating-point behavior, all thirteen allowed
block types against upstream decoding, sparse/block/low-rank algebra, and tiny
full Qwen2/Llama/Qwen3.5 models against decoded dense references. Llama tests
exercise rotary row permutations; hybrid tests exercise norm offsets, log-state
conversion and convolution reshaping. Construction refuses full matrix reads.
The full suite now passes 162 tests. Companion checks cover disabled/scaled
patches, tampering, default/named chat-template text and token-ID identity;
merge checks cover
layout, bounded row writes and immutable bases. Guard tests cover normal/warning/
critical, malformed/missing/denied telemetry, and termination of owned children
including a child in a separate process group while an unrelated sentinel survives.
Counter regressions cover missing/malformed/duplicate telemetry and an anonymous
inactive-heavy fixture that passed the old estimate but fails the corrected guard.

Historical companion tests proved completion-logit identity only: Jinja templates
were inadvertently omitted. The corrected smoke additionally proves identical
chat prompts before/after import and records template/prompt SHA-256 values in
`chat_portability`. Re-export older bundles from their original workspaces for
chat portability; the original artifacts/evidence remain unchanged.

Browser verification used only port 43879 and isolated roots: decoded
logical views, edit contrast, patch disable/enable, partial SVD and companion
reopen. A discovered chart overflow was corrected and rechecked in actual pixels
and DOM (panel width and scroll width both 530 px). Final confirmation exited
cleanly after 48 seconds. Port 8765 and existing workspaces were untouched.

## Unsupported or partial operations

- Unequal grouped heads are validated for the named Qwen3.8 text artifact and tiny
  2:6/16:48 fixtures; arbitrary other layouts remain unsupported. MTP tensors, MoE,
  vision and other GGUF architectures: raw-name inspection where decodable; generation refused.
- GGUF types outside the allowlist, arbitrary 3D/fused layouts, MLX packed/scaled
  safetensors and FP8: no logical edit/inference claim; explicit refusal.
- Qwen3.5/Qwen3.8 standalone export is implemented but lossy; the27B required gain
  preservation check failed. Exact companion is the faithful delivery path. Actual27B
  browser acceptance and behavioral-export warning/gating remain outstanding.
- PEFT LoRA export: written for floating-point `transformers` checkpoints
  (`POST /api/export_adapter`, [DIRECTIONS.md](DIRECTIONS.md#a-lora-adapter-peft)) and checked
  by model-free tests; not yet loaded with PEFT. Refused for a GGUF base (the companion bundle
  is the faithful export there); a llama.cpp LoRA is not written.
- Cross-quantization edit transfer, parameter gradients/training, non-Metal GPU inference,
  Linux native execution: not validated.
- Beyond these checkpoint/scheme pairs: extensibility goal, not universal support.

## Local machine only

Evidence root: `~/.cache/wsp-quantized-ide/evidence/`.
Successful run directories appear in the JSON. Browser images/logs are in
`browser-final`, `browser-final-layout` and `browser-final-confirm`.
Preserved failures: `phase1-attempt1` (pressure parser, setup only),
`phase1-attempt2` (rotary constructor, setup only), `phase2-smol135-q8` and
`phase2-smol135-q8-retry` (probe below downstream quantization resolution), and
`browser-final-layout` (240-second inspection timeout; all owned processes stopped).
The latter has a valid final screenshot but is not a clean teardown pass;
`browser-final-confirm` is the successful replacement. No failure was counted as
an inference pass or bypassed by weakening memory limits.
