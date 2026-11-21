# The atlas

A map of every unit of the loaded model, built once and consulted while editing, so that an
edit starts from "these are the neurons that carry this" instead of from a blank matrix.

```
python -m wsp atlas --model HuggingFaceTB/SmolLM2-135M-Instruct          # build (resumable)
python -m wsp atlas --model NAME --show overview | L28N419 | concepts    # print
python -m wsp atlas --model NAME --stages weights --fetch                # a model still on the Hub
```

or the **Atlas** tab → Build. It is stored in `<workspace>/atlas/` as `.npz` / `.json`.

## What it contains

| Stage | Needs | Gives |
|---|---|---|
| `weights` | nothing run; works for models too large to run and for any store the IDE can read | per neuron: geometry of its read and write vectors, the tokens its write vector promotes and suppresses, the current tokens and the predicted tokens its gate opens on, classes; token concepts; per-layer profile and stages; residual coordinates that are written far more than read; a weights-only record per attention head |
| `wiring` | nothing run; quadratic in neurons, so it runs by default only when cheap (the browser never selects it by default). Saved after every layer; a stopped build continues where it stopped | per neuron: the later neurons that read what it writes, and the earlier ones it reads |
| `observed` | a model that can be run: in this process, or in the model worker for a GGUF model (one window per job; built there only when asked for) | per neuron: the sign of its strong activations, the contexts where it fires hardest; per head: where it attends; next-token loss; a direction on which the final stream is constant (for token shifts); and a test of each weights-only claim |

A neuron is the vectors it reads the stream with and the vector it writes:
`act(g·x) · (u·x) · d` for gated MLPs, `act(r·x + b) · d` for plain ones. The normalisation in
front of the MLP is folded into the reads and the final normalisation into the unembedding
(including LayerNorm centring, and the `1 + weight` gain of Gemma and Qwen3.5 checkpoints).

**Classes** are rules, not judgement. With τ = 5/√d_model (five standard deviations of the
cosine of two random directions; 0.078 at 4,096 dimensions):

| Mechanism | Rule |
|---|---|
| sharpener | cos(value, write) > τ and \|cos(gate, value)\| > τ |
| amplifier | cos(value, write) > τ and \|cos(gate, value)\| ≤ τ |
| damper | cos(value, write) < −τ |
| router | otherwise |

A plain MLP has one read, so it has no amplifiers. **Vocabulary role** (`diffuse`, `few`,
`set_pos`, `set_neg`, `contrast`) and **triggers** compare a neuron's token list with the same
list for 2,048 random directions at its 99.9th percentile (`null.json`). **Concepts** are
spherical k-means clusters of the unembedding (√V / 2 of them, at most 256), named by their
central tokens; some are held together by spelling rather than meaning. **Stages** are the
contiguous runs of layers with the least within-run variance of the class shares, described
by the traits that set each apart; they are not named.

**Read a neuron's direction from its strong activations (`obs_peak`), not from its mean.** A
closed gate still leaks a small value at most positions, often of the opposite sign.

## Using it to edit

| Where | What |
|---|---|
| Atlas tab → Find | neurons by word, concept, class, layers and effect → a weight script in the console |
| Atlas tab → Attribute | for a prompt and a target token, the neurons that push it there → a script that scales the top few |
| Weight scripts | `select(word="red", effect="promote", layers=[19, 31])`, `scale_neurons(L, [...], f)`, `shift_tokens(word_tokens("red"), 4)` |
| Saved selections | Find and Attribute results, or ticked rows, saved as a named selection in one step; the inspector then scales, ablates or reverts them as one patch ([SELECTIONS.md](SELECTIONS.md)) |
| Programmer model | the `atlas` tool (overview, unit, search, concepts, attribute, shift, build) and a section of the brief |

`effect="promote"` keeps the neurons that raise the target when they fire hard. A token shift
uses the model's logit bias where it has one; otherwise it adds a multiple of the fitted
constant direction to the token's unembedding row (on a tied model that also moves the token's
input embedding, and the IDE says so).

The atlas describes the weights it was built from. Layers edited since are flagged in the
tab, on cards and in `Atlas.status()`; rebuild with "current weights" to map an edited model.

## What was checked

- **By construction** (`tests/test_atlas.py`, offline): a neuron written by hand is read back
  as written; attribution per layer equals what that layer's MLP wrote, projected on the
  target; a token shift moves exactly the requested tokens by the requested amount; selections
  scale exactly the selected neurons; and the same build path runs on tiny Llama (grouped
  K/V), GPT-2, GPT-NeoX, Phi-3 and GPT-Neo models, where the stored write signature equals the
  direct logit effect the rest of the IDE computes.
- **On a real model run here**, `HuggingFaceTB/SmolLM2-135M-Instruct` (46,080 neurons): full
  build in about two minutes on four cores. Results are in the table below.
- **Against an independent map**: see "Qwen3.5-9B" below.
- **On a quantized store**: with this branch merged locally onto the quantized-IDE branch
  the combined suite passed, and the weights and wiring stages built on a tiny Q8_0 GGUF
  model, recovering a planted neuron. The atlas only uses `ws.read` / `Ref` / the runner's
  hook points, so it needs nothing from a store beyond logical float reads. The observed
  stage on a GGUF model was not run (no native kernel in this environment). It now runs
  in the model worker (see below); that path is tested on tiny random models and a fake
  decoder, never on a real GGUF model.

## Measured results

**Qwen3.5-9B, generic builder against the earlier hand-written map** (`qwen3.5-9b/`,
393,216 neurons, weights stage only, 4 CPU cores, no GPU: 5,958 s). The two were written
separately; the generic one goes through `Structure`/`Semantic` with no Qwen-specific code.

| Compared per neuron | Agreement |
|---|---|
| gate/value and value/write cosines | correlation 1.0000 in every layer |
| top promoted token, and the top-6 token set | 100% |
| mechanism (sharpener / amplifier / damper / router) | 98.9% |
| output role | 99.3% |
| input-token trigger flag / output-token trigger flag | 99.8% / 99.7% |

The remaining mechanism and role differences are neurons sitting on a threshold: the generic
builder derives its thresholds from the model's width and a random-direction calibration
rather than using the constants fixed by hand for Qwen. Concepts are not compared: the
generic build clusters the unembedding afresh (249 clusters in 24 groups here).

**SmolLM2-135M-Instruct** (46,080 neurons, all three stages: 135 s on the same machine).

- Sign check holds: the weight-predicted firing sign matched the observed peak sign for 82%
  of 2,382 testable neurons.
- Trigger and concept checks fail: 38.4% against 36.0% by chance, and 36.6% against 34.5%.
  This model's token clusters are dominated by one frequent cluster, so `concept=` selection
  is weak on it. Selection by word and per-prompt attribution do not depend on it.
- Bias direction usable (held-out sd 0.08 nats). `shift_tokens(word_tokens("blue"), 6)`
  made " blue" the next token at 81%.
- Attribution: switching off the 5 neurons it ranked highest for " red" on a tomato prompt
  moved the top answer to " orange" (" red" 12.1% to 5.8%).

## Limits

- The bundled probe text is about 3,000–5,000 tokens. Top contexts are examples, and the
  checks have little power on small or toy models ("too few to tell"). Put a larger
  `corpus.txt` in the workspace, or pass `--corpus`.
- Token signatures are the direct path to the vocabulary. They say what a neuron can push,
  not when it fires; attribution answers the second question for one prompt at a time.
- Attention heads get a thin record; recurrent and state-space mixers are not mapped.
- The constant direction for token shifts needs several times more probe positions than
  `d_model`; when the held-out spread is above 0.25 the shift is refused rather than applied.
- Unrecognised architectures have no atlas.

## Wiring: build time and interruption

The wiring stage compares every neuron's write vector with the read vectors of every
neuron in later layers. Its cost grows with the square of the neuron count: on a model
with 64 gated layers of 17,408 neurons over 5,120 dimensions the search is about
6.3 × 10¹⁵ multiply-adds (1.25 × 10¹⁶ floating-point operations, counting a multiply-add
as two; `wiring.wiring_cost` returns that operation count).

**Estimate.** Before the build, the Atlas tab shows an estimate in plain units next to
the wiring box, for example "about 3 h 5 min, plus reading the weights", and one line
on the memory it holds and on what waits. The CLI build logs the same estimate in the
line that also gives the multiply-add count; the MCP build returns that line, with the
lines about saved progress, ahead of the last 12 lines of its log, so an agent sees it
only once the build has returned. It is computed from:

- the time of one comparison step on this machine, timed once per process on random
  data of the model's shape, capped at 4,096 × 4,096 (`wiring.step_timing`: the
  product in the build's number type, then the selection of the best partners),
  scaled by each step's size;
- the time to read the weights, from the per-layer reading time the weights stage
  records (`read_s` in its stage record). A weights stage built before this record
  existed, or not built yet, gives no reading time, and the estimate says "plus
  reading the weights (not yet known)".

Measured against real steps on random data at three sizes (12 threads, float32, a
loaded machine), the scaled probe was off by +49% at 4,096 neurons, −3% at 8,192 and
+20% at 17,408; earlier runs of the same check gave between −38% and +49%. Read it as a
guide to within about half again either way.

**Memory.** The stage holds the normalised read vectors of every reader layer still
to be searched (in the working number type, float32 on CPU), the partner tables, and
one comparison at a time (the writer layer's vectors as read, converted and
normalised, the product and its magnitudes). For the 64-layer model above that is
about 48.5 GB. The figure is approximate: it leaves out the interpreter, the model's
own memory and allocator overhead. Measured once on random data (8,192 neurons over
5,120 dimensions, 3 layers, float32), the build's peak growth was 1.72 GB against an
estimate of 1.71 GB. Nothing checks this against the runtime memory settings before
the build starts; the estimate line is the only warning.

**What waits.** The build holds the workspace lock. Run and Chat (on a model that can
be run here) and edits wait until it finishes; so do inspector calculations.

**One build at a time.** Every atlas build (browser, HTTP, MCP, CLI) takes a lock on
`atlas/build.lock` (`wiring.build_lock`: an `flock`, which the system releases when
the process ends however it ends). A second build of the same atlas, from this IDE or
from another process, is refused at once with "a build of this atlas is already
running"; it does not wait. The lock file stays in the atlas directory and is not part
of the atlas.

Every build also holds the workspace lock for as long as it runs, whoever started it
(the browser, an agent's MCP call, the command line), so the weights cannot change
under it: Run, Chat, edits and inspector reads wait until any build of the atlas
finishes. The Atlas tab shows a build it did not start ("A build of this atlas is
running in this IDE" or "in another process", with the wiring layers done so far) and
offers no Build or Continue while it runs; `GET /api/atlas` reports it as
`build_running`, and the estimate marks saved progress as not usable while it does.
Requests that would otherwise hold the session lock while they wait (opening another
model, downloading weights, saving a test suite, ablation and atlas attribution in the
inspector) are refused at once with that reason instead. A build that ended in an
error shows the error at the top of the Atlas tab. The CLI prints the refusal and
exits with status 1.

**Progress and interruption.** After each writer layer the stage writes that layer's
file, the partner tables of the layers not yet finished, and a progress marker
(`atlas/wiring_partial/progress.json`), each by writing a temporary file and
renaming it. While it runs, the build log, the job state and the browser show the
layers done and the time left at the pace so far. A build that stops (closed IDE,
crash, error) continues on the next build from the first unfinished layer and says
"continuing from layer 23 of 64"; the Atlas overview offers "Continue the wiring
stage". In the tests the result is bit for bit the same as an uninterrupted build.
Saved progress is used only for the same model, the same weights stage, the same
variant, the same enabled patches (each by id, strength, commit time and size, which
survive a restart of the IDE), the same torch version, the same number type and the
same search settings; otherwise it is discarded and the log says why. It does not
record the BLAS library or changes to the IDE's code: progress saved before such a
change is continued as if nothing had changed. Saved files that cannot be read, or
are not of the shapes the build writes, are treated the same way: discarded, with the
reason. `force` (Rebuild) always starts over.

The Atlas overview, the estimate and the build decide from one function
(`wiring.assess`) whether saved progress can be used. When it cannot, the overview
says so with the reason the build will log, and offers "Build the wiring stage again"
instead of a continuation. For edited weights a continuation also rebuilds the weights
stage (the time the last full weights build took is shown, or that it is not known),
and the observed stage then has to be built again.

**What counts as built.** The wiring stage is built when the atlas record lists it,
no progress marker is present, and every layer has its `wiring_LLL.npz`. An atlas
written before this change (all files, no marker) is built. Readers (`Atlas.layer`,
unit cards, `Atlas.status()`) ignore a partly built stage, and the leftovers are
removed when the stage completes.

## IDE integration safeguards

The browser and HTTP build route default to **weights only**. Wiring and observed
probes must be selected explicitly; observed probes load/run the model. CLI/MCP
build defaults remain separately documented and may include observed probes. A
weights build still reads weights and performs vocabulary projections; it is not
metadata-only and may be expensive. Opening the tab only reads saved atlas data.

Build and attribution requests require the current workspace identity/version.
Concurrent identical builds join one job; conflicting active builds are rejected.
Builds hold the workspace lock against edits/unload and publish scoped progress;
the wiring stage's progress (layers done, time left) is in the job state and its
events. The wiring estimate is a separate request (`/api/atlas/estimate`), so
opening the tab never waits for it; it times a small product once per process
and loads no model.
Shutdown interrupts at progress boundaries; a long native operation is not
immediately cancellable, and bounded lifecycle cleanup can report a timeout.
Attribution requires an explicit click, shares identical pending/completed requests,
and ignores results from obsolete views. Changing script options requires another
explicit click before sending the updated script.

### The observed stage on a model in the model worker

A GGUF model runs in the supervised model worker ([runtime notes](RUNTIME.md)),
one sequence at a time. The observed stage runs there one probe window per worker
job (`observe_window`):

- **The worker reduces; the server merges.** The worker runs the window once,
  with batch 1. Each MLP layer's activations are reduced by a hook inside the
  forward pass as the layer runs (sums, squares, positives, positive and negative
  mass, and the window's own top 6 and bottom 3 values with their rows) and are
  not held once reduced. Attention patterns, where the runner gives them, become
  per-head sums. The last layer's stream is normalised there (position 0 left
  out). The logits give the window's loss, summed in float64 16 positions at a
  time, and are dropped. Only these reductions cross; no captures and no logits.
  The server adds them to the same running totals the in-process path uses and
  writes the same files. Both paths call the same reduction and merge functions
  (`wsp/atlas/observe.py`).
- **The bias fit's memory is bounded.** The token-shift direction is fitted on the
  normalised final stream of the first `FIT_ROW_BYTES / (16 · d_model)` probe
  positions (448 MiB of rows: 5,734 positions at d_model 5120, 73,400 at 400), the
  same on both paths. The server then holds about `16 · rows · d_model + 32 ·
  d_model²` bytes for the fit (1.3 GB at d_model 5120), however large the probe. The
  bundled probe (4,953 positions at a context of 128) fits whole for d_model up to
  5,927, so its output is unchanged. A larger probe or model is fitted on its first
  positions only; the log says so ("the bias direction is fitted on the first N of
  M positions, which keeps the fit near X GB of memory") and `bias.json` records
  `positions_seen`. Every position still counts for the per-neuron statistics, the
  head statistics and the loss.
- **Same answer as this process, one window at a time.** On a tiny model, the
  worker path gives the same files as the in-process path run with one window per
  pass: equal arrays, positions and signs, and the same rounded loss. The float32
  partial sums cross unchanged and are added in float64 in the same order. Batched
  in-process passes round differently, so values agree only to float round-off,
  and a tie between equal activations may be kept at another position.
- **Admission.** Each window is one operation, admitted like `run` with named hook
  points: a forward over the window's tokens that holds activations (the `trace`
  forecast). That forecast assumes every layer's activations are retained, so it
  is well above what the job holds. The server checks the lease between windows.
- **Committed weights only.** The job takes `base` or `current` and no uncommitted
  `extra`. The worker syncs the committed patches from each job's manifest, as for
  every job.
- **Defaults.** The browser builds what is ticked. Under the Observed box the tab
  says what the stage runs ("loads and runs the model on about 5,000 tokens of
  probe text, one window at a time", from the status field `observed_note`, sized
  from the probe text without loading the model). A CLI or MCP build without
  stages leaves the stage out on such a model and reports it as skipped ("not run
  by default ... ask for the observed stage explicitly"), because it loads and runs
  the model. `--stages weights,observed` (CLI) or `stages: ["observed"]` (MCP)
  builds it.
- **What waits.** The build holds the workspace lock for its whole length, as every
  build does. Each window's operation and the worker's admission requests run on
  the build's own thread (both locks are re-entrant there). Run, Chat and edits
  wait for the workspace lock until the build ends, then run.
- **Failures.** If the worker is lost (killed by the watchdog, exited, or the
  connection fails), the build stops with "the observed stage stopped at window k
  of W: ... Nothing of this stage was saved; the stages built before it are
  unchanged." If memory admission refuses a window, the message says how much was
  asked for against what was free above the reserve, that this is the
  conservative forecast (the job itself holds far less), and what to do: close
  other software, lower the context setting or lower the preferred reserve in
  Settings, then build again. Either way the error keeps the runtime's record with
  these words as its message: the HTTP job has it as `event`, and the MCP `atlas`
  tool answers with the structured refusal (`{"status": "not_run", "event": …}`)
  as other refused operations do. A server stop ends the stage at the next window
  boundary. Nothing is written until every window has run.
- **Adding the stage to a built atlas.** The Atlas overview offers "Add the observed
  stage" when the atlas is built without it and the model can be run. It sends
  `stages: ["observed"]`, `force: false` and the atlas's own variant. Such a build
  runs only the observed stage: the weights and wiring files and records are kept.
  It is refused, with the reason, when the atlas maps other weights than those
  asked for (base against edited), or when the edited weights it maps have changed
  since it was built (its recorded patches are not the enabled ones). The answer
  then is a rebuild. A base atlas maps the base weights, which edits do not change,
  so the stage can be added to it at any time. When the build ends the tab shows
  the observed results without a reload.

A model that cannot be run at all still has the Observed box disabled with its
reason (`observed_unavailable`), and the build route answers 422 if the stage is
asked for anyway.

### Observed files

The observed files (`probe_ids.npy`, `obs_LLL.npz`, `heads_observed.json`,
`bias_direction.npy`, `bias.json`, `validation.json`) are written only after the
last window has run. Each one goes to a temporary name and is then renamed into
place. A write that fails removes its temporary file. Before the first of them is
replaced, the atlas record drops an earlier observed stage. It records the new one
only after the last file is written. Readers (`Atlas.layer`, `Atlas.json`, unit
cards, the token-shift plan) use observed files only when the record lists the
stage. A build stopped while writing therefore leaves the atlas weights-only,
never a mix.

What an earlier observed stage survives:

- A build of the base weights, not forced, stopped before the observed files are
  written (a lost worker, a refusal, a stop): the earlier stage is kept. Its
  weights stage, if asked for, continues from the saved layer files of the same
  base weights.
- A forced build (Rebuild), or any build of the edited weights other than adding
  the observed stage: the earlier observed record is dropped before any stage
  writes a file, because the weights stage writes the layer files again from
  scratch (for edited weights, from weights the earlier statistics may not
  describe). If that build then stops, the atlas is left with the stages it
  completed, and no observed stage.

The server caches the atlas reader per workspace. Every write of the record drops
that cache, and so does the end of every build, however it ends, so no reader is
older than the record on disk.

The same-token head check computes Spearman's rank correlation with numpy; scipy
is not needed (it is not in the requirements). When a head's weights statistic or
observed statistic is not a number, the correlation is undefined: `validation.json`
then has `"same_token_spearman": null` and a note saying so, and stays valid JSON.

Integration validation uses fake workspaces and DOM/network fixtures in
`tests/test_atlas_transport.py`, `tests/atlas_frontend.test.mjs`,
`tests/test_atlas_wiring.py` and `tests/atlas_wiring_frontend.test.mjs`; no real model
loads, downloads, training or inference were performed for this merge. The observed
stage in the model worker is tested in `tests/test_atlas_observed_worker.py` on tiny
random models (in this process and in `serve_worker` on a socketpair) and a fake
llama-named decoder; `tests/observe_reference.py` is the pre-change implementation,
kept as the reference for the in-process output.
