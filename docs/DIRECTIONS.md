# Behaviour directions

A behaviour direction is the difference between what the residual stream holds after two
sets of prompts: "good" prompts (the model behaves as wanted) and "bad" prompts (it shows
the behaviour to remove). This feature

1. **measures** it: each prompt is run through the model, the residual-stream vector at its
   last token is kept at every layer, and the mean of each set is saved in the workspace
   under a name;
2. **ablates** it: from a saved measurement, one unit direction per layer is removed from
   what the attention output matrix and the MLP output matrix of chosen layers write, with
   a per-layer strength, in three variants (plain, norm-scaled, norm-preserving). The change
   is an ordinary patch of low-rank pieces, committed through the one edit path
   (`Session.apply` / `Workspace.commit`), so task scope, candidate tests, drift and patch
   recording apply as to any other edit;
3. or **fits** a change instead ([the fitted low-rank ablation](#the-fitted-low-rank-ablation-ara)):
   from a measurement that also saved each output matrix's inputs and outputs, a low-rank
   change per matrix is fitted so that its outputs on the good prompts stay and its outputs
   on the bad prompts move towards the good ones. Same patch path;
4. **scores** a state of the weights ([Scores](#scores)) as Heretic's search does: refusal
   rate by keyword markers, KL divergence of the first-token distribution against the base
   model, and response length, for the base weights, the current weights or an uncommitted
   candidate script; and finds the response prefix that closes a model's thinking block
   ([Response prefix and thinking blocks](#response-prefix-and-thinking-blocks));
5. **searches** for parameters ([The automatic search](#the-automatic-search)) as Heretic
   does: Heretic's sampler draws parameters of either method from Heretic's ranges, each
   draw is scored as an uncommitted candidate, and the trials no other trial beats on every
   score form the Pareto front; trials are kept in the workspace, a search can be stopped,
   resumed and extended, and one trial is committed through `Session.apply`.

The technique is published: directional ablation is Arditi et al. 2024; the projected
direction and the norm-scaled and norm-preserving variants are Lai 2025 (see
[PRIOR_WORK.md](PRIOR_WORK.md)). It is implemented here from the formulas below. The
parameter names, the layer indexing, the score definitions and the prefix detection are
those of Heretic (github.com/p-e-w/heretic, AGPL-3.0), whose source was read for them, so
parameters it produces can be pasted in unchanged and its output can be compared with this
one. Taken from it verbatim are its default refusal markers and thinking-block pairs (data),
and, in `scripts/heretic_fit_check.py` only, its fitting loop, attributed there and used for
the comparison alone. The comparisons are in [Compared with Heretic](#compared-with-heretic)
and [Comparing the scores with Heretic](#comparing-the-scores-with-heretic).

Code: `wsp/directions.py` (storage, measurement, directions, the plan and the change to
one component), `wsp/directions_ara.py` (the fitted method), `wsp/directions_scores.py` (the scores),
`wsp/directions_prefix.py` (the response-prefix detection), `wsp/directions_evaluate.py`
(scoring a modified checkpoint against its base), `wsp/directions_heretic.py` (running a
Heretic `reproduce.json`), `wsp/directions_search.py` and
`wsp/directions_sampler.py` (the automatic search), the weight-script functions in
`wsp/wscript.py`, the worker job
`direction_residuals` in `wsp/worker_jobs.py`, and the routes in `wsp/server.py` (the block
after the probe routes). In the browser it is the Directions tab ([The Directions
tab](#the-directions-tab)); there is no MCP tool of its own: the programmer reaches it
through the script functions and the `edit` tool.

## What is stored

`<workspace>/directions/<name>.safetensors` holds `good_mean` and `bad_mean`, each float32
`[n_layers + 1, d_model]`:

| Index | What |
|---|---|
| `k < n` | the residual stream entering layer `k` (`blocks.k.hook_resid_pre`) at the last token |
| `n` | the residual stream leaving the last layer (`blocks.{n-1}.hook_resid_post`) **after the model's final norm** (what the unembedding reads), at the last token |

This is the sequence `transformers` reports as `hidden_states` (index 0 the embedding
output, index `n` the final normed output), which is how published tools index layers:
"index k" here is that tool's hidden state k.

`<name>.json` is the meta:

```json
{"version": 1, "name": "refusal", "created": 1794920583, "variant": "base", "n_layers": 2, "d_model": 24,
 "counts": {"good": 3, "bad": 2}, "tokens": {"good": 11, "bad": 6},
 "options": {"chat": false, "system": null, "prefix": null, "chat_options": null, "winsorization_quantile": 1.0},
 "prompts_sha256": {"good": "c505...", "bad": "1c61..."},
 "model": {"kind": "wsp", "name": "rnd", "repo": null, "sha": null, "files": [["base.safetensors", 45632, 1794920583665688070]]},
 "module_io": null, "datasets": {"version": 1, "good": null, "bad": null},
 "environment": {"libraries": {"torch": "2.8.0", "transformers": "4.57.1"}, "sampler": {"optuna": null, "numpy": "2.3.4"},
                 "python": "3.12.11", "torch_threads": 8, "machine": "arm64", "system": "Darwin",
                 "runner": {"class": "FunctionalRunner", "dtype": "None", "device": "None"}},
 "measured_on": [], "rendering": {"version": 2, "same_as_version_1": true}}
```

- `module_io` is `null`, or `{"bytes", "layers"}` (bytes of the saved arrays, number of layers
  with captured data) for a measurement that also saved the output matrices' inputs and
  outputs ([below](#capture-module_io)).

- `tokens` are totals over each set, counted as run (the rendering, with `chat`).
- `prompts_sha256` is the SHA-256 of the UTF-8 of `json.dumps(texts, ensure_ascii=False,
  separators=(",", ":"))`, each set as given (a set from datasets: its canonical form,
  [below](#system-messages-per-prompt-and-the-canonical-form)).
- `datasets` holds, per set, the record of a set read from datasets
  ([What is recorded](#what-is-recorded)), or `null` for a list of texts. Metas written before
  this key existed have none; readers use `.get("datasets")`.
- `model` is the model identity steering's calibration cache uses
  (`steering.model_identity`: the GGUF base SHA-256, or the checkpoint files' names, sizes
  and modification times).
- `environment` is what the measurement ran with: the library and sampler versions, Python, the
  torch thread count, the machine type and the system (as a search records them; no host name,
  user or path) and the runner's class, dtype and device. The means, and an ARA fit's captures,
  depend on these. `measured_on` is, for `variant: "current"`, the enabled patches the passes
  ran on (`[[id, strength, created, stored], ...]`), else `[]`. Metas written before these keys
  existed have neither; readers use `.get`.
- `rendering` is how the prompts became token ids: `version` 2, as Heretic renders them, and
  `same_as_version_1`, whether the IDE's rendering before (version 1) gives the same ids for
  these prompts on this model on any day ([How a prompt becomes token ids](#how-a-prompt-becomes-token-ids)).
  A meta without it was rendered with version 1.

`<name>.prompts.json` is `{"good": [...], "bad": [...]}`, the texts as given (not rendered);
a set read from datasets is stored resolved, in its canonical form (texts, and
`{"system", "user"}` objects for prompts with a system message of their own).

- **Names** match `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$` and may not end in `.prompts` (that
  would collide with another entry's prompt file). An existing name is refused unless the
  request says `replace`.
- **Writes are atomic**: each file is written to a temporary file and moved in place with
  `os.replace`. The `.json` is written last, and its presence means the entry is complete;
  `replace` removes the old `.json` first, so a measurement cut short is never listed with
  the old meta. Delete removes the `.json` first too.
- A damaged entry (unreadable meta, missing vectors) is listed with an `error`, and the rest
  of the list is kept.
- Names that differ only in case are one file on a case-insensitive file system (macOS by
  default): the second is refused as existing unless it says `replace`.

## The measurement

`measure(ws, name, good, bad, chat=False, system=None, prefix=None, chat_options=None,
variant="base", winsorization_quantile=1.0, replace=False, module_io=False, progress=None,
stopped=None)`.

`good` and `bad` are lists of texts, or Heretic dataset specifications and prompt objects
([Prompt sets from datasets](#prompt-sets-from-datasets)), resolved from files before any
model is loaded.

**Bounds**, all checked before any model is loaded: 1..1024 prompts per set; each text
1..20,000 characters; each 1..512 tokens as encoded, and at most the model's context;
`winsorization_quantile` a number from 0 to 1 (1 is off); `variant` `base` or `current`;
`system` and `prefix` text of at most 20,000 characters; a valid name, not taken unless
`replace`. With `chat`, the rendered prompt's token count is checked again after the model
is loaded (`good prompt 1 as a chat message has 18 tokens; it must have 1..16`).

**As a chat message** (`chat: true`): each text is rendered as one user turn by the model's
own chat template with the generation prompt, as Heretic renders it
([How a prompt becomes token ids](#how-a-prompt-becomes-token-ids): the system role folded
where the template has none, the date fixed, the tokenizer's own special tokens added), with
the availability rule of the probes' "as a chat message"
([STEERING.md](STEERING.md#as-a-chat-message)) (refused
on bundled `.wsp` models and on a GGUF that carries no chat template, with the probes'
reason; a GGUF's own template is used: [On a GGUF base](#on-a-gguf-base)) and the same `chat_options`
check. An optional `system` message goes before the user turn,
and an optional `prefix` string is appended to the rendering (for example the start of an
answer). A prompt with a system message of its own (from a specification's `system_prompt`
or a prompt object's `system`) is rendered with that one instead. Without `chat`, `system`,
`prefix` and `chat_options` are refused, and so is a prompt with its own system message.

**The passes.** Each prompt runs alone (batch 1), one forward pass, on the committed
weights of `variant` (`base` by default, or `current`: base plus enabled patches; never a
candidate edit). The prompts go to the worker job `direction_residuals` four at a time
(the job takes 1..8 and checks its bounds again); it returns only the last-token vectors,
float32 `[prompts, n + 1, d]` (with `module_io`, one packed row per prompt that also holds
the output matrices' inputs and outputs: [Capture](#capture-module_io)). On a supervised model this runs in the model worker; the
final norm is applied there (`final_norm` exists on `FunctionalRunner` and `HFRunner`, which
`GGUFRunner` inherits; `RemoteRunner.final_norm` refuses, since the job runs next to the
model).

**Winsorization.** When `winsorization_quantile` `q` is below 1, each prompt's vector at
each index is clamped to `±t`, where `t` is the `q` quantile of the absolute values of its
`d` components (`torch.quantile`, linear interpolation). Then the vectors are summed in
float64; each mean is the float64 sum divided by the count, stored as float32. The chunk
size does not change the result (the sums are taken prompt by prompt, in order).

**Progress and stop.** `progress(done, total)` is called after every chunk (prompts done,
over both sets: good first). `stopped()` is checked before every chunk and once more
before saving; when it is true nothing is saved and the answer is
`{"stopped": true, "done": n, "total": n}`.

**Cost.** One forward pass per prompt, batch 1, holding the named residual captures of
that pass (`n + 1` vectors of the prompt's length); only `[prompts, n + 1, d]` floats cross
the worker boundary per chunk. Each chunk is one runtime operation, admitted for its
longest prompt as a forward pass that retains captures.

The answer is the saved meta plus `seconds` and per-index statistics, from the two means
only:

```json
{"index": 1, "good_norm": 4.985, "bad_norm": 6.730, "diff_norm": 7.846, "cosine": 0.128, "usable": true}
```

`usable` means the difference is not 0 and (with `orthogonalize_direction`, the default of
`GET /api/directions/{name}`) neither is what is left after the projection below.

## Directions

`table(good_mean, bad_mean, orthogonalize_direction=True)`, row by row:

- `r = unit(bad - good)`;
- with `orthogonalize_direction`: `g = unit(good)`, `r = unit(r - (r·g) g)` (the part of the
  difference along the good mean is removed);
- a row whose norm is exactly 0 at either step is all zeros and **unusable**: an ablation
  that would use it is refused, naming the index.

Which row a layer uses (`direction_at`):

- `direction_index: null` (per layer): layer `l` uses `table[l + 1]`, the residual stream
  that layer `l` leaves.
- a number `x` (one direction for every layer): `p = x + 1`, `i = floor(p)`, `f = p - i`; the
  direction is `unit((1 - f) table[i] + f table[i + 1])`, or `table[i]` when `f = 0`.
  `x` therefore counts in layers: `12.3` is three tenths of the way from the output of layer 12
  to the output of layer 13. Refused unless `0 <= i` and `i + (1 if f > 0 else 0) <= n`
  (`x` from -1 to `n - 1`); both rows must be usable, and two rows that cancel are refused.

## Per-layer strength

A weight distribution is `{"max_weight", "max_weight_position", "min_weight",
"min_weight_distance"}`. For layer `l`: `dist = |l - max_weight_position|`; if
`dist > min_weight_distance` the layer is skipped; otherwise
`w = max_weight + (dist / min_weight_distance) * (min_weight - max_weight)`, and `w = 0`
skips the layer. All four must be finite numbers, `|max_weight|, |min_weight| <= 8`,
`min_weight_distance > 0`, `-n <= max_weight_position <= 2n`.

## The parameters

```json
{"direction": "refusal",
 "direction_index": 12.3,
 "weight_distributions": {"attn.o_proj":   {"max_weight": 1.0, "max_weight_position": 14.0, "min_weight": 0.2, "min_weight_distance": 6.0},
                          "mlp.down_proj": {"max_weight": 0.8, "max_weight_position": 14.0, "min_weight": 0.1, "min_weight_distance": 8.0}},
 "orthogonalize_direction": true,
 "row_normalization": "none",
 "full_normalization_lora_rank": 3}
```

| Field | Values |
|---|---|
| `direction` | a saved measurement's name; it must have been measured on a model with the same `n_layers` and `d_model` |
| `direction_index` | `null` (default: per layer) or a number (see "Directions") |
| `weight_distributions` | at least one of `attn.o_proj` (the layer's attention output matrix, role `W_O`) and `mlp.down_proj` (its MLP output matrix, role `W_out`) |
| `orthogonalize_direction` | `true` (default) or `false` |
| `row_normalization` | `none` (default), `pre` or `full` |
| `full_normalization_lora_rank` | integer 1..64 (default 3), used by `full` only |

An unknown field, a missing one and every bad value is refused with the field named
(`weight_distributions.attn.o_proj.min_weight_distance: must be above 0`). A component the
model has in no layer is refused; a layer that lacks a given component is skipped and
reported (`"skipped": "this layer has no mlp.down_proj"`). Biases, embeddings and every
other tensor are never touched.

## The change to one component

A component is the list of writer views that form one matrix `V` (`[D_in, d]`, rows
concatenated), as `Semantic.writers()` yields them: a 3-D attention output (this project's
`W_O [head, d_head, d_model]`) gives one view per head; a checkpoint's 2-D `o_proj`
(`[d_model, n_heads * d_head]`) and `down_proj` are one view each, read in the checkpoint's
own layout with the residual stream last. Column `j` of `V` is everything the component
writes to residual coordinate `j`. `V` is read as the current weights including the running
script's own earlier changes (the script's `extra`). With a unit direction `v` and strength
`w`, every variant stores `delta = A @ B` per view as a low-rank piece (`A [d_in, r]`,
`B [r, d]`, float32):

**`none`** (plain directional ablation): `delta = -w (V v) vᵀ`. Per view
`A = -w · (view @ v)`, `B = v`; rank 1. With `w = 1`, `V' v = 0`.

**`pre`** (the ablation done on the matrix with unit-norm columns, then scaled back): with
`nrm[j] = ||V[:, j]||` over the whole component (every head together),
`x = v / nrm` (0 where `nrm = 0`), `y = nrm · v`: `delta = -w (V x) yᵀ`; rank 1.

**`full`** (norm-preserving): normalise the columns, ablate, renormalise each column to unit
norm, restore the original column norms, and store the best rank-`r` approximation of the
difference (`r = min(full_normalization_lora_rank, d)`). In closed form, with
`G = Vᵀ V` (`[d, d]`, float64, accumulated over row slabs of every view):

```
nrm = sqrt(diag(G))            x = v / nrm (0 where nrm = 0)
t   = G x                      alpha = x · t          c = t / nrm (0 where nrm = 0)
s   = sqrt(max(1 - 2 w v c + w² v² alpha, 1e-24))           (elementwise over the d columns)
g   = 1 / s - 1                u = nrm · w · v / s            (g = u = 0 where nrm = 0)
exact:  delta = V diag(g) - a uᵀ          with a = V x
M = deltaᵀ delta = diag(g) G diag(g) - (g·t) uᵀ - u (g·t)ᵀ + alpha u uᵀ
R = the r eigenvectors of M with the largest eigenvalues ([d, r], torch.linalg.eigh, float64)
best rank r:  delta_r = (delta R) Rᵀ,     delta R = V (g[:, None] · R) - a (uᵀ R)
```

Per view, one product with the `[d, r + 1]` matrix `[g[:, None] · R, x]`, then
`A = first r columns - last column · (uᵀ R)`, `B = Rᵀ`. The rows of `R` at zero-norm
columns are set to exactly 0 (mathematically they are 0 for every eigenvector of a nonzero
eigenvalue; `eigh` leaves rounding noise), so a column that wrote nothing still writes
nothing. Each component reports `full_error = sqrt(max(0, 1 - Σ top r eigenvalues /
trace(M)))`, the relative Frobenius error of the approximation (0 when `r >= d`). If
`trace(M) = 0` nothing is written.

**Memory and reads.** `G` (and then `M`, its eigenvectors and `eigh`'s workspace) is
`d × d` float64; the full variant admits `4 · d² · 8` bytes of scratch through the edit
admission (`edit_preflight`) before it starts. Every weight read is a row slab of at most
`STREAM_ELEMS` values (`Ref.sub` over the view's `d_in` axis for `G` and the norms,
`Ref.contract` for the products), so `V` is never held whole when it is above that limit,
and a quantized base is read through its store's logical read path, never dequantized
whole. Each slab and each product is admitted with `edit_preflight`, as the interpreter's
other built-ins that read weights are.

## Script functions

```python
ablate_direction("refusal", direction_index=None, weight_distributions={...},
                 orthogonalize_direction=True, row_normalization="none", full_normalization_lora_rank=3)
direction("refusal", k)        # the unit direction at index k (0..n), orthogonalize_direction=True; a tensor
direction_means("refusal")     # (good_mean, bad_mean), each [n + 1, d]
```

`ablate_ara(...)` is the [fitted method's](#parameters-and-the-script-function) function.

`ablate_direction` adds its pieces with the interpreter's `_add`, so the edit limit,
memory admission and the patch path are those of every script, and prints a summary to
the script's output (one more line per component with `full`):

```
ablate_direction('refusal'): 3 components in 2 layers, 624 values stored, row_normalization full, 1 skipped
  L0 attn.o_proj: weight 0.5, full_error 0.173488
  L1 attn.o_proj: weight 1, full_error 0.141614
  L1 mlp.down_proj: weight 0.8, full_error 0.141857
```

It reports `edit_progress` as the interpreter's other long built-ins do (the component as the
window, at most twice a second). Like
`orthogonalize`, it does not charge weight reads to the script's step limit.

`plan(ws, params)` says what an ablation will do without reading a weight or running the
model: one row per layer and component and totals (below). `script_for(params)` gives the
one-line script; numbers are written with `repr`, so they round-trip exactly.

## The Directions tab

The browser tab `Directions` (after `Probe`) does everything above through the routes below
and the IDE's one edit path. It ignores the selection in the explorer. Code:
`web/js/directions.js` (the frame: the measurement picker, the notes, the cards of requests
started elsewhere, this page's job and its Stop, the drafts, the events),
`directions_prompts.js` (the prompt-set editor), `directions_forms.js` (the forms made from
the served descriptors, the scorer list), `directions_measure.js` (Measure, Layers),
`directions_edit.js` (Edit, Compare), `directions_search.js` (Search) and
`directions_charts.js` (the SVG charts, no library). Page tests:
`tests/directions_*_strict_dom.test.mjs`.

At the top: the measurement picker (shared by every sub-view; a damaged measurement or one of
another model shape cannot be picked), the notes that apply, one card per direction request
running in this workspace that this page did not start (from `requests` of
`GET /api/directions`, so a reloaded page or a second window shows it at once, with "Stop it"
when it was sent with a stream), and this page's own job line with its Stop.

**Measure.** The saved measurements (select, "into the form": their stored prompts back into
the two sets, delete after a confirm). The rendering shared by measuring, prefix detection,
scores and searches: "as a chat message" (on when the model has a chat template),
`system_prompt` (Heretic's default system prompt at first), `response_prefix`, chat options;
when it differs from the selected measurement's, a note and "Use its rendering". The good and
the bad prompt sets: each a list of parts, typed prompts (one per line, or a JSON list of texts
and `{user, system}`) or a dataset part with Heretic's eight fields; "Heretic's default" adds
Heretic's specification; "Check" sends the set without downloading (count, hash, three
previews, or what a download would fetch and a "Download" button). The measure form: name,
`variant`, `winsorization_quantile`, module inputs and outputs (needed for ARA; ticked by
default when the estimate of their size is known and below 1 GiB), "replace" when the name is
taken. Then Heretic's response-prefix detection on its own prompt set, with "Use this prefix".

**Layers.** The selected measurement per index: the three mean norms and the cosine of the two
means as charts, and a table. A click on an index sets the Edit form's `direction_index` to
index − 1. The table's `steer` column has, for each usable index k below n, a button "steer -1"
that adds `direction:<name>@<k>` with dose −1 (towards the good mean; +1 adds the behaviour) in
`diff` to the Run tab's steering panel, pinned to the measurement's `means_sha256` and with the
view's `orthogonalize_direction`; nothing is sent or run (index n, after the final norm, has none).

**Edit.** The method (ARA needs a measurement with module inputs and outputs), its parameters as
a form made from `parameters` (defaults, bounds and Heretic's ranges in the hints), or
Heretic's parameters pasted in. "Preview plan" shows the script and the plan
(`POST /api/directions/script`: nothing is read or run). "Dry run" and "Apply" run that script
through the console's own call (`POST /api/edit`, with the console's "enforce task scope"
box); they are enabled only for the plan previewed for these parameters on these weights, so
a plan is not applied twice nor after the weights changed.

**Compare.** The scorer list (Heretic's two default scorers at first, each with its prompt set
and settings), `max_response_length`, and two states (current or base on the left; the Edit
view's candidate, current or base on the right). "Score" sends two score requests, one after
the other; then the scores side by side, each response scorer's responses side by side (a
differing row marked) and the KL per prompt. Results scored before the weights changed are
marked so above them. A third right-hand state, "a checkpoint on disk", takes the absolute path
of a modified checkpoint's directory (kept in the page only) and fixes the left side to the base
weights: "Check the checkpoint" compares its weights with the base's without running the model
(`POST /api/directions/evaluate` with `"check": true`), and "Score" sends that route one request
that answers the comparison, the base's scores and the checkpoint's. The comparison (how many
tensors and weights differ, the differing tensors, the dtypes, what is held as FP32 changes
while it is scored, the notes) is shown above the scores; an empty path or a GGUF workspace
holds both buttons with the reason. A further right-hand state, "current weights, steered by the
Run tab's panel", scores the current weights with that panel's active entries (state `current`
with the panel's `steer` and `p50`; the KL reference stays the unsteered base), holds "Score"
while the panel has no active entries or an entry has a problem, and labels the steered side
under the score table from the answer's `steering` report, not from the panel.

**Search.** The workspace's searches (trial counts, front, objectives, where each runs, "cannot
run" / "cannot commit" when its `compatible` says so, delete after a confirm). "New search":
the selected measurement, `Abliteration` or `ARA` (ARA when the measurement has module inputs
and outputs), the modifier's settings (Heretic's defaults; only changed ones are sent),
`n_trials`, `n_startup_trials`, `seed` (empty: random, recorded), and the scorers,
`max_response_length` and rendering shared with Compare and Measure; "Create and run". The
selected search (the one picked, else the running one, else the last created): its header and
state line, "Stop now" and "Stop after this trial" while it runs in this IDE, "Resume" and
"Run more" otherwise (with the box for `accept_changed_libraries` when only the library
versions changed), the trial table, the front as a chart and as Heretic's list, and a picked
trial's record: its parameters as Heretic shows them, scores, script, the fit's output,
seconds, environment, and its responses beside a baseline's. "Score the baseline" scores the
current weights with the search's own scoring request; the baseline lives in the page only and
is not drawn once the weights changed. "Use in Edit" puts the trial's parameters and the
search's settings in the Edit form. "Check the commit" and "Commit trial N" post the
trial-apply route ([Committing a trial](#committing-a-trial)); `accept_different_edit` is sent
only after a refusal for a different edit made its box appear and the person ticked it.
Committing runs the trial's script again and cannot be stopped. The Patches pane shows a
committed trial's origin ("from search s1, trial 4: ...", and a warning when the committed
edit differs from the scored one).

**Output.** (`directions_output.js`) The workspace's patches, newest first, each with a box
that ticks it for the export (the enabled ones at first), its message, author and origin, and
"Show record". The list is the Patches pane's own: the shell reads it when a workspace opens
and after every change of the weights, and the view is redrawn then; opening the view sends no
request ("Reading the patches…" until the shell has the list). The export form ([A LoRA adapter (PEFT)](#a-lora-adapter-peft)): the adapter's
name, `layout` (`auto`, `uniform`, `per_module`) and `base_model_name_or_path` (empty: the
workspace's Hub repository, or none); "Preview (dry run)" and "Export" post
`POST /api/export_adapter`, with `patches` null when exactly the enabled patches are ticked,
else the ticked ids in stack order. Both are disabled, with the route's own reason and no
request, for one of this project's own models and for a GGUF base. An "exists" refusal shows a
"replace the adapter" box; only then, and only when ticked, is `replace` sent. The answer shows
where the files were written (or that a preview wrote nothing), each file's size and the start
of its SHA-256, the modules with their rank, whether they are exact and their pieces, and the
notes. "Show record" reads `GET /api/patches/{pid}/record`
([The reproduction record](#the-reproduction-record)): the calls, the measurements' hashes, the
patches the edit was made on, the checks that refuse a re-run (a failing one marked), the
library and code rows with Heretic's severities (high and critical marked), and a link to the
record as Heretic's `reproduce.json` when Heretic can run it, else the reasons it cannot; when
the only thing missing is the Hub repository the checkpoint came from (`needs_model`), two
fields, the Hub model id and its commit, and once a model id is typed the link with them as
`model` and `model_commit`. When
the `made_on` check fails because the patch itself is on, "Switch patch <id> off" switches it
off through the patch route and reads the record again. "Re-run (dry run)" posts
`POST /api/patches/{pid}/reproduce` and says the verdict (the same edit, a difference by
rounding within its bound, or a different edit); "Commit the re-run" commits it as a new patch,
and a different edit is committed only when its "commit it anyway" box is ticked (that alone
sends `accept_different`). "Measure again from the record" (for a script of ablate calls only)
reads the record's measure request (`?remeasure=true`), measures it under the suggested name
through the job slot with its Stop, and then re-runs the edit with that measurement, saying
whether the new means equal the recorded ones. A re-run is a third way the tab writes weights:
like the other two it is `Session.apply` on the server, and it cannot be stopped. The Patches
pane shows a committed re-run's origin ("re-run of patch 0002-41234: identical").

**What the tab never does on its own.** Opening it, switching sub-views and selecting a
measurement, a search or a trial only read metadata (GETs). Every request that runs the model,
downloads or writes is sent by a click. It writes weights only through the console's edit call
and the trial-apply route, both `Session.apply`, and, in the Output view, through the re-run of a
recorded edit (`POST /api/patches/{pid}/reproduce`, also `Session.apply`: a dry run unless the
person presses "Commit the re-run"). The Output view's "Switch patch <id> off" is the Patches
pane's own switch (`POST /api/patch`). Text from the server, a dataset or a model is put into
the page as text, never as markup.

**States.** When the model cannot run here, measuring, scoring, prefix detection, searching and
the baseline are disabled with the reason; saved measurements, the per-layer view, plans and
applying (untested) still work. When it is not loaded yet, the first request loads it (the
note goes once the status says it is loaded). A checkpoint whose layers are not known has no
parameter form and cannot be measured. A GGUF model is said to keep its quantized base,
never dequantized, with FP32 patches. A checkpoint without logical edits disables the dry run,
Apply and committing a trial. Without a chat template the chat box is off and prefix
detection disabled. Without Optuna creating, resuming and running more are disabled (reading
and committing still work); another Optuna than the validated one is said. While this page's
job or a request started elsewhere runs, every action that sends a request is disabled except
Stop (the server runs one such request at a time). While a search runs, Apply, committing a
trial, creating or running another search, replacing or deleting that search's measurement and
deleting that search are disabled; measuring and scoring are allowed and slow it down.

**What is kept where.** Measurements and searches are the workspace's. Forms and prompt sets
(never results, streams or jobs) are kept in the page and, as a convenience, in the browser's
`localStorage` under `wsp.directions.v1.<workspace>` (not written above 256 KiB; the tab works
without it). Plans, scores, the prefix result and baselines are the page's only and are gone
after a reload.

**What has been checked.** Page tests in a fake DOM with a fake server and fake events: the
states, the request bodies, the job line and Stop, the live regions (an event repaints them
without redrawing the view), the charts against worked examples, the edit and commit paths,
and that text from outside stays text. No server, model, real browser or real dataset was used
by these tests.

**Checked by hand.** In a browser, on `tiny-random/qwen2.5` (float32, CPU), served offline
from an isolated root under `python -m wsp.memory_guard` (no violation; largest footprint
3.5 GiB). The code was the branch's as of the stream id fix; the one change after it, the
wording of the Resume button for one trial left, was found by this check and has a page test:

- Before a model is open the tab says "Open a model to use this view". With the model open:
  the not-loaded note, the chat box ticked, Heretic's default system prompt. Opening the tab
  loaded nothing.
- Check of Heretic's two default datasets from the local cache: 400 prompts each, with the
  hash and previews, the model still not loaded. With an empty cache: what a download would
  fetch, and Download answered that `HF_HUB_OFFLINE` is set; nothing was written.
- Prefix detection on Heretic's default test prompts (200 responses): no prefix. The
  not-loaded note went once that job had loaded the model.
- Measuring 800 prompts with module inputs and outputs: the progress line counted up and the
  measurement was saved in 1.5 s. A second one stopped part-way: "Stopped after 184 of 800
  prompts; nothing was saved", and one measurement in the list.
- A request started and the page reloaded: its card, the actions disabled with the reason,
  and "Stop it" stopped it. Prefix detection was used for this; a measurement ends too soon on
  this model.
- Layers: five indices, index 0 marked not usable.
- Edit with the ablation defaults: the plan and its script; the dry run (nothing committed;
  6,912 weights in 6 tensors stored as 912 numbers; drift KL 0.00096); Apply made a patch with
  the same numbers.
- Compare, base against current, three typed prompts per scorer: the scores and their change,
  the responses side by side with the differing rows marked, the KL per prompt (0.0010). After
  the weights changed the results were marked as scored before the change.
- Search, ARA with `lora_rank` 3, four trials, two at random first, seed 7. Started while the
  ablation patch was on, its trials failed with ARA's mismatch refusal (the measurement was made
  on the base weights), the run ended after three and the tab showed the reason. With the patch
  off, a second search: "Stop after this trial" stopped it after trial 1; Resume ran trials 2
  to 4, each row arriving as it finished; the front as a chart and as Heretic's list; the
  baseline as a row and a mark; a trial's parameters, script, seconds and responses beside the
  baseline's. "Check the commit" committed nothing. "Commit trial 2" made a patch, the report
  said the committed edit equals the scored one, and the Patches pane showed "from search
  refusal-ara-2, trial 2". After the commit the search was marked as not able to run or commit
  on the changed weights, with the reason.
- Served without Optuna and pyarrow: the Optuna note, the existing searches still readable, and
  Check of a parquet set refused naming `requirements-datasets.txt`.
- No message in the browser's console in any step; no sideways scrolling at 649 px.

Seen there and left as it is: ARA's default `lora_rank` of 50 is above what a very small model
allows (8 on this one). The server refuses naming the limit and the form marks the settings,
but the form does not show the limit before the request is sent.

**Checked by hand, a checkpoint on disk in Compare.** A later check in the same set-up (largest
footprint 0.9 GiB, no violation), with the merged model of a Heretic `full` ablation run on
this base as the checkpoint:

- "Check the checkpoint" was off, with its reason, until a path was typed. Pressed, the job
  line read "Checking the checkpoint", then the comparison: "5 of 51 tensors differ from the
  base (6,656 of 2,462,792 weights)", the five rows, the two dtypes and the bytes held. The
  model was still not loaded.
- Score, Heretic's two default scorers on 5 prompts each from the local cache, 10 new tokens:
  refusals 0/5 and 0/5, KL 0.0007, "applied exactly", the responses side by side and the KL
  per prompt. With a response-length scorer added: mean length 37.4 and 40.0. These are the
  numbers Heretic's `evaluate_model` printed for this merge. The workspace stayed at its
  version with no patch.
- The run's directory, then the base's own directory, as the path: each refused with its text
  and the path field marked.

Seen there and left as it is: the results do not name the path they were scored for, and they
stay on screen when the path is edited afterwards (as any Compare result stays when the form
changes). The browser's console was not read in this check.

**Checked by hand, steering along a direction.** A later check in the same set-up (largest
footprint 0.7 GiB, no violation), on a direction measured from Heretic's two default sets:

- Layers: the `steer -1` button of index 0 is off (not usable). Pressed at index 2, the note
  read "direction:refusal@2 -1 added to the Run tab's steering panel; Run, Chat and the Probe
  tab's Next token use it." The Run tab's panel then held the entry `direction:refusal@2`, add,
  dose -1, unit "difference of the means", "pushes a direction entering layer 2", and the
  header "the edited model is steered: direction:refusal@2 -1".
- The panel's Check listed the entry with the dose under "constant added" and the vector's
  length (0.0322) under "output column length", headings written for a neuron, and did not
  show what one unit is. That was changed after this check: a direction entry's row now reads
  "vector of length …" and "unit diff: one unit is …". The change is covered by page tests
  and was not looked at again in a browser.
- Run: the base side labelled "unsteered", the edited side "steered: direction:refusal@2 -1"
  with "next token (steered)". On this random-weight model the push (length 0.03) changed the
  order of two of the ten next tokens and not the eight generated tokens.
- Probe, Next token, with the panel's entries: "steered: direction:refusal@2 -1.
  KL(unsteered ‖ steered) = 1.001e-6 nats" and the two token tables.
- Compare, the current weights against "current weights, steered by the Run tab's panel":
  both sides scored (refusals 0/8 and 0/8, KL 0 and 1.2e-6, mean length 26.8 and 27.4), the
  label "steered: direction:refusal@2 -1" under the table, the workspace still at its version
  with no patch.
- The measurement made again under the same name with another winsorization quantile, then
  Run: refused with "measurement 'refusal' changed since this entry was made: its means are
  55da30285ff7… now and the entry names d3d5dbfa8bf0…; add the entry again from the Layers
  view, or drop means_sha256, to steer along the new measurement".

Seen there and left as it is: the Run tab shows that refusal with the error's class name in
front ("SteeringError: …"), as it shows every steering refusal. The Chat mode and the
browser's console were not checked.

Not checked by hand: GGUF and quantized checkpoints, a model that cannot run here, a real
download, a search running in another process, more than 200 trials, and the page served on
another address than the loopback one.

## Routes

Everything that needs no model is checked first and answers 400 with the reason
(`{"error": "DirectionError: ..."}`); a missing name answers 404 JSON. The examples are
from the tiny random test model (2 layers, `d_model` 24).

### `POST /api/directions/measure`

The `measure` arguments, plus an optional `stream`. An unknown field is refused. `good` and
`bad` take dataset specifications too ([Prompt sets from datasets](#prompt-sets-from-datasets));
the events count the resolved prompts.

```json
{"name": "refusal", "good": ["abc", "hello", "fox"], "bad": ["xyz", "qqq"], "stream": "dir-1"}
```

```json
{"version": 1, "name": "refusal", "created": 1794920583, "variant": "base", "n_layers": 2, "d_model": 24,
 "counts": {"good": 3, "bad": 2}, "tokens": {"good": 11, "bad": 6}, "options": {"...": "..."},
 "prompts_sha256": {"...": "..."}, "model": {"...": "..."}, "seconds": 0.012,
 "stats": [{"index": 0, "good_norm": 1.402, "bad_norm": 1.467, "diff_norm": 1.558, "cosine": 0.411, "usable": true},
           {"index": 1, "good_norm": 4.985, "bad_norm": 6.730, "diff_norm": 7.846, "cosine": 0.128, "usable": true},
           {"index": 2, "good_norm": 3.365, "bad_norm": 3.771, "diff_norm": 4.352, "cosine": 0.260, "usable": true}]}
```

- **Stop.** With a `stream`, the request is registered like a Run request's, so
  `POST /api/generation/stop {"stream": "dir-1"}` stops it before its next chunk (a stop
  sent before the request arrives applies when it does). Nothing is saved; the answer is
  `{"stopped": true, "done": 4, "total": 11}`.
- **Events** on `/api/events`: `measuring` when it starts and then at most every 0.25 s,
  then one of `measured`, `stopped` or `failed` (with `error`):

  ```json
  {"type": "directions", "workspace": "rnd", "name": "refusal", "state": "measuring", "seconds": 0.0, "done": 0, "total": 5, "stream": "dir-1"}
  {"type": "directions", "workspace": "rnd", "name": "refusal", "state": "measured", "seconds": 0.013, "done": 5, "total": 5, "stream": "dir-1"}
  ```

### `GET /api/directions`

`{"directions": [meta, ...], "limits": {...}, "parameters": {...}, "requests": [...]}`: the
metas (no prompts, no vectors), a damaged entry as `{"name", "error"}`, the bounds, the
methods' parameter fields for the open model, and the direction requests running on it. The
bounds:

```json
{"limits": {"prompts_per_set": 1024, "characters_per_text": 20000, "tokens_per_prompt": 512, "prompts_per_job": 8,
            "prompts_per_chunk": 4, "name": "^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$", "max_abs_weight": 8.0,
            "full_normalization_lora_rank": [1, 64], "components": ["attn.o_proj", "mlp.down_proj"],
            "row_normalization": ["none", "pre", "full"], "variants": ["base", "current"],
            "scores": {"plugins": ["KeywordRate", "KLDivergence", "ResponseLength"], "states": ["base", "current", "candidate"],
                       "max_response_length": [1, 2048], "scorers_per_request": 8, "script_characters": 100000,
                       "keyword_markers": 256, "chain_of_thought_blocks": 16, "kl_reference_keep": 8,
                       "model_worker_candidate_bytes": 2147483648,
                       "settings": {"KeywordRate": {"score_name": {"kind": "text64", "default": "Refusals"},
                                                    "keyword_markers": {"kind": "markers", "default": ["disclaimer", "sorry", "..."]},
                                                    "print_responses": {"kind": "bool", "default": false}},
                                    "KLDivergence": {},
                                    "ResponseLength": {"strip_responses": {"kind": "bool", "default": true},
                                                       "short_response_threshold": {"kind": "count", "default": 2},
                                                       "short_response_score_length": {"kind": "number", "default": 500.0},
                                                       "print_responses": {"kind": "bool", "default": false}}},
                       "max_response_length_default": 100},
            "search": {"...": "...", "settings": {"...": "..."}, "defaults": {"n_trials": 100, "n_startup_trials": 30}},
            "datasets": {"fields": ["dataset", "commit", "config", "split", "column", "prefix", "suffix", "system_prompt"],
                         "formats": [".arrow", ".csv", ".geoparquet", ".gpq", ".json", ".jsonl", ".ndjson", ".parquet",
                                     ".tsv", ".txt"],
                         "specifications_per_set": 16, "file_bytes": 2147483648, "files_per_specification": 1000,
                         "text_bytes": 268435456, "listed_files": 100000, "download_bytes": 4294967296,
                         "disk_reserve": 2147483648, "heretic_defaults": {"...": "..."}}}}
```

`limits.datasets.heretic_defaults` are Heretic's default prompt sets and system message
([Prompt sets from datasets](#prompt-sets-from-datasets)), published for a client to fill in;
the IDE never applies them by itself.

`limits.scores.settings` gives each scorer's settings with their kind (`text64`, `markers`,
`bool`, `count`, `number`) and Heretic's default (`keyword_markers`: the 33 markers of
[Scores](#scores)); `max_response_length_default` is the `max_response_length` a request
without one gets. `limits.search.settings` and `limits.search.defaults` are described with
the search's routes ([The automatic search](#the-automatic-search), "Routes").

**`parameters`**: the fields of both methods' parameters for the open model, for a form, or
`null` when the model's layers or output components are not known (no ablation or ARA can be
planned there). Field order is the order of a form; the ablation's per-component fields are
listed per component the model has. On the tiny test model:

```json
{"parameters": {"n_layers": 2,
  "ablation": {"fields": [
      {"name": "direction_index", "kind": "number", "default": null, "nullable": true, "null_label": "a direction per layer",
       "min": -1, "max": 1, "min_exclusive": false, "choices": null, "applies_when": null, "heretic_range": [0.4, 0.9], "log": false},
      {"name": "orthogonalize_direction", "kind": "boolean", "default": true, "...": "..."},
      {"name": "row_normalization", "kind": "choice", "default": "none", "choices": ["none", "pre", "full"], "...": "..."},
      {"name": "full_normalization_lora_rank", "kind": "integer", "default": 3, "min": 1, "max": 64,
       "applies_when": {"row_normalization": "full"}, "...": "..."}],
    "components": [
      {"name": "attn.o_proj", "fields": [
         {"name": "max_weight", "kind": "number", "default": 1.15, "min": -8.0, "max": 8.0, "heretic_range": [0.8, 1.5], "...": "..."},
         {"name": "max_weight_position", "kind": "number", "default": 0.8, "min": -2, "max": 4, "heretic_range": [0.6, 1.0], "...": "..."},
         {"name": "min_weight", "kind": "number", "default": 0.575, "min": -8.0, "max": 8.0, "heretic_range": null, "...": "..."},
         {"name": "min_weight_distance", "kind": "number", "default": 1.0, "min": 0, "max": null, "min_exclusive": true,
          "heretic_range": [1.0, 1.0], "...": "..."}]},
      {"name": "mlp.down_proj", "fields": ["... max_weight 0.625 (range [-0.25, 1.5]), min_weight 0.3125, the rest as above"]}]},
  "ara": {"fields": [
      {"name": "start_layer_index", "kind": "integer", "default": 0, "min": 0, "max": 1, "heretic_range": [0, 1], "...": "..."},
      {"name": "end_layer_index", "kind": "integer", "default": null, "nullable": true, "null_label": "to the last layer",
       "min": 1, "max": 2, "heretic_range": [2, 2], "...": "..."},
      {"name": "steer_bad_behavior_weight", "kind": "number", "default": 0.01, "min": 0, "max": 1000.0,
       "heretic_range": [0.0001, 1.0], "log": true, "...": "..."},
      "..."]}}}
```

Every field has the same keys: `name`, `kind` (`number`, `integer`, `boolean`, `choice`),
`default`, `nullable` (and its `null_label`), `min`, `max`, `min_exclusive` (the bound itself
is refused), `choices`, `applies_when` (the field is used only when another field has that
value), `heretic_range` (the range Heretic's search draws it from on this model, as
[The search spaces](#the-search-spaces) computes it, unrounded) and `log` (Heretic draws it on
a log scale); a key that does not apply is `null` (`false` for the three flags). The bounds are
the method's own checks ([The parameters](#the-parameters), [Parameters and the script
function](#parameters-and-the-script-function)); `max_weight_position` is from `-n` to `2n`.
The defaults:

- ablation, scalar fields: the defaults `ablate_direction` applies (`row_normalization`
  `none`, not the search's `full`);
- ablation, distribution fields (required by the API, so they have no default of their own):
  Heretic's draw at the middle of each range, transformed as Heretic transforms a draw, with
  `direction_scope` `per layer` (so `direction_index` `null`): `max_weight` the middle of its
  range clamped at 0, `max_weight_position` and `min_weight_distance` the middle of theirs,
  `min_weight` half of `max_weight` (Heretic draws a fraction of `max_weight`, so it has no
  range of its own); each rounded to four decimals;
- ARA: the defaults of `ablate_ara`. `neighbor_count`'s range is Heretic's
  `min(15, good prompts, bad prompts)` for sets of the largest size; a measurement with fewer
  prompts narrows it.

**`requests`**: the last event of every measure, scores, prefix or dataset-fetch request
running on the open workspace (whoever started it), in the order they started, each exactly as
its event was published (without the event's time `t`):

```json
{"requests": [{"type": "directions", "workspace": "rnd", "name": "refusal", "state": "measuring", "seconds": 1.2,
               "done": 4, "total": 11, "stream": "dir-1"},
              {"type": "direction_datasets", "workspace": "rnd", "state": "fetching", "seconds": 0.4, "phase": "file",
               "file": "data/train-00000-of-00001.jsonl", "done_bytes": 1024, "total_bytes": 4096, "stream": "fetch-1"}]}
```

A request is listed from its first event until it ends, however it ends (an answer, a refusal,
a stop or an error); the entry is updated before each of its events is published, so a client
that refetches on an event finds that event listed, and nothing after a terminal one
(`measured`, `scored`, `detected`, `fetched`, `stopped`, `failed`). With a `stream`, any page
can stop it with `POST /api/generation/stop`. A dataset request is listed only while it
downloads (`"download": true`); a search is not listed here (it has `job`, below).

### `GET /api/directions/{name}`, `GET /api/directions/{name}/prompts`, `DELETE /api/directions/{name}`

The meta plus `stats` (`?orthogonalize_direction=false` for `usable` without the
projection); the stored prompt lists `{"good": [...], "bad": [...]}`; delete answers
`{"ok": true, "deleted": "refusal"}`. A missing name: 404 `{"error": "no direction named 'nothing'"}`.

### `POST /api/directions/script`

The ablation parameters (`"method": "ablation"`, the default when `method` is absent, or
`"method": "ara"` with [its parameters](#parameters-and-the-script-function)); answers the
script and the plan. Nothing is read or run; apply
the script through `POST /api/edit` (with `dry_run` to see the candidate's tests first).

```json
{"direction": "refusal", "direction_index": null, "row_normalization": "full",
 "weight_distributions": {"attn.o_proj":   {"max_weight": 1.0, "max_weight_position": 1.0, "min_weight": 0.5, "min_weight_distance": 1.0},
                          "mlp.down_proj": {"max_weight": 0.8, "max_weight_position": 1.0, "min_weight": 0.0, "min_weight_distance": 1.0}}}
```

```json
{"script": "ablate_direction(\"refusal\", direction_index=None, weight_distributions={\"attn.o_proj\": {\"max_weight\": 1.0, \"max_weight_position\": 1.0, \"min_weight\": 0.5, \"min_weight_distance\": 1.0}, \"mlp.down_proj\": {\"max_weight\": 0.8, \"max_weight_position\": 1.0, \"min_weight\": 0.0, \"min_weight_distance\": 1.0}}, orthogonalize_direction=True, row_normalization=\"full\", full_normalization_lora_rank=3)",
 "plan": {"direction": "refusal", "row_normalization": "full",
          "rows": [{"layer": 0, "component": "attn.o_proj", "weight": 0.5, "skipped": null, "direction": "index 1", "views": 2, "stored": 216},
                   {"layer": 0, "component": "mlp.down_proj", "weight": null, "skipped": "weight 0", "direction": "index 1", "views": 0, "stored": 0},
                   {"layer": 1, "component": "attn.o_proj", "weight": 1.0, "skipped": null, "direction": "index 2", "views": 2, "stored": 216},
                   {"layer": 1, "component": "mlp.down_proj", "weight": 0.8, "skipped": null, "direction": "index 2", "views": 1, "stored": 192}],
          "totals": {"layers": 2, "components": 3, "views": 5, "stored": 624, "skipped": 1}}}
```

`direction` is `"index k"` or, with a fractional `direction_index`, `"between i and i+1 (f)"`.
`stored` is the number of values the pieces will hold (`d_in + d` per view for `none` and
`pre`, `(d_in + d) · r` for `full`); a piece that comes out all zero is not stored, and the
patch's own compaction may store a small tensor's pieces densely when that is smaller.

### `POST /api/directions/scores`

Scores one state of the weights ([Scores](#scores)); nothing is committed. Fields: `state`
(`base`, `current` or `candidate`, required), `script` (with `candidate` only, 1..100,000
characters), `chat`, `system`, `prefix`, `chat_options` (as for `measure`; defaults false,
null, null, null), `max_response_length` (1..2048, default 100), `scores` (1..8 entries) and
`stream`. Each entry has `plugin` (`KeywordRate`, `KLDivergence`, `ResponseLength`, or
Heretic's import paths `heretic.scorers.keyword_rate.KeywordRate` and so on), `prompts`
(1..1024 texts, or dataset specifications: a `[scorer.<Class>.prompts]` table pastes in,
[Prompt sets from datasets](#prompt-sets-from-datasets)), optionally `instance_name` (Heretic's rules, at most 64 characters) and
`optimization` (`minimize` / `maximize` / `none`, echoed, for the search), and the scorer's
own settings under Heretic's names: `score_name`, `keyword_markers`, `print_responses` for
KeywordRate; `strip_responses`, `short_response_threshold`, `short_response_score_length`,
`print_responses` for ResponseLength. A Heretic `[[scorers]]` entry plus its
`[scorer.<Class>]` table's keys pastes in. To reproduce Heretic's rendering pass
`"chat": true, "system": "You are a helpful assistant."`. On the tiny model:

```json
{"state": "candidate", "script": "mlp(1).W_out *= 0.5", "max_response_length": 4,
 "scores": [{"plugin": "KeywordRate", "prompts": ["abc", "hello"]},
            {"plugin": "KLDivergence", "optimization": "minimize", "prompts": ["ab", "zz"]},
            {"plugin": "ResponseLength", "instance_name": "harmless", "prompts": ["quiet"], "print_responses": true}],
 "stream": "sc-1"}
```

```json
{"state": "candidate", "version": 1, "seconds": 0.179,
 "candidate": {"stored": 960, "changed": [{"param": "blocks.1.mlp.W_out", "changed": 960, "of": 960, "norm": 2.3478, "max_abs": 0.2545,
                                           "rank": 24, "kinds": ["block"], "stored": 960}],
               "output": "", "commit_check": {"ok": true, "error": null}, "note": null},
 "rendering": {"chat": false, "system": null, "prefix": null, "chat_options": null},
 "generation": {"runner": "FunctionalRunner", "where": "this process", "generation": "argmax loop (wsp.model.generate); no end-of-text token",
                "first_token": "one forward pass, last row", "max_response_length": 4, "settings": {}},
 "scores": [{"plugin": "KeywordRate", "name": "Refusals", "optimization": "none", "value": 0.0, "display": "0/2", "matches": 0, "total": 2},
            {"plugin": "KLDivergence", "name": "KL divergence", "optimization": "minimize", "value": 0.03087875572964549, "display": "0.0309",
             "by_definition": false, "per_prompt": [0.009344379417598248, 0.052413132041692734],
             "reference": {"key": "258c2b10...", "cached": false, "recomputed": null, "bytes": 208}},
            {"plugin": "ResponseLength", "name": "Mean response length - harmless", "optimization": "none", "value": 4.0, "display": "4.0",
             "short": 0, "responses": [{"prompt": "quiet", "response": "cznw", "tokens": 4, "length": 4}]}],
 "done": 7, "total": 7}
```

- `name` is Heretic's: the scorer's name, with `" - <instance_name>"` when one is given.
  Numbers are not rounded; `display` is Heretic's `md_display` (KL `0 (by definition)` for
  state `base`, where no second pass runs and `per_prompt` is null).
- `responses` only with `print_responses`: KeywordRate rows `{prompt, response, tokens, match}`,
  ResponseLength rows `{prompt, response, tokens, length}` (`length` after the short rule).
- `candidate` is null for `base` and `current`. `commit_check` says whether `Session.apply`
  would commit this script (the task scope, then the largest-patch setting); it never stops
  the scoring. A script that changes nothing scores as `current`, with the note
  `the script changed no weights`.
- `done` / `total` count the units of the request (reference rows computed, distinct
  response prompts, KL rows of the state).
- **Stop** as for `measure`, checked before every unit (one response or one first-token pass):
  `{"stopped": true, "done": 1, "total": 7}`; a KL reference being computed is discarded.
- **Events**: type `direction_scores`, state `scoring` when it starts (`total` null) and then
  at most every 0.25 s with `phase` (`reference`, `responses`, `logits`), then one of `scored`,
  `stopped`, `failed` (with `error`):

  ```json
  {"type": "direction_scores", "workspace": "rnd", "state": "scoring", "seconds": 0.014, "phase": "reference", "done": 1, "total": 7, "stream": "sc-1"}
  {"type": "direction_scores", "workspace": "rnd", "state": "scored", "seconds": 0.179, "done": 7, "total": 7, "stream": "sc-1"}
  ```

**Steering the scored state.** Two more fields, `steer` (a steering specification:
[STEERING.md](STEERING.md#the-steering-specification), with direction entries,
[Steering along a direction](#steering-along-a-direction)) and `p50` (as for `/api/run`), score
the state as the steered model behaves. `steer` absent, `null` or `[]` is an unsteered request:
`p50` is then ignored and the answer has exactly the keys above. Otherwise, with everything else
and before the model is loaded: state `base` is refused (`SteeringError: the base model is
never steered; steer the current variant`), then `p50` and every entry are checked with
steering's own texts. After the model is loaded, the route prepares the steering before the
scoring operation starts (a calibration, when a neuron `resid` entry needs one, runs there,
outside the operation, and publishes no `steering` events). In the operation every response of
the state is generated with the push, and every KL row is the steered first-token distribution
(`next_logits` with the steering where the runner has it, else one forward pass with the
steering hooks, its last row); the KL reference stays the unsteered base. `candidate` with
`steer` steers the candidate's weights. The answer gains `steering`, as a Run reply's report:
`{"variant": "current", "resolved": true, "entries", "calibration", "layers", "neurons",
"heads", "mlps", "directions", "reads"}`, without the wire form. The unknown-field message still
names only the fields above and `stream`; `recalibrate` is not a field of this route. A Python
caller of `score` that passes no prepared steering gets it prepared inside the operation; with a
neuron `resid` entry under a runtime supervisor that is a calibration inside the scoring
operation, which is not tested.

### `POST /api/directions/prefix`

Heretic's response-prefix detection ([Response prefix and thinking blocks](#response-prefix-and-thinking-blocks)).
Fields: `prompts` (required, 1..1024 texts, or dataset specifications: Heretic's
`response_prefix_test_prompts` list pastes in), `system`, `chat_options`, `variant` (`base`, the
default, or `current`), `max_response_length` (default 100), `chain_of_thought_blocks` (0..16
pairs, default Heretic's four), `print_responses`, `stream`. It always reads the prompts as a
chat message, so it needs the model's own chat template. On the tiny model with a stub
template (`xx<system>zz<user>yy`):

```json
{"prompts": ["abc", "ab"], "max_response_length": 3, "stream": "pf-1"}
```

```json
{"response_prefix": "m", "found_by": "common_prefix", "block": null,
 "steps": {"template": {"rendering": "xxYou are a helpful assistant.zzThis is a dummy prompt.yy", "system_role": true},
           "common_prefix": {"responses": 2, "prefix": "m"}},
 "generation": {"runner": "FunctionalRunner", "...": "..."}, "seconds": 0.003, "done": 2, "total": 4}
```

`found_by` is `template_closed`, `template_initializer`, `common_prefix`,
`common_prefix_block` or `none`; `steps.recheck` (`{responses, additional}`) appears when a
block was closed through the prefix. With `print_responses` each step also has `texts`. Stop
and events as for scores, type `direction_prefix`, states `detecting` / `detected` /
`stopped` / `failed`, phases `common_prefix` / `recheck`; `total` is `2 N`, the most it can run.

The score answer has `datasets`, one set record or `null` per scorer, and the prefix answer
`datasets`, the record of its prompts or `null` ([What is recorded](#what-is-recorded)); with
`print_responses`, a response row's `prompt` is the canonical prompt (a text, or
`{"system", "user"}`).

### `POST /api/directions/datasets`

Resolves a prompt set the way measure, scores, prefix and searches will, and is the one route
that fetches dataset files ([Prompt sets from datasets](#prompt-sets-from-datasets)). Fields:
`prompts` (required: any value a prompt field takes), `system` (text or null, default null:
the request's system message the canonical form is taken under), `download` (default false),
`texts` (default false: answer the first three prompts only), `stream`. An unknown field is
refused. It needs an open workspace (its file lists of Hub datasets live there) and no model.

Everything cached (a local text file at `/data/prompts.txt` with five lines):

```json
{"prompts": {"dataset": "/data/prompts.txt"}}
```

```json
{"cached": true, "count": 5, "prompts_sha256": "5f0c...", "datasets": {"version": 1, "count": 5, "prompts_sha256": "5f0c...", "parts": [{"...": "..."}]},
 "preview": ["alpha", "beta", "gamma"], "fetched": [], "fetched_bytes": 0, "seconds": 0.001}
```

With `"texts": true` the answer also has `prompts`, the whole canonical set. Something not
cached answers **200** with what a fetch would get, and nothing is fetched:

```json
{"prompts": {"dataset": "mlabonne/harmless_alpaca", "split": "train[:400]", "column": "text"}}
```

```json
{"cached": false, "would_fetch": [{"field": "prompts", "repository": "mlabonne/harmless_alpaca", "revision": "main", "commit": null, "files": null,
                                   "requests": ["the repository's file list at main (one Hub API request)", "README.md",
                                                "the data files of split train (names and sizes known after the file list)"]}],
 "fetched": [], "fetched_bytes": 0, "seconds": 0.002}
```

With a file list in the workspace, `files` lists the missing files with their bytes and
`requests` names them. Any other refusal of the set (a bad split, an unknown column, a
dataset card that says more rows than the cache holds, ...) answers 400 as usual.

`"download": true` fetches what is missing ([The fetch](#the-download-gate-and-the-fetch)),
then answers as the first case, with what it fetched:

```json
{"cached": true, "count": 400, "prompts_sha256": "...", "datasets": {"...": "..."}, "preview": ["...", "...", "..."],
 "fetched": [{"repository": "mlabonne/harmless_alpaca", "commit": "02c6a92cfcf11bb0c387334f8146d149d65b587f", "path": "README.md", "bytes": 388},
             {"repository": "mlabonne/harmless_alpaca", "commit": "02c6a92cfcf11bb0c387334f8146d149d65b587f", "path": "data/train-00000-of-00001.parquet", "bytes": 971794}],
 "fetched_bytes": 972182, "seconds": 1.9}
```

- **Stop.** With a `stream`, `POST /api/generation/stop` stops the fetch before its next file:
  `{"stopped": true, "done_bytes": 388, "total_bytes": 972182, "fetched": [...]}`. The files
  fetched stay in the cache (they are complete); the next request fetches the rest.
- **Events** (only with `download`): type `direction_datasets`, state `fetching` at most every
  0.25 s with `phase` (`list` for the file list, `file` while a file arrives), `file`,
  `done_bytes`, `total_bytes`; then `fetched` (with `fetched` and `fetched_bytes`), `stopped`
  (with `done_bytes`, `total_bytes`, `fetched`) or `failed` (with `error`):

  ```json
  {"type": "direction_datasets", "workspace": "rnd", "state": "fetching", "seconds": 0.0, "phase": "list", "file": null, "done_bytes": 0, "total_bytes": 0, "stream": "ds-1"}
  {"type": "direction_datasets", "workspace": "rnd", "state": "fetched", "seconds": 1.9, "fetched": [...], "fetched_bytes": 972182, "stream": "ds-1"}
  ```

### `POST /api/directions/evaluate`

Heretic's `evaluate_model` ([Scoring a modified checkpoint](#scoring-a-modified-checkpoint-heretics-evaluate_model)):
a modified checkpoint on disk is read, never written, and scored as an uncommitted change on
top of the loaded model, against the base. The directory may also be a PEFT LoRA adapter
(the base plus the adapter, [An adapter as the modified model](#an-adapter-as-the-modified-model))
or another workspace (its base with its enabled patches,
[Another workspace as the modified model](#another-workspace-as-the-modified-model)); the
answer's `input` says which it took. Fields:

| Field | Values |
|---|---|
| `path` | required: the absolute path of a local directory holding the modified checkpoint, a PEFT LoRA adapter or a workspace (at most 4096 characters) |
| `check` | `false` (default) or `true`: compare the weights only; no model is loaded, nothing is scored |
| `scores`, `chat`, `system`, `prefix`, `chat_options`, `max_response_length` | as for `POST /api/directions/scores` (`state` and `script` are not fields here); `scores` is required unless `check` is true, and with `check` it is checked and its prompts are encoded for the tokenizer comparison |
| `stream` | as for every direction route |

The path is read with the trust `POST /api/open` and a local dataset path have
([What the IDE reads and sends](#what-the-ide-reads-and-sends)); like every non-GET route it
is refused to a foreign browser origin when the IDE is served normally.

**The order of the checks.** Through the route: an open workspace; the body (unknown field,
`path`, `check`, `scores` present); then, still before any weight is read beyond the headers,
the workspace (GGUF, weights on the Hub, a base tensor that is not floating point); the path
(absolute, exists, a GGUF file, another file, the kind of directory: a PEFT adapter, a
workspace, a checkpoint, more than one of them, none; an adapter and a workspace then take
their own checks, listed under their sections, and a checkpoint goes on with the safetensors files: the
index (`model.safetensors.index.json`, its entries must be file names in the directory),
else `model.safetensors`, else every `*.safetensors`; `config.json` for a `transformers`
base; the base's own files, compared by real path; `config.json` readable; quantization; the
safetensors headers; `generation_config.json` readable); the tensor names (missing, then
extra), shapes and dtypes; the configuration table; the generation configuration; the scorers
(the scores route's own refusals); and, unless `check`, the model (a model that cannot run
keeps its own refusal). Then pass 1 (non-finite values) and the forecast. The Python
function `evaluate(ws, body)` checks the workspace before the body.

On the tiny model, a checkpoint whose `blocks.1.mlp.W_out` is the base's times 1.5 (its
`tokenizer_meta.json` copied):

```json
{"path": "/data/merged", "max_response_length": 4,
 "scores": [{"plugin": "KeywordRate", "prompts": ["abc", "hello"], "print_responses": true},
            {"plugin": "KLDivergence", "prompts": ["ab", "zz"]}],
 "stream": "ev-1"}
```

```json
{"check": false, "path": "/data/merged", "input": {"kind": "checkpoint"}, "version": 1, "seconds": 0.06,
 "compare": {"tensors": 38, "differ": 1, "changed": 960, "of": 10666,
             "rows": [{"param": "blocks.1.mlp.W_out", "changed": 960, "of": 960, "norm": 2.3478414623960826,
                       "max_abs": 0.25452715158462524, "relative": 0.5000000027882282}],
             "dtypes": {"base": {"F32": 38}, "modified": {"F32": 38}}, "tied": [], "not_loaded": [], "seconds": 0.009,
             "held": {"tensors": 1, "weights": 960, "bytes": 3840, "applied_bytes": 11520, "fits": true, "why": null},
             "config": {"same": true, "other": []}, "generation_config": {"same": true},
             "tokenizer": {"prompts": 4, "differ": 0, "template_same": true},
             "reconstruction": {"inexact": 0, "max_abs": 0.0}},
 "rendering": {"chat": false, "system": null, "prefix": null, "chat_options": null},
 "generation": {"runner": "FunctionalRunner", "...": "..."},
 "base": {"state": "base", "scores": [{"plugin": "KeywordRate", "name": "Refusals", "value": 0.0, "display": "0/2", "...": "..."},
                                      {"plugin": "KLDivergence", "name": "KL divergence", "value": 0.0, "display": "0 (by definition)", "...": "..."}]},
 "modified": {"state": "candidate", "scores": [{"plugin": "KeywordRate", "name": "Refusals", "value": 0.0, "display": "0/2", "...": "..."},
                                               {"plugin": "KLDivergence", "name": "KL divergence", "value": 0.028733290266245604, "display": "0.0287",
                                                "per_prompt": [0.005139241926372051, 0.052327338606119156], "...": "..."}]},
 "changes": [{"name": "Refusals", "base": 0.0, "modified": 0.0, "change": 0.0},
             {"name": "KL divergence", "base": 0.0, "modified": 0.028733290266245604, "change": 0.028733290266245604}],
 "datasets": [null, null], "notes": [], "done": 6, "total": 6}
```

- `input`: which kind of directory `path` was read as: `{"kind": "checkpoint"}`; for an
  adapter `{"kind": "adapter", "peft_type", "r", "lora_alpha", "use_rslora",
  "base_model_name_or_path", "modules"}`, `modules` one entry per module in the workspace's
  tensor order with `module` (PEFT's name), `param` (the tensor it changes), `r`, `alpha`,
  `scaling`, `transposed` (a Conv1D weight) and `dtype` (of the factors); for a workspace
  `{"kind": "workspace", "name", "workspace_kind", "same_base", "patches", "disabled",
  "candidate"}`, `patches` its enabled patches in stack order (`id`, `message`, `strength`),
  `disabled` how many others it has, `candidate` `"patches"` (its own pieces are scored) or
  `"difference"` (pass 2's `other − current`). In every answer, a check's too.
- `compare`: pass 1's `tensors` (compared), `differ`, `changed` and `of` (weights, over the
  compared tensors), `rows`, the stored `dtypes` of the compared tensors on each side, `tied`,
  `not_loaded` (the differing tensors that the loaded text model does not load, sorted; `[]`
  when there are none, in every answer, also a check's; they are in `rows` and in the counts
  but get no piece) and its `seconds`; `held`, the forecast (`bytes` = 4 W, `applied_bytes` =
  12 W, `why` the refusal text when it does not fit); `config` (`same`, and `other`, the noted keys);
  `generation_config`; `tokenizer` (`prompts`, `differ`, null without a tokenizer file,
  `template_same`); `reconstruction` (score answers only).
- `base` and `modified` are `POST /api/directions/scores` answers' `scores` (with `responses`
  where `print_responses` is set); `modified.state` is `candidate`, or `current` when the
  checkpoint equals the current weights. `changes[i].change = modified − base`, unrounded.
  `datasets` has one set record or `null` per scorer; `done` / `total` are the checkpoint's
  evaluation's.
- With `"check": true` the answer is `{"check": true, "path", "input", "version", "compare" (without
  `reconstruction`), "notes", "seconds"}`; `held.fits` and `held.why` say whether a score
  request would be refused for memory.
- **Stop** as for scores, checked before every tensor of each pass and before every unit of the
  scores: `{"stopped": true, "done": 0, "total": 38}`, the `done` / `total` of the step it
  stopped in (pass 1's tensors, pass 2's tensors, the reference or a state's units). Nothing is
  written but a KL reference that was completed.
- **Events**: type `direction_scores`, states `scoring` / `scored` / `stopped` / `failed` as for
  scores, with the phases `compare` (pass 1, `total` the compared tensors), `reference`,
  `base_responses`, `responses` and `logits`; the request is listed in `GET /api/directions`
  `requests` while it runs.

Refusals (400, `DirectionError: <text>`), each exact:

| When | Text |
|---|---|
| the open workspace is GGUF | `scoring a modified checkpoint needs a floating-point base: this workspace's base is GGUF (quantized) and is never dequantized to compare it with another checkpoint` |
| base weights on the Hub | `the open model's weights are still on the Hub; pull them first (the comparison reads every base tensor)` |
| a base tensor not floating point | `the open model's tensor <name> is stored as <dtype>; only a checkpoint of floating-point tensors can be compared` |
| the body is not an object | `the request must be an object` |
| an unknown field (the first, sorted) | `<field>: unknown field; the fields are path, check, scores, chat, system, prefix, chat_options, max_response_length and stream` |
| `path` missing, not text, too long | `path: required: the absolute path of the modified checkpoint's directory` / `path must be text` / `path is longer than 4096 characters` |
| `check` not a boolean; `scores` missing without `check` | `check must be true or false` / `scores: required (unless check is true)` |
| `path` not absolute | `path must be an absolute path of a local directory; a Hub id is not looked up (give its snapshot directory in the Hugging Face cache)` |
| nothing at `path` | `path: no such directory: <path>` |
| a `.gguf` file | `path names a GGUF file; the modified model must be a directory of floating-point safetensors files (a GGUF model is not read as floating-point weights)` |
| another file | `path names a file; give the checkpoint's directory` |
| more than one kind of directory (the kinds in the order adapter, workspace, checkpoint; at most 3 files each, then ` and <k> more`) | `path holds more than one kind of model: <a PEFT adapter (adapter_config.json)>; <a workspace (workspace.json)>; <checkpoint weights (<files>)>; give the directory of one` |
| none of them | `path holds no model the IDE reads: no PEFT adapter (adapter_config.json), no workspace (workspace.json) and no safetensors weights (model.safetensors, an index, or *.safetensors files); it holds <n> entries: <names>` (sorted, a directory with `/`, at most 8 names then ` and <k> more`; `1 entry`; `it is empty`) |
| an index naming no file | `path has no safetensors weights (model.safetensors, an index, or *.safetensors files)` |
| an index entry that is not a file in the directory (a name with a path separator included) | `the checkpoint's index names <file>, which is not in path` |
| the index unreadable, or without a `weight_map` of file names | `the checkpoint's index model.safetensors.index.json cannot be read: <error>` |
| a `transformers` base and no `config.json` | `path has no config.json` |
| the base's own files | `path holds the open workspace's own base checkpoint: there is nothing to compare` |
| `config.json` unreadable (a `transformers` base) | `the modified checkpoint's config.json cannot be read: <error>` / `the modified checkpoint's config.json cannot be read: it is not an object` |
| quantization declared | `the modified checkpoint's config.json declares quantization (<quant_method, or "quantization">); its integer storage is not read as floating-point weights` |
| the safetensors headers unreadable | `the checkpoint's safetensors headers cannot be read: <type>: <error>` |
| a `generation_config.json` unreadable | `the modified checkpoint's generation_config.json cannot be read: <error>` / `the open model's generation_config.json cannot be read: <error>` |
| base tensors missing | `the modified checkpoint lacks <n> tensor(s) of the base, for example <name>; it is not the same architecture` |
| extra tensors | `the modified checkpoint has <n> tensor(s) the base does not, for example <name>; it is not the same architecture` |
| a shape | `<name> has shape [<a>, <b>] in the modified checkpoint and [<c>, <d>] in the base` |
| a dtype | `<name> is stored as <dtype> in the modified checkpoint; only floating-point tensors are compared (a quantized checkpoint is not dequantized)` |
| a configuration row | `the modified checkpoint's config.json differs from the base's in <row> (base <v1>, modified <v2>): it would run differently from the configuration the IDE runs it with, the base's` |
| a greedy generation key | `the modified checkpoint's generation_config.json differs from the base's in <key> (base <v1>, modified <v2>): Heretic's evaluate_model would generate with the modified checkpoint's settings and the IDE would use the base's; this is not supported yet` |
| a NaN or inf (pass 1) | `<name> contains NaN or inf in the modified checkpoint` |
| the held changes do not fit (score requests) | `the modified checkpoint differs from the current weights in <t> tensors (<W> weights); scoring it holds them as FP32 changes applied to the loaded model, about <12 W bytes> at 12 bytes per weight, and memory allows <w> weights now (<available> available, <preferred> kept free by the Admission headroom reserve). Free memory on the machine, or score a checkpoint that changes fewer tensors` |
| memory not measured (score requests) | `memory could not be measured (<reason>), so the modified checkpoint's <W> changed weights are not loaded` |

`<v1>` and `<v2>` are the JSON of the normalised values; weight counts have thousands
separators; byte amounts are written as GiB or MiB. The refusals of an adapter and of another
workspace are listed under [An adapter as the modified model](#an-adapter-as-the-modified-model)
and [Another workspace as the modified model](#another-workspace-as-the-modified-model); after
them, every check above from the tensor names on applies to them too.

Notes (in `notes`, in this order), each exact:

| When | Text |
|---|---|
| an adapter names another base (both known) | `the adapter names <name> as its base (base_model_name_or_path); it is applied to the open workspace's base, <repo>` |
| a workspace over another base | `the workspace at path is over another base (its weight files are not this workspace's): its base with its <p> enabled patch(es) is compared with this base as a checkpoint` |
| the tokenizers encode some texts differently | `the modified checkpoint's tokenizer encodes <k> of the <N> prompt texts differently from the base's; it is scored with the base's tokenizer, as Heretic's evaluate_model does` |
| no (readable) tokenizer file | `the modified checkpoint has no tokenizer file; it is scored with the base's tokenizer, as Heretic's evaluate_model does` |
| the chat templates differ | `the modified checkpoint's chat template differs from the base's; prompts are rendered with the base's template, as Heretic's evaluate_model does` |
| no tensor differs | `the modified checkpoint's weights equal the base's in every tensor` |
| patches are enabled | `<p> patch(es) are enabled in this workspace: the modified checkpoint is scored as itself (its weights replace the current ones for this scoring), against the base weights` |
| a tied head absent from the checkpoint | `<names> is not in the modified checkpoint; the base ties it to the token embedding, so it is not compared` |
| a tied head only in the checkpoint | `<names> is in the modified checkpoint but not in the base; the base ties it to the token embedding, so it is not compared` |
| a tied head in both | `<names> is in both checkpoints; the base ties it to the token embedding, so the model does not use it and it is not compared` |
| one differing tensor the text model does not load | `1 differing tensor belongs to a part the text model does not load (<part>) and is not scored: <name>` (`<part>` such as `the vision tower` or `the multi-token prediction head`) |
| several differing tensors the text model does not load | `<k> differing tensors belong to parts the text model does not load (<parts>) and are not scored: <names>` (`<parts>` the distinct parts joined by `; `, one part when they all belong to one; at most 8 names, then ` and <k> more`) |
| other configuration keys differ | `the modified checkpoint's config.json also differs in <keys>; these do not change how the IDE runs it` (at most 8 keys, then ` and <k> more`) |
| the applied change is inexact (score answers) | `<n> weights of the modified checkpoint come out differently by at most <x> once applied as a change in float32` (`<x>` to 6 significant digits) |
| the runner's patches take the side path (score answers: a bfloat16 or float16 run, or `WSP_PATCH_APPLY=side`) | `this run is in <dtype>: the checkpoint's changes are applied as float32 differences beside the <dtype> base weights (the side path), not as the checkpoint's weights rounded to <dtype>, which is what loading it would give` |

## Compared with Heretic

On 2026-11-18 the same model, prompts and parameters were run through Heretic
(github.com/p-e-w/heretic at `64e9991`) and through this feature, and compared stage by
stage. Numbers: [`directions-heretic-validation.json`](directions-heretic-validation.json).

**The model** is `tiny-random/qwen2.5` (4 layers, `d_model` 8, 2.46M parameters, random
weights, the real Qwen2.5 tokenizer and chat template): Heretic's own test fixture. It
checks the arithmetic and the plumbing. It says nothing about behaviour on a trained model.

**Setup.** Heretic: its fixture configuration (5 good and 5 bad prompts, 5 KL prompts, 2
trials, the first trial of the Pareto front exported as a merged model) with float32, the
CPU and batch size 1; one run per `row_normalization`, and one more with another seed whose
exported trial uses a direction per layer. The IDE: the same checkpoint files opened as a
workspace, the same prompts read as a chat message with Heretic's system prompt, the
exported trial's parameters pasted into `ablate_direction` unchanged.

| Stage | Result |
|---|---|
| Token ids of every prompt as fed to the model | identical, 15 of 15 |
| Mean residuals, each of the 5 indices | relative difference at most 3.0e-7 |
| Directions | cosine at least 1 − 4.2e-11 at indices 1 to 4; index 0 is zero in both (every prompt ends in the same template token) |
| Which tensors change | the same 5 in both; every other tensor exactly unchanged in both |
| The weight change, from each tool's own measurement | relative difference 1.1e-5 to 1.4e-5 |
| The weight change, from Heretic's means in both | 1.9e-7 (`none`, `pre`), 6.5e-7 and 2.5e-6 (`full`) |
| Base model's first-token logits | largest difference 1.6e-7 |
| KL of the edited model's first token against the base | agree to between 5 and 3 significant digits (5.85249e-4 against 5.85252e-4; worst 5.74116e-4 against 5.74513e-4) |

- The 1e-5 in the end-to-end row comes from the measurement, not from the ablation: the two
  tools' means differ by rounding (different torch and transformers versions), and the
  direction is the difference of two means that are only 3 to 5% apart, which magnifies it.
  With the same means the weight changes agree to rounding.
- Heretic's KL is measured with its adapter attached, the IDE's with the patch added to the
  weights; the scores differ by at most 4e-7 nats.
- The final-norm convention of index `n` and the chat rendering with a system message ran on
  a real `transformers` model here for the first time, and match.

Not covered by this comparison: a trained model; a tokenizer that adds a beginning-of-text
token (Qwen's does not; aligned since, and compared on a fixture:
[How a prompt becomes token ids](#how-a-prompt-becomes-token-ids)); a response prefix or
a thinking block (the prefix was empty); the difference between this exact rank-`r`
approximation and Heretic's randomized one, which cannot show on 8-wide matrices; GGUF;
the refusal score (0 of 5 in Heretic on random weights; not computed here); winsorization.

To run it again: `scripts/heretic_dump.py` (with Heretic's interpreter, in a finished
Heretic run directory) writes what Heretic measured; `scripts/heretic_compare.py` (with
the IDE's interpreter) compares. Both carry their usage. Run every model load under
`python -m wsp.memory_guard`, with a reservation above the IDE's own forecast for the
model (4.86 GiB for this one), an isolated root and an isolated model cache.

### ARA against Heretic

The fitted low-rank ablation (below) was compared the same way on the same day and the
same model. Numbers:
[`directions-ara-heretic-validation.json`](directions-ara-heretic-validation.json).

**Setup.** Heretic's fixture configuration with its ARA modifier, `lora_rank = 4`, float32,
the CPU, batch size 1, a merged export; three runs: seed 12345, seed 1 (its exported trial
fits two layers), and seed 12345 with `preserve_row_magnitudes = false`. The IDE: a
measurement with `module_io`, then the exported trial's parameters, Heretic's settings and
Heretic's seed pasted into `ablate_ara`, applied through `Session.apply`.

**What agrees exactly.** Given the same captured inputs and outputs and the same torch
version, the IDE's fit is Heretic's fit, bit for bit:

| Check | Result |
|---|---|
| Token ids of every prompt | identical, 15 of 15 in each run |
| Captured module inputs and outputs, 8 components | all 8 captured by both; relative difference at most 7.0e-7 (inputs), 5.5e-7 (outputs) |
| Initial `A` | identical |
| Which tensors change | the same in both: `o_proj` of each layer in the range. `down_proj` is left unchanged by both: on this model the squared norm of its starting gradient is 9e-13 to 1e-10, below L-BFGS's default `tolerance_change` of 1e-9, so the optimiser stops before its first step (for `o_proj` it is 1.2e-9 to 2.9e-7). Every other tensor exactly unchanged in both |
| The IDE's fit against Heretic's fitting loop, same inputs, one interpreter (`scripts/heretic_fit_check.py`) | identical in every element at every budget tried (1×1 to 5×20 iterations), in float32 and float64, under torch 2.14.1 and 2.15.0 |
| The whole IDE (`Session.apply`, the patch read back) under Heretic's library versions, from Heretic's captures, against Heretic's merged model | 0 of 2048 elements differ, in each of the 4 fitted matrices |

**What does not, and why.** With each tool's own captures and its own library versions,
the fitted changes differ:

| Run, matrix | change / weight | relative difference of the two changes | objective: start → Heretic, IDE |
|---|---|---|---|
| seed 12345, layer 2 | 2.0e-4 | 4.4e-2 | 1.86382e-6 → 1.77388e-6, 1.77413e-6 |
| seed 1, layer 2 | 9.6e-2 | 0.38 | 4.409e-5 → 1.738e-5, 1.351e-5 |
| seed 1, layer 3 | 1.5e-3 | 1.0e-3 | 3.012e-5 → 2.33172e-5, 2.33172e-5 |
| no row magnitudes, layer 2 | 2.1e-4 | 5.1e-2 | 1.86382e-6 → 1.77409e-6, 1.77400e-6 |

This is the fit's sensitivity, not a difference between the tools. Heretic's own fitting
loop, fed its own captures:

- lands 0.3% to 33% away from Heretic's export when it is run under torch 2.15.0 instead
  of 2.14.1 (the IDE's fit lands on the same point, bit for bit);
- moves by 0.4% to 14% after a single iteration, and by up to 37% at the full budget, when
  its inputs are changed in their seventh digit (relative 1e-7). The two tools' captures
  differ by 7e-7, and still by 6.4e-7 when the IDE runs under Heretic's library versions.

Most of the first-iteration sensitivity comes from the overcorrect term: at the start each
bad output is at distance zero from its own captured value, so the direction of that term's
gradient is rounding noise (see "Known limits of the fit"). With
`overcorrect_relative_weight` set to 0 the same change of the inputs moves the first
iteration by 1e-5 to 2e-4. Long fits still drift apart without it (up to 45% at some
budgets). Heretic's search draws `overcorrect_relative_weight` from 0 to 1.3, so its trials
almost always have it.

The KL scores are all between 3e-7 and 1.5e-6 nats: these edits barely move a random
model. From each tool's own captures they differ as the weights do (3.4e-7 against 6.0e-7,
1.06e-6 against 7.7e-7, 5.3e-7 against 5.5e-7). With identical weights (the last row of the
first table) they differ by at most 3.5e-8, where the two tools' base logits already differ
by 1.4e-7.

**Two differences were found and removed by this comparison.** As first built, the IDE's
loss gave the same value as Heretic's but rounded differently: it summed the row norms over
`V + AᵀBᵀ` instead of over `W + B A`, and it built the scale before the two outputs instead
of after them (the order in which the three gradient paths into `A` and `B` are added up).
The fits agreed to 5e-8 after one iteration and then drifted to 1e-2..1e-1. Both are now
done in Heretic's order, and a test holds the fit to equality with a loop in that order
(`test_the_fit_takes_exactly_the_steps_of_a_loop_in_heretics_order`).

What this means for a comparison on a larger model: per-weight equality can be expected
only from the same captures under the same torch version (`scripts/heretic_fit_check.py`
checks exactly that, without loading the model); otherwise the comparison has to be of the
objective reached and of behaviour scores.

Not covered: a trained model (whether its fits are as sensitive is not known; the fit
check prints it); a module with a bias; a `down_proj` that the optimiser actually moves;
`lora_rank` above 4; bfloat16; everything listed as not covered above.

To run it again: `scripts/heretic_dump.py` as above (it also writes Heretic's captured
module inputs and outputs for an ARA run), then `scripts/heretic_compare_ara.py` (the whole
IDE against the dump and the merged model) and `scripts/heretic_fit_check.py` (the fit
alone, tensors from files, under either tool's interpreter).

### On a trained model

On 2026-11-18 both methods were compared on `Qwen/Qwen2.5-0.5B-Instruct` (24 layers,
`d_model` 896, 494M parameters, stored in bfloat16, run by both tools in float32 on the
CPU). Numbers:
[`directions-heretic-validation-qwen05b.json`](directions-heretic-validation-qwen05b.json).

**Setup.** Heretic: 64 good and 64 bad prompts, 32 prompts per score, responses of at most
30 tokens, 4 trials, the first trial of the Pareto front exported as a merged model; one
run each for directional ablation with `row_normalization` `full` and `none`, and one for
ARA with its defaults (`lora_rank` 50). The base model refused 30 of the 32 scored prompts.
The IDE: as on the fixture.

**Directional ablation** agrees:

| Stage | Result |
|---|---|
| Token ids | identical: 64 + 64 measured prompts, 32 KL prompts |
| Mean residuals, each of the 25 indices | relative difference at most 1.4e-6 |
| Directions | cosine at least 1 − 1.0e-10 at indices 1 to 24 |
| Which tensors change | the same 31 in both; the other 259 exactly unchanged in both |
| The weight change, `none` | relative difference 3.2e-6 to 3.9e-6; from Heretic's means at most 6.5e-7 |
| The weight change, `full` | median 1.4e-5, at most 3.9e-4 (the same from Heretic's means) |
| Base model's first-token logits | largest difference 8.0e-5 on logits up to 23 |
| KL of the edited model | 0.0132517 against 0.0132516 (`full`), 0.0133851 against 0.0133852 (`none`) |

The 3.9e-4 in the `full` row is the first sight of a difference the 8-wide fixture could
not show: both tools store a rank-3 approximation of the norm-preserving change, Heretic
from a randomized decomposition, the IDE the best one. The approximation itself is much
coarser than that difference: the IDE's `full_error` (the distance of the rank-3 change
from the exact one) is 3.8% to 6.4% for the last 11 of the 31 components (the lines of
the others were not kept).

**ARA** agrees exactly where the fixture did, and is as sensitive:

| Check | Result |
|---|---|
| Captured module inputs and outputs, 48 components | all captured by both; relative difference at most 7.9e-6 |
| Which tensors change | the same 8 in both (`o_proj` and `down_proj` of layers 12 to 15: here `down_proj` is fitted); the other 282 exactly unchanged in both |
| The IDE's fit against Heretic's loop, same inputs, one interpreter | identical in every element, every budget, float32 and float64, both torch versions |
| The whole IDE under Heretic's library versions, from Heretic's captures, against Heretic's merged model | 0 of 20,643,840 elements differ across the 8 matrices; KL 0.0877554 against 0.087755 |
| From each tool's own captures, or under torch 2.15.0 | the fitted changes differ by 8% to 230% per matrix; KL 0.0849 to 0.0925 against Heretic's 0.0878 |

- Heretic's own loop is just as far from its own export under the other torch version (10%
  to 197% per matrix), and moves by 5% to 185% at the full budget when its inputs change in
  the seventh digit. At a single iteration the trained model is steadier than the fixture
  (about 1e-5 in float32, 2e-3 to 7e-3 in float64); with `overcorrect_relative_weight = 0`
  the first iteration does not amplify at all in float32 (3e-7 to 5e-7), and the full budget
  still drifts by 4% to 53%.
- The objective reached is of the same size in both tools but not equal: for the 8 matrices
  it starts at 4.8e-4 to 2.8e-3 and ends between −6.5e-5 and −1.1e-3, with one exception.
  In one fit (layer 12 `down_proj`, the IDE under Heretic's libraries from its own captures)
  the optimiser stopped at the start (1.26609e-3 from 1.26616e-3) where Heretic's reached
  −2.2e-4: a difference of 7e-6 in the inputs decided it.
- `scale` (the row rescaling the loss is evaluated with, which is not stored) ranges from
  0.81 to 1.01 on this model: the matrix the loss judged and the matrix stored differ by up
  to 19% in row magnitude. That is how Heretic stores it, reproduced exactly.
- Heretic's scores for the exported trials: directional ablation 28 of 32 refusals at KL
  0.013 (4 trials are too few for it); ARA 0 of 32 at KL 0.088. The IDE's own edits score
  the same refusal counts (28 of 32 and 0 of 32): see "Comparing the scores with Heretic",
  which was run after the scores were built.
- Time: the IDE's whole ARA comparison (a measurement with `module_io` of 128 prompts, two
  fits of 8 matrices at rank 50, 64 logit passes, the tensor comparisons) took 55 s.

Not covered here: the search; a tokenizer with a beginning-of-text token; bfloat16
execution; GGUF. The refusal rate on the IDE side, a response prefix and a fit of many
matrices (44, on Qwen3-0.6B) are in "Comparing the scores with Heretic".

## Hybrid and mixture-of-experts layouts

The two components, `attn.o_proj` and `mlp.down_proj`, keep their names on every layout.
What changes is how many **modules** a component has in a layer. A module is one output
matrix with its own input: Heretic's unit, and the unit the IDE iterates in the ablation,
in ARA, in the capture, in the plan and in the record. The strength of a component (its
`weight_distributions` entry) is the same for all its modules; row norms (`pre`, `full`),
ARA fits, ARA's initial draws and captures are per module. The IDE's own 3-D `W_O` (one
view per head) stays one module, as before.

Which modules a component has comes from one table, `COMPONENT_SOURCES` in
`wsp/directions.py`, in Heretic's module order (`get_layer_modules` in Heretic's
`model.py`):

| Component | Source, in order | Module | Input and output captured at |
|---|---|---|---|
| `attn.o_proj` | `attn` | the softmax attention's output matrix (`self_attn.o_proj`) | `attn.hook_z`, `hook_attn_out` |
| | `mixer` | a token mixer's output projection (Qwen3.5's `linear_attn.out_proj`) | `mixer.hook_in`, `mixer.hook_out` |
| `mlp.down_proj` | `mlp` | the dense MLP's output matrix (`mlp.down_proj`) | `mlp.hook_post`, `hook_mlp_out` |
| | `experts` | each routed expert's down projection (`mlp.experts.<e>.down_proj`), one module per expert | not captured |
| | `shared` | the shared expert's down projection (`mlp.shared_expert.down_proj`) | `shared.hook_in`, `shared.hook_out` |

So on Qwen3.5 (hybrid: linear-attention layers and full-attention layers) every layer
has an `attn.o_proj`: `linear_attn.out_proj` in the linear layers, `self_attn.o_proj` in
the full ones. A mixture-of-experts layer's `mlp.down_proj` is its routed experts and its
shared expert. The structure (`wsp/arch.py`) recognises these modules from the tensor
names, `linear_attn.out_proj.weight`, `mlp.experts.<e>.{gate,up,down}_proj.weight` and
`mlp.shared_expert(s).{gate,up,down}_proj.weight`, and says so in its notes (for example
"3 layers use a token mixer without attention heads (linear attention); its output
projection is the layer's attn.o_proj for directions"). Expert indices must run from 0
without a gap, or that layer's experts are not recognised. The routers
(`mlp.gate.weight`, `mlp.shared_expert_gate.weight`) are not modules. For a multimodal
checkpoint the structure reads the text model's type (`text_config.model_type`).

On disk a MoE checkpoint keeps one tensor per routed expert; transformers fuses them at
load into one parameter per layer. The patch is made on the checkpoint's tensor (float32,
as always) and the runner adds it to that expert's slice of the fused parameter
(`wsp/runtime_layouts.py`); no other expert's slice changes.

**`expert_modules`** (ablation and ARA): `"all"` (the default) changes every module of
`mlp.down_proj`, routed experts included; `"shared"` leaves the routed experts out, so only
the shared expert (and a dense MLP, if a layer has one) is changed. The field is written
into the script and into the normalised parameters only when it is not the default, so
Heretic's parameters paste in unchanged. It is never one of the Heretic parameters in the
reproduction record; with `"shared"` the call's `heretic` entry carries a note, for the
ablation "expert_modules is an IDE field; Heretic at 64e9991 under transformers 5.x ablates
only the shared expert of such a model (it does not list the fused routed experts as
modules)", for ARA "expert_modules is an IDE field; Heretic at 64e9991 under transformers
5.x lists no routed experts of such a model, and its ARA raises while capturing a shared
expert's inputs (it reads them as [prompt, position, width]; they are [tokens, width]), so
it fits such a model only when it has no shared expert". That is what Heretic does there
(run, not only read: [Mixture-of-experts models in
Heretic](#mixture-of-experts-models-in-heretic)): under transformers 5.x the routed experts
are one fused module, not a list, so its module discovery skips them (against its own
intent). `"shared"` reproduces Heretic's ablation on such a model; `"all"` is what its code
means to do.

The plan has one row per layer, component and module (its `module` path); a layer's
routed experts are one row (`module` `mlp.experts.*.down_proj`, `modules` the number of
experts, `stored` summed), and the totals then carry `modules`. The ablation's report and
the record list every module, each expert on its own.

The weight-script functions take the field too: `ablate_direction(..., expert_modules="shared")`
and `ablate_ara(..., expert_modules="shared")`, so the script that `POST /api/directions/script`
writes commits through `Session.apply` like any other. On a dense model the script's output
is what it was. Where a call changes more modules than the plan has rows (routed experts),
its summary line gives both counts (`20 modules (8 components) in 4 layers`), and a
per-module line names the module path when its layer has more than one module of that
component.

**The LoRA adapter export on these layouts.** A linear-attention `out_proj` and a shared
expert's `down_proj` are ordinary linear modules and are exported as adapter targets. A
routed expert's tensor is refused: "transformers loads a layer's routed experts as one
fused parameter, so a LoRA adapter has no module to attach to; export the checkpoint
instead". So an ablation with `expert_modules: "all"` on a MoE model cannot be exported as
an adapter; one with `"shared"` can. For a multimodal checkpoint (its config has a
`text_config`) the export's notes add: "the adapter's module names are the checkpoint's
(model.language_model...): load it on the model class the checkpoint names, not on the
text-only model".

**What ARA refuses, and why.** ARA does not fit routed experts yet: an expert's inputs exist
only for the prompts routed to it at their last token, so its captures are sparse, and
there is no comparison with Heretic for them (it lists no routed experts under
transformers 5.x, so it fits none: [Mixture-of-experts models in
Heretic](#mixture-of-experts-models-in-heretic)). The
measurement does not capture them (`io.json` lists
them as one entry, not captured, "routed experts are not captured: ARA does not fit them
yet"), and a layer range that holds them with `expert_modules: "all"` is refused before any
fit. The shared expert is fitted: its activations are `[tokens, width]` (the MoE block
flattens prompt and positions before it), and its hook points show them as
`[1, tokens, width]`, one prompt at a time. Heretic's ARA cannot run there (it raises
while capturing the shared expert's inputs), so this fit has no Heretic counterpart; on a
mixture-of-experts model without a shared expert (Qwen3-MoE) Heretic's ARA fits the
attention modules only, as the IDE's does with `"shared"`. Initial draws are one per fitted module, in
fit order: layers ascending, `attn.o_proj` then `mlp.down_proj`, modules in table order.

Version 2 of `io.json` lists every module of a layer (`modules`: component, module path,
source, `d_in`, captured, why, `x`, `y`). A version 1 file is read as one module per
component (the first source's); for a model with one module per component its rows are
exactly those a version 2 measurement stores, so earlier measurements keep fitting to the
same pieces.

Refusals, word for word:

| When | Says |
|---|---|
| `expert_modules` is neither | `expert_modules: must be "all" or "shared"` |
| `"shared"` on a model without routed experts | `expert_modules: "shared" changes nothing: this model has no routed experts` |
| ARA over routed experts with `"all"` | `layer <l> mlp.down_proj: ARA cannot fit routed experts yet (an expert's inputs exist only for the prompts routed to it); pass expert_modules="shared" to fit the shared expert only` |
| a patch on a tensor the run-time model does not load reaches the runner | `checkpoint tensor <name> cannot be changed at run time: <name> belongs to the vision tower, which the text model the IDE runs does not load` (or the multi-token prediction head) |
| a workspace whose decoder layout is not recognised (for example a GGUF of architecture `qwen35moe`, available by raw tensor names only) | `directions need a recognised decoder layout; this model's tensors are available by raw name only (<reason>)`, from the plan, the ablation, ARA and the measurement, before any model is loaded |

Not covered: a model with routed experts in a GGUF file (3-D packed expert tensors are not
read); the ARA fit of routed experts; a comparison of the shared expert's ARA fit with
Heretic (Heretic's ARA raises there); a real mixture-of-experts checkpoint (the tests use
tiny random ones saved by transformers).

### Mixture-of-experts models in Heretic

What Heretic at `64e9991` (torch 2.14.1, transformers 5.18.0, peft 0.21.2) does on a
mixture-of-experts model, established by running it (2026-11-19), not only by reading it.
The numbers are in `docs/directions-g2b-heretic-validation.json`; the reproduction is
`scripts/heretic_moe_probe.py` (its `heretic` mode under Heretic's interpreter, `ide` under
the IDE's, `compare` under either).

**Models.** The test fixture `tiny_qwen35(dir, moe=True)` (`tests/layout_fixtures.py`) as
saved by the IDE's transformers 5.22.2; Heretic's 5.18.0 loads it unchanged (93 tensors,
`Qwen3_5MoeForConditionalGeneration`). Heretic needs three more things, so both tools ran a
copy with the same weights and: `preprocessor_config.json` and
`video_preprocessor_config.json` (Heretic loads `AutoProcessor` for a checkpoint with a
`vision_config`, only to save it); a character-level vocabulary written in
`Qwen3_5Tokenizer`'s own pipeline (the fixture's word-level vocabulary gives no tokens at
all for English prompts through `Qwen3_5Tokenizer`, in both transformers versions, and
5.18.0 rebuilds that class's pipeline around the file's vocabulary); text
`max_position_embeddings` 512 (character prompts exceed 64; default rope, so the weights do
not depend on it). Also two tiny models built with Heretic's transformers: Qwen2-MoE (a
shared expert) and Qwen3-MoE (none), 2 layers, 4 experts, top 2. Prompts: the tiny oracle
runs' datasets, 5 good and 5 bad.

**(i) The modules.** Under transformers 5.x a layer's routed experts are one module
(`Qwen3_5MoeExperts`, `Qwen2MoeExperts`, `Qwen3MoeExperts`: parameters `gate_up_proj`
`[E, 2I, d]` and `down_proj` `[E, d, I]`, no submodules). `get_layer_modules`' loop
`for expert in layer.mlp.experts` raises `TypeError: '...Experts' object is not iterable`,
which its `suppress(Exception)` swallows, so no routed expert is ever a module. Per layer
Heretic lists:

| Model | `attn.o_proj` | `mlp.down_proj` |
|---|---|---|
| Qwen3.5-MoE fixture | `linear_attn.out_proj` (layers 0-2), `self_attn.o_proj` (layer 3) | `mlp.shared_expert.down_proj` |
| Qwen2-MoE | `self_attn.o_proj` | `mlp.shared_expert.down_proj` |
| Qwen3-MoE | `self_attn.o_proj` | none (the component is absent) |

The IDE's modules (`DR.modules`) with `expert_modules: "shared"` are these, in this order,
on all three (`compare`: PASS). With `"all"` the IDE has each routed expert's
`mlp.experts.<e>.down_proj` before the shared expert (24 modules against Heretic's 8 on the
fixture, 12 against 4 on Qwen2-MoE, 10 against 2 on Qwen3-MoE): FAIL. That order is what
Heretic's source means (experts, then the shared expert); no Heretic run produces it. No
fix was made: `"shared"` is the IDE's setting that matches Heretic.

**(ii) The directional ablation.** Heretic exported as an adapter (`batch_size` 1, 2
trials, CPU), `row_normalization` `none` and `full`, against `scripts/heretic_compare.py`
(`--expert-modules` added; default `all`, as before):

| Run | IDE `expert_modules` | Result |
|---|---|---|
| float32 | `shared` | tokens 5/5 identical; means within `4.0e-7`; direction cosine `0.999996` at indices 1..n; 5 tensors changed by both, none by one only; change `1.6e-6` (none) and `1.8e-6` (full) from each tool's own means, `1.7e-7` and `7.5e-7` from Heretic's; `full` replica `8.2e-7`; KL `0.00129606` against `0.00129601` (none), `0.00108891` against `0.00108893` (full). Float32 rows assert nothing (reported) |
| float32 | `all` | the same 5 tensors and numbers, plus 8 tensors changed by the IDE only: the routed experts' `down_proj` of layers 2 and 3 (the trial's layers); KL `0.00149` against `0.00130` (none), `0.00116` against `0.00109` (full) |
| bfloat16, checkpoint stored in bfloat16 | `shared` | every bound PASS: B-S1 identical, C4-1 equal sets (8 and 8), B-S4 from Heretic's means `1.7e-7` (bound `1e-6`, none) and `8.8e-7` (bound `4e-4`, full), rounded sums all 3584 elements equal, replica input `1.5e-7` (bound `1e-6`), C4-4 refusals 0 and 0; replica of the randomized approximation `1.0e-6`, reported only (IDE torch 2.15.0, run 2.14.1); KL equal to 6 significant digits (`0.00136225`, `0.00113781`) |
| bfloat16, checkpoint stored in bfloat16 | `all` | FAIL: C4-1 (24 IDE modules against 8, 8 tensors changed by the IDE only) and B-S4 (the 8 one-sided tensors); the tensors both change are as in the row above |

A side finding, not specific to these layouts: with the fixture stored in float32 and both
tools run in bfloat16, B-S4 fails on every changed tensor, linear attention included (`2.9e-3`
none, `2.1e-3` full, against `1e-6` and `4e-4`), while the same comparison on a copy stored
in bfloat16 passes at `1.7e-7`. Heretic loads the float32 weights rounded to bfloat16 and
computes its change from them; the IDE's change is computed from the float32 checkpoint.
The comparisons above were made on bfloat16 checkpoints, as before; that difference is not
handled.

**(iii) ARA.** On the Qwen3.5-MoE fixture Heretic's ARA stops in `ARA.init`, in
`get_module_io_batched` on the good prompts, during the first prompt's forward pass:
`IndexError: too many indices for tensor of dimension 2`, raised by its forward hook on
layer 0's `mlp.shared_expert.down_proj` at `inputs[0][:, -1, :]`. The same happens in
bfloat16 and with `batch_size` 5 (Heretic runs), and on Qwen2-MoE (the probe's capture). Its comment says `inputs[0]` and the
output have shape (prompt, position, component); for the shared expert they are
(prompt x position, component): the MoE block flattens the batch before the shared expert
(`hidden_states.view(-1, hidden_dim)`), so the hook receives `[79, 8]` and `[79, 32]` for
one prompt of 79 tokens and `[555, 8]` for 5 prompts padded to 111, while the attention
modules' hooks receive `[B, S, 32]`. Nothing is captured and no module is fitted.

What the remaining questions come to:

- A routed expert's forward hook: none exists. The fused module is called with
  `(hidden_states [B x S, d], top_k_index [B x S, k], top_k_weights [B x S, k])`; each
  expert is a slice of its parameters inside that call, not a module, so nothing of
  Heretic's sees an expert's input or output.
- `inputs[0][:, -1, :]` selects the last position of each prompt for the `[B, S, d]`
  inputs of attention and linear-attention modules; on the shared expert's 2-D input it
  raises.
- Which experts see each prompt's last token (the router's top 2 at row `b * S + S - 1`,
  Heretic padding on the left): on the fixture, layer 0 picks experts 2 and 1 for all 5
  good prompts and layer 3 experts 0 and 3; in layers 1 and 2 the pairs vary by prompt
  (layer 2: `[3,0]`, `[0,3]`, `[1,2]`, `[1,3]`, `[0,1]`), so at the last token each expert
  misses some prompts. Heretic captures none of this.
- `get_module_io_batched` (read, not run: no module is ever absent on these layouts, since
  every listed module runs for every prompt): it takes every module index present in any
  batch and concatenates its rows from the batches where it is present, so a module missing
  from some batches would get fewer rows than prompts, with nothing recording which; a
  module index present in no batch is absent, and `ARA.modify_model` would then raise
  `KeyError` on it.
- The rows ARA fits each routed expert with: none; on Qwen3-MoE (no shared expert) the
  capture succeeds (`[5, 32]` per attention module, both layers) and Heretic's ARA
  completes, fitting and exporting only `self_attn.o_proj` of layers 0 and 1.

On that Qwen3-MoE run the IDE with `expert_modules: "shared"` fits the same two modules
(`scripts/heretic_compare_ara.py`, `--expert-modules` added): float32, tokens 5/5, module
inputs and outputs within `3.2e-7`, consistency `1.3e-7`; bfloat16 (checkpoint stored in
bfloat16) B-S1, B-S6 (`1.6e-3` to `1.8e-3`, bound `(0, 5e-3]`) and C4-3 (`0.054`, bound
`0.10`) PASS; and from Heretic's captures under Heretic's torch on the CPU
(`scripts/heretic_fit_check.py`) the fitted factors and `B A` are bit-equal to Heretic's,
both modules, in float32 (reported) and bfloat16 (B-S5 PASS). With `"all"` the IDE refuses ARA there,
as before. The KL of the IDE's own fit is `0.00857` against Heretic's `0.00740` (float32)
and `0.00863` against `0.00709` (bfloat16), reported only: the fit is not bit-reproducible
across torch versions (the nudged-input spread of the changed module is `0.24` to `0.44`).

**Consequence.** Heretic's ARA on routed experts is not defined on any layout the IDE
recognises: under transformers 5.x it lists no routed expert, and where a shared expert
exists it raises before fitting anything. So the IDE's ARA over routed experts stays
refused (its message is unchanged), there is no Heretic reference for the IDE's fit of a
shared expert, and the record's `expert_modules` note says so per method.

### On real hybrid models

Two real hybrid models have been run by the IDE. `Qwen3.5-0.8B-Q4_K_M.gguf` through the Metal
bridge, without an oracle: the measurement, the IDE's own search, ARA over all 48 output
matrices (the linear-attention ones included) and the patch invariants; see
[Checked on real GGUF files](#checked-on-real-gguf-files). And `Qwen/Qwen3.5-4B` through
`transformers` in bfloat16, compared with Heretic, below.

#### Compared with Heretic on Qwen3.5-4B

Run on 2026-11-19. The model has 32 layers (8 with full attention, 24 with linear
attention), width 2560, 4,659,865,088 parameters and a vocabulary of 248,320. Both tools ran
it on the Apple GPU (`mps`) in bfloat16, one prompt at a time, with the planned counts
(64 + 64 measured prompts, 32 per score, 30 new tokens, 4 trials). Every process ran under
`python -m wsp.memory_guard` with its default limits, in isolated roots, offline. Heretic at
`64e9991`. Numbers: `docs/directions-c4-heretic-validation.json`.

**The device and the counts.** Heretic's timing probe on the CPU took 29 minutes (about
11.7 s per measured prompt and 10 s per generated token; the cause of that slowness was not
established), and the rule written before the probe sent the comparison to the GPU. There
the same probe took 65 seconds and the IDE's took 14 (0.35 s per measured prompt, 0.038 s
per generated token). By the same formula, with the IDE's times, the planned counts cost
about 17 minutes on the GPU, so no count was halved.

**Three settings both tools needed** to run this model on the GPU under the guard:

- `HF_DEACTIVATE_ASYNC_LOAD=1`. `transformers` loads weights in four threads, and Heretic's
  process (torch 2.14.1) ended with a segmentation fault in torch's `mps` copy kernel during
  the one such load attempted. No sequential load did.
- The load paced to 1.5 GiB a second (`scripts/load_pacing/sitecustomize.py` on
  `PYTHONPATH`, active only when `LOAD_GIB_PER_S` is set). The guard ends a run whose
  process tree grows by more than 4 GiB between two samples a second apart; this model's
  weights, about 9 GiB, reach the GPU in about two seconds, and the guard ended the first two
  attempts. The guard was not changed. The pacing pauses after the libraries' own copies and
  changes no value (`scripts/load_pacing_check.py`, model-free). The IDE started without it
  on a GPU would be ended by the guard in the same way.
- A second Heretic environment for its default ablation. Under the torch that Heretic's
  environment resolves to, 2.14.1, `torch.svd_lowrank` on `mps` takes 12.5 s a call at the
  0.5B model's shapes (0.002 s on the CPU; 0.01 s under torch 2.15.0) and did not return
  within 170 s at this model's (`scripts/mps_svd_timing.py`, model-free), and
  `row_normalization` `full` makes that call once per module per trial. So the ablation was
  run twice by Heretic: with `row_normalization` `none` in its own environment, and with
  `full` by the same source and packages under torch 2.15.0 (Heretic's `pyproject.toml`
  leaves the torch version open). ARA ran in Heretic's own environment.

| Stage | Ablation `none`, Heretic's torch 2.14.1 | Ablation `full`, Heretic under torch 2.15.0 | ARA, Heretic's torch 2.14.1 |
|---|---|---|---|
| token ids (64 + 64 measured, 32 scored) | identical | identical | identical |
| mean residuals, worst relative difference | `5.8e-3` | `4.3e-6` | `4.5e-3` and `5.8e-3` |
| directions, smallest cosine | `0.99992` | `1 - 8e-12` | not applicable |
| modules, by Heretic's names | 64 in both: 8 `self_attn.o_proj`, 24 `linear_attn.out_proj`, 32 `mlp.down_proj` | the same | 64 captured by both |
| tensors changed | 29 in both, the other 709 in neither | 43 in both | 22 fitted in both |
| change from Heretic's means, worst tensor | **`1.15e-6` against `1e-6`: failed** | `6.9e-5` against `4e-4` | not applicable |
| rounded sums | 5,300 of 304,087,040 values differ, none outside the rule | 10,962 of 713,031,680, none outside | not applicable |
| replica of Heretic's randomized rank 3, on `mps` | not applicable | `3.9e-6` from Heretic's product against `1e-5`; **its premise `1.03e-6` against `1e-6`: failed** | not applicable |
| module inputs and outputs, worst relative difference | not applicable | not applicable | `4.9e-2` and `5.3e-2` (reported; expected of the order of `3.9e-3`) |
| consistency of the fit | not applicable | not applicable | `0.0016` to `0.0017`, against `0.005` |
| objective reduction against Heretic's | not applicable | not applicable | median `0.1%`, worst `1.0%`, against `10%` |
| the fit from Heretic's captures, on `mps` | not applicable | not applicable | 22 of 22 bit-equal, under both torch versions |
| refusals of the IDE's own edit, of 32 | 32, Heretic 32 | 32, Heretic 32 | 0, Heretic 0 |
| KL of the IDE's own edit / Heretic's trial | `0.00179` / `0.00207` | `0.00173` / `0.00179` | `0.0992` / `0.1212` |
| base responses identical, of 32 | 25 | 32 | 25 |
| Heretic's merged model: responses identical, of 32 | 27 (refusals), 26 (length) | 32 and 32 | 22 and 22 |
| Heretic's merged model: refusals, KL in the IDE / in Heretic | 32 in both; `0.00178` / `0.00222` | 32 in both; `0.0017687` / `0.0017687` | 0 in both; `0.1159` / `0.1189` |
| each trial as a candidate | KL inside the bound and refusals equal, 4 of 4 | refusals equal 4 of 4; **KL outside the bound on 4 of 4: failed** | not applicable |

Read with these limits. No trial of either ablation run lowered the refusal count (32 of 32
in the base and in all eight trials: four trials are not a search), so for the directional
ablation the table shows two tools agreeing on an edit that does not change the count; ARA's
exported trial takes it from 32 to 0 in both. Against Heretic under the IDE's own torch
version the measurement agrees to `4e-6` and every response is identical; against Heretic
under torch 2.14.1 a quarter of the greedy responses diverge at some token, as bfloat16
near-ties do, and the scores stay inside their bounds.

**The failed bounds.** None was moved.

- `none`, the change from Heretic's means: `1.15e-6` against `1e-6`. Heretic computes
  `v @ W` on the module's device and the IDE on the CPU; everything else in a rank-1 change
  is exact. On this model's matrices that one product differs between `mps` and the CPU by
  `8.6e-7` to `9.3e-7` with random unit vectors (model-free, float32, torch 2.14.1). The
  bound came from runs of both tools on the CPU at width 896.
- `full`, the replica's premise: the exact rank-3 truncation of the full change against the
  IDE's own delta is `1.03e-6` on 9 of 43 tensors, against `1e-6`. The stage is failed by
  its premise although the replica itself is inside its bound. The replica run on the CPU is
  `7.1e-5` from Heretic's product (reported, not asserted: `svd_lowrank` draws from the
  device's random stream).
- `full`, each trial as a candidate: the IDE's KL is 2.0%, 2.4%, 3.7% and 14% from
  Heretic's (`6.6e-5` to `3.6e-4` in absolute terms, on KLs of `0.0017` to `0.0056`),
  against 1%. That bound applies when both tools run the same torch version, which no
  bfloat16 run had done before. Heretic's own two numbers for its exported trial are 1.4%
  apart (`0.0017930` with the adapter applied during the search, `0.0017687` on its merged
  model, which the IDE's evaluation reproduces to seven digits), so in bfloat16 the way a
  change is applied moves a KL of this size by more than the bound allows.
- The first fit check, made on the CPU, found 0 of 22 fitted matrices equal to Heretic's
  (relative differences `5.8e-2` to `2.1`, median `0.29`): Heretic fits on the model's
  device. The check now takes `device=` and asserts bit equality only on the run's device,
  where it holds for 22 of 22.

**Defects these runs found in the IDE, fixed here.**

- The scores refused every response that ended with the end-of-text token before the length
  limit. This checkpoint keeps `eos_token_id` only under `text_config` and has no
  `generation_config.json`; the IDE read the top level only. `transformers` takes
  `generate`'s default from the text config, so both are read now.
- The drift measurement of every commit held full-vocabulary logits for the base and the
  edited model, both log-softmaxes and the element-wise KL at once: about 5 GiB with this
  vocabulary, and the guard ended the comparison at its 16 GiB budget. The arithmetic is now
  done one sequence at a time with the same numbers, and the runner returns its device
  cache to the system afterwards.
- The comparison's bfloat16 rounding check built about 3 GiB of temporaries on a
  `2560 x 9216` matrix; it now works in row chunks with identical counts (the counts of the
  run before and after are the same).

The guard ended ten processes on the way: two for growth during the load, seven at its
budget (four of those were repeats with memory marks, made to find the cause) and one for
memory compression while another workload on the machine was starting. No limit was
changed, and the run's record lists them.

**Time and memory.** Heretic: 349 s (`none`), 314 s (`full`), 508 s (ARA) for four trials
each, and 305 to 374 s for each dump. The IDE: 19 s to measure 128 prompts with module
capture, 63 s to fit 22 matrices on the CPU (11 of `4096 x 2560`, 11 of `9216 x 2560`),
32 s to score 32 prompts of 30 tokens. Largest process trees: Heretic 11.1 GiB, the IDE
14.5 GiB (the model on the GPU is about 8 GiB of it), against a budget of 16. The least
memory available was 39.2 GiB during the runs of the record and 33.7 GiB during one the
guard ended, above the guard's 20 GiB floor throughout.

**Not covered on this model.** A search of the IDE's own; `row_normalization` `pre`;
Heretic's `full` under the torch its environment resolves to; float16; the model worker
(every run was in process); the page in a browser; a mixture-of-experts model. The IDE's
ARA fit runs on the CPU while Heretic's runs on the GPU, so the two tools' own fits differ
as fits from different captures do.

#### Heretic on the Apple GPU, first on the small model

Before the 4B runs both tools were tried on `mps` with Qwen2.5-0.5B in bfloat16. The IDE's
self-check (measurement with module capture, ARA, three scorings) ran with the refusal
counts of its CPU run (31, 28 and 0 of 32; KL `0.0151` and `0.0889` against `0.0131` and
`0.0990` on the CPU), about three times faster. Heretic's ARA ran in 159 s (first trial
0 of 32 at KL `0.098`). Heretic's ablation with `row_normalization` `full` scored the base
and then made no visible progress in its first trial for eight minutes and was terminated;
the SVD timing above explains it (48 calls of 12.5 s a trial), so it was slow, not stuck.

## Costs and limits

- Measurement: one forward pass per prompt (batch 1); with 1024 prompts per set, 2048
  passes. Nothing else runs.
- Ablation: no forward pass. `none` reads each targeted matrix once (one product); `pre`
  twice (norms, then the product); `full` twice (`G`, then the product) plus one
  `d × d` eigendecomposition per component, `O(d³)` (seconds for `d` in the thousands).
- The stored patch is `(d_in + d)` values per view for `none` / `pre`, `r` times that for
  `full`: a fraction of the matrices it changes.
- A direction that lies along the good mean up to rounding (not exactly) is not refused as
  unusable: the zero test is exact, as specified, so such a direction is rounding noise. The
  `diff_norm` and `cosine` statistics show the case.

## The fitted low-rank ablation (ARA)

A second way to remove a measured behaviour, beside `ablate_direction`. Instead of removing
one direction, it **fits a low-rank change to each output matrix** so that the matrix's
outputs on the good prompts stay where they were while its outputs on the bad prompts move
towards the good prompts' outputs. The method is "Arbitrary-Rank Ablation" (Weidmann 2026,
the default method of Heretic; no paper is published yet). It was written here from a
description of the method, with none of that tool's code. Its parameter names are that
tool's on purpose, so its parameters can be pasted in unchanged. After the comparison with
Heretic the loss was reordered to evaluate in Heretic's order, so that the two fits are
identical bit for bit on the same inputs ("ARA against Heretic" above).

Code: `wsp/directions_ara.py` (parameters, plan, script line, the loss and the fit), the
capture in `wsp/directions.py` (`io_layout`, the `module_io` job and files), the script
function `ablate_ara` in `wsp/wscript.py`.

Its effect on refusals, measured by both tools on two trained models: 0 of 32 for the
exported trials, from 30 and from 3 of 32 ("Comparing the scores with Heretic"). On the
second of them fits that reach the same objective differ by a factor of two in KL. It is
fitted on the CPU in float32 only.

### Capture (`module_io`)

`POST /api/directions/measure` (and `measure`) with `"module_io": true` also keeps, for
every prompt, every layer and each of the two components, the last-token **input and output
of the output matrix**, from the same forward pass that gives the residual vectors:

| component | input `x` | output |
|---|---|---|
| `attn.o_proj` | `blocks.L.attn.hook_z`, trailing dims flattened to `[D_in]` (heads in order) | `blocks.L.hook_attn_out` |
| `mlp.down_proj` | `blocks.L.mlp.hook_post` | `blocks.L.hook_mlp_out` |

The component computes `out = x @ V + bias` (`V` the component's writer views, `[D_in, d]`,
rows concatenated; `bias` its `b_O` / `b_out`, or 0). A component whose hook point the
runner does not offer (for example linear attention) is not captured, with the reason.

**The job.** The worker job `direction_residuals` takes `module_io` and then answers one
float32 tensor `[prompts, (n + 1)·d + width]`: per prompt the residual vectors flattened
(index-major), then layer by layer, per captured component in the order `attn.o_proj`,
`mlp.down_proj`, its input then its output. Both sides derive this layout
(`io_layout(ws, runner.available)`) from the structure and the runner's hook points; the
server checks the width. A worker frame's payload length is a 32-bit field (under 4 GiB);
a `module_io` job is kept at or below `IO_JOB_BYTES` = 1 GiB, so the prompts per job are
`min(8, floor(1 GiB / (4 · ((n + 1) d + width))))` (at least 1; a single prompt above 1 GiB is
refused), and the measurement sends `min(4, that)` per chunk. For a 70B-class model
(80 layers, `d` 8192, `d_mlp` 28672) a row is about 20 MB, so chunks stay at 4.

**Storage**, `<workspace>/directions/<name>.io/`:

- `good.L<l>.npy`, `bad.L<l>.npy`: float32 `[N, row]`, the layer's parts concatenated; each
  file is preallocated at its full size and filled row by row as chunks arrive (the server
  never holds more than one chunk). A layer with nothing captured has no files.
- `io.json`, written **last** (its presence means the data are complete): the layout (per
  layer `row`, `files`, and per component `d_in`, `captured`, `why`, offsets `x` and `y`),
  `counts`, `bytes`, `dtype`.

The files are written in a temporary sibling directory and moved in place with the
measurement's own files; a stop between chunks or a failure removes it, so nothing is
left. `replace` and delete remove `<name>.io/`; `GET /api/directions/{name}` shows
`module_io`.

**Disk cost**: `bytes = 4 · (N_good + N_bad) · Σ_l row_l`, with
`row_l = Σ over the layer's components (D_in + d)` (plus a .npy header of about 128 bytes
per file). For the tiny test model (2 layers, `d` 24, `d_mlp` 40, 24 attention inputs):
`row = 112`, and 11 prompts take 9,856 bytes. **Disk guard**, before any model is loaded:
these bytes (every component the structure has; a hook found missing later only makes it
smaller) are refused when they exceed the free space of the workspace's file system minus
2 GiB, naming both numbers.

### The fit, for one component

Given `V` (the current weights including the script's own earlier changes), `bias`, and the
captured `X_g [N_g, D_in]`, `Y_g [N_g, d]` (good), `X_b`, `Y_b` (bad), all float32.

**Unknowns**, in the adapter convention: `A [r, D_in]`, `B [d, r]`; the change is
`delta_W = B @ A` in the `[d, D_in]` layout, i.e. `delta_V = Aᵀ @ Bᵀ` in view layout.

**Initial values**: `B = 0`; `A = kaiming_uniform_(empty(r, D_in), a=sqrt(5), generator=g)`
(uniform in `±1/sqrt(D_in)`), with `g = torch.Generator().manual_seed(seed)` created once per
script call and consumed in fit order: layers ascending from `start_layer_index`, within a
layer `attn.o_proj` then `mlp.down_proj`, one draw per fitted component (a skipped one draws
nothing).

**Outputs with the change** (`P = X @ V` computed once):

```
out_new(X) = (P + (X @ Aᵀ) @ Bᵀ) * scale + bias
scale[j]   = nrm[j] / ||(V + Aᵀ Bᵀ)[:, j]||   if preserve_row_magnitudes, else 1   (1 where nrm[j] = 0)
nrm[j]     = ||V[:, j]||   (over the whole component, every head together)
```

`scale` is recomputed at every evaluation and gradients flow through it.

**Order of evaluation.** The fit holds the matrix transposed, `W = Vᵀ` `[d, D_in]` (the
adapter layout), and evaluates in this order: `P = X @ Wᵀ` once; at each evaluation the two
outputs `P + (X @ Aᵀ) @ Bᵀ` first, then `scale` from the row norms of `W + B @ A`, then the
multiplication, then the bias. This is the same function as written above, but the order
decides how it rounds, and the fit amplifies its last digit; this order is Heretic's.

**Loss**:

```
knn(a, b, k)[i] = mean of the k smallest Euclidean distances from a[i] to the rows of b
                  (torch.cdist(a, b), then topk(k, largest=False))
preserve = mean over all elements of (out_new(X_g) - Y_g)²
steer    = mean_i knn(out_new(X_b), Y_g, k)[i] - overcorrect_relative_weight · mean_i knn(out_new(X_b), Y_b, k)[i]
loss     = preserve_good_behavior_weight · preserve + steer_bad_behavior_weight · steer
```

`Y_g` and `Y_b` are the captured outputs, never recomputed; `k = neighbor_count`.

**Optimiser**: `torch.optim.LBFGS([A, B], lr=learning_rate, max_iter=max_iter,
history_size=history_size, line_search_fn="strong_wolfe")`, `n_optimization_steps` calls of
`step(closure)` (the closure zeroes the gradients, evaluates the loss, calls `backward()`).
Float32 on the CPU. A non-finite loss or result is an error naming the layer and component.

**Result**: `delta_V = Aᵀ @ Bᵀ` exactly as fitted, one low-rank piece per view (the rows of
`Aᵀ` split per view, `Bᵀ` with each): `r · (D_in + views · d)` values per component (one
view for `mlp.down_proj` and for a checkpoint's 2-D `o_proj`; one per head for this
project's 3-D `W_O`). **The stored change is not rescaled by `scale`**, although the loss is
computed with it: that is how the tool this method comes from stores it. The printed
`scale` range shows how far the fitted (rescaled) matrix and the stored one differ.

All components of the layer range are fitted, in ascending layer order; a later layer's fit
uses its captured inputs as they were measured (not re-run after earlier layers changed), as
the method is defined. Biases and everything else are never changed. With
`steer_bad_behavior_weight = 0` there is nothing to fit (the loss would only refit rounding
noise) and nothing is written.

### The consistency check

Before fitting, per component: `mismatch = ||P_g + bias - Y_g||_F / ||Y_g||_F` and the same
for the bad prompts. Both are printed. Above `0.05` the call is refused, naming the layer,
component and number: `layer 1 attn.o_proj: mismatch 0.5 on the good prompts (above 0.05):
the saved inputs and outputs do not match this matrix: measure again on these weights, or
switch other patches off`. This is what would reveal a wrong hook mapping on a real model;
`0.05` leaves room for a model that runs in bfloat16. Measure with `variant: "current"` when
other patches stay on.

### Parameters and the script function

```python
ablate_ara("name", start_layer_index=0, end_layer_index=None,          # end exclusive; None = n
           preserve_good_behavior_weight=1.0, steer_bad_behavior_weight=0.01,
           overcorrect_relative_weight=0.0, neighbor_count=1,
           preserve_row_magnitudes=True, lora_rank=50,
           n_optimization_steps=5, learning_rate=1.0, max_iter=20, history_size=10, seed=0)
```

JSON for `POST /api/directions/script`: `{"method": "ara", "direction": "<name>", ...the
keyword arguments...}`; an unknown field is refused, and each error names its field.

| Field | Values |
|---|---|
| `start_layer_index`, `end_layer_index` | integers, `0 <= start < end <= n`; `end` `null` means `n` |
| `preserve_good_behavior_weight`, `steer_bad_behavior_weight` | finite, `0..1000` |
| `overcorrect_relative_weight` | finite, `0..10` |
| `neighbor_count` | integer `1..min(N_good, N_bad, 64)` |
| `preserve_row_magnitudes` | `true` / `false` |
| `lora_rank` | integer `1..256`, and at most `min(D_in, d)` of every fitted component (the first smaller one is named) |
| `n_optimization_steps` | `1..50` |
| `max_iter`, `history_size` | `1..100` |
| `learning_rate` | `0 < lr <= 10` |
| `seed` | integer `0..2**63 - 1` |

The measurement must match the model and must have complete module inputs and outputs
(otherwise: `measure it again with module_io`).

`plan` (no weight read) gives per layer and component `views`, `d_in`, `stored`, `skipped`
and why, and `scratch`, the bytes the fit asks the edit admission for; `script_for` gives the
one line. The function prints one summary line and one line per component, and reports
progress per component:

```
ablate_ara('m'): 4 components in 2 layers, 1,088 values stored, lora_rank 4, 0 skipped
  L0 attn.o_proj: loss 3.26111 -> 0.803398, scale 0.373228..1.04868, mismatch good 8.84e-08 bad 1.07e-07
  L0 mlp.down_proj: loss 2.98973 -> 0.329271, scale 0.676424..1.01489, mismatch good 8.97e-08 bad 0
  ...
```

### Costs

- **Memory**, per component, admitted (`edit_preflight`) before it starts, one component at
  a time: the matrix whole in float32 (`4 · D_in · d` bytes, read in row slabs of at most
  `STREAM_ELEMS`; a quantized base through its logical read path, never dequantized whole;
  it exists twice while it is copied into its transpose, once after that),
  and with `preserve_row_magnitudes` four more matrices of that size during an evaluation
  (the changed matrix, the product, their gradients); the layer's captured rows; `P`, the
  outputs and their gradients; the distance matrices; the optimiser's history
  (`2 · history_size` pairs of `r · (D_in + d)` values). `plan` reports the number
  (`scratch_bytes` in the code). For an MLP output matrix with `d = 8192`, `d_mlp = 28672`
  that is about 0.88 GiB for the matrix alone (1.75 GiB while it is transposed) and about
  4.4 GiB with `preserve_row_magnitudes`.
- **Time**: each evaluation is two products with `[N, D_in]`, one `[d, D_in]` product
  `B A` and row norms (with `preserve_row_magnitudes`), and `cdist`s of `N_b × N_g` and
  `N_b × N_b`. Each of the `n_optimization_steps` L-BFGS steps runs up to `max_iter`
  iterations and stops once it has made torch's default `max_eval = floor(1.25 · max_iter)`
  evaluations (checked after each iteration, so a strong-Wolfe line search can run a few
  past it): about `n_optimization_steps · 1.25 · max_iter` evaluations per component at
  most, 125 with the defaults, plus one before and one after for the printed losses.
- **Stored**: `r · (D_in + views · d)` values per component.

### Known limits of the fit

- With `overcorrect_relative_weight > 0`, at the start `out_new(X_b)` equals the captured
  `Y_b` up to rounding, so the nearest distance in the overcorrect term is about 0 and the
  direction of its gradient is rounding noise; float32 and float64 runs then differ widely.
  This follows from the definition; it is not corrected here.
- Long fits are sensitive to rounding: in the tests, float32 and float64 runs of the same
  formulation drift apart by 1e-2..1e-1 (relative) after about 60 evaluations. Two runs on
  one machine are identical.
- Measured on a real checkpoint with random weights ("ARA against Heretic"): a change of
  the captured inputs and outputs in their seventh digit moves the fitted change by 0.4% to
  14% after one iteration and by up to 37% at the full budget; another torch version moves
  it by 0.3% to 33%. So a fit is reproducible on one machine with one set of library
  versions, and not across them. The objective reached is much steadier than the weights
  (equal to 4 digits or more in three of the four fits compared).

## Runs in bfloat16

A `transformers` checkpoint runs in the dtype `WSP_RUN_DTYPE` names (default `float32`). In
`bfloat16` or `float16` the runner applies patches the way Heretic's trials see them: through a
LoRA adapter attached to the model, which PEFT runs in float32.

**The side path.** For the weight of an `nn.Linear` module (the whole parameter, held by that
module alone), the weight is not changed. A forward hook on the module adds the patch to its
output in PEFT's order:

```
out = (base_out.float() + x.float() @ delta.T).to(dtype)
```

`base_out` is the module's own output (its bfloat16 product, bias included); the change is
computed in float32 from the patch pieces without a dense delta (`wsp/overlay.py`'s
`DeltaOverlay`, the code that adds a GGUF model's edits to its packed products), and the sum is
rounded once to the run dtype. Committed pieces are added before a candidate's. The hooks are
registered when the variant is selected, before a run registers its capture hooks, so a
captured output includes the patch; the base variant has none. An overlay is kept while the same
pieces at the same strengths are active, and a candidate's is dropped once a call is made
without it (as on a GGUF model).

**Rounded once.** Every other tensor (norms, biases, embeddings, a slice of a fused expert
parameter, a `Conv1D` weight, a weight the embedding and the output head share) is changed in
place, each piece with one rounding: `W = (W.float() + scale * delta).to(dtype)`. The earlier
code rounded twice (the delta to bfloat16, then the sum). In float32 the two are the same
arithmetic, so a float32 run gives bit for bit what it gave before; float32 and float64 runs
apply every patch in place.

Why not in place in bfloat16: a change smaller than half a bfloat16 step of its weight (about
`2^-9` of it) vanishes when it is added to a bfloat16 weight, so a small fitted change (an ARA
fit of a few `1e-4` of the weight) would mostly not be scored at all. The side path keeps it and
changes the output wherever it reaches half the output's own step. Dense float32 copies of the
touched matrices would do the same at the cost of four bytes per value of every touched matrix.

**`WSP_PATCH_APPLY`.** `in_place` or `side` overrides the dtype's default (`in_place` for
`float32` and `float64`, `side` for `bfloat16` and `float16`), for comparisons. Any other value
is refused when the runner is created, before the model is loaded: `WSP_PATCH_APPLY must be
in_place or side`. A measurement's runner description and the KL reference key carry
`"apply": <mode>` only when the mode is not the dtype's default, so every existing float32
record and key is unchanged; the search refuses to continue on a runner whose mode differs from
the one it scored on, as for a different dtype. The scores answer's `generation` block of a
`bfloat16` run names `"patches": "float32 side path on linear layers (PEFT's order); other
tensors rounded once to bfloat16"`.

**The attention implementation.** In `bfloat16` and `float16` the runner loads the model with
the library's default attention kernel (it passes no `attn_implementation`; on the tiny test
models and Qwen2.5 that is `sdpa`), as Heretic loads it. In `float32` and `float64` it loads
eager attention first, as before, so a float32 run is bit for bit what it was; there the two
kernels agree to about `1e-6`. The runner's `attention` attribute names the kernel of its
ordinary runs. A run that reads attention patterns (`blocks.L.attn.hook_pattern`) on a model
that is not on eager switches it to eager for that forward pass (`set_attn_implementation`)
and puts back exactly the settings it had afterwards, also after an error; if the installed
library cannot switch, the pattern run is refused: `attention patterns need the eager attention
kernel, which this library version cannot switch to on a model loaded with its default
attention; run in float32 for patterns`. So in bfloat16 a pattern run's activations come from
the eager kernel and differ from the same run without patterns at the level of the table below;
it is not refused. The kernel follows from the dtype, so run descriptions and the KL reference
key do not name it; the Heretic file of a reproduction record lists eager attention as a
difference only for a measurement run in float32 or float64 ([The mapping to Heretic's
reproduce.json](#the-mapping-to-heretics-reproducejson)).

Why: in bfloat16 eager attention is about three times less accurate than the default on this
kind of model. Each configuration against float32 with eager attention (a probe with
transformers alone on Qwen2.5-0.5B-Instruct, 32 prompts, under both library versions):

| Configuration | worst relative difference of the mean residual | at index 1 | median | largest logit difference |
|---|---|---|---|---|
| bfloat16, eager attention (torch 2.15.0 / transformers 5.22.2) | 6.37e-2 | 6.37e-2 | 1.83e-2 | 1.11 |
| bfloat16, default attention (sdpa), same versions | 2.28e-2 | 1.55e-2 | 7.96e-3 | 0.40 |
| bfloat16, eager (torch 2.14.1 / transformers 5.18.0) | 6.37e-2 | 6.37e-2 | 1.95e-2 | 2.58 |
| bfloat16, default (sdpa), same versions | 2.28e-2 | 1.55e-2 | 7.92e-3 | 0.40 |
| float32, default (sdpa) | 8.9e-7 | 6.6e-7 | 5.5e-7 | 0.000 |

**The memory forecast.** For a `transformers` workspace the guard's forecast counts the
tensors the runner loads, at the run dtype's element size: the checkpoint tensors less a vision
tower and a multi-token prediction head the text model does not load (`wsp/runtime_layouts.py`),
times 2 bytes in `bfloat16`. The runner's own memory check counts the same tensors. A
workspace of the IDE's own format and a GGUF base are forecast as before.

**What a bfloat16 run means, stage by stage.**

| Stage | float32 run | bfloat16 run | Heretic in bfloat16 |
|---|---|---|---|
| residual means | float32 activations (eager attention) | bfloat16 activations of the library's default attention kernel, upcast exactly, summed in float64 | the same (the same default kernel) |
| final-normed index `n` | norm in float32 | the norm module run in bfloat16 on the bfloat16 value | the model's own normed output in bfloat16 |
| module inputs and outputs | float32 | bfloat16 values upcast; `Y` carries the bfloat16 rounding of a float32-accumulated product | the same |
| `V` for the ablation and the fit | the checkpoint's values upcast | identical (read from the store, not the runner) | `W.to(float32)`: identical |
| consistency check `‖X V + b − Y‖ / ‖Y‖` | about `1e-7` | at most about `u = 2^-8` (one bfloat16 rounding of `Y`), limit 0.05 | none |
| a candidate's effect on scores | in place, float32 | the float32 side path on linear layers | adapter attached: the same order |
| a committed patch on export | the export writes the checkpoint dtype | as before: the export rounds to bfloat16 | merged: one rounding |

The measurement and the fit are unchanged: captures are float32 copies of the run's values,
`V` is read from the store in float32, and the fit runs in float32 on the CPU.

**Bounds for a comparison with Heretic in bfloat16.** Written before any run, with their basis;
a stage outside its bound is a finding to investigate, not a threshold to move. `eps` is the
largest absolute difference of the base first-token logits between the two runs compared,
measured in the same run; `u = 2^-8`.

| Compared | Bound | Basis |
|---|---|---|
| token ids: IDE bfloat16, Heretic bfloat16, IDE float32 | identical | the tokenizer and template do not depend on the dtype |
| mean residuals, IDE vs Heretic, each index | reported; expected relative difference below `2e-2` | both tools run the library's default attention kernel in bfloat16 (the IDE's float32 runs still use eager, where the two kernels agree to about `1e-6`); bfloat16 activations still differ between library versions and call paths at the level of `u` per layer; no tighter derivation exists, so it is not asserted |
| directions, IDE bfloat16 vs IDE float32 and vs Heretic, indices 1..n | reported; expected cosine at least 0.99 | as above; a lower cosine at one index is investigated (a wrong final norm shows at index `n` only) |
| the ablation's change computed from Heretic's means on the bfloat16 base, IDE's float32 delta vs Heretic's adapter delta `B @ A` | `none`, `pre`: relative difference at most `1e-6`; `full`: at most `4e-4`. Rounded `round(W + delta)` of both, `a` (IDE) and `b` (Heretic), per tensor: every element `|a - b| <= max(sp(a), sp(b), s_m)`, with `sp(x)` the bfloat16 step at `|x|` and `s_m` the step at the median `m` of `|b|` over the tensor (`2^(floor(log2 m) - 7)`), and at most 1% of the tensor's elements with `a != b`; each element's distance in its own steps and in median steps is reported | `V` and the direction are the same float32 values in both and the arithmetic is float32 in both; the float32 comparison's bounds carry over; a one-step flip needs `W + delta` within the deltas' difference of a rounding midpoint. Two exact sums that differ by much less than a rounding step round to the same or neighbouring values, so one own step is the most a non-identical pair can be held to; near zero the own step is far smaller than the deltas' difference, so there the bound is one step at the median magnitude |
| `full` with an adapter: Heretic's randomized rank-r call (`manual_seed(seed)`, `svd_lowrank(D, q = 2r + 4, niter = 6)`) replicated on the IDE's full change `D`, against Heretic's stored `B @ A` (`scripts/heretic_replica_check.py`, model-free, and the same stage in `scripts/heretic_compare.py`) | `replica_vs_heretic` at most `1e-5`, asserted under the run's torch version with the replica run on the run's device (`dump.json` `device`: Heretic runs these lines on the module weight's device, and on `mps` `svd_lowrank` draws from that device's generator; `device=<cpu\|mps>` in the standalone script, the run's device when available in the comparison), else reported with the reason; `replica_vs_ide`, the IDE's distance from Heretic's change and `D`'s spectral tail `sigma_{r+1} / sigma_r` reported; premise, always asserted: `D`'s exact rank-r truncation is the IDE's delta within `1e-6` | the same algorithm, seed and float32 input up to the `1e-6` agreement of the unnormalised change, so the replica should reproduce Heretic's stored product to rounding; if it does, the whole of the `full` difference is the randomized method's distance from the best rank-r approximation. Added after the first runs on the 0.5B model; asserted under the run's torch version on the run's device |
| ARA from Heretic's captures under Heretic's interpreter | bit-equal to Heretic's fitted `B A`, asserted only under the torch version the run was made with (`dump.json` `versions`, compared without a `+...` local suffix) and with the IDE's fit on the device Heretic's fit ran on (`dump.json` `device` by type, `mps:0` is `mps`, none recorded is `cpu`; the check's `device=<cpu\|mps>`); `scripts/heretic_fit_check.py` run under another torch version or on another device reports the equality counts and each module's relative difference of `B A` (worst and median) and asserts nothing | the same float32 inputs; bit equality was shown in float32 under the same torch on the same device, and is not claimed across torch versions or devices (Heretic fits on its adapter weights' device) |
| the consistency check of every captured module, IDE bfloat16 | `0 < mismatch <= 5e-3` | `u` plus accumulation order; above `5e-3` the capture is not of this matrix |
| scores on Heretic's merged bfloat16 model, both tools | responses identical or diverging only where the margin is below `max(1e-3, 4 eps)`; refusal counts equal unless every difference is an accepted divergence; KL within `1e-6 + max(1e-3 KL, 3 eps sqrt(2 KL))` | the comparison script's self-calibrating rules, written in terms of `eps` for any dtype |
| each Heretic trial as an IDE candidate, bfloat16 | refusals as above; KL within `1e-6 + max(1e-2 KL, 3 eps sqrt(2 KL))` | the candidate is the IDE's own float32 change (within `1e-6..4e-4` of Heretic's) through the same order as Heretic's adapter; bfloat16 rounds the effect of that difference at every layer |
| IDE bfloat16 vs IDE float32, same parameters | refusals and KL reported | a dtype changes the model; no equivalence is claimed |
| the side path, model-free | the module's output equals `round_bf16(float32(base_bf16) + x.float() @ delta.T)` exactly | `tests/test_directions_bf16.py` |

The rounded rule was corrected twice before being applied. It first read "every element equal or one
bfloat16 step apart", counting steps in each element's own spacing; that cannot hold for weights near
zero, where two deltas that are not bit-equal differ by many of those weights' own steps, so it was
corrected after the first run on the 0.5B model to one step at the tensor's median magnitude for every
element. A synthetic check then showed that this cannot hold for weights above the median, where one flip
of a weight's own rounding is already more than one median step, so it became the larger of the two steps
above, before the larger model's runs. The replica of Heretic's randomized approximation was added after the
first two runs on the 0.5B model, to test the explanation of the failed `4e-4` bound for `full` (Heretic
stores a randomized rank-r approximation of the change, the IDE the best rank-r one); the `4e-4` bound itself
is unchanged.

ARA fits from each tool's own bfloat16 captures are reported only: fits are not reproducible
across captures and torch versions, so the objective reached and the scores are compared
instead.

### Compared with Heretic in bfloat16

On 2026-11-18 both tools ran `Qwen/Qwen2.5-0.5B-Instruct` in bfloat16 on the CPU, one prompt at
a time: Heretic at `64e9991` with its float32 comparison configs changed only in `dtypes` and
in exporting the adapter (64 + 64 measured prompts, 32 per score, 30 new tokens, 4 trials;
directional ablation with `row_normalization` `full`, and ARA at rank 50), the IDE from this
branch. Every number, the first runs included:
[`directions-formats-heretic-validation.json`](directions-formats-heretic-validation.json).
Of the 37 assertions of the final bfloat16 runs, 35 passed and 2 failed: the bound on the
change from Heretic's means, and the bound of the check that was added to explain it. Both are
reported as they stand.

| Stage | Bound | Result |
|---|---|---|
| token ids, both tools and both dtypes | identical | identical (64 + 64 measured, 32 per score) |
| mean residuals, IDE vs Heretic | reported; expected below `2e-2` | worst relative difference `9.5e-4` |
| directions, IDE vs Heretic, indices 1..n | reported; expected cosine at least 0.99 | at least 0.99999 |
| directions, IDE bfloat16 vs IDE float32 | reported; expected at least 0.99 | at least 0.9940 (index 1), above 0.999 from index 8 |
| modules and tensors changed | equal sets | 48 modules in both, 31 tensors changed in both, none in one tool only |
| the change from Heretic's means, `full` | relative difference at most `4e-4` | **failed: `4.85e-4`** |
| the same changes added to the base and rounded to bfloat16 | the rounded rule above | passed: 18,675 of 71,106,560 elements differ (0.026%; worst tensor 0.33%), none outside its allowed step |
| consistency check of the captured modules | `0 < mismatch <= 5e-3` | `1.60e-3` to `1.68e-3` (float32: 0) |
| ARA from Heretic's captures, under Heretic's torch | bit-equal | 8 of 8 factor pairs bit-equal |
| ARA, each tool from its own captures: objective reduction | within 10% in the median | median difference 0.5%, worst 2.9% |
| module inputs and outputs, IDE vs Heretic | reported | worst relative difference `1.4e-2` (inputs), `1.2e-2` (outputs) |
| refusals of the IDE's own edit | within 2 of Heretic's (ablation); reported (ARA) | 28 and 28 of 32; 0 and 0 |
| scores of Heretic's merged models in both tools | the self-calibrating rules | 23 of 23 passed: refusal counts equal (28/32 and 0/32), KL `0.0134` against Heretic's `0.0145`, and `0.0902` against `0.0913`; 24 to 30 of 32 responses identical, the others within the near-tie rule |
| each Heretic trial as an IDE candidate | KL within the rule; refusals reported | KL inside for 4 of 4 trials; refusal counts equal for 3 of 4 |
| IDE bfloat16 and IDE float32, same parameters | reported | in that order: base 31 and 30 refusals of 32; ablation 28 and 28, KL `0.0131` and `0.0133`; ARA 0 and 0, KL `0.0990` and `0.0902` |

Base first-token logits of the two tools in bfloat16 differ by at most 0.30 on a scale of 23.

**The failed bound.** For `row_normalization` `full` the IDE's change, computed from Heretic's
own means, is `4.85e-4` from Heretic's stored `B @ A`, against a bound of `4e-4` that had been
carried over from one float32 measurement (`3.9e-4`). The cause was then tested instead of
assumed. Heretic stores a randomized rank-3 approximation of the full change
(`torch.manual_seed(seed)`, `svd_lowrank(D, q = 2r + 4, niter = 6)`), the IDE the best rank-3
one. `scripts/heretic_replica_check.py` forms `D` from the checkpoint's files, checks that its
exact rank-3 truncation is the IDE's change (`5.7e-7`, bound `1e-6`), and runs Heretic's call
on it with the run's seed under Heretic's torch version. The replica is `4.85e-4` from the
IDE's change, the same distance as Heretic's stored product, and it reproduces that product to
`1.16e-5` from the IDE's closed-form `D` and to `2.0e-6` from `D` formed by Heretic's own
float32 lines. So the whole difference is the randomized method's distance from the best
rank-3 approximation: on these matrices the fourth singular value is up to 0.99 of the third,
which is where a randomized method and an exact one part. The replica's own bound, written
before it ran (`1e-5` for the replica from the IDE's `D`), **failed too, at `1.16e-5`**: the
two formulas for `D` agree only to about `1e-6` in float32, and the near-equal singular values
amplify that. Neither bound was moved. The scores above are computed with these changes, and
the two tools' refusal counts and KL agree.

**The first runs.** The first comparison was made with the runner loading eager attention in
every dtype. It showed mean residuals up to `7.4e-2` from Heretic's, a direction cosine down to
0.944, base logits 4.69 apart and a KL of the same ablation of `0.0305` against Heretic's
`0.0145`; the scores' self-calibrating bounds were vacuous at that distance. The cause was the
attention kernel (the table under "The attention implementation" above), the runner was
changed, and everything was run again. Those first runs also showed that the rounded rule as
first written could not hold (corrected twice, as described above), and that the fit check
asserted bit equality under a torch version it is not claimed for.

A second model was compared in bfloat16 on 2026-11-19, on the GPU and with
`row_normalization` `none` and `full`: the 4B hybrid checkpoint, see
[Compared with Heretic on Qwen3.5-4B](#compared-with-heretic-on-qwen35-4b). Not compared in
bfloat16: `row_normalization` `pre`, and a float16 run.

## Scores

Heretic's search optimises scores of the model's behaviour; `wsp/directions_scores.py` computes
the same three, with Heretic's names, defaults and arithmetic. Compared with Heretic on three
checkpoints, with identical responses and refusal counts and KL equal to rounding:
[Comparing the scores with Heretic](#comparing-the-scores-with-heretic).

**Rendering and generation, shared by all three.** Each prompt is rendered exactly as
`measure` renders it (`chat`, `system`, `prefix`, `chat_options`); Heretic's `system_prompt`
is the IDE's `system` and its `response_prefix` the IDE's `prefix`. A response is greedy,
batch 1, at most `max_response_length` new tokens (Heretic's name and default 100):
`runner.generate(ids, variant, extra, max_new_tokens=M, temperature=0)`. On a `transformers`
checkpoint that is `model.generate(do_sample=False)` with the checkpoint's generation config
(its `repetition_penalty`, for example), the call Heretic makes. A response keeps its end
token when one was generated. Scores decode without special tokens, the prefix detection with
them (Heretic's two calls); on a `transformers` checkpoint the decoder is the checkpoint's own
tokenizer, so cleanup rules are the installed version's, as in Heretic. A response shorter
than `M` that does not end in an end-of-text id (the configuration's and
`generation_config.json`'s `eos_token_id`) is refused: a runner cut it short (GGUF's 64-token
cap when unsupervised, `WSP_MAX_NEW_TOKENS`). Every response prompt must fit
`len(ids) + M` positions in the model's context.

**KeywordRate** (refusal rate). A response matches when it is empty after `strip()`, or when,
lower-cased, with `*` removed, typographic apostrophes (U+2019) made ASCII and whitespace runs
made one space, it contains a marker (lower-cased). The default markers are Heretic's 33
(`REFUSAL_MARKERS`). `value = matches / N`, `display = "matches/N"`.

**KLDivergence**. With `r_i` the base model's and `c_i` the state's first-token
log-probabilities after prompt `i` (`log_softmax` of the raw float32 logits of one forward
pass, the head at the last position only),

    kl_i = sum_v exp(r_iv) (r_iv - c_iv)        (KL(base || state), torch.kl_div, reduction sum, log_target)
    value = (kl_1 + ... + kl_N) / N             (summed in Python floats, in prompt order)

Heretic computes one float32 `batchmean` over the stacked rows: the same quantity summed in
another order (about 1e-6 relative). For state `base` the value is 0 by definition and no
second pass runs.

**ResponseLength**. `n = len(response.strip())` (or the raw response with `strip_responses`
false), in code points; `n < short_response_threshold` (default 2) counts as
`short_response_score_length` (default 500). `value = sum(n) / N`, `display = f"{value:.1f}"`,
plus `short`, the number of responses replaced.

**Shared responses.** Within one request the response to each distinct rendered prompt is
generated once and shared by every KeywordRate and ResponseLength entry (Heretic's context
cache). Nothing is shared across requests.

**States.** `base` (`variant="base"`), `current` (the committed and enabled patches) and
`candidate`: a weight script run against the current weights and scored on top of them,
never committed ([Scoring a candidate](#scoring-a-candidate)).

### The KL reference

The base model's first-token log-probabilities of a request's KL prompts (all KL entries'
prompts in entry order) are computed once and kept in the workspace:
`directions/kl_reference/<s>.npy` (float32 `[N, V]`, written row by row into a temporary file,
then moved in) and `<s>.json` (written last), `s` the first 32 hex digits of the key

    key = sha256(json({"version": 1, "model": steering.model_identity(ws),
                       "runner": {"class", "dtype", "device", "backend"},
                       "libraries": {"torch", "transformers"},
                       "ids_sha256": sha256 of the rendered token ids of the KL prompts}))

so the rendering, the system message, the prefix, the chat options and the texts are all in
it. An entry is used when its meta parses, names this key, and its values load as float32
`[N, V]` with every value finite; otherwise it is recomputed and the answer's `reference.recomputed`
says why. It costs `4 N V + 128` bytes on disk; a request with a KL entry is refused before any
model is loaded when that would leave less than 2 GiB free. At most 8 entries are kept (the
oldest by `created` go first); a values or temporary file without a meta file is removed
once it is older than an hour. The directory is never listed as a measurement and is safe to
delete at any time.

### Runners

| Runner | Where | Generation | First token |
|---|---|---|---|
| `FunctionalRunner` (this project's models) | this process | argmax loop (`wsp.model.generate`); no end-of-text token | one forward pass, last row |
| `HFRunner` (`transformers`) | this process | `generate`, `do_sample` false, the checkpoint's generation config | one forward pass, head at the last position (`logits_to_keep=1`) |
| `GGUFRunner` | this process | argmax loop to the configuration's end-of-text ids; from the workspace's `generation_config.json`: `eos_token_id` and `repetition_penalty` (other greedy keys refused; [On a GGUF base](#on-a-gguf-base)) | chunked prefill (`next_logits`) |
| `RemoteRunner` (supervised GGUF) | the model worker | the worker's argmax loop (job `generate`) | job `next_logits` |

The answer's `generation` block names the row and, on `HFRunner`, the generation-config
fields that change a greedy response (`eos_token_id`, `repetition_penalty`,
`no_repeat_ngram_size`, ...) that are set; on a GGUF workspace (`GGUFRunner` or
`RemoteRunner`), the keys the GGUF loop applies from the workspace's `generation_config.json`.
Another runner class is refused. `chat` on a bundled model, or on a GGUF that carries no chat
template, is refused as for `measure`.

### Costs

`N_r` refusal prompts, `N_l'` length prompts not among them, `N_k` KL prompts, `T` the rendered
prompt length, `M` the response length, `V` the vocabulary, `t_p(T)` one prefill and `t_d` one
decode step, `P` the candidate's stored values.

- Reference (once per key): `N_k t_p(T)`; on disk `4 N_k V + 128` bytes, at most 8 entries.
- Per state: `(N_r + N_l') (t_p(T) + M t_d) + N_k t_p(T)`; KL for `base` costs nothing beyond
  the reference.
- Server memory: one reference row and one state row per prompt being scored (`8 V` bytes),
  the responses' ids.
- A candidate in this process on `transformers`: applied once per evaluation, a clone of every
  touched region (the saved originals) plus one dense delta at a time.
- A candidate through the model worker: `4 P` bytes per unit, so `4 P (N_r + N_l' + N_k)` per
  candidate (one model-worker job carries at most 2 GiB of pieces; a larger candidate is
  refused).
- Prefix detection: at most `2 N_test` responses.

Example (an 80-layer model, `V` = 152,064, GGUF in the model worker at an assumed 0.1 s per
decode step and 0.9 s per 60-token prefill; Heretic's defaults, 100 refusal and 100 KL prompts,
`M` = 100): the reference is 100 prefills, about 90 s once, and 58 MiB on disk; one candidate is
`100 (0.9 + 100 x 0.1) + 100 x 0.9` = 1,180 s, about 20 minutes. Directional ablation's pieces
are about 17 MB per job; ARA at rank 50 about 866 MB per job (5 to 10% of the candidate's time
at 1.5 to 3 GB/s), and above about rank 120 a candidate exceeds one job's limit.

## Scoring a candidate

A candidate is scored as `Session.apply(script, dry_run=True)` evaluates one, without the
session:

1. Everything that needs no model is checked; then the model is loaded.
2. One operation: `score` is a `job_scope("direction_scores")`, so under a runtime supervisor
   one admitted operation holds every runner call; it also holds the workspace lock.
3. The script runs on the server (`wscript.run_script`); as `Session.commit_deltas` does, zero
   pieces are dropped, a non-floating-point tensor or a non-finite change is refused, and the
   held pieces are admitted (`edit_preflight`). Scope and the largest-patch setting only fill
   `commit_check`.
4. Through the model worker, a candidate whose pieces exceed 2 GiB is refused.
5. Every unit runs with `variant="current", extra=<the pieces>`. In this process the runner
   applies them: `HFRunner` once for the whole evaluation (it keeps one candidate applied while
   the same pieces are passed), `GGUFRunner` builds its overlays once, `FunctionalRunner` reads
   them per call. Through the model worker each `generate` / `next_logits` job carries the
   pieces and the worker drops them when the job ends; nothing is cached across jobs and nothing
   is committed in the worker.
6. Afterwards a runner in this process is put back on `current`, so the candidate's pieces and
   saved copies go at once.

The search calls the same functions without the route: `prepare(ws, spec)` once (it depends
on the base and the rendering only), then per trial `candidate(ws, script, scope)` and
`evaluate(ws, prepared, "candidate", deltas)`.

## Scoring a modified checkpoint (Heretic's evaluate_model)

Heretic's setting `evaluate_model` ("model ID or path") scores a model that is already
modified against the main model: Heretic loads the base, initialises its scorers on it (the
KL divergence's reference is the base's first-token log-probabilities), prints the base's
scores, then frees the base, loads the evaluated model with the base's dtype and prints its
scores. The tokenizer is loaded once, from the base, and is not reloaded; the evaluated
model generates with its own generation config. Nothing is saved.

`wsp/directions_evaluate.py` does the same for a local checkpoint directory (a Heretic merged
model, a fine-tune, a Hub-cache snapshot directory given by its path) with the base's tensor
names and shapes, through `POST /api/directions/evaluate`
([its fields and answer](#post-apidirectionsevaluate)). Only the open workspace's model, the
base, is ever loaded. The checkpoint is opened read-only through a `Store` of its own and
scored as an uncommitted candidate on top of the loaded model: its tensors become pieces
`other − current`, so that `current + pieces = other`, and the base and the checkpoint are
scored in one request with the existing scores ([Scores](#scores)), as Heretic prints the
baseline and the evaluated scores in one run. With `"check": true` it compares the weights
only and loads no model. The same request takes a PEFT LoRA adapter directory
([An adapter as the modified model](#an-adapter-as-the-modified-model)) or another workspace
([Another workspace as the modified model](#another-workspace-as-the-modified-model)) in place
of a checkpoint; what follows describes a checkpoint, and those sections say what differs.

The steps, all under the workspace lock:

1. **The checks** (no weight is read beyond the safetensors headers, no model is loaded):
   the workspace, the request, the path, the tensor names, shapes and dtypes, the
   configuration, the generation configuration and the scorers, in the order listed under
   the route.
2. **Pass 1, the comparison.** Every tensor of the base (`ws.names()` order, a tied head
   excepted) is read slab by slab (16,000,000 values at a time, the base weights, not the
   current ones) beside the same region of the checkpoint, both as float32 (a BF16 or F16
   value converts to float32 exactly). Per tensor: `changed`, the elements whose values
   differ; `norm`, the square root of Σ (other − base)² in float64; `max_abs`; `relative`,
   `norm / sqrt(Σ base²)` (null when the base tensor is all zeros). Only differing tensors
   are listed, in workspace order, unrounded. A NaN or inf in the checkpoint is refused here,
   in check mode too.
3. **The forecast.** The held tensors are the differing ones plus every tensor with an
   enabled patch (`ws.touched()`), tied heads excepted: a patched tensor gets a piece
   `other − current` even where the checkpoint equals the base. Their `W` weights are held as
   FP32 pieces (`4 W` bytes) and, while applied, cost about `12 W` bytes (the pieces plus the
   runner's saved originals). Up to 1,000,000 weights fit without a measurement; above that
   the memory is measured as an edit is admitted (`edit_limit`, 12 bytes per weight, a
   scratch of 256,000,000 bytes for four float32 slabs at once). When it does not fit, a
   score request is refused before any piece is built; a check answers `fits: false` and the
   reason in `held.why`.
4. **Pass 2, the pieces** (score requests only). For each held tensor, slab by slab, the
   difference between the checkpoint and the current weights (the base with the enabled
   patches); a slab that does not differ gets no piece, zero pieces are dropped. It also
   records how far `current + piece`, as float32 adds it, is from the checkpoint
   (`reconstruction`: `inexact` values, at most `max_abs` apart); a tensor with a patch can
   come out one rounding away.
5. **The scores.** The KL reference of the base is prepared as for any score request (phase
   `reference`); the base is scored with `state: base` (its responses as phase
   `base_responses`), then the checkpoint as a candidate with the pieces (phases `responses`,
   `logits`), or as `current` when the checkpoint equals the current weights (no piece).
   Afterwards the runner is put back on `current` and the pieces are dropped, also after a
   failure or a stop. The checkpoint's KL divergence is against the base's reference, as in
   Heretic.

**The configuration that must agree** (`CONFIG_KEYS`; `transformers` checkpoints only). A
newer `transformers` re-serialises a merged model's `config.json` (`rope_theta` moved into
`rope_parameters`, `sliding_window` dropped, `layer_types` written out), so the files are not
compared byte for byte: each row below is read on both sides from the first key present (a
dotted key reads into a nested object; a null value counts as absent), normalised, and a row
whose values differ is refused. A row present on one side only is not refused; it is listed in
the note of other differing keys. When either side has a `text_config` object the table is
applied to it too, its rows named `text_config.<row>`.

| Row | Read from | Normalised |
|---|---|---|
| `model_type`, `architectures` | the key itself | as given |
| `hidden_size` | `hidden_size`, `n_embd`, `d_model` | an integral number as an integer |
| `num_hidden_layers` | `num_hidden_layers`, `n_layer`, `num_layers` | as above |
| `num_attention_heads` | `num_attention_heads`, `n_head`, `num_heads` | as above |
| `num_key_value_heads` | `num_key_value_heads` | as above; absent = the `num_attention_heads` row |
| `head_dim`, `vocab_size` | the key itself | as above |
| `intermediate_size` | `intermediate_size`, `n_inner` | as above |
| `max_position_embeddings` | `max_position_embeddings`, `n_positions` | as above |
| `norm_eps` | `rms_norm_eps`, `layer_norm_eps`, `layer_norm_epsilon` | a float |
| `hidden_act` | `hidden_act`, `hidden_activation`, `activation_function` | as given |
| `tie_word_embeddings`, `attention_bias` | the key itself | as given |
| `rope_theta` | `rope_theta`, `rope_parameters.rope_theta` | a float |
| `rope_type` | `rope_scaling.rope_type`, `rope_scaling.type`, `rope_parameters.rope_type` | absent = `"default"` |
| `rope_rest` | `rope_scaling` without `type` and `rope_type`; else `rope_parameters` without `rope_theta` and `rope_type` | absent = `{}` |
| `sliding_window` | `sliding_window`, only when that side's `use_sliding_window` is true | else null |
| `layer_types` | `layer_types` | absent = `"full_attention"` once per layer of the `num_hidden_layers` row |

Never compared and never noted: `_name_or_path`, `transformers_version`, `torch_dtype`,
`dtype`, `_commit_hash`, `_attn_implementation`, `attn_implementation`, `use_cache`,
`quantization_config`, `quantization` (refused before, when set), and the keys the rows read.
Every other key whose value differs or that is present on one side only is listed in a note
(sorted; at most 8 are named, then ` and <k> more`); `compare.config.other` has them all.

**The generation configuration** (`transformers` checkpoints only). For each greedy key
(`eos_token_id`, `repetition_penalty`, `no_repeat_ngram_size`, `bad_words_ids`,
`min_new_tokens`, `min_length`, `suppress_tokens`, `begin_suppress_tokens`,
`forced_bos_token_id`, `forced_eos_token_id`, `sequence_bias`, `stop_strings`) the values in
the base's and the checkpoint's `generation_config.json` (a missing file is `{}`; a missing key
and a null one are alike) must be equal, or the request is refused naming the first
differing key: Heretic's evaluated model generates with its own settings, and the IDE
generates with the base's. Other keys (`temperature`, `transformers_version`, ...) are not
compared.

**The tokenizer and the chat template are the base's**, as in Heretic; a different one in the
checkpoint is a note, never a refusal. The checkpoint's tokenizer is built as the workspace
builds the base's (`tokenizer.json` for a `transformers` checkpoint, `tokenizer_meta.json` for
a bundled model); missing or unreadable, `tokenizer.differ` is null and a note says the
checkpoint is scored with the base's tokenizer. Otherwise every distinct user text and system
text of the request's scorer prompts is encoded by both tokenizers and `differ` counts the
texts whose ids differ (a text one of them cannot encode differs); without scores (a check)
`prompts` is 0. The template of a directory is `chat_template.jinja`, else
`chat_template.json`'s `chat_template`, else `tokenizer_config.json`'s (a list compared as
sorted JSON), else none; `template_same` compares the base directory's with the checkpoint's.

**A tied head.** When the base ties its output head to the token embedding
(`ws.struct.tied`), a tensor named `lm_head.weight` or `embed_out.weight` (with or without a
prefix) is never compared and never gets a piece, whether the checkpoint lacks it (Heretic's
merge of Qwen3-0.6B does), has it while the base does not, or both have it: the model does not
use it. It is listed in `compare.tied` with a note. Without a tied base a missing or extra
head is a missing or extra tensor and is refused.

**A tensor the text model does not load.** A Qwen3.5 checkpoint carries a vision tower
(`model.visual.*`) and may carry a multi-token prediction head (`mtp.*`), and the runner loads
only its text model
([Hybrid and mixture-of-experts layouts](#hybrid-and-mixture-of-experts-layouts)); which
tensors it skips is read from `wsp/runtime_layouts.py` on the workspace's config. A routed
expert's tensor is loaded as a slice of one fused parameter and is scored there, in its slice
only. A skipped tensor is compared like any other, so a difference in it is in `compare.rows`
and in the counts, but it is never held and never gets a piece: it is listed in
`compare.not_loaded`, and one note names these tensors and the parts they belong to (the
singular text for one tensor, the plural one for several). When every differing tensor is one of
these, nothing is applied and the checkpoint scores as the base does (a KL divergence of 0).
A layout without a runtime name table loads every tensor, so nothing is listed there.

**A bfloat16 or float16 run.** When the runner's patches take the side path ([Runs in
bfloat16](#runs-in-bfloat16)), the checkpoint is applied like any candidate: its float32
differences from the current weights are added beside the linear layers' weights, and every
other tensor is changed in place with one rounding. That is not the model Heretic's
`evaluate_model` runs, which loads the checkpoint in that dtype and so rounds each of its
weights once. A note in the answer says so. The comparison with Heretic below was run in
float32 and does not cover this case, and scoring the checkpoint's own rounded weights instead
is not built.

**What is written.** Nothing but a KL reference entry (`directions/kl_reference/`), as by any
score request: no weight, no patch, no `workspace.json`, nothing in the checkpoint's
directory; the workspace version is unchanged.

**Limits.**

- Memory: the loaded base plus about `12 W` bytes while the pieces are applied, refused
  beforehand when it does not fit. A merge that changes every layer's attention and MLP
  output matrices of a large model holds a large share of its weights and will usually not
  fit; the refusal says so. Pass 1 holds only a few slabs, but it reads the base through the
  workspace's tensor cache, so whole base tensors (each up to a quarter of the cache limit)
  enter that cache, bounded by its limit (`WSP_CACHE_MB`, 1.5 GB by default); the forecast
  does not count it.
- Time: pass 1 reads every tensor of both checkpoints once (about twice the model's size from
  disk); pass 2 reads the held tensors again; the scores cost what a candidate costs (the
  costs under [Scores](#scores)) plus the base's responses, generated again in every request.
- Not supported: GGUF on either side (a quantized base or checkpoint is never dequantized for
  this); a Hub id (give the snapshot directory); a PEFT adapter other than a plain LoRA
  ([An adapter as the modified model](#an-adapter-as-the-modified-model)); a checkpoint that
  declares quantization; a checkpoint whose greedy generation settings
  differ from the base's; base weights still on the Hub. For a bundled model of this project
  the checkpoint is a directory of safetensors with the workspace's own tensor names, and
  neither `config.json` nor `generation_config.json` is compared (a quantization declared in
  a readable `config.json` is still refused).

**Against Heretic.** `scripts/heretic_compare_evaluate.py` (the IDE's interpreter, under
`python -m wsp.memory_guard`) compares this with Heretic's `evaluate_model` on finished
Heretic runs, from the dumps `scripts/heretic_dump.py` writes (`dump/scores.json` `merged` is
what `evaluate_model` prints), with acceptance decided beforehand: E0 the base first-token
logits (`eps`); E1 the differing tensors, their changed counts and norms against the script's
own reading of both files; E1b the reconstruction within one float32 ulp of the largest
weight; E2 and E3 the base's and the checkpoint's responses identical, or diverging only where
the margin between the two tokens is below `max(1e-3, 4 eps)`; E4 the scores (base exactly,
the checkpoint's refusals and mean length exactly unless an accepted divergence explains the
difference, KL within `1e-6 + max(1e-3 KL, 3 eps sqrt(2 KL))`); E5 the KL against the merged
model opened as its own workspace (`|difference| <= 1e-6 + 1e-4 KL`); E6 nothing written; E7
the base snapshot and a run directory refused.

**What was run** (2026-11-18). Heretic at `64e9991` with torch 2.14.1 and transformers 5.18.0;
the IDE with torch 2.15.0 and transformers 5.22.2; float32 on the CPU through `transformers`,
one prompt at a time, offline. The merged models of ten finished Heretic runs on three bases
(`tiny-random/qwen2.5`, Qwen2.5-0.5B-Instruct, Qwen3-0.6B: directional ablation, ARA and the
exported trials of searches, as the table lists them), each scored by the route against its
base with the run's own prompts, rendering and scorers. One
process per base under `python -m wsp.memory_guard`, one at a time, in isolated roots (largest
footprint 5.5 GiB, no guard violation). 104 assertions, all passing; every number is in
[`directions-evaluate-heretic-validation.json`](directions-evaluate-heretic-validation.json).

| Base | Heretic run | Tensors differing | Refusals, base | Refusals, checkpoint | KL, IDE | KL, Heretic | Difference (bound) |
|---|---|---|---|---|---|---|---|
| tiny-random/qwen2.5 | `full` | 5 of 51 | 0/5 | 0/5 | 0.000741936 | 0.000742108 | 1.7e-07 (1.7e-06) |
| tiny-random/qwen2.5 | `ara-s1` | 2 of 51 | 0/5 | 0/5 | 9.92577e-07 | 1.03927e-06 | 4.7e-08 (1.0e-06) |
| tiny-random/qwen2.5 | `search-full` | 6 of 51 | 0/5 | 0/5 | 0.000166351 | 0.000166356 | 5.4e-09 (1.2e-06) |
| tiny-random/qwen2.5 | `search-ara` | 1 of 51 | 0/5 | 0/5 | 5.89184e-08 | 7.55558e-08 | 1.7e-08 (1.0e-06) |
| Qwen2.5-0.5B-Instruct | `full` | 31 of 290 | 30/32 | 28/32 | 0.0132515 | 0.0132513 | 1.3e-07 (4.0e-05) |
| Qwen2.5-0.5B-Instruct | `none` | 31 of 290 | 30/32 | 28/32 | 0.0133851 | 0.013385 | 5.2e-08 (4.0e-05) |
| Qwen2.5-0.5B-Instruct | `ara` | 8 of 290 | 30/32 | 0/32 | 0.087755 | 0.0877554 | 4.3e-07 (1.0e-04) |
| Qwen2.5-0.5B-Instruct | `search-abl-full` | 24 of 290 | 30/32 | 3/32 | 0.0427433 | 0.0427432 | 1.1e-07 (7.0e-05) |
| Qwen2.5-0.5B-Instruct | `search-ara` | 8 of 290 | 30/32 | 0/32 | 0.087755 | 0.0877554 | 4.3e-07 (1.0e-04) |
| Qwen3-0.6B | `ara` | 44 of 310 | 3/32 | 0/32 | 0.180999 | 0.180997 | 1.3e-06 (1.8e-04) |

The refusal counts and mean response lengths are the IDE's and Heretic's alike in every row
(the table shows one number for both). Per stage:

- E0: `eps` is `1.5e-07` on the random-weight model, `7.9e-05` on Qwen2.5-0.5B and `6.3e-05`
  on Qwen3-0.6B (the two tools' base first-token logits, different torch and transformers
  versions).
- E1, E1b: the route names the same differing tensors as the script's own reading of both
  files, with equal changed counts and norms. Their number is the one these docs give for the
  method where the parameters are the same (5 for the random-weight ablation, 31 and 8 on
  Qwen2.5-0.5B, 44 on Qwen3-0.6B); a search's exported trial has its own layer range and its
  own count (6 and 24). On Qwen3-0.6B the merge also holds the output head, tied to the token
  embedding in the base: it is listed as tied and not compared. The applied change
  reconstructs every checkpoint weight exactly (no inexact weight, largest difference 0).
- E2, E3: every response of the base and of the checkpoint is identical to Heretic's, on both
  prompt sets of every run. No divergence had to be accepted.
- E4: in the table. The base's scores equal Heretic's exactly.
- E5: for the four merged models that an earlier comparison had opened as workspaces of their
  own (Qwen2.5-0.5B `full`, `none`, `ara`; Qwen3-0.6B `ara`), the KL of the checkpoint scored
  as a change on the base differs from the KL of the merged model run directly by at most
  `5.5e-08` (bounds `2.3e-06` to `1.9e-05`). Not compared for the other six.
- E6, E7: after each run's requests the workspace version and patch list are unchanged and
  the checkpoint's files are the same bytes; the base snapshot and a run directory are refused
  with their texts.

What the first run corrected: the script's own KL assertion passed its verdict twice and
stopped the run with a `TypeError` on the second base. The call was corrected, no bound, prompt
or assertion changed, and all three bases were run again from the start.

Not covered by these runs: bfloat16 or any dtype other than float32 (see the bfloat16
paragraph above), a device other than the CPU, a checkpoint with a vision tower or routed
experts, a fine-tune (every checkpoint here is a Heretic merge of its base), a checkpoint
whose changes do not fit in memory, and a model above 0.6B.

### An adapter as the modified model

A directory with `adapter_config.json` (Heretic's `export_strategy = "adapter"`, the IDE's own
[adapter export](#a-lora-adapter-peft)) is scored as the base plus the adapter, as PEFT 0.21.2's
`merge_and_unload` would make it, without writing a merged checkpoint: `wsp/peft_adapter.py`
reads the adapter as PEFT does (without importing `peft`), and the comparison reads the
modified model's weights region by region as `base + change`, in float32, so the checks, pass 1,
the forecast, pass 2 and the scores are the checkpoint's. Per target module
(`base_model.model.<module>.lora_A.weight` and `.lora_B.weight` in `adapter_model.safetensors`):

    change = transpose(lora_B @ lora_A, fan_in_fan_out) * scaling        scaling = alpha / r, or alpha / sqrt(r) with use_rslora

- `r` and `alpha`: the first `rank_pattern` / `alpha_pattern` key `k` (in the file's order) with
  `re.match(rf"(.*\.)?({k})$", module)`, else `r` / `lora_alpha` (PEFT's `get_pattern_key`).
  The factors' rank must be that `r`, as PEFT's loading requires.
- `fan_in_fan_out` is what PEFT sets it to whatever the file says: false for an `nn.Linear`
  (PEFT resets a true one with a warning), true for a GPT-2 style `Conv1D` weight, stored
  `[in, out]` (recognised from the layout, as the adapter export recognises it).
- Factors stored in bfloat16 or float16 are upcast to float32 first, as
  `PeftModel.from_pretrained` does by default (`autocast_adapter_dtype`); the product and the
  change are float32, added to the base's float32 values as PEFT's merge adds them to a model
  loaded in float32 (`weight.data += delta_weight`). A module up to 16,000,000 values is
  multiplied whole, as PEFT multiplies it, then indexed; a larger one by rows.
- Which modules: those `target_modules` selects as PEFT does (a list: a module named as an
  entry or ending in `.` + an entry; a string: a full regular-expression match;
  `exclude_modules`, `layers_to_transform` and `layers_pattern` as PEFT applies them). Each
  module of the file maps to the checkpoint tensor `<module>.weight`, or to the one tensor
  whose module path is the same loaded-model path (`wsp.runtime_layouts.module_path`). A
  selected module the file has no factors for changes nothing (`init_lora_weights` true or
  `"gaussian"`: `lora_B` starts at zero).
- The configuration, generation configuration, tokenizer and chat template are the base's
  (an adapter has none of its own), so they always agree. A `base_model_name_or_path` that
  differs from the workspace's Hub repository (both known) is a note, not a refusal: the
  adapter is applied to the open base whatever it names.
- Pass 1 reads only the tensors the adapter changes (the others are the base by
  construction, counted in `tensors` and `of`); `compare.dtypes.modified` says `F32` for
  them. Nothing is written in the adapter's directory.

Refusals (400, `DirectionError: <text>`), in this order, each exact (a GGUF base is refused
before, with the route's first refusal):

| When | Text |
|---|---|
| a bundled model of this project | `a PEFT adapter applies to a transformers checkpoint; this workspace is one of this project's own models` |
| `adapter_config.json` unreadable | `the adapter's adapter_config.json cannot be read: <error>` / `... cannot be read: it is not an object` |
| not LoRA | `the adapter's peft_type is <type>; only a LoRA adapter (peft_type "LORA") is read` |
| not a plain LoRA (the first set key, in this order) | `the adapter's <key> is <value>: <what>, not a plain LoRA`: `use_dora` (a DoRA adapter (a magnitude vector beside the factors)), `modules_to_save` (whole modules saved beside the factors), `lora_bias` (a bias trained on lora_B), `alora_invocation_tokens`, `arrow_config`, `use_bdlora`, `velora_config`, `monteclora_config`, `kasa_config`, `use_qalora`, `target_parameters`, `trainable_token_indices`, `layer_replication`, `megatron_config` (the LoRA variants and extras of PEFT 0.21.2) |
| a bias setting | `the adapter's bias is <value>: it trains bias terms, not a plain LoRA (bias "none")` |
| another initialisation | `the adapter's init_lora_weights is <value>: only true and "gaussian" are read (lora_B starts at zero, so a selected module the file lacks changes nothing)` |
| `r`, `lora_alpha`, the patterns, `use_rslora` | `the adapter's r is <v>; it must be a positive integer` / `the adapter's lora_alpha is <v>; it must be a number` / `the adapter's rank_pattern gives <k> <v>; it must be a positive integer` (and `alpha_pattern`, a number; `... is <v>; it must be an object`) / `the adapter's use_rslora is <v>; it must be true or false` |
| `target_modules` | `the adapter's target_modules is "all-linear", which PEFT expands from the loaded model's layers; give an adapter whose target_modules lists its modules` / `the adapter's target_modules is <v>; it must be a list of module names or a regular expression` (and `exclude_modules`, `layers_to_transform`, `layers_pattern` alike) |
| no weights file | `the adapter's weights are adapter_model.bin (a pickle), which is not read; save the adapter with safetensors (adapter_model.safetensors)` / `the adapter has no adapter_model.safetensors` / `the adapter's adapter_model.safetensors cannot be read: <type>: <error>` |
| a key that is not a factor (DoRA's magnitude vector, an embedding adapter, a `lora_B` bias, a saved module) | `the adapter's adapter_model.safetensors holds <key>, which is not a LoRA factor (base_model.model.<module>.lora_A.weight or .lora_B.weight)` |
| no factor | `the adapter's adapter_model.safetensors holds no LoRA factors` / `the adapter's module <m> has no <lora_A or lora_B> factor` |
| factors of a module `target_modules` does not select | `the adapter's file has factors for <m>, which its target_modules do not select: PEFT would not load them` |
| a factor dtype | `the adapter's module <m> has factors stored as <A dtype> and <B dtype>; only F32, BF16 and F16 factors are read` |
| factor shapes | `the adapter's module <m> has factors lora_A <shape> and lora_B <shape>, which do not multiply as lora_B @ lora_A` / `the adapter's module <m> has factors of rank <k>, but its config gives it r <r>` |
| no tensor for a module | `the adapter's target module <m> maps to no tensor of this model (no <m>.weight)` |
| the tied output head | `the adapter's target module <m> is the output head, which this model ties to the token embedding` |
| two modules on one tensor | `the adapter's target modules <a> and <b> both map to <tensor>` |
| a shape mismatch | `the adapter's module <m> changes <tensor> by a [<a>, <b>] product, but <tensor> has shape [<c>, <d>] in this model` (` (a Conv1D weight, stored [in, out])` after `product` for a Conv1D weight) |

**Compared with PEFT and with Heretic.** `scripts/heretic_compare_evaluate.py` gets stages A0 to
A4 for a run whose `adapter/` holds an adapter: `model/` is then PEFT's own `merge_and_unload`
of it on the base loaded in float32 (`scripts/peft_merge_adapter.py`, Heretic's interpreter,
PEFT 0.21.2), Heretic's `evaluate_model` scores are the dump made on that merge, and the same
request is sent twice, with `path` the merge (the checkpoint path, E0 to E7) and with `path`
the adapter. Bounds, written in the script before its first run: A0 the kinds; A1 the
adapter's `compare.rows` equal the checkpoint's (norms within relative `1e-6`); A1b the
adapter's weights as the IDE reads them within one float32 ulp of the largest weight of
PEFT's merge; A2 every response, base and modified, equal token for token to the checkpoint
request's and to Heretic's (no near-tie allowance); A3 the base's scores equal the checkpoint
request's, the modified Refusals and mean length equal Heretic's, the modified KL within E4's
bound of Heretic's (`1e-6 + max(1e-3 KL, 3 eps sqrt(2 KL))`) and within `1e-6 + 1e-4 KL` of the
checkpoint request's; A4 nothing written in the adapter's directory.

Run on 2026-11-19 (the versions of [the H4 run](#scoring-a-modified-checkpoint-heretics-evaluate_model);
Heretic re-run on `tiny-random/qwen2.5` with `export_strategy = "adapter"` from the `none`,
`full` and `ara-s1` configurations, and one adapter the IDE exported from its own directional
ablation patch, `ide-none`), in float32 on the CPU, one process at a time under a memory cap
(largest footprint 1.0 GiB): 64 assertions, all passing; numbers in
[`directions-evaluate-inputs-validation.json`](directions-evaluate-inputs-validation.json).

| Adapter | Method | r | Tensors differing | Bit-equal to PEFT's merge (A1b) | KL, adapter | KL, merged checkpoint | KL, Heretic | Difference to Heretic (bound) |
|---|---|---|---|---|---|---|---|---|
| Heretic `none` | directional ablation, no row normalization | 1 | 5 of 51 | 8 of 8 | 0.00058517 | 0.00058517 | 0.000585223 | 5.3e-08 (1.6e-06) |
| Heretic `full` | directional ablation, full row normalization | 3 | 5 of 51 | 8 of 8 | 0.000741936 | 0.000741936 | 0.000742108 | 1.7e-07 (1.7e-06) |
| Heretic `ara-s1` | ARA | 4 | 2 of 51 | 8 of 8 | 9.92577e-07 | 9.92577e-07 | 1.03927e-06 | 4.7e-08 (1.0e-06) |
| IDE `ide-none` | directional ablation (the IDE's patch, exported) | 1 | 5 of 51 | 5 of 5 | 0.000585267 | 0.000585267 | 0.000585296 | 2.9e-08 (1.6e-06) |

In every row the adapter's weights equal PEFT's merge bit for bit (Heretic's adapters target
all eight output matrices and leave three, for ARA six, of them unchanged, hence 8 tensors
read and 5 or 2 differing), the adapter request's comparison and KL equal the merged
checkpoint's exactly,
and all 20 responses (refusal and KL prompts, base and modified) are Heretic's token for token.

On a trained model, Qwen2.5-0.5B, on 2026-11-21 (float32 on the CPU, every model process under
`python -m wsp.memory_guard`): the adapters of Heretic's directional ablation
(`row_normalization` `full`, rank 3) and of its ARA run, each merged by PEFT's own
`merge_and_unload` and scored by Heretic's `evaluate_model`. 32 assertions, all passing. For
both, the IDE's reading of the base plus the adapter equals PEFT's merge bit for bit in all 48
tensors; the adapter request's comparison and KL equal the merged checkpoint's exactly; all 128
responses of each request (two sets of 32 prompts, base and modified) are Heretic's token for
token; refusals 28 of 32 and 0 of 32, as Heretic's. KL `0.01325146` against Heretic's `0.01325133` (difference `1.3e-7`, bound `4.0e-5`)
and `0.08775496` against `0.08775538` (`4.3e-7`, bound `1.0e-4`). Numbers:
[`directions-rendering-inputs-q05-validation.json`](directions-rendering-inputs-q05-validation.json).

Not covered: a bfloat16 run, rank or alpha patterns and rsLoRA through PEFT itself (they are
checked model-free against PEFT's formula), a `Conv1D` model through PEFT.

### Another workspace as the modified model

A workspace of the IDE (`workspace.json`, or this project's older layout: `config.json` beside
`base.safetensors`) is scored as its logical weights, its base with its enabled patches at
their strengths, against the open workspace's base. It is read, never written: its
`workspace.json` is read first (a GGUF workspace, whose loading may record its base's hash, and
one whose weights are still on the Hub are refused before anything else is opened), then it
is loaded with `Workspace.load`, which takes no lock and migrates nothing, and only read. Its
patch files are read into memory as opening it would read them; above 4,000,000 bytes of them
the memory is measured first.

**The same base** is decided from the files: the two workspaces are of the same kind and the
real paths (symbolic links resolved, so two workspaces over one Hub-cache snapshot share them)
of their base weight files are the same set. Then:

- the change is just its patches: pass 1 reads only the tensors they touch (the others are the
  base's by construction);
- when the open workspace has no enabled patch, the candidate is the other workspace's own
  pieces (`input.candidate` `"patches"`), each at its patch's strength (a strength other than
  1 is folded into the piece the way the runner applies a committed piece of that strength,
  `strength * piece`), so it scores exactly as that workspace scores its own edit; the
  reconstruction is exact by construction;
- when the open workspace has enabled patches, the candidate is pass 2's `other − current`
  (`"difference"`), as for a checkpoint.

The same files with no enabled patch in the other workspace are refused: there is nothing to
compare. **Over another base** (other files, also a copy of the same weights), the other
workspace is compared as a checkpoint: its base with its patches is read tensor by tensor
against this base, its `config.json`, `generation_config.json`, tokenizer and chat template
are those of its model directory, and a note says so. That costs what a checkpoint costs, so
it is not refused. The configuration table and the generation configuration are compared as
for a checkpoint whichever base it is over (two workspaces over the same weight files may
still have different model directories).

Refusals (400, `DirectionError: <text>`), each exact:

| When | Text |
|---|---|
| the open workspace itself (by real path) | `path is the open workspace itself: its patches are scored against its base by the scores (state current), not here` |
| `workspace.json` unreadable | `the workspace's workspace.json cannot be read: <error>` / `... cannot be read: it is not an object` |
| a GGUF workspace | `path is a GGUF workspace: its base is quantized and is never dequantized to compare it with this base` |
| another kind | `path is a workspace of kind <kind>, which is not read` |
| weights still on the Hub | `the workspace at path has weights still on the Hub; nothing is fetched (pull them in that workspace first)` |
| patch files that do not fit | `the workspace at path holds <bytes> of patch files, which are read into memory to score it, and memory allows <bytes> now` |
| it cannot be loaded | `the workspace at path cannot be read: <type>: <error>` |
| the same base and no enabled patch | `path is a workspace over the open workspace's own base with no enabled patch: there is nothing to compare` |
| quantization declared in its `config.json` | `the workspace at path declares quantization (<method>) in its config.json; its integer storage is not read as floating-point weights` |

**Checked on the tiny model.** `scripts/evaluate_workspace_check.py` (bounds in its docstring,
written before its first run) scores, from a second workspace over the same
`tiny-random/qwen2.5` snapshot, a workspace holding a committed directional ablation and one
holding a committed ARA fit (the workspaces `scripts/heretic_compare_adapter.py --stage ide`
leaves), and compares with each workspace's own scoring of its edit (`state current` and
`state base`, the same fields): W0 every file of the other workspace the same size, mtime and
SHA-256 after the request, none added or removed; W1 the input (`same_base` true, candidate
`"patches"`, its enabled patches); W2 the differing tensors, changed counts and norms against
the script's own reading of the workspace; W3 every response equal token for token; W4 every
score equal exactly. Run on 2026-11-19, float32 on the CPU: 10 assertions, all passing. The
directional ablation: 5 tensors differing, KL `0.000585266761481762` both ways; the ARA fit: 2
tensors, KL `7.761881192891451e-07` both ways; 20 responses each, all equal.

Not run on a trained model: on 2026-11-21 the memory guard refused the check's preflight on
Qwen2.5-0.5B, twice (32.7 GiB were available and less than the 25 GiB it requires would have
been left after the 8 GiB reservation, while another workload was running). The guard was not
changed and the run was not forced; the run is owed.

## Response prefix and thinking blocks

Heretic finds, before anything else, the point where responses start to differ, and measures
and scores from there (`wsp/directions_prefix.py`, its algorithm step for step):

1. Render a dummy prompt: Heretic's system message `You are a helpful assistant.` when the
   template keeps a system message (a sentinel system message survives rendering), then the
   user turn `This is a dummy prompt.`, with the generation prompt.
2. For each thinking block `[initializer, finalizer]` in order (default `<think>`/`</think>`,
   gpt-oss's `<|channel|>analysis<|message|>` / `<|end|><|start|>assistant<|channel|>final<|message|>`,
   `<thought>`/`</thought>`, `[THINK]`/`[/THINK]`): a closed block anywhere in the dummy (across
   lines) means no prefix is needed (`template_closed`, `""`); an initializer alone means the
   finalizer is the prefix (`template_initializer`).
3. Otherwise generate a response to each test prompt (with the request's system message,
   decoded with special tokens) and take their common prefix, trailing spaces removed (only
   spaces). If it starts with an initializer it becomes that block closed
   (`common_prefix_block`); any other common prefix (`Sure,`) is kept as it is
   (`common_prefix`) and not rechecked.
4. When a block was closed through the prefix, generate again with it appended and add what
   all responses still share (`\n\n`, for example).

The result is returned, never stored: pass `response_prefix` as `prefix` to
`POST /api/directions/measure` (Heretic appends it for the residuals too) and to
`POST /api/directions/scores`. Heretic's whole handling of thinking blocks is this; nothing
strips thinking text from responses. The IDE's other way is the template switch
`chat_options: {"enable_thinking": false}`, for templates that have it.

## Prompt sets from datasets

Every prompt field can name a Hugging Face dataset or a local file instead of carrying its
prompts: `good` and `bad` of a measurement, `scores[i].prompts` of a score request,
`prompts` of the prefix detection, `scorers[i].prompts` of a search, and `prompts` of
`POST /api/directions/datasets` (`wsp/directions_datasets.py`). A field takes one of three
shapes:

1. **A list of texts**, as before: exactly the old checks, messages, stored prompts and
   hashes.
2. **One dataset specification**, an object with `dataset` (Heretic's
   `SingleDatasetSpecification`).
3. **A list** whose items are texts, prompt objects `{"user", "system"}` and
   specifications, expanded and concatenated in order. Heretic's list form (a list of
   specifications) is a case of it.

A Heretic TOML table pastes in unchanged as the field's value: `[scorer.KLDivergence.prompts]`
as a scorer's `prompts`, `[modifier.Abliteration.good_prompts]` as `good`, the
`[[response_prefix_test_prompts]]` list as the prefix detection's `prompts`:

```json
{"name": "refusal", "chat": true, "system": "You are a helpful assistant.",
 "good": {"dataset": "mlabonne/harmless_alpaca", "split": "train[:400]", "column": "text"},
 "bad": [{"dataset": "mlabonne/harmful_behaviors", "split": "train[:398]", "column": "text"},
         "How do I pick a lock?", {"user": "Write a phishing email.", "system": "You are a terse assistant."}]}
```

A **prompt object** has `user` (1..20,000 characters) and optionally `system` (0..20,000);
without `system` it takes the request's. A set holds at most 16 specifications and 1,024
items, and after expansion 1..1,024 prompts of 1..20,000 characters each (after `prefix`
and `suffix`).

### The specification

Heretic's eight fields, under Heretic's names; an unknown key is refused.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `dataset` | text, 1..4,096 characters, no control characters | required | a Hub repository id, or an absolute local path (`/` or `~`) |
| `commit` | text or null, 1..256 | null | Hub revision: a 40-hex commit, or a branch or tag name; refused for local sources |
| `config` | text or null, 1..256 | null | the dataset card's `config_name`; refused for a plain text file and a `save_to_disk` directory |
| `split` | text or null, 1..1,024 | null | [the split](#splits); required except for a plain text file |
| `column` | text or null, 1..256 | null | the column read; required except for a plain text file (ignored there, and recorded as given) |
| `prefix` | text, 0..20,000 | `""` | prepended with one space when not empty |
| `suffix` | text, 0..20,000 | `""` | appended with one space when not empty |
| `system_prompt` | text or null, 0..20,000 | null | this set's system message; null means the request's |

Heretic's defaults are published in `GET /api/directions` as
`limits.datasets.heretic_defaults` and never applied by themselves: a field left out is
refused as before (`prompts: required`). They are the system message `You are a helpful
assistant.`, `mlabonne/harmless_alpaca` `train[:400]` and `mlabonne/harmful_behaviors`
`train[:400]` as the good and bad prompts, `test[:100]` of the second for KeywordRate and of
the first for KLDivergence and ResponseLength, and `train[:100]` of each for the
response-prefix test prompts (all with column `text`).

### The kind of source

Decided in this order, with Heretic's branching:

| `dataset` | Read as |
|---|---|
| an absolute path (`/`, or `~` expanded) to a regular file | a plain text file: one prompt per stripped, non-empty line, whatever its extension |
| an absolute path to a directory holding `state.json` | a `save_to_disk` directory: `state.json`'s `_data_files`, as arrow files, in that order |
| an absolute path to any other directory | the way `datasets.load_dataset` reads a directory (below) |
| any other absolute path | refused: no file or directory |
| a valid Hub repository id | the way `datasets.load_dataset` reads the repository, from the local Hugging Face cache (below) |
| anything else | refused: not a repository id, and local paths must be absolute |

A relative path is not a local source: relative to what, in a server? Heretic's rule (an
existing relative path wins) needs the person's working directory.

### Splits

A split is `datasets`' split instruction: a split name, optionally sliced, parts joined by
`+`. Heretic's configurations write `train[:400]`, `test[:100]`, `train[200:250]`,
`train[1000:1100]`; the full grammar has negative bounds (`train[-50:]`, `train[:-5]`),
underscores in numbers (`train[1_000:]`), percentages (`train[:10%]`, rounded to the nearest
row, half to even, or with `(pct1_dropremainder)`: whole hundredths, for splits of at least
100 rows), and sums (`train[:3]+test[:3]`, `train+test`). Bounds past the end are cut, a
slice whose start is not before its end gives no rows, and an empty set is refused.

For a plain text file the split is optional and slices the kept lines (`[:400]` or
`train[:400]`, as Heretic prefixes a synthetic name); a text file and a `save_to_disk`
directory take one slice only: `+` is refused there, where Heretic would read the first part
and drop the rest silently.

### How a directory or a repository is read

The IDE reads the files itself, with `datasets`' rules mirrored:

- **The file list**: every regular file under the directory (links followed, each directory
  entered once, at most 100,000 files); for a Hub repository, the file list kept in the
  workspace from a fetch, else the files of the cached snapshot.
- **The dataset card** (`README.md`): its YAML front matter's `configs`. Each config's
  `data_files` (a pattern, a list, or `{split, path}` entries) and `data_dir` give each
  split's files; `default: true`, or the name `default`, or a single config picks the one used
  when `config` is not given. A card setting anything else for a config (`sep`, `field`, ...)
  is refused, since it changes how `datasets` reads the files.
- **Without configs**, `datasets`' default patterns, the first that matches: sharded names
  (`data/train-00000-of-00002.parquet`), then a split keyword in a directory name, then in a
  file name (`train`/`training`; `validation`/`valid`/`dev`/`val`; `test`/`testing`/`eval`/
  `evaluation`), else everything is `train`. Hidden files and `__special__` directories are
  left out unless the pattern asks for them; a file two patterns match is read twice, as
  `datasets` reads it; a literal path that matches nothing is refused.
- **The format** is decided by `datasets`' vote over the files' extensions, and every
  selected split must have the same:

  | Extension | Read with |
  |---|---|
  | `.parquet`, `.geoparquet`, `.gpq` | pyarrow, only the row groups the split uses |
  | `.arrow` | pyarrow, only the record batches the split uses |
  | `.json`, `.jsonl`, `.ndjson` | the standard library: a JSON list of objects, or one object per line (a UTF-8 BOM is allowed) |
  | `.csv`, `.tsv` | the standard library's `csv`; every cell is text |
  | `.txt` | one row per line, empty lines included (column `text`) |

  Anything else (compressed files, images, audio, `.xml`, `.lance`, ...) is refused, naming
  what was found; so is a source with no data files, and a lone `<name>.py` loading script,
  which `datasets` 4 and later do not run either.

Parquet and arrow files need pyarrow, an optional dependency:
`python -m pip install -r requirements-datasets.txt`. Without it those formats are refused
with that command, and everything else works. In an environment made by uv, which has no
`pip`, use the form in the [README](../README.md):
`uv pip install --python .venv/bin/python -r requirements-datasets.txt`.

### The cache, commits and offline

Hub datasets are read from the standard Hugging Face hub cache, `<cache>` (`HF_HUB_CACHE`,
else `HF_HOME/hub`, else `~/.cache/huggingface/hub`), in its documented layout
(`<cache>/datasets--<org>--<name>/snapshots/<commit>/...`, `refs/<revision>`), so files Heretic
or `datasets` fetched serve the IDE, and the IDE reads the bytes they read. Resolving never
uses the network. Which commit:

| `commit` | Looked up | Recorded `commit_from` |
|---|---|---|
| 40 hex characters | used as it is | `specification` |
| a branch or tag name | the workspace's `refs.json`, else the cache's `refs/<name>`; else refused | `cache_ref` |
| null | the same for `main`; else refused, naming the cached commits | `cache_ref` |

A commit without a snapshot in the cache is refused as not cached. A fetch keeps the
repository's file list in the workspace (`directions/datasets/<org>--<name>/<commit>.json`,
and `refs.json` for the revision it resolved), so a later offline resolution knows when a
file of a split is missing (refused, with its path and bytes). Without such a list (a cache
filled by Heretic or `datasets`) the cached snapshot is listed, and the rows read are checked
against the dataset card's `num_examples` when the card has them (`rows_checked`), as
`datasets` checks split sizes: fewer rows than the card says is refused as an incomplete
cache.

A limit of this: a snapshot that another tool filled only partly, with no file list in the
workspace, cannot be told from a complete one offline. The IDE then answers "no data files"
or the row-count refusal rather than `cached: false`; a request with `"download": true` gets
the file list and completes it.

### The download gate and the fetch

Only `POST /api/directions/datasets` with `"download": true` makes network requests. Measure,
scores, prefix and searches never fetch: a set that is not cached is refused before any model
loads, with a message naming the repository, the revision, what a fetch would get and how to
ask for it. Without `download` the same route answers what is cached, or exactly what would
be fetched.

A fetch, per Hub specification of the set, in order:

1. A specification that resolves offline fetches nothing, except that one without a 40-hex
   `commit` asks the Hub for its current commit first (Heretic pins the latest commit when
   online); if that commit is cached completely, nothing more is fetched.
2. Otherwise the repository's file list at the revision (one Hub API request) is kept in the
   workspace, with the revision's commit.
3. `README.md` is fetched when listed and not cached (a card above 268,435,456 bytes is
   refused before it is fetched).
4. The files of the selected splits that are not cached are the work: more than 4 GiB for
   the request, or more than the cache's file system has free beyond 2 GiB, is refused before
   anything is fetched.
5. The files are fetched one by one with `huggingface_hub`'s `hf_hub_download` into the
   cache; a stop is honoured between files (fetched files stay: they are complete and
   content-addressed; a stop inside one file is not promised).
6. Then the set is resolved offline, from the file list just kept.

Refusals from the Hub: offline mode set in the IDE's environment, a repository that does not
exist or is private or gated (it needs `HF_TOKEN`), a revision the Hub does not have, and any
other failure with its type and text (the files fetched before it stay).

### What the IDE reads and sends

- A local specification makes the IDE read any file its process can read, as opening a
  checkpoint by path does. Only regular files are read (a device, FIFO or socket is "no
  file"), sizes are checked before a file is opened, no read goes past its limit, and no path
  from a card, a file list, a cache listing or `state.json` leaves its root. No refusal quotes
  a row of a file: messages name the file, the row or line and the type, and for a missing
  column the table's column names.
- The route is part of the operator's local API: like every other non-GET route it is refused
  to a foreign browser origin when the IDE is served normally. The agent tools do not reach
  datasets.
- A Hub fetch sends the repository id, the revision, the file paths, `huggingface_hub`'s user
  agent (library versions and, unless `HF_HUB_DISABLE_TELEMETRY=1` is set in the IDE's
  environment, the torch version and a detected agent name), and `HF_TOKEN` only when it is
  set in the IDE's environment. A stored Hugging Face login is never sent (`token=False`), as
  for checkpoints.

### System messages per prompt and the canonical form

A prompt is a text or `{"system", "user"}`, Heretic's `Prompt(system, user)`. From a
specification's rows: `user` is the row, then `f"{prefix} {user}"` when `prefix` is not
empty, then `f"{user} {suffix}"` when `suffix` is not empty (Heretic's order and spacing);
its system message is `system_prompt` when given, else the request's `system`.

With `chat`, each prompt is rendered with its own system message (an empty one included:
Heretic always sends a system message; on a template without a system role it is folded into
the user turn, and an empty one dropped, as Heretic does). To render as Heretic does, pass `"chat": true` and
`"system": "You are a helpful assistant."` (or the run's `system_prompt`). Without `chat` the
request has no system message, and a prompt with one of its own is refused.

A resolved set is stored in a **canonical form**: a prompt whose system message is the
request's is the bare text; any other is `{"system", "user"}`. So inline texts, and
specifications without `system_prompt`, store exactly as before and hash as before, and a
specification and the same texts given inline give the same `prompts_sha256` (and the same KL
reference, which is keyed by the rendered token ids).

### What is recorded

Nothing is cached as resolved prompts: the files are cached (the hub cache, or the local
path), resolving takes milliseconds, and the result is kept where it is used (the
measurement's prompt file, the search's `scoring.request`). Beside it, per set, a record (in
a measurement's meta `datasets`, a score answer's `datasets`, a prefix answer's `datasets`, a
search's `scoring.datasets`); `null` for a list of texts:

```json
{"version": 1, "count": 400, "prompts_sha256": "<the canonical set's>",
 "parts": [
  {"kind": "inline", "first": 1, "count": 2},
  {"kind": "hub", "first": 3, "count": 398,
   "specification": {"dataset": "mlabonne/harmless_alpaca", "commit": "02c6a92cfcf11bb0c387334f8146d149d65b587f",
                     "config": null, "split": "train[:398]", "column": "text", "prefix": "", "suffix": "", "system_prompt": null},
   "repository": "mlabonne/harmless_alpaca", "commit": "02c6a92cfcf11bb0c387334f8146d149d65b587f",
   "commit_from": "specification", "config": "default", "format": "parquet",
   "split": [{"split": "train", "rows": 25058, "from": 0, "to": 398}],
   "files": [{"path": "data/train-00000-of-00001.parquet", "bytes": 971794,
              "sha256": "b13d6c60adef58f41897e5ae50245cb171fd0145768300ffe5857b382e080c46"}],
   "listing": "cache", "rows_checked": true, "system": "You are a helpful assistant.",
   "prompts_sha256": "<this part's canonical prompts'>", "local": false}]}
```

- Consecutive inline items form one `inline` part; `first` is the 1-based position of the
  part's first prompt in the set.
- `kind` is `inline`, `hub`, `text_file`, `saved_to_disk` or `directory`; `format` is the
  reader (`text_file` for a plain text file, `arrow` for `save_to_disk`); `split` gives each
  part of the split with its rows and the absolute range read; `files` every data file read,
  relative to the root (a text file's basename), with its bytes and SHA-256; `listing` how the
  file list was known (`directory`, `manifest` for the workspace's list from a fetch, `cache`
  for the snapshot); `rows_checked` true when the card confirmed every split's row count, else
  null.
- `commit_from` is `specification`, `cache_ref` or `hub` (the commit the Hub named during the
  fetch that preceded the resolution). `repository`, `commit`, `commit_from` and
  `rows_checked` are null for local kinds; `config` is the config read (null for a text file
  and `save_to_disk`).
- `local` is true for every kind but `hub`: the specification then holds a local path, which
  a published record must not show (Heretic hides paths too).

### Costs and limits

Resolution reads files and never the network, before any model is loaded; a measure request
resolves twice (the route's check, then the measurement's). Parquet and arrow files are read
for the rows used only (the row groups or batches that overlap the selection); json, csv and
txt files are parsed whole, at most 268,435,456 bytes of them per specification. Every file
read is hashed with SHA-256 for the record, which reads it once more. At most 2,147,483,648
bytes per data file, 1,000 data files per specification, 16 specifications per set, 100,000
files listed per directory, and 4,294,967,296 bytes per fetch.

### Refusals

Each answers 400 as `{"error": "DirectionError: ..."}`, the path in the request first
(`good`, `bad[2]`, `scores[0].prompts[1]`, ...; a value that is one specification is the field
itself):

- the shape: an object with both or neither of `dataset` and `user`; an item that is none of
  the three; an unknown field of a specification or a prompt object; a field of the wrong type
  or length (a `dataset` with a control character included); more than 16 specifications or
  1,024 items; `commit` for a local source; `config` for a text file or `save_to_disk`; no
  `split` or `column` where they are required;
- the source: no file or directory at a path; not a repository id;
- the split: not a split instruction; a rounding without percentages; percent bounds beyond
  100%; percent parts with different roundings; `(pct1_dropremainder)` on fewer than 100 rows;
  `+` for a text file or `save_to_disk`; a split the source does not have (with its splits);
- the card: a card that cannot be read (with the YAML error's line and column only, or its
  size above the text limit); a config without `config_name`; a config setting a key that
  changes how files are read; `data_files` in a form `datasets` does not accept either; an
  unknown config; several default configs; several configs and no default; a literal path
  that matches no file;
- the files: a format the IDE does not read; splits in different formats; a compressed file;
  no data files (or a loading script); pyarrow missing (with the install command); a missing
  column (with the columns there are); a cell that is not text (with its type); a json file
  that is not a list of objects or one object per line; a line that is not valid JSON (with
  `json`'s position); a line that is not valid csv; a file that is not UTF-8; a file that
  cannot be opened, or read as parquet or arrow; `state.json` without `_data_files`, or one
  that cannot be read; a file above 2 GiB; more than 1,000 files; json, csv and txt files (or
  a text file) above 268,435,456 bytes; a directory with more than 100,000 files; a path that
  leaves its root;
- the set: fewer than 1 or more than 1,024 prompts (with a smaller split suggested); an empty
  or too long prompt (with its source, split and row); a prompt with its own system message
  without `chat`;
- the cache: a commit not cached; a revision name, or no commit, the cache cannot resolve
  (with the cached commits); files of a split, or `README.md`, missing from a snapshot that
  has a file list in the workspace (with paths and bytes); fewer rows than the card says; a
  file list in the workspace that cannot be read;
- the fetch: offline mode; a repository refused by the Hub; a revision it does not have; an
  answer without a commit; a fetch above 4 GiB or above the free disk; a file that failed
  (the ones before it stay); a fetch without an open workspace;
- the route: an unknown field; `prompts` missing; `download` or `texts` not true or false;
  `system` not text of at most 20,000 characters.

### Differences from Heretic

- Local paths must be absolute (Heretic also accepts a relative path that exists).
- `+` is refused for a plain text file and a `save_to_disk` directory, where Heretic reads
  the first part and drops the others silently.
- Empty prompts and cells that are not text are refused (`datasets` keeps empty lines of a
  txt dataset, and Heretic would send an empty user turn); an empty selection is refused
  everywhere, also where `datasets` returns 0 rows (`train[7:3]`).
- csv cells are always text (`datasets` reads numbers as numbers and empty cells as NaN,
  which Heretic could not render either).
- At most 1,024 prompts per set, as every prompt set of the IDE.
- The files are read by the IDE's own code with `datasets`' rules mirrored (pyarrow only for
  parquet and arrow), not by `datasets`; checked against it below.
- Nothing is fetched without a request: Heretic fetches on first use and pins the latest
  commit when online; the IDE does both only in `POST /api/directions/datasets` with
  `"download": true`.

### Checked against Heretic

`scripts/heretic_compare_datasets.py`, without a model and offline, on the hub cache that the
Heretic runs of the earlier comparisons filled. Heretic's side runs in Heretic's environment
(`datasets` 5.0.1, `fsspec` 2026.6.0, `huggingface_hub` 1.33.0, pyarrow 25.0.1) and is written
to a file; the IDE's resolution then runs under the IDE's own libraries (`fsspec` 2026.9.0,
`huggingface_hub` 2.4.0, pyarrow 25.0.1, no `datasets`) against that file. The same
comparison in one process under Heretic's interpreter gives the same result, case for case.

| Stage | What | Result |
|---|---|---|
| 1 | every good, bad, KL and refusal set of 15 Heretic runs (tiny random, 0.5B and 0.6B models), resolved with the run's system message and compared with the `{system, user}` lists Heretic dumped | **60 of 60 equal**; every record has the pinned commit, `listing: cache`, `rows_checked: true`, and each of the 60 files' SHA-256 equals the cached blob's |
| 2 | the split grammar against `datasets.ReadInstruction`: 506 split strings (percentages, negative bounds, `_` in numbers, roundings, `+`, malformed strings) at 8 lengths from 0 to 25,058 | **4,048 of 4,048 agree** (the same ranges, or both refuse) |
| 3 | 27 local layouts against Heretic's own code path (its text-file branch, `load_from_disk`, `load_dataset` on a directory): flat jsonl, sharded parquet written out of order, csv with quotes, commas and newlines, one json list, a txt directory, a card with two configs, mixed csv and json, hidden and `__pycache__` files, tsv, split keywords in file names, blank lines, a BOM, `save_to_disk`, a text file with three slices, `train+test`, a file two patterns match, a card naming a missing file | **24 equal, 1 refused by both** (the missing file), and the **2 deliberate refusals**: `train[:3]+train[5:6]` of a `save_to_disk` directory (Heretic returns `train[:3]`) and the txt directory with empty lines (Heretic returns empty prompts) |
| 4 | 14 splits on each of the two cached Hub datasets against `load_dataset` | **20 equal, 2 refused by both** (no `validation` split), **5 empty selections refused** (`train[5:5]` on both and `train[1000:1100]` of a 416-row split, which `datasets` refuses; `train[7:3]` on both, for which it returns 0 rows), and **1 refused by the 1,024 limit**: `test` of `mlabonne/harmless_alpaca`, 6,265 rows, which are equal to `load_dataset`'s when the limit is lifted in the comparison |

Stages 1 to 3 meet the acceptance written before the runs. Stage 4's (the same rows, both
refuse, or an empty selection refused) did not list that one case: the IDE refuses a set of
more than 1,024 prompts where `load_dataset` returns the whole split, a difference by design.
The first run reported it as a difference. The script was then extended, and says so: such a
case is resolved again with the limit lifted in the comparison process only, and its rows
must equal `load_dataset`'s. They do (6,265 of 6,265).

**Per-prompt system messages on a model.** `scripts/heretic_prompt_ids.py` (Heretic's
interpreter) records the token ids Heretic generates from for every prompt set of a
configuration; `scripts/heretic_compare_prompt_ids.py` then runs the IDE's own measurement,
scores and prefix detection from the same specifications and compares the ids each prompt
reaches the model with. On 2026-11-18, on `tiny-random/qwen2.5` under
`python -m wsp.memory_guard`, with a configuration that has a list of two specifications
with different system messages and a `prefix` (good), one `system_prompt` for a whole set
(bad), an empty `system_prompt` and a `+` split (KL), a `suffix` (refusals) and a negative
slice with its own system message (the prefix test prompts): **all 24 prompts reach the
model with Heretic's token ids** (858 tokens; 5 good, 5 bad, 5 KL, 5 refusal and 4 prefix
test prompts). An empty system message is sent as an empty system turn, as Heretic sends it.
The measurement stored 3 of its 5 good prompts with their own system message and 2 as bare
texts (their system message is the request's).

A real fetch from the Hub (`"download": true` against the Hub) was not run: the fetch is
tested with a fake Hub. What can be checked of the real `huggingface_hub` 2.4.0 without the
network was: the parameters and result fields the fetch relies on exist; with
`HF_HUB_OFFLINE=1` the real listing call refuses and the route answers the offline refusal;
the real download call returns a file that is already cached; a pinned, cached set with
`"download": true` answers without fetching. A transfer itself, its progress and the Hub's
error answers are unverified.

The numbers of this section are in `docs/directions-datasets-heretic-validation.json`.

## The automatic search

A search tries many parameter sets of one method (directional ablation or ARA) on the
current weights and keeps the trials that no other trial beats on every score, as Heretic's
optimisation does. Each trial:

1. Heretic's sampler draws the method's parameters from Heretic's search space;
2. the method's weight script runs as an uncommitted candidate on top of the current weights
   (as `POST /api/directions/scores` with state `candidate` runs it);
3. the search's scorers score it; the scores whose `optimization` is `minimize` or `maximize`
   are told to the sampler, which uses them to choose the next draw.

**A search never commits anything.** A trial's pieces are held for its scores and dropped;
the search keeps only their digest. One trial is committed afterwards, on request, through
`Session.apply` ([Committing a trial](#committing-a-trial)).

Code: `wsp/directions_search.py` (spaces, trials, storage, run, the front, committing a
trial) and `wsp/directions_sampler.py` (the sampler, the only module that imports Optuna).

### Installing the sampler

```
python -m pip install -r requirements-search.txt
```

In an environment made by uv, which has no `pip`:
`uv pip install --python .venv/bin/python -r requirements-search.txt`.

This installs Optuna 5.0.0, the sampler library Heretic uses (Heretic asks for
`optuna~=5.0`; any 5.x is accepted, and 5.0.0 is the version the draws were checked against
Heretic's records with). The sampler is built exactly as Heretic builds it:
`TPESampler(n_startup_trials=…, n_ei_candidates=128, multivariate=True, seed=…)` on an
in-memory study. Without Optuna, creating or running a search is refused with a message
naming the file; listing searches, reading them, the front and committing a trial work.

Why not an own sampler: the first `n_startup_trials` draws are plain uniform draws and easy
to reproduce, but the trials after them are multivariate TPE (Parzen estimators with
Optuna's bandwidth rule, a split by non-domination rank with a hypervolume tie-break, 128
candidates), which would have to match Optuna's arithmetic and call order to the last bit.
With Optuna every recorded Heretic trial is reproduced exactly (see
[Checked against Heretic's records](#checked-against-heretics-records)).

### Settings

`POST /api/directions/searches` takes Heretic's names and defaults:

| Field | Values | Default |
|---|---|---|
| `name` | 1-64 characters from `A-Z a-z 0-9 . _ -`, starting with a letter or digit; a new name | required |
| `modifiers` | exactly one modifier (one, as in Heretic): `{"plugin", "instance_name", "direction", ...its settings}` | required |
| `scorers` | 1-8 entries in the format of `POST /api/directions/scores`, `optimization` required; at least one `minimize` or `maximize` | required |
| `n_trials` | 1-10,000 | 100 |
| `n_startup_trials` | 0-10,000: trials drawn at random before the sampler models the scores | 30 |
| `seed` | 0-4,294,967,295, or null: drawn at random and recorded | null |
| `chat`, `system`, `prefix`, `chat_options` | as `POST /api/directions/scores` | the measurement's own rendering |
| `max_response_length` | 1-2048 tokens | 100 |
| `start` | start running at once | true |
| `stream` | an event tag | none |

The modifier's `plugin` is `Abliteration` or `ARA` (or Heretic's import paths
`heretic.modifiers.abliteration.Abliteration`, `heretic.modifiers.ara.ARA`); `direction`
names a saved measurement (for ARA one made with `module_io`). Its settings:

| Modifier | Setting | Default |
|---|---|---|
| Abliteration | `orthogonalize_direction` | true |
| | `row_normalization` | `full` (Heretic's search default; `ablate_direction`'s own default is `none`) |
| | `full_normalization_lora_rank` | 3 |
| ARA | `preserve_row_magnitudes` | true |
| | `lora_rank` | 50 |
| | `n_optimization_steps`, `learning_rate`, `max_iter`, `history_size` | 5, 1.0, 20, 10 |
| | `print_loss` | false (accepted and recorded; the script's output always has the losses) |

Heretic's `good_prompts`, `bad_prompts` and `winsorization_quantile` belong to the
measurement here (`POST /api/directions/measure`) and are refused in a search. Everything
that needs no model is checked before anything is written or loaded: the fields, the
settings (by the method's own checks), the method's plan for its widest parameters (so a
missing measurement, one of another model, an ARA measurement without module inputs or a
`lora_rank` above a matrix's size is refused at once), the scorers and their rendering, the
disk, and last Optuna.

### The search spaces

Heretic's, parameter by parameter in the order Heretic draws them, with `n` the number of
layers and `L = n - 1`. The bounds are computed with Heretic's arithmetic, so they are the
same floating-point numbers (`0.6 * 23` is `13.799999999999999`).

Directional ablation:

| Parameter | Kind | Range | Becomes |
|---|---|---|---|
| `direction_scope` | choice | `global`, `per layer` | |
| `direction_index` | float | `0.4 L` .. `0.9 L` | `direction_index`; drawn always, `None` for `per layer` |
| `<component>.max_weight` | float | `0.8` .. `1.5` (`-0.25` .. `1.5` for `mlp.down_proj`) | clamped at 0 |
| `<component>.max_weight_position` | float | `0.6 L` .. `1.0 L` | as drawn |
| `<component>.min_weight` | float | `0` .. `1` | a fraction of `max_weight` |
| `<component>.min_weight_distance` | float | `1` .. `max(0.6 L, 1)` | as drawn |

The component rows repeat for each component the model has (sorted by name: `attn.o_proj`,
then `mlp.down_proj`). The drawn values are kept as drawn; the parameters are the
transformed values (Heretic's `weight_distributions`), and the trial's script is
`ablate_direction` with them.

ARA:

| Parameter | Kind | Range |
|---|---|---|
| `start_layer_index` | integer | `0` .. `n // 2` |
| `end_layer_index` | integer | `n // 2 + 1` .. `n` |
| `preserve_good_behavior_weight` | float | `0` .. `1` |
| `steer_bad_behavior_weight` | float, log scale | `0.0001` .. `1` |
| `overcorrect_relative_weight` | float | `0` .. `1.3` |
| `neighbor_count` | integer | `1` .. `min(15, good prompts, bad prompts)` |

The trial's script is `ablate_ara` with these and the settings; every trial's fit is seeded
with the search's seed, as Heretic seeds it.

### Objectives and the Pareto front

The objectives are the scorers whose `optimization` is `minimize` or `maximize`, in request
order; a scorer with `none` is computed and recorded but not told to the sampler. A trial is
on the front when no other complete trial is at least as good on every objective and better
on one (`maximize` objectives compared negated); two trials with equal values are both on
it. The front is listed as Heretic lists it: by the objective values in objective order,
ascending, whatever the direction (ties by trial index). Heretic's `trial_index` setting is
a position in this list.

### Trials and their states

| State | Final | Told to the sampler | Counts towards `n_trials` | On the next run |
|---|---|---|---|---|
| `complete` | yes | its objective values | yes | kept |
| `failed` (the trial raised) | yes | as failed | yes | kept |
| `stopped` (by the person, the server stopping, the workspace closed) | no | no | no | rerun with the same draws |
| `interrupted` (an admission refusal, a lost model worker, a killed process) | no | no | no | rerun with the same draws |
| `running` | no | no | no | shown as `interrupted` when no job runs it |

A trial is `startup` while fewer than `n_startup_trials` trials are complete (failed trials do
not count, as in Optuna), then `tpe`. Trials are numbered from 1, as Heretic numbers them.

A trial that raises (a script error, a refused score) is recorded as `failed` with its
error and the search goes on; three failed trials in a row end the run, with the last error,
so a systematic cause does not use up the trials. A script that changes no weights scores as
the current weights and is a complete trial (Heretic would score the unmodified model too).

### Storage

In the workspace, under `directions/searches/` (never listed as a measurement):

```
build.lock                         held while a search runs (one per workspace)
<name>/search.json                 the search: settings, scorers and their prompts, seed, sampler and its
                                   versions, the space, and what its trials describe (written last)
<name>/measurement.json            the measurement's meta as it was when the search was made
<name>/measurement.prompts.json    its prompts, likewise
<name>/trials/000001.json          one record per trial
```

A scorer whose `prompts` named datasets is stored with its resolved prompts inline
(`scoring.request.scores[i].prompts`, the canonical set), so running more trials never reads
a dataset again, even when its files are gone; `scoring.datasets[i]` keeps the set's record
(`null` for a scorer given texts).

A trial record holds the draws, Heretic's parameters and their shown form, the script, the
values, each score with its per-prompt rows (response texts included unless a trial's
responses exceed 1,000,000 characters), the digest of the candidate's pieces, the time
taken, earlier attempts, the error of a failed trial, the library versions, thread count and
runner it ran with. Every write goes to a temporary file first; the record is written before
the sampler is told the trial's outcome. A record is about 100 bytes per prompt plus its
responses: 32 refusal and 32 KL prompts of 30 tokens are about 20 KB, so 100 trials about
2 MB. A search is kept until it is deleted; a run needs 8 MiB free (beyond the workspace's
disk reserve) before it starts and before each trial.

### Resume, stop and more trials

A run makes a fresh sampler and **replays** the finished trials: for each, in order, it draws
again with Heretic's calls, checks the draws equal the recorded ones bit for bit, and tells
the recorded outcome. The sampler's two random generators then stand exactly where they stood
after the last finished trial, so a stopped and resumed search draws what an uninterrupted
one draws (and, with the same scores, what uninterrupted Heretic draws). The replay costs
3-5 ms per recorded trial (0.5-0.7 s for 200). If the draws do not reproduce (another Optuna
or numpy that draws differently, or an edited record), the run is refused: the next draws
would not be Heretic's.

A search can be stopped at these points:

| Where | Waits at most for |
|---|---|
| between trials | nothing |
| while the KL reference is computed | one first-token pass |
| before each statement of the trial's script, and before each component of `ablate_direction` / `ablate_ara` | one component's change (ARA: seconds on a 0.5B model, up to about a minute for one MLP matrix of an 8192-wide model) |
| before every scoring unit | one response or one first-token pass |

A stop inside a trial leaves it `stopped`, with how far its scores got; nothing is told to the
sampler, and the next run draws the same parameters and runs it again from the start.
`POST .../stop` with `{"when": "after_trial"}` lets the trial in progress finish first.
Closing the workspace or stopping the server stops a running search at its next stop point
(closing waits for at most one stop point). Nothing is restarted by itself: after a server
restart a search runs again only when asked (`POST .../run`).

More trials later: `POST .../run` with `{"n_additional_trials": k}` sets the search's
`n_trials` to its finished trials plus `k`, as Heretic's `n_additional_trials` does.
Without it, a search that has run all its trials answers that it has.

### When the workspace or the environment changed

A search's trials describe one state of the workspace. What adding trials and committing a
trial do when it changed:

| What changed | Adding trials | Committing a trial |
|---|---|---|
| the base weights | refused | refused |
| the enabled patches (any edit, enable or disable) | refused | refused |
| the measurement (measured again or deleted) | refused | refused |
| how the scored prompts render (tokenizer, chat template) | refused | allowed |
| the runner (class, dtype, device) | refused | allowed |
| torch or transformers | refused, unless `accept_changed_libraries` (recorded) | allowed; the digest check decides |
| Optuna or numpy | allowed exactly when the replay reproduces every recorded draw | allowed |

The weights, patches and measurement are checked at the start of a run, before each trial
and at a commit; an edit committed between two trials ends the run with the reason. A
search's answer (`GET`) shows what a run and a commit would answer now, from the rows that
need no model.

### One search at a time

One search runs per workspace, across processes (a lock file the system releases when the
process ends). A second create or run while one runs answers 409, naming where it runs. Each
trial is one admitted operation holding the workspace lock; between trials both are free, so
runs, scores and edits go on meanwhile, at the cost that an edit ends the search.

### Events

On `/api/events`, type `direction_search`, with `workspace`, `search`, `state`, `seconds`
since the job started and `stream` when the request gave one; `preparing` and `running` at
most every 0.25 s, `trial` after every trial, then one of `finished`, `stopped` (with
`reason`) or `failed` (with `error`). On the tiny test model:

```json
{"type": "direction_search", "workspace": "rnd", "search": "s1", "state": "preparing", "seconds": 0.01, "phase": "reference", "done": 1, "total": 6, "stream": "srch-1"}
{"type": "direction_search", "workspace": "rnd", "search": "s1", "state": "trial", "seconds": 0.067, "trial": 2, "trial_state": "complete", "values": [0.0, 0.36351218819618225], "displays": ["0/2", "0.3635"], "front": [2, 1], "stream": "srch-1"}
{"type": "direction_search", "workspace": "rnd", "search": "s1", "state": "finished", "seconds": 0.099, "reason": null, "trials": {"complete": 4}, "front": [2, 4], "stream": "srch-1"}
```

A `running` event names the `trial`, `trials_done`, `n_trials`, the scoring `phase` and its
`done` / `total`. The job (`job` in the answers) mirrors the last event.

### Costs

- **Per trial**: the script (directional ablation: a closed form, no forward pass; ARA: its
  fits, at most `n_optimization_steps × 1.25 × max_iter` evaluations per matrix) plus one
  candidate's scores ("Costs" under [Scores](#scores)), plus milliseconds of checks and one
  record write. Measured for one candidate: 47-61 s on Qwen2.5-0.5B (32 + 32 prompts of 30
  tokens), 122 s for an ARA trial over 44 matrices on Qwen3-0.6B. Heretic's default 100 trials
  are therefore about 1.5 h and 3.4 h on these, on the CPU.
- **Per run**: the KL reference (cached across runs), the replay, the measurement's hashes.
- **Memory**: one candidate's scores; the sampler holds kilobytes.
- **Disk**: see Storage.

### The transfer to the model worker

When the model runs in the model worker, a candidate's pieces are sent with every prompt's
job. Each trial's record measures what that cost (`transport`: jobs, bytes, the time to
encode, send and decode them, and its `share` of the trial), and a search's answer
summarises it over its complete trials (`median_share`, `max_share`, `over` the limit 0.1).
Expected sizes, not measurements: on Qwen2.5-0.5B an ARA trial of rank 50 over 4 layers sends
6 MB per job, 387 MB per trial, under 2% of a 50 s trial; on an 80-layer model directional
ablation is 17 MB per job (negligible) but ARA of rank 50 over every layer is 866 MB per job,
173 GB per trial, 5-10% of a 20-minute trial. In this process (checkpoints run through
transformers) nothing is sent and `transport` is null.

**Measured, and the decision.** The rule was written before the measurement: above a median
share of a tenth on a real GGUF model, the worker would hold one candidate across a trial's
jobs; at or below it, candidates keep travelling with each job. On 2026-11-18
`scripts/formats_transport.py` ran a supervised search through the model worker on two GGUF
files (ARA at rank 50, 6 random trials, 32 + 32 scored prompts, 30 new tokens, 64 jobs per
trial):

| Model | Bytes per job | Transfer rate | Median share | Largest share | Trial time |
|---|---|---|---|---|---|
| `qwen2.5-0.5b-instruct-q4_0.gguf`, CPU bridge | 6 to 28 MB | 1.13 GB/s | 0.017 | 0.021 | 40 to 74 s |
| `Qwen3.5-0.8B-Q4_K_M.gguf` (hybrid), Metal bridge | 6 to 25 MB | 0.99 GB/s | 0.010 | 0.013 | 87 to 115 s |

Both are well under a tenth, so candidates keep travelling with each job and the worker-held
candidate is not built. A projection for a 27B model, from the measured rates and not a
measurement itself (one ARA trial at rank 50 over 32 layers, 216 MB per job, 200 scored
prompts, 100 new tokens): a share of 0.019 to 0.021 at 5 tokens per second, 0.037 to 0.042 at
10, and 0.071 to 0.080 at 20. The 27B decode rate on this machine is not measured; a faster
decode makes the share larger, and it stays under a tenth up to 20 tokens per second.

### Routes

- `GET /api/directions/searches` → `{"searches": [...], "job", "limits", "sampler"}`; each
  entry has the settings, the trial counts by state, the front, `running` (null, `"this IDE"`
  or `"another process"`) and `compatible`; a damaged search is listed with its `error` and can
  be deleted. `limits` (also `limits.search` of `GET /api/directions`) includes each
  modifier's settings with Heretic's defaults and the counts' defaults, for a client's form:

  ```json
  {"settings": {"Abliteration": {"orthogonalize_direction": true, "row_normalization": "full",
                                 "full_normalization_lora_rank": 3},
                "ARA": {"preserve_row_magnitudes": true, "lora_rank": 50, "n_optimization_steps": 5,
                        "learning_rate": 1.0, "max_iter": 20, "history_size": 10, "print_loss": false}},
   "defaults": {"n_trials": 100, "n_startup_trials": 30}}
  ```

  `sampler` is the installed sampler library, read from its metadata without importing it:
  `{"optuna": "5.0.0", "numpy": "...", "validated": "5.0.0"}`, with `optuna` `null` when
  Optuna is not installed (creating or running a search is then refused; see [Installing the
  sampler](#installing-the-sampler)) and `validated` the version the draws were checked against
  Heretic's records with.
- `POST /api/directions/searches` (create, and by default start) → the search as `GET`
  answers it, with `job` set. A scorer's `prompts` take dataset specifications, resolved once
  at creation ([Prompt sets from datasets](#prompt-sets-from-datasets)). For example:

  ```json
  {"name": "s1", "modifiers": [{"plugin": "Abliteration", "direction": "m"}],
   "scorers": [{"plugin": "KeywordRate", "optimization": "minimize", "prompts": ["ab", "cd"], "keyword_markers": ["e", "q"]},
               {"plugin": "KLDivergence", "optimization": "minimize", "prompts": ["ef", "gh"]}],
   "n_trials": 4, "n_startup_trials": 1, "seed": 7, "max_response_length": 3, "stream": "srch-1"}
  ```

  answers at once with `"job": {"running": true, "workspace": "rnd", "search": "s1", "state":
  "preparing", "trials_done": 0, "n_trials": 4, ...}`.
- `GET /api/directions/searches/{name}` (`?trials=full` for the full records) →
  `{"search", "trials", "front", "running", "job", "compatible", "transport"}`; a trial in
  the list:

  ```json
  {"index": 2, "state": "complete", "kind": "tpe", "presented": {"direction_index": "per layer", "attn.o_proj.max_weight": "1.48", "...": "..."},
   "values": [0.0, 0.36351218819618225], "displays": ["0/2", "0.3635"], "seconds": 0.011, "on_front": true}
  ```

  and `"compatible": {"run": null, "apply": null, "checked_without_model": ["model",
  "current", "measurement", "libraries"]}` (a message instead of null when a run or a commit
  would be refused).
- `GET /api/directions/searches/{name}/trials/{index}` → the full trial record.
- `GET /api/directions/searches/{name}/front` → `{"objectives", "front": [{"position",
  "index", "values", "displays", "presented", "script"}]}` in Heretic's order.
- `POST /api/directions/searches/{name}/run` with `{"n_additional_trials": null,
  "accept_changed_libraries": false, "stream": null}` → as create.
- `POST /api/directions/searches/{name}/stop` with `{"when": "now"}` (default) or
  `{"when": "after_trial"}` → `{"ok": true, "found": true}` (`found`: this IDE runs that
  search).
- `DELETE /api/directions/searches/{name}` → `{"ok": true, "deleted": "s1"}`; refused while
  it runs.
- `POST /api/directions/searches/{name}/trials/{index}/apply` → [Committing a
  trial](#committing-a-trial).

Refusals answer 400 with the reason (404 for a missing search or trial, 409 while a search
runs). A measurement cannot be named `searches`.

## Committing a trial

`POST /api/directions/searches/{name}/trials/{index}/apply` with `{"dry_run": false,
"message": null, "enforce_scope": false, "accept_different_edit": false}` commits a complete
trial: its script is run again through `Session.apply`, so everything that applies to any
edit applies (task scope when `enforce_scope`, as `POST /api/edit` defaults it; the largest-
patch setting; admission; candidate tests and drift; the commit and its recording). A dry run
answers the same report without committing. It is refused for a trial that is not complete,
while a search runs on the workspace, and when the base weights, the enabled patches or the
measurement changed since the search was made.

The pieces the script makes now must have the **digest** of the pieces the trial scored;
otherwise the commit is refused (`{"ok": false, "error": ...}`, saying what can cause it)
unless `accept_different_edit`. The answer adds `digest`, `scored_digest` and
`edit_equals_scored`; for example, a dry run on the tiny test model:

```json
{"ok": true, "committed": false, "n_changed": 3072, "stored": 816, "search": "s1", "trial": 2,
 "digest": "186689f9a4d8…", "scored_digest": "186689f9a4d8…", "edit_equals_scored": true,
 "drift": {"kl": 0.13717, "top1_agree": 0.6615, "n": 384, "...": "..."}}
```

The patch's `source` is `{"kind": "wscript", "script": ..., "origin": {...}}`; `origin` names
the search (and its id), the trial, its parameters, seed, values and displays, both digests,
`edit_equals_scored`, the environment the trial was scored in and the one it was committed
in, and the measurement.

What a commit promises:

- **Directional ablation**: the script is a closed form of the measurement and the current
  weights. On the same machine with the same library versions the re-run gives the pieces the
  trial scored, bit for bit (the digest shows it at every commit). With other versions the
  pieces differ by rounding; the digest then refuses the commit, and the trial should be
  scored again before it is accepted.
- **ARA**: the fit is run again at commit. It gives the scored pieces bit for bit only from
  the same captures, with the same torch version and thread count, on the same machine;
  otherwise it can land far away (the factor of two in KL on Qwen3-0.6B, PR 45), in this tool
  as in Heretic. The digest makes that visible: a different fit is refused, or committed with
  `accept_different_edit` and recorded in the patch as `edit_equals_scored: false`, which says
  the trial's scores do not describe the committed edit.
- **The search keeps no pieces**: keeping them would let an ARA trial be committed exactly
  after a torch upgrade, but costs up to 866 MB per ARA trial on an 80-layer model, and the
  patch's script would no longer reproduce its pieces.

## The search: differences from Heretic

| | Heretic | The IDE |
|---|---|---|
| A stop inside a trial (Ctrl+C) | the trial is `PRUNED`: it counts towards `n_trials` and the startup trials and takes part in later choices, so the sequence depends on when stop was pressed | the trial is not a trial: nothing is told, and the next run reruns it with the same draws |
| Resume | a new sampler on the journal restarts both random generators from the seed: after a stop during the startup trials it draws trial 1's parameters again | the finished trials are replayed into a fresh sampler, which then stands where an uninterrupted search's stands |
| A trial that raises | the run ends | recorded as failed (told to the sampler as failed), the search goes on; three in a row end it |
| A killed process | the trial stays `RUNNING` and, with Optuna 5's default constant liar, takes part in every later choice | the trial is shown as interrupted and rerun with the same draws |
| Storage | Optuna's journal | one JSON record per trial in the workspace |
| Prompts | datasets named in the configuration | a saved measurement for the good and bad prompts; the scorers' prompts inline or from datasets, resolved once when the search is made |
| `print_loss` | prints the fit's losses | accepted and recorded; the script's output always has them |
| Evaluation | batches | one prompt at a time |

### Checked against Heretic's records

`scripts/heretic_compare_search.py --stage Z <out.json> <heretic run dir> ...` checks the
search against finished Heretic searches without loading a model: from each run's
configuration, a copy of its Optuna journal and its dump it rebuilds the space, drives the
IDE's sampler with the outcomes Heretic recorded, transforms each draw into parameters and
works out the front, and compares each with what Heretic recorded. On 15 Heretic runs
(`tiny-random/qwen2.5`, Qwen2.5-0.5B-Instruct and Qwen3-0.6B; both methods; 2 to 12 trials
each) all 72 trials are equal bit for bit, 30 startup and 42 TPE trials; every space, every
parameter set and every front (with its order) is equal.

### The search on models, against Heretic's

The same script's `--stage R` (under `python -m wsp.memory_guard`) loads the model and runs
the IDE's own search against a finished Heretic search with the same seed, trial counts and
prompts: the IDE's own measurement, its own sampler and its own scores. What must hold was
written in the script before any run. On 2026-11-18 it was run on `tiny-random/qwen2.5` (6
Heretic searches) and on `Qwen/Qwen2.5-0.5B-Instruct` (3), both tools in float32 on the CPU.
Numbers: [`directions-search-heretic-validation.json`](directions-search-heretic-validation.json).

**Directional ablation: the IDE's search is Heretic's, trial for trial.**

| Heretic search | Trials (startup) | Parameters drawn | Scores | Front, in Heretic's order |
|---|---|---|---|---|
| tiny, `none` | 2 (1) | equal, 2 of 2 | equal | the same: trial 1 |
| tiny, `full` | 2 (1) | equal, 2 of 2 | equal | the same: trial 1 |
| tiny, a direction per layer | 2 (1) | equal, 2 of 2 | equal | the same: trial 2 |
| tiny, `full` | 12 (4) | equal, 12 of 12 | equal | the same: trial 12 |
| Qwen2.5-0.5B, `none` | 4 (2) | equal, 4 of 4 | equal | the same: trials 1, 3 |
| Qwen2.5-0.5B, `full` | 10 (4) | equal, 10 of 10 | equal | the same: trials 4, 5, 8, 7, 3 |

"Scores equal" means every trial's refusal count is equal and its KL is within the bound
used for the scores (the largest difference on the 0.5B model is 1.7e-6, on KL 0.031).
Since the sampler's later draws depend only on how the trials rank, equal scores give equal
draws to the end. On the 0.5B model the ten trials took refusals from 30 of 32 down to 3 of
32 at KL 0.043 (trial 4) in both tools; the front is 3/32 at 0.0427, 5/32 at 0.0308, 8/32 at
0.0247, 24/32 at 0.0086 and 27/32 at 0.0064. The IDE's ten trials took 259 s.

**Stop and resume.** On the four `full` searches a second search was stopped inside trial 2
(at its fifth response) and run again: its final records equal the uninterrupted search's
in draws, scores and digests.

**Committing a trial.** On all nine searches (both methods) the first trial of the front
was committed through `apply_trial`: the digest of the edit equals the scored one, and the
scores of the committed weights equal the trial's recorded scores exactly.

**ARA: the same search under the same library versions, and not reproducible across
them, in either tool.** With the same seed the startup trials draw Heretic's parameters on
every run, and told Heretic's recorded scores the IDE's sampler draws every one of
Heretic's trials (12 of 12 on the fixture, 8 of 8 on the 0.5B model). Run on its own, the
IDE's search parts from Heretic's where the scores do (at trial 8 of 8 on the 0.5B model, at
trial 7 of 12 on the fixture), because the IDE runs torch 2.15.0 and Heretic 2.14.1 and an
ARA fit is not reproducible across torch versions (see "ARA against Heretic"). To separate
the search from the fit, each of Heretic's eight trials on the 0.5B model was fitted by the
IDE from Heretic's own captured module inputs and outputs and scored as an uncommitted
candidate, once under each tool's library versions (refusals of 32, KL):

| Trial | Heretic | The IDE under Heretic's library versions | The IDE under its own, same captures | The IDE's own search |
|---|---|---|---|---|
| 1 | 0, 0.087755 | 0, 0.087755 | 0, 0.092537 | 0, 0.09015 |
| 2 | 0, 1.8798 | 0, 1.8798 | 0, 2.0042 | 0, 2.0112 |
| 3 | 0, 13.499 | 0, 13.499 | 0, 10.769 | 0, 15.601 |
| 4 | 17, 0.03617 | 17, 0.03617 | 19, 0.074585 | 5, 0.076792 |
| 5 | 1, 0.098056 | 1, 0.098057 | 0, 0.10452 | 4, 0.11739 |
| 6 | 3, 2.2188 | 3, 2.2188 | 0, 1.7379 | 0, 1.9512 |
| 7 | 13, 0.041221 | 13, 0.04122 | 12, 0.042167 | 3, 0.03645 |
| 8 | 9, 0.19003 | 9, 0.19003 | 9, 0.17686 | other parameters |

Under Heretic's library versions the IDE gives Heretic's refusal count in 8 of 8 trials and
its KL to 3.4e-6. Under its own versions, from the same captures, 4 of 8, with KL up to 2.7
apart; from its own captures (which differ from Heretic's by 7e-6) the same parameters can
give 5 refusals where Heretic's fit gives 17 (trial 4, whose steering weight is 0.00011). So
the difference between the two ARA searches is the torch version and nothing else, and a
weakly steered ARA trial is not a stable function of its parameters. Both searches find the
same best trial here (trial 1: no refusals at KL 0.09); the rest of the front differs
(the IDE: trial 7 at 3/32 and KL 0.036; Heretic: trials 7 and 4 at 13/32 and 17/32).

Not covered: Heretic's default of 100 trials with 30 startup trials (the longest search
compared has 12); a search through the model worker compared with Heretic's (two have run on
real GGUF files since, for the transfer measurement, with no oracle); Qwen3-0.6B on
the model (its Heretic search is in the model-free check only); objectives other than
refusals and KL.

## How a prompt becomes token ids

Every prompt the Directions feature measures or scores becomes token ids in one place,
`directions.render_ids` over `wsp.directions_rendering` (plan item F4): measurements with and
without module inputs, scores (with or without steering), the response-prefix detection,
searches, scoring a modified checkpoint, and the comparison scripts. A prompt read as a chat
message is rendered as Heretic renders it in `Model.generate` (64e9991):

1. **Messages.** The system message (the request's `system`, or the prompt's own; none when there
   is neither) and the user turn. When the template has no system role, each system message is
   folded into the next user turn (joined by a blank line) and an empty one is dropped
   (Heretic's `_apply_system_role`). "No system role" is Heretic's probe: a system message
   `HERETIC_SYSTEM_ROLE_PROBE` must survive into a rendering; a template that raises has none.
   The probe runs once per request, the first time a prompt has a system message, with the
   request's `chat_options` (Heretic has none).
2. **Text.** The model's own template with the generation prompt and the date fixed at
   2026-01-01 00:00 UTC (Heretic's `strftime_fixed_date`), then the prefix appended.
3. **Ids.** The text tokenized by the checkpoint's `transformers` tokenizer called with its
   defaults, as Heretic calls it, so the tokenizer's own special tokens are added: a template
   that prints a beginning-of-text token the tokenizer also adds gives two, in both tools.
   Without a `transformers` tokenizer, the vocabulary's encoding with its special tokens.

A prompt not read as a chat message is encoded as before. The Chat and Run tabs and the probes
render with `wsp.probes`, unchanged (the messages as given, today's date, no special tokens
added for a BPE vocabulary).

**The date.** Heretic passes `strftime_now=strftime_fixed_date` to `apply_chat_template`.
`transformers` 5.22 (the IDE's) turns every template argument into plain data first, so a
function arrives as the text `<function>` and a template that reads the date fails (Heretic's
own call would fail the same way under 5.22; under its pinned 5.18 it works). The IDE fixes the
clock instead: while a renderer renders, `transformers`' own `strftime_now` reads 2026-01-01
00:00 UTC in that thread (the module's `datetime`, replaced once by a subclass whose `now()`
answers the fixed date only inside `directions_rendering.fixed_date()`); every other thread,
the Chat tab's included, keeps the real time.

**Every difference found between the two tools' prompt-to-ids paths:**

| Difference | Status |
|---|---|
| (a) Heretic tokenizes the rendered text with the tokenizer's defaults (its special tokens added); the IDE without them | aligned |
| (a') Heretic's tokenizer is the `transformers` one, the IDE's was `tokenizer.json` read by `tokenizers` directly (they can differ where `transformers` rebuilds a tokenizer's parts from `tokenizer_config.json`) | aligned: the same call |
| (b) Heretic renders prompts with a fixed date; the IDE with today's | aligned (by the clock, above) |
| (c) Heretic folds system messages where the template has no system role and drops empty ones; the IDE passed them through (a raising template failed the request) | aligned |
| Heretic's system-role probe and its thinking-block dummy prompt render without the fixed date | the same in the IDE |
| An empty system message is an empty system turn where the template has a system role | the same (was) |
| The response prefix is appended to the rendered text, then the whole is tokenized; an empty one is nothing | the same (was) |
| The IDE can send no system message (`system` null: the template's default system turn appears); Heretic always sends one (`system_prompt`, default `You are a helpful assistant.`) | not aligned, by design: pass `system` to render as Heretic; the Heretic file of such a record is refused |
| `chat_options` (thinking switches) are IDE-only, and reach the system-role probe too | not aligned, by design; with none, the same |
| Heretic batches with left padding (pad = end token when the tokenizer has none) and an attention mask; the IDE runs one prompt at a time | the same ids; comparisons run Heretic at `batch_size = 1` |
| The IDE refuses a rendered prompt longer than 512 tokens or the context; Heretic has no limit | a refusal, not other ids |
| Heretic's interactive chat renders with today's date | not a measured prompt |
| The libraries: Heretic 64e9991 runs `transformers` 5.18.0 and `tokenizers` 0.23.2, the IDE 5.22.2 and 0.23.3 | not controllable; the same ids on every fixture |

**Measurements and records.** A measurement's meta records `rendering`: `{"version": 2,
"same_as_version_1": <bool>}`. Version 1 is how the IDE rendered before (a meta or record
without the entry). `same_as_version_1` is found when the measurement is made, from the
prompts themselves: they are rendered the version-1 way too, and it is true only when every
prompt gives the same ids and no template read the date (so the answer holds on any day); a
version-1 rendering that fails (a template that raises on a system message) is false. Prompts not
read as a chat message are always true. The reproduction record copies the entry with the
measurement.

Records identify prompts by their texts (`prompts_sha256`), the rendering by its options
(`chat`, `system`, `prefix`, `chat_options`), and record token totals, not ids. A re-run from
the recorded means renders nothing, so it is unaffected. Only measurement mode renders again
(`?remeasure=true`, then `reproduce` with `measurement`): there the two renderings must be
compatible, the same version or a version-1 one and a version-2 one with `same_as_version_1`
true, else the re-run is refused with the reason ([Re-running an edit](#re-running-an-edit)). On
a model where nothing differs (Qwen) a record made before is re-run from a new measurement as
before; where the ids would differ it is refused, never re-run silently. The Heretic file of a
record rendered with version 1 states the difference. A search keeps its own guard: it stores
the hash of its scored ids at its first run and refuses more trials when they render
differently, which is what a search scored before this change meets on such a model.

Why this and not another mechanism: the decision needs a model's template and tokenizer, which
exist when a measurement is made, not when a record is read; a marker alone would refuse every
old record, the Qwen ones included, though nothing changed for them; a request switch "render
as before" is not needed, because an old record's own means re-run it exactly, and it would
leave two renderings to keep alive. Version 1 is kept only inside `render_ids` (`_version`),
to make that comparison.

**On a GGUF base.** A GGUF goes through the same renderer: `GGUFRunner` inherits
`chat_prompt` (the GGUF's template, saved in the workspace's `model/`) and `hf_tokenizer` (the
tokenizer `transformers` converted from the GGUF), so (b) and (c) apply and are aligned as on a
checkpoint. (a): for a byte-level (`gpt2`) vocabulary the converted tokenizer adds no special
tokens, even when the GGUF asks for a beginning-of-text token (`tokenizer.ggml.add_bos_token`;
checked on a synthetic GGUF), so the ids hold only what the template prints, where llama.cpp's
own tokenizer would add one more. Heretic has no GGUF path to compare with, the conversion is
`transformers`', and this is left as it is. A SentencePiece (`llama`) GGUF vocabulary was not
checked.

**Checked against Heretic.** On 2026-11-19, on fixtures derived from `tiny-random/qwen2.5` by
changing only tokenizer and template files (`scripts/heretic_compare_prompt_ids.py
make-fixture`: `none`; `bos`, a tokenizer that adds `<|begin_of_text|>` and a template that
prints it; `date`, a template that prints the date in its system turn; `system-raises`;
`system-drops`), `scripts/heretic_prompt_ids.py` recorded the ids Heretic generates from and
`scripts/heretic_compare_prompt_ids.py` compared the IDE's (measurement with module inputs, KL
and refusal scores, prefix detection; per-set system messages, an empty one, a prefix, a suffix
and the response prefix `Sure,`). Bound, written before the runs: ids identical for every prompt
of every set after the change, and every altered fixture differing before it.

| Fixture | Before (prompts equal, of 24) | After | `same_as_version_1` |
|---|---|---|---|
| `none` | 24 | 24 | true |
| `bos` | 0 (one beginning-of-text token where Heretic has two) | 24 | false |
| `date` | 0 (today's date) | 24 | false |
| `system-raises` | 0 (every request failed: `System role not supported`) | 24 | false |
| `system-drops` | 5 (the KL set, whose system message is empty) | 24 | false |

End to end on the `bos` fixture (`scripts/heretic_compare.py`, Heretic's directional ablation,
`row_normalization` none, adapter export): in bfloat16 every bounded stage passes (token ids
identical 5/5/5; the modules equal; the change from Heretic's means 1.37e-7 against 1e-6; the
rounded sums equal in all 6,656 elements; refusals 0 and 0), the means are identical and the KL
scores equal (0.00042721); in float32 (no bounds) the ids are identical, the means within
3.2e-7 and the change within 1.2e-5. With the code before this change the same bfloat16 run
fails B-S1 (0 of 5 identical) and its means differ by 2e-2. Numbers:
`docs/directions-f4-heretic-validation.json`.

On a trained model whose template none of this touches, Qwen2.5-0.5B, the three bfloat16
comparisons of [Runs in bfloat16](#runs-in-bfloat16) were run again on the new renderer on
2026-11-21, against the same Heretic runs: every assertion line is the earlier record's,
number for number (31 lines: token ids, the weight change, the ARA fit, responses, refusals 28
and 0 of 32, KL), its one failure included (`4.85e-4` against `4e-4`). Numbers:
[`directions-rendering-inputs-q05-validation.json`](directions-rendering-inputs-q05-validation.json).

Not checked: a trained model whose template differs from Qwen's (Gemma 2, Llama 3.x), and
Heretic at a batch size above 1.

## Scores: differences from Heretic

| Difference | Reason | Effect |
|---|---|---|
| In float32 the IDE loads `transformers` models with eager attention, Heretic with the default (sdpa); in bfloat16 and float16 both load the default ([Runs in bfloat16](#runs-in-bfloat16)) | the attention-pattern hooks need eager (a bfloat16 pattern run switches to eager for its own pass) | float32: logits differ at rounding level (about `1e-6`); a greedy response can flip at a near-tie |
| Heretic scores a trial with its LoRA adapter attached; the IDE adds the pieces to the weights | the IDE's one patch form | rounding level |
| KL summed per prompt, then in Python floats; Heretic one float32 `batchmean` | bounded memory, per-prompt values | about 1e-6 relative |
| Heretic fixes the date by passing `strftime_now` to the template; the IDE fixes the clock transformers' own `strftime_now` reads, in the rendering thread ([How a prompt becomes token ids](#how-a-prompt-becomes-token-ids)) | transformers 5.22 turns a function passed to a template into the text `<function>` | none: the same date, 2026-01-01 UTC (the dummy prompt and the system-role probe without it, in both) |
| Heretic evaluates in batches with left padding; the IDE one prompt at a time | the IDE's admission model | compare with Heretic at `batch_size = 1` |
| Score settings are inline in each request entry, not in TOML tables | one request, no global state | names and defaults identical |
| GGUF runners do plain argmax and apply only `eos_token_id` and `repetition_penalty` from the workspace's `generation_config.json` (other greedy keys refused) | a GGUF carries no generation config; the file is copied into the workspace's `model/` by hand | Heretic cannot run GGUF; `generation` says what was applied |

### Comparing the scores with Heretic

`scripts/heretic_dump.py` (Heretic's interpreter) now also writes `dump/scores.json` and
`dump/scores.safetensors` from Heretic's own classes: the dummy prompt, the prefix-test
responses without and with the run's prefix, and on the base model and on the run's merged
model the responses (texts and ids), flags and scores of `KeywordRate`, `ResponseLength` (on
the KL prompts) and `KLDivergence`, plus the generation config. `scripts/heretic_compare_scores.py`
(the IDE's interpreter, under `python -m wsp.memory_guard`) compares stage by stage, with
acceptance decided beforehand: S0 the base first-token logits (`eps`), S1 token ids identical,
S2 the detected prefix identical, S3 responses identical or diverging only where the margin
between the two tokens is below `max(1e-3, 4 eps)`, S4 flags and scores, S5 the merged model
(KL within `1e-6 + max(1e-3 KL, 3 eps sqrt(2 KL))`), S6 every trial of an ablation run as an
uncommitted candidate.

On 2026-11-18 both were run on three checkpoints, both tools in float32 on the CPU, Heretic
at `batch_size = 1`. Numbers:
[`directions-scores-heretic-validation.json`](directions-scores-heretic-validation.json).

| Stage | `tiny-random/qwen2.5` | `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen/Qwen3-0.6B` (a thinking model) |
|---|---|---|---|
| Heretic runs scored | ablation `full`, ARA | ablation `full` and `none`, ARA | ARA |
| Prompts per score | 5 | 32 | 32 |
| S0: base first-token logits, largest difference | 1.5e-7 | 7.9e-5 | 6.3e-5 |
| S1: token ids | identical | identical | identical, with and without the prefix |
| S2: detected prefix | none in both | none in both | `<think></think>\nOkay,` in both |
| S3: responses on the base model | identical | identical | identical |
| S4: base refusals; mean length | 0/5; 37.4 | 30/32; 133.53125 | 3/32; 143.9375 |
| S5: Heretic's merged model, responses | identical | identical | identical |
| S5: refusals, in both tools | 0/5 and 0/5 | 28/32, 28/32 and 0/32 | 0/32 |
| S5: KL, IDE / Heretic | 0.00074194 / 0.00074211; 9.9e-7 / 1.04e-6 | 0.0132515 / 0.0132513; 0.0133850 / 0.0133850; 0.0877550 / 0.0877554 | 0.1809987 / 0.1809974 |
| S6: each trial as an uncommitted candidate | 2 of 2 equal | 4 of 4 equal, in both ablation runs | not applicable (ARA) |

Every stage passed its acceptance on every one of these float32 runs (in bfloat16 on the 4B
hybrid model S6 failed against Heretic under the same torch version: see
[Compared with Heretic on Qwen3.5-4B](#compared-with-heretic-on-qwen35-4b)). No response diverged anywhere (the
prefix-test, refusal and length responses, on the base and on the merged models), so the
near-tie rule of S3 was never needed, and the refusal counts and mean lengths are equal
exactly. On Qwen3-0.6B the detected prefix ends in the first word of an answer, which is why
its base model counts only 3 refusals in 32: that is Heretic's detection, reproduced.

**The IDE alone, end to end.** The table above scores Heretic's weights. For the exported
trial of three runs the IDE also did everything itself: its own measurement, the trial's
parameters as a script, scored as an uncommitted candidate.

| Model, method | The IDE's own edit | Heretic's exported trial |
|---|---|---|
| Qwen2.5-0.5B, directional ablation `full` | 28/32 refusals, KL 0.0132515 | 28/32, KL 0.0132516 |
| Qwen2.5-0.5B, ARA | 0/32, KL 0.0902 | 0/32, KL 0.0878 |
| Qwen3-0.6B, ARA | 0/32, KL 0.0931 | 0/32, KL 0.1810 |

After each, no tensor was touched and the workspace version was unchanged. Scoring the base
state (64 responses of at most 30 tokens, 32 logit passes) took 53 s on the 0.5B model and
86 s on Qwen3-0.6B; a candidate, including running its script, 47 to 61 s and 122 s; the
measurement 9 s and 18 s.

**The Qwen3-0.6B ARA trial** has the same refusal count at half the KL. This is the
instability of the fit described in "ARA against Heretic", at full size: the trial fits 44
matrices over 22 layers with a steering weight of 0.93 (the 0.5B trial: 8 matrices, 0.00066).
What was checked:

- The inputs agree: token ids identical for all 160 prompts (with the prefix), mean
  residuals to 6.2e-7, the captured inputs and outputs of all 56 components to 5.3e-6.
- Under Heretic's torch (2.14.1) from Heretic's captures, the IDE's fit and Heretic's loop
  both give Heretic's merged weights in every one of the 115,343,360 elements of the 44
  matrices (`scripts/heretic_fit_check.py`, no model loaded).
- Under the IDE's torch (2.15.0), from the same captures, Heretic's own loop lands 9% to
  213% (median 26%) from its own export, per matrix. The IDE's fit equals that loop element
  for element.
- Three fits from inputs that are equal to rounding give three KL values: 0.0931 (the IDE,
  its own captures), 0.1269 (the IDE, Heretic's captures, torch 2.15.0), 0.1810 (Heretic).
  The objective they reach is the same: summed over the 44 matrices −71.4, −69.5 and −69.8,
  from 277.9 at the start; per matrix the reduction differs from Heretic's by about 0.5% in
  the median and at most 7%.

So with these parameters the loss does not pin the KL down: fits that are equally good by
the objective differ by a factor of two in what they do to harmless prompts. A trial's
scores are reproducible only from the same captures under the same torch version, in either
tool. The search (plan item E) and the reproduction record (H3) therefore have to keep the
library versions with every trial. The row rescaling the loss is evaluated with, which is
not stored, went down to 0.62 on this model (layer 21 `o_proj`, among the 16 matrices whose
lines were kept).

Not covered in this comparison: GGUF runners and a real model through the model worker (run
since without a bit-level oracle, see [Checked on real GGUF files](#checked-on-real-gguf-files));
the classifier and benchmark scorers; the three rendering differences of the table above (none arises on Qwen
tokenizers and templates); Heretic at a batch size above 1; a response that diverges at a
near-tie (none did); the refusal count of the fit made from Heretic's captures under torch
2.15.0 (its KL only).

## Steering along a direction

A saved measurement can also be used at run time, without changing a weight: a steering
entry `direction:<name>@<k>` adds a multiple of the measurement's direction at index `k` to
the residual stream while the model runs. It is one more kind of entry in the IDE's steering
([STEERING.md](STEERING.md#the-steering-specification)), so it goes wherever a steering
specification goes: `POST /api/run` and `/api/chat` (the edited reply), `/api/probe/reads`
and `/api/probe/next`, `POST /api/steer/resolve`, and the scores
([`POST /api/directions/scores`](#post-apidirectionsscores)). It may be mixed with neuron,
MLP and head entries in one specification. Heretic has no steering. Code:
`_direction_entry`, `resid_hookable` and `add_vector` in `wsp/steering.py`; tests:
`tests/test_steering_directions.py`.

```json
{"steer": [{"addr": "direction:refusal@12", "mode": "add", "dose": -1, "unit": "diff",
            "orthogonalize_direction": true, "means_sha256": "5f0c...(64 hexadecimal digits)"}]}
```

| Field | Values |
|---|---|
| `addr` | `direction:<name>@<k>`: `<name>` a measurement name ([What is stored](#what-is-stored)), `<k>` ASCII digits without a leading zero |
| `mode` | `add`, required; nothing else |
| `dose` | as for every entry: a finite number, magnitude at most 1e6 |
| `unit` | `diff` (the default; absent or `null`) or `raw` |
| `orthogonalize_direction` | `true` (the default) or `false` |
| `means_sha256` | optional: the SHA-256 of the measurement's means file, 64 lowercase hexadecimal digits |

The last two fields are accepted on direction entries only; on a neuron, MLP or head entry
they are an unknown field, as before.

**Where.** The vector is added to the residual stream entering layer `k`,
`blocks.k.hook_resid_pre`, at every position: the point where the measurement read its index
`k` (the table in [What is stored](#what-is-stored)). So `k` runs from 0 to `n - 1`; index `n`,
after the final norm, is read by no layer and is refused. The index counts as in the
measurement, not as `direction_index` does: an ablation with `direction_index: null` uses at
layer `l` the row `l + 1` (what layer `l` writes), while `direction:<name>@<k>` pushes what layer
`k` reads, which is what layer `k - 1` wrote.

**Which vector.** `v = table(good_mean, bad_mean, orthogonalize_direction)[k]`, the unit
direction of [Directions](#directions) at index `k`: the row the Layers view marks usable and,
with the projection, the one the script function `direction(name, k)` returns. With `d = bad_mean[k] - good_mean[k]`, in float64:

    diff:  c = dose * (d · v) * v        one unit is the separation of the two means along v
    raw:   c = dose * v                  one unit is the unit vector

With `orthogonalize_direction` (the default), `d · v` is the norm of the part of the difference
orthogonal to the good mean; without it, `‖d‖`. A negative dose pushes towards the good mean
(removes the behaviour); a positive dose pushes towards the bad mean (adds it). `unit: "resid"`
(the neurons' unit, a multiple of the layer's residual norm) is not offered for a direction, so
no direction entry needs a calibration.

**Several entries.** One entry steers one index. Several entries at one layer are summed in
float64 and the sum is cast to float32 once; entries at other layers make a push per layer, so
a span of layers is several entries. A direction's push at a layer composes with that layer's
neuron, MLP and head entries, which act at other points of the layer.

**The pin.** The means can change: measuring the same name again with `replace` writes new
ones. `GET /api/directions/{name}` answers the means file's SHA-256 as `means_sha256`; an entry
that carries it is refused when the file's hash is another one now, and an entry without it
steers along whatever the measurement holds. Every report of a direction entry names the
hashes it used (`means_sha256`, `meta_sha256`).

**Resolved on the server.** The server reads the measurement (only its files; no model is
needed) and turns the entries into the wire form's `add_vector`, `d_model` numbers per layer
([STEERING.md, "The wire form"](STEERING.md#the-wire-form)). The model worker gets only those
numbers, checks them again, and adds them with a forward hook at the layer's input
([STEERING.md, "Where steering runs"](STEERING.md#where-steering-runs)). The float32 vector is
cast to the stream's dtype and device and added. On GGUF it is an activation change in the
runner; no weight is read or dequantized for it.

**Report.** `/api/steer/resolve` describes a direction entry as `{index, addr, mode, dose,
kind: "direction", unit, layer, direction, measurement_index, orthogonalize_direction, scale,
norm, means_sha256, meta_sha256, value}`: `layer` and `measurement_index` are both `k`, `scale`
is `d · v` for `diff` and 1 for `raw`, `norm` is the length of this entry's own vector `c`
(before the sum with other entries), `value` is the dose. The answer's `directions` counts the
direction entries; `layers` counts each steered layer once. The steering report of a Run or
Chat reply and of steered scores carries the same entries and the same count.

**Refused**, each with the entry named (`steer entry 1 (direction:refusal@12): ...`), as a 400
`SteeringError` before any model is loaded; the checks run in this order:

1. The fields: an unknown field (`unknown field scale`); a `mode` that is not `add`, or none:
   `a direction is added: mode must be 'add'`; a missing or bad dose, with the texts of every
   entry (`dose is missing`, `dose 1e+07 is larger than 1e+06`); `unit 'resid' is not available
   for a direction; use 'diff' (the difference of the two means along the direction, the
   default) or 'raw' (the unit direction times the dose)`; any other unit: `unit must be 'diff'
   or 'raw' for a direction`; `orthogonalize_direction must be true or false`; a
   `means_sha256` that is not 64 lowercase hexadecimal digits (`null` too): `means_sha256 must
   be the 64 hexadecimal digits of the measurement's means file`.
2. The address: `a direction is written direction:<measurement>@<index>, for example
   direction:refusal@12` (a name outside the measurement names' characters, a leading zero,
   no `@`).
3. The measurement: `no direction named 'refusal'`; a name a measurement cannot have gets the
   measurements' own name rule; files that cannot be read get the measurement's own text; a
   measurement of another model shape: `direction 'refusal' was measured on a model with 24
   layers and d_model 896; this model has 28 layers and d_model 1024`; a pin that differs:
   `measurement 'refusal' changed since this entry was made: its means are <hash now>… now
   and the entry names <pin>…; add the entry again from the Layers view, or drop
   means_sha256, to steer along the new measurement`, where `<hash now>` and `<pin>` are the
   first 12 digits of each hash.
4. The index and the hook point: `index 24 is the residual stream after the final norm, which
   no layer reads; steer at an index from 0 to 23`; `index 30 does not exist (the measurement
   has indices 0 to 24)`; `layer 12 has no hook point at its input (hook_resid_pre), so a
   direction cannot be added there` (a checkpoint layer whose structure names no decoder-layer
   module).
5. The row: `index 12 of direction 'refusal' is unusable: the good and bad means do not differ
   there (after the projection, when it is asked for)`.

After every entry is read, each layer's summed push is checked once: a component above 1e9 in
magnitude (or not finite) is refused as `the push at layer 12 has a component of 1e+10; at
most 1e+09`, named by the last entry at that layer. The base model is never steered, on every
route and every runner: a steering request for the base variant is refused with `the base model
is never steered; steer the current variant`, and a base-only chat request with `steering
applies to the edited model only, and this request runs the base model only`
([STEERING.md](STEERING.md#steering-a-run-or-chat-reply)).

**Steered scores.** `POST /api/directions/scores` with `steer` scores the current weights, or
a candidate on top of them, as the steered model behaves: every response is generated with the
push, and every KL row is the steered first-token distribution. The KL reference stays the
unsteered base model, so KL measures how far the steered state is from the base model the
reference was computed on; the cached reference is the one an unsteered request uses. State
`base` with `steer` is refused (the base is never steered). Details in the route's paragraph
under [`POST /api/directions/scores`](#post-apidirectionsscores).

**What it does not do.** No weight, patch, `workspace.json` or model file changes, and nothing
appears in the patch stack; to keep a change, apply an ablation instead. There is no run-time
projection (removing the direction from the stream at every layer, directional ablation without
weights): a direction entry only adds. One index per entry: no weight kernel over layers as an
ablation's weight distributions have. The automatic search does not steer. The Run tab's
steering panel and the Layers view do not offer direction entries yet; a request carries them.

**Costs.** Each request reads the measurement's two means (`2 (n + 1) d` float32 values) and
hashes its means file once per direction entry; the push costs one addition of `d` values per
position per steered layer. With a model worker, each `generate` or `next_logits` job carries
`d_model` numbers per steered layer in its JSON arguments.

**Checked**, model-free, in `tests/test_steering_directions.py`: the wire holds the vector
computed in the test from the saved means (both units, with and without the projection, the
sign of the dose); on this project's tiny model the stream entering layer `k` is the unsteered
one plus the float32 vector at every position and the layers before it are unchanged; on a tiny
random `transformers` model a pre-hook on decoder layer `k` sees the pushed input and steered
generation equals a greedy loop written in the test; entries add up and compose with a neuron
push; every refusal text, through `/api/run`, `/api/chat`, `/api/probe/next` and
`/api/steer/resolve`, before any model is built; the worker's checks of `add_vector`; the stub
model worker's job carries the numbers and no measurement name and gives the in-process tokens;
a changed measurement is noticed through the pin; the report's fields and hashes; steered scores
against the test's own greedy loop and float64 KL; the base never steered; and weights, patches,
`workspace.json` and the version unchanged after steered Run, Chat, Probe and scores.

**On a real model** (2026-11-18). `scripts/steer_direction_check.py` on Qwen2.5-0.5B-Instruct,
float32 on the CPU through `transformers`, offline, in an isolated root under
`python -m wsp.memory_guard` (footprint 2.4 GiB, no violation). Heretic has no steering, so
this is a behaviour check, not a comparison: the prompts, system prompt and scorer settings are
those of a finished Heretic ablation run on this model (64 + 64 measured, 32 refusal prompts,
32 harmless prompts, 30 new tokens). The direction "refusal" is measured once; the layer is
the index in the middle half of the model with the largest difference of the means relative to
their norms (index 18 of 24, unit `diff` scale 12.15). Every number is in
[`directions-steering-check.json`](directions-steering-check.json).

| Steering | Refusals | Harmless prompts refused | KL on the harmless prompts | Mean length |
|---|---|---|---|---|
| none | 30/32 | 2/32 | 0 | 133.5 |
| `steer: []` | 30/32 | 2/32 | 0 | 133.5 |
| `direction:refusal@18`, dose -0.5 | 23/32 | 0/32 | 0.198 | 132.8 |
| dose -1 | 1/32 | 0/32 | 0.683 | 132.1 |
| dose -2 | 0/32 | 0/32 | 2.385 | 134.3 |
| dose +1 | 32/32 | 27/32 | 2.120 | 138.9 |
| indices 17, 18 and 19, -0.5 each (no criterion) | 0/32 | 0/32 | 1.168 | 133.1 |

The three criteria were written before any run:

- B1, refusals down by at least 8 of 32 at a KL of at most 0.3 for one of the doses -0.5, -1,
  -2: **failed**. Dose -0.5 keeps the KL at 0.198 and removes 7 refusals, one short; dose -1
  removes 29 at a KL of 0.683; dose -2 removes all 30 at 2.385. Neither the doses nor the bound
  were changed. What it shows: on this model, with this layer rule and these three doses, a
  push at one layer moves the refusals and the next-token distribution of harmless prompts
  together, and no tried dose reaches the pair the criterion asked for. For scale, on the
  same prompts the exported trial of that Heretic ablation run scored 28/32 at a KL of 0.013
  and the ARA run 0/32 at 0.088 (see
  [Scoring a modified checkpoint](#scoring-a-modified-checkpoint-heretics-evaluate_model)).
- B2, harmless prompts refused up by at least 4 of 32 at dose +1: passed (2 to 27).
- B3, nothing changed: passed. The workspace version and the patch list are the same before and
  after, the snapshot's weight file has the same SHA-256, and the answer with `steer: []`
  equals the unsteered one.

The check ran twice. The first run's script asserted the version to be 0; a freshly opened
workspace is at version 1 (1 before and 1 after), so the invariant was corrected to
"unchanged" and the check run again from the start. Every count, KL, length, response and
means hash of the second run equals the first's.

**Not checked.** One model, one measurement, one layer rule, float32 on the CPU. A real GGUF
file has not been steered (the wire is checked on the stub worker only), the worker protocol's
size limits for `add_vector` have not been read, and steered scores have not been run on GGUF
or in bfloat16.

**Known limits.** The means' hash is read separately from the means, so a measurement replaced
between the two reads could pair one hash with the other means. The steered KL rows go through
a full forward pass of the hooked model (the last row of `run`), or `next_logits` where the
runner has it, while unsteered KL rows use the first-token path; on `transformers` the two may
differ slightly, so a steered KL at dose 0 need not be exactly the unsteered one.

## A LoRA adapter (PEFT)

Heretic offers "Export the LoRA only (can be merged later)" for a trial: a PEFT adapter, which
`PeftModel.from_pretrained(base, directory)` loads on top of the unchanged base model. The IDE
writes the same kind of adapter from the workspace's patches, whatever made them (a direction
ablation, an ARA fit, a committed search trial, a hand edit), with `POST /api/export_adapter`
or `wsp.adapter_export.export(ws, name, ...)`. It writes the two PEFT files itself and does not
import `peft`: the format is two small files, and `peft` would need the base model loaded only to
write them. Code: `wsp/adapter_export.py`.

**What is exported.** By default every enabled patch at its strength, in stack order, so the
adapter turns the base weights into the current ones, as `GET /api/export` does for a patch
file. Or a list of patch ids, each at its stored strength; a switched-off patch may be listed
(it is included, with a note), a patch of strength 0 may not. Only the tensors the patches
change become target modules (Heretic's adapter lists every attention and MLP output module of
every layer, with zero factors where nothing changed). Each changed tensor must be a linear
layer's weight matrix, decided per role of the tensor's parts (`TARGETS`):

| Role | Exported | Refused as |
|---|---|---|
| `W_Q`, `W_K`, `W_V`, `W_in`, `W_gate`, `W_up` | when stored `[out, in]` (the residual axis is axis 1) | a Conv1D weight stored `[in, out]` (GPT-2): PEFT needs `fan_in_fan_out`, which this export does not write |
| `W_O`, `W_out` | when the residual axis is axis 0 | the same |
| `W_U` | when not tied to the embedding | the unembedding tied to the token embedding |
| `W_E`, `W_pos` | never | the token embedding (PEFT's embedding adapters are not written) / a position embedding |
| norms, biases, tensors outside the recognised structure | never | a normalisation weight / a bias / not a matrix the IDE recognises |

The module name is the tensor's name without `.weight` (for dense Llama and Qwen2 checkpoints
that is the `transformers` module path). A fused tensor such as `qkv_proj` is one module. The
acceptance against PEFT covers the model type `qwen2`; any other model type is exported with
`validated_family: false` and a note, because PEFT refuses an adapter whose module names do
not exist in the model it is loaded on.

**From pieces to factors.** PEFT's change of a module is `lora_B @ lora_A * (lora_alpha / r)`;
the IDE writes `lora_alpha = r` for every module, so the scaling is 1. Every piece kind is
turned into factors exactly, without an SVD or a tolerance (`FACTORS`), for a piece on rows
`rows` and columns `cols` of `W [out, in]` at strength `s`:

| Piece | Rank in the adapter | `lora_B [out, k]` | `lora_A [k, in]` |
|---|---|---|---|
| low-rank `A [m, r] @ B [r, n]` | `r` | rows `rows` = `s · A` | columns `cols` = `B` |
| block `values [m, n]`, `m <= n` | `m` | one-hot: `lora_B[rows[i], i] = s` | row `i`, columns `cols` = `values[i]` |
| block, `m > n` | `n` | rows `rows` = `s · values` | one-hot: `lora_A[j, cols[j]] = 1` |
| sparse | `min(rows, columns)` | first the box of its distinct rows and columns, then the block rule | |

A product with one-hot factors is one value plus exact zeros, so a block or sparse change comes
back bit for bit. The pieces of one tensor, from one patch or several, are concatenated along
the rank axis in stack order, so the module's rank is the sum of its pieces' ranks; it must not
exceed `min(out, in)`. The strength is folded into `lora_B`: for a strength that is a power of
two (1, 0.5, -2, ...) that is exact; for any other strength each factor value is rounded once
where the IDE rounds the scaled product instead, and the module says `exact: false`. The IDE
stores a directional ablation as `A = -w (V v)`, `B = v` where Heretic writes `lora_B = -w v`:
the products agree, the factors do not (ARA's factors are Heretic's own).

"Exact" is meant per piece: each piece's product `lora_B @ lora_A` over its own rank columns
is the IDE's change of that piece bit for bit (at a power-of-two strength). When one module
holds several pieces, PEFT forms the sum of their products in one matrix product, in another
order than the IDE adds them, so the module's change is bit-equal only when every product and
partial sum is exactly representable (as in the tests, which use small integers); otherwise it
agrees to float32 rounding.

**Rank layout** (`layout`): `uniform` pads every module with exact zeros to the largest rank
`r` and writes no patterns (Heretic's layout); `per_module` keeps each module's rank and writes
`rank_pattern` and `alpha_pattern` (equal, scaling 1) for the modules whose rank is not `r`,
each key the module name with `re.escape`, because PEFT reads the keys as regular expressions;
`auto` (the default) is `uniform` when every module has the same rank (any single-direction
patch) and `per_module` otherwise, so a mixed export (an ARA fit of rank 50 beside a rank-1
patch) is not padded to 50 everywhere. The keys are written in module-name order.

PEFT matches a key on the end of a module's name: it gives a module the rank of the first key
`k` with `re.match(rf"(.*\.)?({k})$", module)`, so an escaped key `a` also matches every module
whose name ends in `.a`. When a target `m` ends in `.a` for a target `a` that has a key and
another rank (`pattern_collision`), PEFT would give `m` the wrong rank: `auto` then writes the
`uniform` layout, with the note `module names <a> and <m> end alike and have different ranks,
...`, and `per_module` is refused (`layout 'per_module' cannot be written: PEFT matches a rank
pattern on the end of a module's name, and <m> ends in .<a> with another rank; use 'uniform' or
'auto'`). The same rule decides which modules PEFT attaches an adapter to: a checkpoint module
that is not a target but whose 2-D weight's module name ends in `.<target>` (`also_matched`)
gets an empty adapter too (its `lora_B` is PEFT's zero initialisation, so it changes nothing)
and PEFT warns that its weights are missing. The export is allowed, with one note per such
module (at most 5, then `and <n> more`). The IDE writes the full module names; PEFT itself
shortens a long target list to the shortest endings that match the same modules when it loads
the file (the target list of Heretic's own adapter for a 24-layer model is `["o_proj",
"down_proj"]`).

**The files**, in `<workspace>/export/adapters/<name>/` (`name` as a measurement's name),
written in a hidden temporary directory beside it and moved in place; a failure leaves nothing,
and with `replace` the old adapter stays until the new one is in place. (`POST
/api/export_checkpoint` writes the whole checkpoint to `<workspace>/export/<name>`, so it
refuses the name `adapters`, before writing anything.)

- `adapter_config.json`: `json.dumps(indent=2, sort_keys=True)` with no trailing newline, as PEFT
  writes it, with an explicit minimal set of keys: every key that changes the forward pass
  (`r`, `lora_alpha`, `target_modules` sorted, `rank_pattern`, `alpha_pattern`, `bias: "none"`,
  `lora_dropout: 0`, `fan_in_fan_out: false`, `use_dora`, `use_rslora`, `lora_bias`: false, ...)
  plus `peft_type: "LORA"`, `task_type: "CAUSAL_LM"`, `base_model_name_or_path`, `revision`. Not
  PEFT's full field dump, which ties the file to one PEFT version; `peft_version` is filled in by
  PEFT on load.
- `adapter_model.safetensors`: float32, metadata `{"format": "pt"}`, keys sorted,
  `base_model.model.<module>.lora_A.weight` `[r, in]` and `.lora_B.weight` `[out, r]`, written one
  module at a time.
- `wsp_adapter.json` (PEFT ignores it): where the adapter came from: the patches with their
  piece digests, whether each has a reproduction record, the base's repository, commit and files
  (with their SHA-256 when the record capture has cached it; the export never hashes the base),
  the layout, per module its rank, `exact` and pieces, the sizes and SHA-256 of the two PEFT
  files, and the notes.

**`base_model_name_or_path`**: the request's value; else the workspace's Hub repository id, with
the Hub commit as `revision`; else `null`, with the note that no Hub repository is known (pass a
name, or load the base yourself and give `PeftModel.from_pretrained` the directory). A local path
is never written: Heretic writes the path it loaded, which for a local model is a path of the
machine the file was made on.

**A GGUF base is refused.** The patches were computed against the GGUF's reconstructed weights,
so an adapter for the floating-point checkpoint the GGUF came from would not be the edit the IDE
tested, and the dequantized model it would apply to exactly is never built (see
[QUANTIZED.md](QUANTIZED.md)). The faithful export is the WSP companion bundle (`GET
/api/export`); or apply the edit to the floating-point checkpoint. One of this project's own
models is refused too (`GET /api/export` writes the whole edited model). The export never reads
a base weight, never writes a weight, a patch file or `workspace.json`, and never dequantizes.

**The route.** `POST /api/export_adapter` with `name` (required unless `dry_run`), `patches`
(null or a list of ids), `layout` (`"auto"`), `base_model_name_or_path` (null or 1..256
characters without control characters), `replace` (false), `dry_run` (false); any other field is
refused with the list. The answer, also for a dry run (`written: false`, no `dir`, `files: []`):
`dir`, `files` (name, bytes, SHA-256 of each of the three), `layout`, `r`, `rank_pattern`,
`bytes` (the safetensors size), `modules`, `patches`, `base_model_name_or_path`, `revision`,
`family`, `validated_family`, `notes`. The notes: no Hub repository is known; a patch is switched
off and included; a patch was made on top of patches this adapter does not include (from its
reproduction record, when it has one); a strength that is not a power of two; a model type the
acceptance has not loaded; module names that end alike with different ranks (the uniform layout
is used); a module PEFT will attach an empty adapter to.

Refusals (400, `AdapterError: ...`; an unknown patch id is 404):

| When | Says |
|---|---|
| one of this project's own models | use the existing export (`GET /api/export`) |
| a GGUF base | the paragraph above, naming the companion bundle |
| a checkpoint with packed or scaled (non-float) weights | which dtype in which tensor; nothing the IDE does not decode is exported |
| nothing selected / an empty list / a listed patch of strength 0 | `nothing to export: ...` / `patch <id> has strength 0: it changes nothing` |
| a changed tensor that is not a linear layer's weight | `patch <id> changes <tensor>, <reason>: a LoRA adapter holds changes of linear layers' weight matrices only` |
| a module above rank `min(out, in)` | export fewer patches at once |
| a sparse piece spread over a box above 16,000,000 weights | too wide to factor exactly |
| not enough disk (the safetensors size plus 1 MiB, keeping 2 GiB free) | the bytes needed, free and kept free |
| an adapter of that name exists, without `replace` | pass replace, or choose another name |
| `layout: "per_module"` with module names PEFT's rank patterns cannot tell apart | use `uniform` or `auto` |
| a bad `layout`, `base_model_name_or_path`, `name` or `patches` | the rule it breaks |

Checked by `tests/test_adapter_export.py` (model-free: the llama-named fake checkpoint as an `hf`
workspace): the files read back and the change rebuilt with PEFT's own rank and alpha rule
written out in the test, against the workspace's delta, exactly for block, sparse and
power-of-two strengths. Loading the files with PEFT and comparing them with Heretic's own
adapter loads models, so the suite does not do it: see "Compared with Heretic" under
[the mapping to Heretic's `reproduce.json`](#the-mapping-to-heretics-reproducejson).

## The reproduction record

Heretic writes `reproduce.json` beside an uploaded model: the settings, the trial's parameters
and scores, the library versions and the hashes of the weights, so that `heretic --reproduce`
can make the same model again. The IDE keeps that knowledge for every committed edit whose
script read a measurement, through every edit path (the Directions tab, the weight-script bar,
the agent's edit tool, a committed search trial, the command line: they all end in
`Session.apply`). Code: `wsp/directions_record.py`.

**When.** The interpreter lists each successful call of `ablate_direction`, `ablate_ara`,
`direction` and `direction_means` (`run_script(...)["calls"]`: the function, the method, the
normalized parameters, the measurement's hashes, and for the two ablations what was stored).
When the list is not empty and the edit is not a dry run, `Session.apply` prepares the record
before the commit (reading the measurements' prompts and means, the weights it was made on, the
base, the libraries; nothing is written yet) and writes it after a commit. A dry run, an edit
the scope or a check refuses, and a script that read no measurement write nothing. A record
that cannot be prepared or written never undoes or fails the commit: the report says
`record_error: "<type>: <message>"`; otherwise it says `record: {"version": 1, "calls": <n>,
"digest": <the pieces' digest>}`.

**The files**, in `<workspace>/directions/records/`:

- `<patch id>.json`: the record, written atomically.
- `prompts/<sha>.json`: the measurement's prompt file as it was, `{"good": [...], "bad": [...]}`,
  named by the SHA-256 of `json.dumps(prompts, ensure_ascii=False, separators=(",", ":"),
  sort_keys=True)`; written once.
- `means/<sha>.safetensors`: a byte copy of the measurement's means file, named by its SHA-256;
  written once. A measurement measured again later or deleted leaves the record and its
  snapshots as they were. An ARA measurement's module inputs and outputs are not copied (they
  can be gigabytes): `io_sha256` (the hash of their `io.json`) identifies them.
- `base.json`: `{"files": [[name, bytes, mtime_ns, sha256], ...]}`, the base files' hashes.

Snapshots are never removed (a few KB to a few MB each).

**The record** (version 1):

| Field | Holds |
|---|---|
| `version` | `1` |
| `patch` | `{id, created, message, author}` of the committed patch |
| `created` | when the record was written |
| `script`, `script_sha256` | the script as committed, and the SHA-256 of its UTF-8 |
| `rebuildable` | every call is `ablate_direction` / `ablate_ara` and the script (stripped) is exactly their one-line scripts, so it can be rebuilt for another measurement |
| `calls` | each call as the interpreter listed it, plus `heretic` (`modifier` `Abliteration` or `ARA`, Heretic's trial `parameters` with ARA's open `end_layer_index` resolved to the number of layers, and the modifier `settings`) and `seed` (ARA's, else null); `null` both for the two reads |
| `calls[].measurement` | `name`, `meta_sha256`, `means_sha256`, `io_sha256`, `prompts_sha256`, `created`, `counts`; from the meta `variant`, `tokens`, `options`, `datasets`, `module_io`, `model`, `environment`, `rendering`, `measured_on` (null when an older meta has none); `prompts_snapshot`, `means_snapshot` |
| `pieces` | `{digest, stored, tensors}`: the pieces' digest (as a search trial's), the values stored, the tensors changed |
| `made_on` | the enabled patches before the commit, `[[id, strength, created, stored], ...]` |
| `base` | `{kind, name, repo, sha, base_sha256, files: [{name, bytes, mtime_ns, sha256}], hashed, hash_seconds}` (below) |
| `environment` | the libraries, sampler, Python, thread count, machine type and system (as a search records them), the installed versions of torch, transformers, safetensors, numpy, tokenizers, optuna and pyarrow, `platform.platform()`, the Python implementation, the loaded runner, and `code_sha256`: the SHA-256 of `wsp/directions.py`, `wsp/directions_ara.py`, `wsp/wscript.py`, `wsp/delta.py`, `wsp/workspace.py` |
| `origin` | the patch's `source.origin` (a search trial's, for example), or null |
| `search` | for a committed search trial: the search's name, id, counts, seed, sampler, objectives, rendering, `max_response_length`, its scorers (plugin, instance, optimization, settings, the prompts' SHA-256 and dataset records; no prompt texts), the finished trials, and the trial's index, values, displays and scores; else null |
| `heretic` | the Heretic commit the semantics are Heretic's at (`64e9991`) |

A record belongs to its patch while `patch.id` and `patch.created` are the patch's. A patch id
is the patch count and the time modulo 100,000 seconds, so it can come back after a deletion;
the old record is then stale (answered as `record: null, stale: true`), and the next direction
edit committed with that id overwrites it.

**The base, by content.** A GGUF base is identified by its `base_sha256` (computed when it is
opened). For a checkpoint, each local weight file is hashed with SHA-256, one MiB at a time,
and the hash is cached in `base.json` by file name, size and modification time; a file whose
size or time differs from its cache entry is hashed again. Weights still on the Hub have
`sha256: null` and are identified by `repo` and `sha`. Hashing takes about a second per GB, so
inside a commit it is done only when the local files hold at most `HASH_AT_COMMIT_MAX_BYTES`
(8 GiB): the first direction edit on a workspace pays it once. Above that nothing is hashed at
commit time; the record still holds each file's name, size and modification time, with
`sha256: null` and `hashed: false`, and a hash computed later (by a re-run) is the file's hash
at commit time exactly when the file still has the recorded size and time. `hash_seconds` is
the time spent hashing in that commit.

### `GET /api/patches/{pid}/record`

The record and what changed since. It never hashes a weight file and writes nothing. 404 `no
patch '<pid>'`; a patch without a record answers 200 with `record: null` (`stale` as above) and
the rest null.

```json
{"record": {...}, "stale": false,
 "checks": [{"key": "base", "then": {...}, "now": {...}, "same": true, "effect": "refuse", "message": null},
            {"key": "made_on", "then": [], "now": [["0002-41234", 1.0, 1794920600.12, 912]], "same": false, "effect": "refuse",
             "message": "the enabled patches are not those patch 0002-41234 was made on (then: none; now: 0002-41234). Switch patch 0002-41234 off (and the others back as they were) to re-run it on the weights it was made on"},
            {"key": "measurement:refusal", "then": {...hashes...}, "now": {...}, "same": true, "effect": "refuse", "message": null}],
 "environment": [{"name": "torch", "then": "2.8.0", "now": "2.8.0", "severity": "high", "same": true}],
 "remeasure": null, "remeasure_unavailable": null, "heretic": null}
```

- `record.base.files[]` gains `sha256_from`: `"commit"`, `"later, file unchanged since"` (a null
  hash filled from the cache because the file still has the recorded size and time), or null.
- `checks` are the rows a re-run is refused on when they differ. `base`: the kind and
  `base_sha256`, or each file's name, size and hash (repository and commit for Hub files):
  `the base weights are not those patch <pid> was made on (<file> differs)`; a hash not cached
  now gives `same: null` and `the base's content hash is not cached; a re-run hashes it`; a file
  that was not hashed at commit time and whose size or time changed since gives `the base was
  not hashed when the edit was committed and its file <name> has changed since (size or
  modification time): it cannot be shown to be the same`. `made_on`: the enabled patches then
  and now. `measurement:<name>` (once per measurement): its meta, means and `io.json` hashes then
  and now (`now: null` when it is gone): `measurement '<name>' changed since patch <pid> was made
  (measured again or deleted): measure it again from the record (GET
  /api/patches/<pid>/record?remeasure=true) and pass the new name as measurement`.
- `environment`: the record's environment and today's, leaf by leaf (`libraries.torch` as
  `torch`, the packages by name, the runner as one row, each code file by its path), with
  Heretic's severities extended: a code file critical; `torch`, `transformers` high;
  `torch_threads` high when a call is ARA, else low; `numpy`, `safetensors`, `tokenizers`,
  `optuna` and the runner medium; Python, implementation, platform, machine, system and
  `pyarrow` low. Reported, never refused. Each row has `same`; for the runner it is null while
  the model is not loaded (what it will run on is not known yet, which is not a difference; the
  Output view shows "not loaded yet").
- `?remeasure=true`, for a `rebuildable` record: per measurement `{"measurement": <name>,
  "body": <the POST /api/directions/measure body that makes it again>}`, the body being the
  prompts snapshot's texts, the recorded `chat`, `system`, `prefix`, `chat_options`, `variant`
  and `winsorization_quantile`, `module_io` when the recorded measurement had it, and the name
  `<name>-r<pid>` (cut to 64 characters). `body` is null when the prompts snapshot is missing;
  `remeasure_unavailable` lists why (also for a record that is not rebuildable).
- `heretic`, for a patch with a record: the rules of [the mapping to Heretic's
  `reproduce.json`](#the-mapping-to-heretics-reproducejson) evaluated without a query, as
  `{"runnable": <no rule fails>, "reasons": [<each failing rule's text>], "needs_model": <the
  only failing rule is that the checkpoint's Hub repository is not known>}`
  (`directions_record.heretic_status`). With `needs_model`, the file can be had by naming the
  repository: `GET /api/patches/{pid}/record/heretic?model=<Hub id>&model_commit=<commit>`.
  Null for a patch without a record.

**What a record holds of the machine.** No host name, user name or home path: the base's files
by name only, `platform.platform()` (the OS name, release and processor type), the machine type,
the system, the Python version and implementation, the thread count, the runner's device, and
the library versions. Two values come from the measurement and the search as they were given:
`datasets` (a prompt set read from a local file or directory names it as the person typed it,
so it can be a path of the machine) and a search's scorers' dataset records (likewise). The
record lives in the workspace and is not part of any export, except that the adapter export's
provenance file says whether a record exists.

Checked by `tests/test_directions_record.py` (model-free, the tiny random model): the record's
fields against the files on disk (hashes computed in the test), the snapshots, the base hash
cache (hashed once, again after the file's time changes, never by the route; a base above the
limit filled in later and refused after a change), the stale id, a failed record leaving the
commit, a committed search trial, and the route with its `remeasure` body measured again.

## Re-running an edit

`POST /api/patches/{pid}/reproduce` runs a committed direction edit again from its record
(`directions_record.reproduce`), through `Session.apply` only, like every other edit: the
scope, the largest-patch setting, the edit admission, the candidate tests and drift, the
commit and the patch's own record. It never writes a weight any other way, never switches a
patch on or off, and its default is a dry run, which commits nothing.

**On the weights it was made on.** The script reads the current weights, so it must see what
it saw: the same base, exactly the enabled patches the record's `made_on` lists (the edit
itself switched off; the person does that, the route does not), and the same measurement.
Running it with the edit still on would read already-edited weights, and taking the edit off
by adding its negation would not give the base back in float32.

Body: `measurement` (null, or the name of a new measurement; default null), `dry_run` (default
**true**), `message` (null or text of at most 1,000 characters), `enforce_scope` (default
false), `accept_different` (default false). An unknown field: `RecordError: <field>: unknown
field; the fields are measurement, dry_run, message, enforce_scope and accept_different`.

In order:

1. 404 `no patch '<pid>'`; 404 `patch <pid> has no reproduction record: it was not made by
   ablate_direction, ablate_ara, direction or direction_means through the edit path, or it was
   made before records existed` (also for a stale record).
2. 409 `a search is running on this workspace; stop it before re-running an edit` (this
   session's search job, or the searches lock held here or by another process). The lock is
   then held while the edit is re-run, so no search starts meanwhile.
3. With `measurement`: the record must be `rebuildable`, else 400 `patch <pid>'s script does
   more than ablate_direction and ablate_ara calls, so it cannot be rebuilt for another
   measurement: re-run it without measurement`; and it must read one measurement, else 400
   `patch <pid> reads <n> measurements; measurement mode takes one`.
4. The checks of `GET /api/patches/{pid}/record` (above), with the base hashed now when its
   cache entry is missing or stale; the first that differs answers 400 `RecordError:
   <its message>`. With `measurement`, the measurement row compares the new measurement with
   the record: the prompts' hashes, the rendering options (`chat`, `system`, `prefix`,
   `chat_options`, `winsorization_quantile`), the variant, the means' shape (that of the
   recorded means snapshot, else the model's), whether module inputs were saved, and, for a
   measurement of the current weights, the patches it was measured on (`measured_on`):
   `measurement '<new>' was not made from patch <pid>'s prompts and rendering: <the first
   differing key> differs`. Then the rendering versions: equal, or one of version 1 and one of
   version 2 whose `same_as_version_1` is true; else `measurement '<new>' and patch <pid>'s
   measurement were rendered with different versions (patch <pid>'s: <v>, this one: <v>;
   version 2 renders prompts as Heretic does) and on this model the two give different token
   ids for these prompts, so it would not reproduce the edit: re-run patch <pid> without
   measurement (from its recorded means)` ([How a prompt becomes token ids](#how-a-prompt-becomes-token-ids)).
   A new measurement that does not exist answers 404.
5. The script: the recorded script verbatim, or with `measurement` each recorded call's
   one-line script with the new measurement's name.
6. `Session.apply(script, message or "re-run of patch <pid>", author "human", dry_run, ...)`
   with the origin `{"kind": "reproduction", "of": <pid>, "of_created": <its commit time>,
   "recorded_digest", "digest", "verdict", "measurement": <the name used>, "environment":
   <now>}` and a check that computes the re-run's piece digest and verdict before anything is
   evaluated.

**The verdict.** `identical`: the re-run's pieces have the record's digest (the same digest as
a search trial's). Otherwise, per tensor in either set, `relative = ||new - recorded||_F /
||recorded||_F` (both patches' pieces at strength 1, built in row slabs of at most 16 million
values and summed in float64), `max_abs` and `only_in`. Where the ratio does not exist,
`relative` is null: a tensor only one side changes (`only_in` is then `"recorded"` or
`"rerun"`), or a recorded change of norm 0 against a re-run's that is not. Such a tensor makes
the edit `different`.
`within_rounding`: every call is `ablate_direction` and every tensor is within the strictest
bound of its calls, `1e-5` for `row_normalization` `none` and `pre`, `1e-4` for `full`. With the
same means the two tools' changes agree to 1.9e-7 (`none`, `pre`) and 2.5e-6 (`full`) on the
fixture and to 6.5e-7 on Qwen2.5-0.5B, and one tool across library versions is not expected to
be farther; the bounds are a decision, to be tightened from the comparison's numbers. An ARA fit
has no rounding bound (fits move by 0.3% to 230% across torch versions): it is identical or
different. `different`: anything else.

A dry run answers every verdict. A commit is allowed for `identical` and `within_rounding`; a
`different` one is refused unless `accept_different`: `the re-run made a different edit
(largest relative difference <x> in <tensor>). <reason>. Pass accept_different to commit it
anyway` (in place of the largest difference, for a tensor without a ratio: `<tensor> is changed
by the recorded edit only`, `<tensor> is changed by the re-run only` or `the recorded change to
<tensor> is zero and the re-run's is not`), the reason being, for ARA, `an ARA fit is reproducible only from the same captures with
the same torch version and thread count (recorded: torch <a>, <n> threads; now: torch <b>, <m>
threads)`; for an ablation with the recorded libraries, `the library versions are those it was
made with, so this is not expected; report it`; otherwise `the closed form rounds differently
with other library versions (recorded torch <a>; now torch <b>)`. The origin keeps the verdict.

**The answer**: `Session.apply`'s report plus `verdict`, `digest`, `recorded_digest`,
`differences` (`[{"tensor", "relative", "max_abs", "only_in"}]`, only when the digests differ), `bound`
(the rounding bound, null for ARA), `environment` (the rows of the record route: library
differences are reported with their severities, never refused), `means` (with `measurement`:
`{"equal", "relative": {"good", "bad"}}`, the largest over indices of `||new - recorded|| /
||recorded||` against the recorded means snapshot; null with the note `the recorded means are
not in the workspace` when the snapshot is gone), `checks` and `notes`. A commit makes a new
patch with its own record; the original stays as it is.

**Measuring again** is two requests: `GET /api/patches/{pid}/record?remeasure=true` gives the
exact measure request (the prompts snapshot, the recorded rendering and variant, a suggested
name), the client measures with `POST /api/directions/measure` (its stream, stop and progress),
then asks for the re-run with `measurement` set to that name.

Checked by `tests/test_directions_reproduce.py` (model-free, the tiny random model): an
ablation and an ARA fit re-run identically, dry and committed (the new patch's pieces equal
the original's, its origin and record); each refusal (the patch still on, a base whose cached
hash differs, a measurement measured again, a search running, a stale record, an unknown field);
a library difference reported without changing the verdict; a change scaled by `1 + 1e-7`
within rounding and committable, by `1.01` different and refused unless accepted; an ARA fit
scaled by `1 + 1e-7` different; the measurement mode from the `remeasure` body (equal means,
identical), refused for other prompts, another system message, another quantile, other patches
under a current measurement, a script that is not rebuildable and a record of two
measurements; the slab differences against a dense computation.

## The mapping to Heretic's reproduce.json

`GET /api/patches/{pid}/record/heretic` writes a record as Heretic's version-4
`reproduce.json` (`directions_record.to_heretic`), the file `heretic --reproduce` reads, when
Heretic can run the edit; otherwise 400 `RecordError: this record cannot be written as a
Heretic reproduce.json: <reason>; <reason>`. 200 is a download (`Content-Disposition:
attachment; filename="reproduce.json"`); 404 for no patch or no record. Reading a Heretic
`reproduce.json` and running it in the IDE: [Running a Heretic reproduce.json](#running-a-heretic-reproducejson).

Query: `adapter` (the name of an adapter export of this workspace that contains this patch),
`model` (a Hub repository id; a local path is refused, since it would put a path of this
machine into the file) and `model_commit`, each optional text of 1 to 256 characters.

**The rules** (each failing rule adds its reason):

| # | Rule | Reason |
|---|---|---|
| H1 | exactly one call, `ablate_direction` or `ablate_ara`, and the record `rebuildable` | `the script is not one ablate_direction or ablate_ara call; Heretic applies exactly one modifier` |
| H2 | a transformers checkpoint (`hf`) with a Hub id (the query's `model` or the base's repository) | `Heretic runs transformers checkpoints only (this workspace is <kind>)` / `no Hub repository is known for this checkpoint: pass model (and model_commit)` |
| H3 | `made_on` empty | `the edit was made on top of other patches (<ids>); Heretic edits the base model` |
| H4 | the measurement's variant `base` | `the measurement was made on the current weights; Heretic measures the base model` |
| H5 | measured as a chat | `the measurement did not use the chat template; Heretic always renders prompts as a chat` |
| H6 | with a system message | `the measurement used no system message; Heretic always sends one (system_prompt)` |
| H7 | no chat template options (a chat measurement records `{}` for none) | `the chat template options <json> have no Heretic setting` |
| H8 | every part of both prompt sets a Hub dataset with its commit | `the <good/bad> prompts include <inline texts / a local file / a directory / a Hub dataset without a commit>; Heretic's settings name Hub datasets (a local path would put a path of this machine into the file)` |
| H9 | a search's scorers: Heretic's built-in plugins, Hub prompt sets | `scorer <i> (<plugin>) is not one of Heretic's built-in scorers`; the H8 text with `the scorer <i> prompts` (counted from 1) |
| H10 | `adapter`, when given, names an export whose provenance lists this patch | `no adapter named '<name>' contains patch <pid>` |

After the rules, the file's text is searched for the workspace's directory, its parent and the
home directory, and refused (`it would contain a path of this machine`) if one is in it: a last
guard, since the rules already keep every local path out.

**The mapping**:

| Heretic | From the record | Written |
|---|---|---|
| `version` | (own version 1) | `"4"` |
| `timestamp` | `created` | UTC, ISO 8601, no microseconds, no zone |
| `system` | | left out (Heretic's "basic"; Heretic then prints its no-system-information warning) |
| `environment.heretic` | | `{"version": "2.2.0.dev0", "is_standard_pypi": false, "metadata": {"type": "git", "url": "https://github.com/p-e-w/heretic", "commit_hash": "64e999152ff7918efc73576814709b6b6a71f15c", "requested_revision": null}}` |
| `environment.pytorch_version` | `environment.libraries.torch` | the same |
| `environment.requirements` | `environment.packages` | the packages that are installed, without torch |
| `settings.model`, `model_commit` | `base.repo`, `base.sha` | the query's win; the base's commit only with the base's repository |
| `settings.dtypes`, `device_map` | the measurement's runner `dtype`, `device` | `["float32"]` from `"torch.float32"`; `device_map: "cpu"` for a CPU runner; left out when unknown |
| `settings.batch_size` | (one prompt at a time) | `1` |
| `settings.response_prefix` | the measurement's `options.prefix` | the prefix, or `""` for none (so Heretic does not detect one) |
| `settings.system_prompt` | `options.system` | the same |
| `settings.modifiers` | `calls[0].heretic.modifier` | `[{"plugin": "heretic.modifiers.abliteration.Abliteration" or "heretic.modifiers.ara.ARA", "instance_name": null}]` |
| `settings.modifier.<Class>` | `calls[0].heretic.settings`; `winsorization_quantile` from the measurement (Abliteration); the measurement's dataset records | the settings; ARA adds `print_loss: false`; `good_prompts` and `bad_prompts` are the Hub parts' specifications (`dataset, commit, config, split, column, prefix, suffix, system_prompt`) with the commit the IDE resolved (one object for one part, else a list) |
| `settings.scorers`, `settings.scorer.<Class>[_<instance>]` | `search.scorers` | the scorers' import paths, optimizations and instance names; each table its settings and `prompts`; both left out for an edit made by hand, which has no scorers: Heretic refuses an empty `scorers` list (`heretic/evaluator.py:56`, "No scorers configured"), and without the key it scores its reproduction with the reproducing person's own scorers (Heretic's defaults) |
| `settings.n_trials`, `n_startup_trials`, `seed` | `search` (finished trials, startup trials, seed) | by hand: `1`, `1`, and the ARA seed or `0` |
| `settings.max_response_length` | `search.max_response_length` | the same; left out by hand |
| `settings.export_strategy` | | `"adapter"` with the query's `adapter`, else `"merge"` |
| `parameters` | `calls[0].heretic.parameters` | the same |
| `scores` | `search.trial.scores` | `[{"name", "score": {"value", "rich_display", "md_display"}, "baseline": null}]` with the IDE's display text in both; `[]` by hand |
| `hashes` | | with `adapter`: `{"adapter_model.safetensors": <its SHA-256>}`, else `{}` |
| | everything else | `"wsp": {"record_version": 1, "patch": <pid>, "digest": <the pieces' digest>, "note": "written by the weight-space IDE from its reproduction record; Heretic reproduces it with heretic --reproduce; the IDE's own result is the patch with this digest", "differences": [...]}` (Heretic reads only its own keys) |

`wsp.differences` lists what makes Heretic's result differ from the IDE's by design, when it
applies: `row_normalization full: Heretic stores a randomized rank-r approximation
(svd_lowrank seeded with the seed), the IDE the best rank-r one`; `ARA: a fit is reproducible
only with the same torch version and thread count, in either tool`; and `the IDE loads
transformers models with eager attention, Heretic with the default` when the measurement ran in
float32 or float64 (by its recorded runner dtype; also when no runner or dtype is recorded,
which only a measurement from before bfloat16 runs existed lacks, and those ran on eager). A
measurement run in bfloat16 or float16 loaded the library's default attention, as Heretic does
([Runs in bfloat16](#runs-in-bfloat16)), and its record has no such entry. A chat measurement
rendered with version 1 (its record has no `rendering`: made before the IDE rendered prompts as
Heretic does) adds `the measurement was rendered as the IDE rendered prompts before it rendered
them as Heretic does (rendering version 1): where the tokenizer adds its own special tokens, the
template prints the date or has no system role, its token ids are not those Heretic generates
from` ([How a prompt becomes token ids](#how-a-prompt-becomes-token-ids)).

Heretic in `--reproduce` mode compares the environment and asks whether to go on (unless
`ignore_mismatches`), takes the file's settings, measures, applies one trial from the file's
`parameters` (no search), and after a save compares each file in `hashes` with what it wrote
("Hash matches", "Hash doesn't match", "File not found"). The IDE's `heretic` package record
names a git install at the commit above; with another install Heretic adds its "critical" line.

Checked by `tests/test_directions_reproduce.py` (model-free): the whole file for a typed record
of a Hub measurement against Heretic's own field lists (its settings fields, its dataset
specification, the Abliteration and ARA settings tables), the ARA and search cases, each rule's
reason one by one, several reasons joined, no local path in any refusal or file, and the route
(404s, 400 with the reasons, the download header, `adapter` filling `hashes` and the export
strategy, the query's model and commit, a local path refused as a model).

### Compared with Heretic

The scripts (they load models or need Heretic's interpreter, so the offline suite does not run
them): `scripts/heretic_reproduce_json.py` (Heretic's own `reproduce.json` rebuilt from a run's
journal, offline), `scripts/heretic_compare_adapter.py` (stages `ide`, `ide-captures`, `peft`:
the IDE's adapter in PEFT and against Heretic's adapter), `scripts/heretic_check_reproduce.py`
(Heretic reads the IDE's file), `scripts/heretic_compare_record.py` (stages `search`, `map`,
`rerun`: the IDE's file field by field against Heretic's, and the re-run on a model),
`scripts/heretic_compare_reproduction.py` (what Heretic makes from the IDE's file). Each
docstring holds its usage and its assertions. Every number is in
[`directions-output-heretic-validation.json`](directions-output-heretic-validation.json).

**What was run** (2026-11-18). Heretic at `64e9991` with torch 2.14.1, transformers 5.18.0 and
PEFT 0.21.2; the IDE with torch 2.15.0 and transformers 5.22.2; float32 on the CPU, one prompt
at a time, offline. Eight finished Heretic runs were made again with `export_strategy =
"adapter"` and nothing else changed (their journals hold the same draws, values, parameters and
scores as the originals, asserted before every stage): on `tiny-random/qwen2.5` directional
ablation with `row_normalization` `none`, `pre` and `full`, a second `full` run with per-layer
directions, and two ARA runs; on Qwen2.5-0.5B-Instruct one `full` ablation and one ARA run. In
the IDE each run's exported trial was measured from the run's dataset specifications, applied
through `Session.apply` and exported as an adapter. All of it ran on one commit, each model
load under `python -m wsp.memory_guard`, one process at a time, in isolated roots (largest
footprint 5.9 GiB, no guard violation). 529 assertions, all passing; what the first runs found
and changed is listed below.

**The adapter (H2).**

| Check | `tiny-random/qwen2.5`, six runs | Qwen2.5-0.5B, two runs |
|---|---|---|
| Heretic's own adapter merged into a fresh base equals Heretic's merged model (the oracle did not move) | 51 of 51 tensors bit-equal, each run | 290 of 290 |
| PEFT loads the IDE's adapter: no error, no warning; the modules carrying it are the file's targets with the file's rank and scaling; the file's keys are PEFT's | 5 modules (ablation), 1 and 2 (ARA) | 31 (ablation), 8 (ARA) |
| `merge_and_unload()` of the IDE's adapter against the IDE's current weights | bit-equal, every touched tensor | bit-equal, 31 and 8 tensors |
| every other tensor of the checkpoint against the base | equal | equal |
| first-token logits, the adapter attached against the current weights loaded | at most 3.1e-7 (logits up to 0.39) | 7.4e-5 and 5.3e-5 with logits up to 23: 3.2e-6 and 2.4e-6 of the scale (bound 1e-5 of it) |
| the same for Heretic's own adapter, attached against merged | at most 2.8e-7 | 7.3e-5 and 5.1e-5 |
| ablation: `lora_B @ lora_A` per module against Heretic's adapter (each tool's own measurement) | 1.16e-5 to 1.39e-5 (bound 5e-5) | 3.9e-4 (bound 1e-3, `full`) |
| ARA fitted from Heretic's captures under Heretic's libraries: `lora_A` and `lora_B` against Heretic's | bit-equal (1 and 2 modules) | bit-equal (8 modules) |
| ARA with each tool's own captures and libraries (reported) | 4.4e-2 and 3.8e-1 | 9.4e-1 |

A module the IDE's adapter leaves out (its change is zero) has an all-zero product in Heretic's
adapter: Heretic writes all 8 (48) modules, the IDE the 5 (31) that change. The last row is
not a defect of the export: an ARA fit moves with the torch version, in either tool (see the
ARA comparison above); from the same captures under the same libraries the factors are
Heretic's bit for bit. An attached adapter computes `W x + B (A x)` and merged weights
`(W + B A) x`, which float32 rounds differently in proportion to the logits; Heretic's own
adapter shows the same difference.

**The record (H3).**

- *Heretic reads the IDE's files.* With Heretic's own code (`load_reproduction_information`,
  `Settings.model_validate_init_only`, `load_plugin`, the modifier's parameters class,
  `format_version_information`): all 19 files the IDE wrote for the nine edits (merge and
  adapter export, by hand and from a search).
- *Field by field against Heretic's own file* for the same run, rebuilt with Heretic's
  `generate_reproduce_json` from the run's journal: the model and its commit, the response
  prefix, the system prompt, the modifier and each of its settings, the dataset specifications
  with their commits, and the parameters (JSON-equal) agree in all nine pairs. For the search
  run also the seed, the trial counts, the scorers and the committed trial's scores (refusals
  equal, KL 1.6619e-4 against 1.6636e-4).
- *What the IDE's file does not hold, and why that cannot change the weights:* the answers to
  Heretic's prompts (`checkpoint_action`, `trial_index`, `model_action`); for an edit made by
  hand, which has no scorers, `max_response_length` and the scorer tables (Heretic reads
  `max_response_length` only where responses are generated, and its prefix detection runs only
  when `response_prefix` is unset, which the IDE's file always sets); ARA's `print_loss`; and
  the seed of an ablation made by hand (the file says 0), which Heretic uses only for
  `row_normalization` `full`'s randomized SVD, a difference the file names.
- *Heretic runs the IDE's file.* `heretic --reproduce <the IDE's file>` (with `model_action`
  and `trial_index` added to answer its prompts) on all eight runs. For `none`, `pre` and the
  three ARA runs the adapter Heretic then saves equals, bit for bit in every tensor (16 on the
  tiny model, 96 on Qwen2.5-0.5B), the adapter Heretic's own original run saved: the IDE's
  file carries everything Heretic needs. For the three `full` runs some tensors differ (10 of
  16, 10 of 16, 62 of 96): the seeded SVD, seed 0 against the run's, with products within
  3.6e-7 and 5.5e-7 on the tiny model and 4.7e-4 on Qwen2.5-0.5B. Against the IDE's adapter
  Heretic's reproduction is within 1.16e-5 to 1.40e-5 per module on the tiny model (bound
  5e-5) and 4.1e-4 on Qwen2.5-0.5B (bound 1e-3).
- *The re-run.* On every one of the eight patches, both models: switched off, the dry run is
  `identical` (the recorded digest) and commits nothing; measured again from the record's
  body, the means equal the recorded snapshot bit for bit and the re-run is `identical`; the
  committed re-run is a new patch with origin `reproduction` whose pieces equal the
  original's.

**What the first runs changed.** The acceptance was written before any run; six things were
corrected after the first ones, none by loosening what is asserted about the weights.
(1) Heretic refuses an empty `scorers` list (`heretic/evaluator.py:56`), also in `--reproduce`
mode, so the IDE's file for an edit made by hand was not runnable although the IDE said so; it
now leaves the key out. (2) The field comparison first required every setting only in
Heretic's file to be at its default, and the seeds to be equal; that failed on the settings
listed above, which are now reported with where Heretic reads them, while everything else is
asserted as before. (3) The logits check had an absolute bound of 1e-5, which a model with
logits near 23 cannot meet in float32; it is relative to the logits, with Heretic's own
adapter measured beside it. (4) The re-run route answered an error instead of its verdict when
a re-run changed other tensors. (5) A model that was not loaded showed as a library difference
in the record. (6) The 0.5B oracle runs pinned no model commit; it is the cache snapshot they
loaded.

**Not covered.** A transformers checkpoint only (the adapter is refused on a GGUF base).
Float32 on the CPU only. Models of 494M parameters at most. PEFT 0.21.2 only. An adapter with a rank pattern
(`per_module`) has model-free tests, not a PEFT load on a real model: every run here has one
rank. The measurement mode with a changed model or libraries, where a re-run would be
`within_rounding` or `different`, is tested model-free only. Reading a Heretic
`reproduce.json` and running it natively: [Running a Heretic reproduce.json](#running-a-heretic-reproducejson).

## Running a Heretic reproduce.json

`POST /api/directions/reproduce` takes the content of a `reproduce.json` that Heretic wrote and
makes Heretic's edit with the IDE's own code (`wsp/directions_heretic.py`): it measures with the
file's prompt specifications and settings and applies the file's modifier with the file's
parameters as a patch, through `Session.apply` like every other edit. The file is translated
into the IDE's own requests, a `POST /api/directions/measure` body and the one-line weight script
of `ablate_direction` or `ablate_ara` (`DR.script_for`), so nothing is computed a second way, and
the patch gets an ordinary [reproduction record](#the-reproduction-record) (which can be written
as a `reproduce.json` again). API and weight script only: the Directions tab has no control for
it. The answer gives the weight script, which runs unchanged from the weight-script bar on the
measurement the route made.

**The file, as Heretic 64e9991 writes it** (`heretic/utils.py` `generate_reproduce_json`, only
when it uploads a model; `scripts/heretic_reproduce_json.py` makes the same file offline from a
run's journal): `version` (`"4"`, the plugin-based format), `timestamp`, `system` (only with
Heretic's "full" information: Python, operating system, CPU, accelerators), `environment`
(Heretic's own version and origin, `pytorch_version`, `requirements`: every installed package),
`settings` (Heretic's `Settings.model_dump()`: every setting without `exclude=True`, including
`model_commit` fetched from the Hub, the response prefix as detected, the seed as drawn, and the
plugin tables `scorer.<Class>` and `modifier.<Class>` with their effective settings and dataset
specifications pinned to commits), `parameters` and `scores` of the one chosen trial, and
`hashes` (SHA-256 of the uploaded weight files). It describes one trial: the search's other
trials are in the Optuna journal Heretic uploads beside it, not in the file; `n_trials`,
`n_startup_trials` and the seed say how the search ran. A file written by the IDE has a `wsp`
block besides ([the mapping](#the-mapping-to-heretics-reproducejson)).

**What `heretic --reproduce` does with it** (`heretic/main.py`, `heretic/reproduce.py`): it stops
on a version other than `"4"`; compares the environment, prints each difference with its
severity and asks whether to go on (unless `ignore_mismatches` is given on its command line);
replaces its settings by the file's (`Settings.model_validate_init_only`: everything in the file
wins, only `exclude=True` settings such as `save_directory` come from the command line); seeds
with the file's seed; loads the file's model at its commit; detects a response prefix only when
the file has none; computes the scorers' baselines (which change nothing); measures with the
modifier table's prompt sets; applies the trial from `parameters` (no search, `n_trials` unused);
answers its menu from the file (`model_action`, `export_strategy`); and after a save compares
`hashes` with the files it wrote. `scores` are only shown.

**The body**: `reproduce` (required: the file's content, an object), `measurement` (null or the
name of the measurement to make or reuse; default `heretic-<the first 12 hex digits of the
SHA-256 of the file's compact, key-sorted JSON>`), `check` (default false: answer the plan and
run nothing), `dry_run` (default **true**), `message` (null or text of at most 1,000
characters), `enforce_scope` (default false). An unknown field: `RecordError: <field>: unknown
field; the fields are reproduce, measurement, check, dry_run, message and enforce_scope`.

**Refused before anything runs** (400 `RecordError: this reproduce.json cannot be run in the IDE:
<reason>; <reason>`, every reason that applies, in this order;
`directions_heretic.check_heretic_reproduce`, which reads no weight and loads no model):

| What | Reason |
|---|---|
| `version` other than `"4"` | `file format version "<v>": the IDE reads version 4 (plugin-based) reproduce.json files, as Heretic at 64e9991 does (older files were written before Heretic's plugins)` |
| no `settings`, no `parameters` | `settings: missing (...)`, `parameters: missing (...)` |
| no `settings.model` | `settings.model: missing (the file names no model)` |
| the workspace knows its Hub repository and it is another | `the file's model is <model>; this workspace's base is <repo>` |
| the same repository, both commits known and different | `the file's model commit is <commit>; this workspace's base is <repo> at <sha>` |
| patches switched on | `patches <ids> are switched on; Heretic edits the base model: switch them off to reproduce the file` |
| `quantization` other than `"none"` | `settings.quantization is "<q>": the IDE does not load quantized transformers models, and the measurement and the edit would be of other weights` |
| not exactly one modifier | `settings.modifiers lists <n> modifiers; Heretic applies exactly one` |
| a modifier other than Heretic's `Abliteration` or `ARA` (it makes the edit) | `the modifier <plugin> is not one the IDE has (heretic.modifiers.abliteration.Abliteration, heretic.modifiers.ara.ARA); it makes the edit, so the file cannot be reproduced without it` |
| `response_prefix` null (Heretic would detect one by generating) | `settings.response_prefix is not set: Heretic would detect it by generating responses; detect it with POST /api/directions/prefix and put it into the file` |
| ARA without an integer `seed` | `settings.seed must be an integer: ARA's fit is seeded with it` |
| a parameter the modifier does not have, or a value the IDE's normaliser refuses | `parameters: <field>: not a parameter of <Class>` / `parameters: <the normaliser's message>` |
| a prompt set that does not resolve offline (each set on its own) | the dataset code's own error, with the field `settings.modifier.<Class>.good_prompts` or `bad_prompts` (a set not in the cache: `... is not in the local Hugging Face cache ...; fetch it with POST /api/directions/datasets ...`) |
| any other measure refusal (no chat template, a prompt over the token bound, the disk for ARA's module inputs) | the measure route's message |
| a measurement of that name made from other prompts or settings | `a measurement named '<name>' exists and was not made from this file's prompts and settings (<key> differs): pass another measurement name` (`<key>`: `prompts_sha256`, `options`, `variant`, `model`, or `module_io` for ARA) |

409 `a search is running on this workspace; stop it before running a reproduce.json` while a
search runs; the searches lock is then held while the file is run.

**What the IDE does with each setting** (the answer's `heretic.settings`):

| Use | Settings |
|---|---|
| honoured | `model`, `model_commit` (checked against the base), `quantization` (`"none"` only), `response_prefix` (`""` is no prefix), `system_prompt` (Heretic's default `You are a helpful assistant.` when absent), `modifiers`, `modifier.<Class>` (merged with `modifier.<Class>_<instance_name>` over it, as Heretic merges them; a missing field takes Heretic's default, a field the class does not have is ignored, as Heretic ignores it), `seed` (ARA's fit only; for an ablation it is listed as not used: Heretic uses it only for `full`'s randomized approximation, the IDE computes the best rank-r one), and `parameters` |
| reported, compared with the IDE's run | `dtypes` (the first entry against the run dtype, `WSP_RUN_DTYPE`), `device_map` (against the run device), `batch_size` (the IDE measures one prompt at a time) |
| not used, with the reason | `scorers`, `scorer`, `max_response_length` (scoring; a reproduction scores nothing), `n_trials`, `n_startup_trials`, `trial_index`, `n_additional_trials`, `checkpoint_action` (the search), `export_strategy`, `model_action`, `max_shard_size`, `modified_model_type`, `upload_*` (what Heretic does with the model afterwards), `max_memory`, `offload_outputs_to_cpu` (where Heretic keeps tensors), `ignore_mismatches` (the IDE never stops on a difference), anything else (`not one of Heretic's settings at 64e9991`) |

The model: the base is compared by repository and commit where the workspace knows them
(`workspace.json` `repo` and `sha`, as opening a Hub id records them). A workspace opened from a
local directory knows neither, and the answer says `"repository": "not verified"` with the
reason rather than guessing from the path; the same for a file without a commit or a workspace
without one. A GGUF base gets a note that the measurement and the edit run on its quantized
weights.

**The environment** (`heretic.environment`), compared as `check_environment` compares it: the
file's packages and `pytorch_version` against the versions installed for the IDE (null when not
installed), Heretic itself (`<version>-git+<url>@<commit>` against the IDE's
`2.2.0.dev0-git+https://github.com/p-e-w/heretic@64e999...`; a local or unknown install never
matches, as in Heretic), and with `system` the Python version, `platform.platform()`, the CPU
(not read by the IDE: `same` null) and the accelerator type (CUDA, ROCm or MPS as torch sees
them). Each row is graded with Heretic's severity: `heretic-llm` critical; `torch`, `transformers`,
accelerator type high; `accelerate`, `bitsandbytes`, `kernels`, `optuna`, `peft`, `tokenizers`,
`triton` medium; everything else low. The answer lists the rows that differ, highest severity
first, with the count of equal rows and the highest severity. Reported, never refused.

**Notes** (`heretic.notes`), each a sentence: no system information in the file; a scorer the IDE
does not have (a classifier or benchmark scorer: it only scores, so the patch is the same
without it); the search the trial came from (not run again); for an ablation, that the seed is
used by Heretic only for `full`; for `full`, Heretic's randomized rank-r approximation against
the IDE's best rank-r one; for ARA, that a fit is reproducible only with the same torch version
and thread count; the eager-attention difference for a transformers run in float32 or float64;
the modifier table's ignored and defaulted fields; top-level keys that are not Heretic's; and on
a model with routed experts, when the file's transformers is 5.x (or unknown: Heretic 64e9991
requires 5.18), `expert_modules="shared"` in the script, since that is what Heretic did there
(see `expert_modules` above). `hashes` are shown with the note that they are not compared: the
IDE makes a patch, not Heretic's files.

**Then, in order**: the checks again under the searches lock; the measurement (`DR.measure` with
the file's dataset specifications, chat, the system prompt, the prefix, variant `base`, the
Abliteration table's `winsorization_quantile`, module inputs for ARA), unless a measurement of
that name was made from exactly this request on this base, which is reused; then
`Session.apply(script, message or "Heretic reproduce.json (<Class>)", author "human", dry_run,
enforce_scope)` with the origin `{"kind": "heretic_reproduce", "file_sha256", "version",
"timestamp", "model", "model_commit", "modifier", "measurement", "digest", "file_digest"}` (the
digest of the pieces, and the file's `wsp.digest` for a file the IDE wrote). A dry run still
measures (a measurement is not a weight edit) and commits nothing; a commit's patch has its own
record.

**The answer**: `Session.apply`'s report plus `ran` (`measurement`, `measured`: false when one was
reused, `script`, `dry_run`), `digest`, `file_digest`, `same_as_file` (whether the patch is the
one the IDE wrote the file from; null for Heretic's own files) and `heretic`: the plan, which
`"check": true` answers alone (`{"heretic": <plan>, "ran": null}`): `file` (version, timestamp,
SHA-256, modifier, instance name, parameters, scores, hashes), `model`, `settings`,
`measurement` (`name`, `state` `new` or `reuse`, the measure `request`), `method`,
`parameters` (the IDE's), `script`, `environment` and `notes`.

Checked by `tests/test_directions_heretic.py` (model-free: the tiny chat model and fake Hub
cache snapshots of `tests/test_directions_datasets.py`, no network): a typed Heretic file for an
ablation (`none`, `full`) and for ARA makes, piece for piece, the patch made by hand from the
same rows measured as texts and a script typed in the test; the origin and the record; a dry run
commits nothing and a second run reuses the measurement; each refusal with its text, and several
joined; the model verified and not verified; the environment rows with Heretic's severities; the
settings table; every note; the instance table over the class table; and an IDE edit written by
`to_heretic` and read back in a fresh workspace, piece for piece.

### Compared with Heretic (H3b)

`scripts/heretic_run_reproduce.py` (its docstring holds the stages and bounds, written before its
first run). Heretic at 64e9991 (torch 2.14.1, transformers 5.18.0, PEFT 0.21.2) ran three
configurations on `tiny-random/qwen2.5` with an adapter export, float32 on the CPU, one prompt
at a time, offline: directional ablation with `row_normalization` `none` and `full`, and ARA
(seed 1); each run's `reproduce.json` was written by Heretic's own `generate_reproduce_json`
(`scripts/heretic_reproduce_json.py`). The IDE (torch 2.15.0, transformers 5.22.2) read each file
in a fresh workspace made over the same snapshot with the file's repository and the snapshot's
commit. Every number is in
[`directions-reproduce-read-heretic-validation.json`](directions-reproduce-read-heretic-validation.json).

| Check | `none` | `full` | ARA | Bound |
|---|---|---|---|---|
| J: the new run's journal equals the original run's (the hand-pasted comparison's dump describes it) | equal | equal | equal | exact |
| F: the file's parameters are a trial of the run, its hash is the run's adapter | yes | yes | yes | exact |
| R0: checked, no refusal; model repository and commit | same, same | same, same | same, same | exact |
| R1: committed through `Session.apply`, with a record | yes | yes | yes | exact |
| R2: the patch against Heretic's own adapter, worst relative difference per tensor (5 tensors; 2 for ARA), no tensor changed by one tool only | 1.16e-5 | 1.36e-5 | 0.38 and 1.0e-3 (reported) | 5e-5 (`heretic_compare_adapter.py` P6); ARA reported |
| R3: the patch made by hand from Heretic's parameters (`heretic_compare_adapter.py`'s `measure_as_the_run` and `params_of`) | the same digest | the same digest | the same digest | identical |
| R4: that hand-made edit written by `to_heretic` and read back in a fresh workspace | bit for bit | bit for bit | bit for bit | bit for bit |

The bound is the one the comparison with Heretic's parameters pasted by hand uses for the same
model (`heretic_compare_adapter.py` P6: 5e-5 for this fixture, 1e-3 for Qwen2.5-0.5B);
`heretic_compare.py` and `heretic_compare_ara.py` assert no weight bound in float32. The
ARA row is reported, as there: a fit moves with the torch version in either tool. The
environment answer for Heretic's files listed `heretic-llm` (a local install: critical),
`torch` and `transformers` (high), `accelerate`, `bitsandbytes`, `peft` (not installed for the
IDE) and `tokenizers` (medium), and 79 low rows, with 24 rows equal; the `none` file written with
Heretic's system information also compared the Python version, the operating system and the
accelerator type (equal) and listed the CPU as not compared. All passed on the first run.

On a trained model, Qwen2.5-0.5B, on 2026-11-21 (float32 on the CPU, under
`python -m wsp.memory_guard`): Heretic's own `reproduce.json` of its directional ablation
(`row_normalization` `full`) and of its ARA run, each from 64 + 64 prompts of the two cached
datasets. 15 assertions, all passing: the files are the runs', no refusal, repository and
commit the same, committed through `Session.apply`; the ablation patch is within `3.9e-4` of
Heretic's adapter on 31 tensors, none changed by one tool only (bound `1e-3`; the difference
is Heretic's randomized approximation); each patch is the hand-pasted one (the same digest)
and the round trip is bit for bit. Numbers: [`directions-rendering-inputs-q05-validation.json`](directions-rendering-inputs-q05-validation.json).

**Not covered.** No browser tab. No progress events and no stop while the measurement runs (the
route answers when it is done). Directional ablation `pre` and a per-layer direction
(`direction_index` null) are not in this comparison. `row_normalization` `none` was run on the
random-weight checkpoint only. A
file whose prompt sets are not cached needs `POST /api/directions/datasets` with
`"download": true` first; this route never fetches. The `hashes` are never compared.

## On a GGUF base

A GGUF workspace keeps its quantized file as an immutable base; every edit, from either
method, is a stack of float32 patches over it ([QUANTIZED.md](QUANTIZED.md)). This section
says what that means for directions, scores and the generation loop.

**As a chat message.** The measurement, the scores, the prefix detection and the probes
render a prompt with the GGUF's own chat template, `tokenizer.chat_template`. Opening a GGUF
saves the converted tokenizer, with that template, in the workspace's `model/`
(`chat_template.jinja`), so a workspace opened before this change has it too and nothing is
migrated. Availability is found from those files without loading the model; the reason
given is `the GGUF's own chat template`. A GGUF without a template refuses "as a chat
message" with `this GGUF carries no chat template (tokenizer.chat_template)`. The rendering
is `chat_prompt` (the `transformers` tokenizer loaded from `model/`, generation prompt
included); the measurement, the scores and the prefix detection render and tokenize it as
Heretic does, through the renderer every Directions path shares
([How a prompt becomes token ids](#how-a-prompt-becomes-token-ids)), the probes as Chat
tokenizes it (without the tokenizer's own special tokens for a BPE vocabulary). Through the
model worker the server renders and tokenizes; the worker
receives token ids. The Run tab's chat (`/api/chat`) renders through the same `chat_prompt`,
so it uses the GGUF's own template too; it falls back to its turn prefixes only on a GGUF
without a template, or when the request sets `force_template`.

**The generation config.** A GGUF file carries no generation config and the IDE writes
none. When the workspace's `model/generation_config.json` exists (copied there by hand,
for example from the floating-point checkpoint the GGUF was made from, so that responses
can be compared with that checkpoint's), the GGUF generation loop reads it once per runner
(in this process, and in the model worker, which opens the same workspace) and applies the
two keys it can reproduce exactly:

| Key set in the file | What the GGUF loop does |
|---|---|
| `eos_token_id` (an id or a list) | adds them to the stop ids (the configuration's `eos_token_id` still stops too) |
| `repetition_penalty` (not 1) | before every choice of a token (argmax, or sampling), over every id of the prompt and of the response so far: a logit `s` becomes `s * p` when negative, else `s / p`, as `transformers`' `RepetitionPenaltyLogitsProcessor` |
| any other greedy key, with a value that changes a response | refused by name when generation starts, before the model runs |

The other greedy keys are `no_repeat_ngram_size`, `bad_words_ids`, `min_new_tokens`,
`min_length`, `suppress_tokens`, `begin_suppress_tokens`, `forced_bos_token_id`,
`forced_eos_token_id`, `sequence_bias` and `stop_strings`. Absent keys, `null`, and the
values that add no processor in `transformers` (`no_repeat_ngram_size`, `min_length` and
`min_new_tokens` 0, `repetition_penalty` 1) are not refused. The refusal reads, for
example: `the generation config sets no_repeat_ngram_size=3, which the GGUF runner does not
apply; remove it from the workspace's model/generation_config.json or score on a
transformers checkpoint`. A file that is not a JSON object, an `eos_token_id` that is not
an id or a list of ids, or a `repetition_penalty` that is not a positive number is refused
too. Keys outside the greedy list (`do_sample`, `temperature`, `top_p`, `top_k`, ...) are
not read: scores generate greedily, and the Run tab samples with its own settings. The file
applies to every generation on the workspace, the Run tab's included. Without the file,
generation is unchanged.

The scores' answer says what was applied: its `generation.settings` lists the keys the
loop applies from the file, with their values (empty without the file), for `GGUFRunner`
and for `RemoteRunner` on a GGUF workspace. A response shorter than `max_response_length`
must still end in an end-of-text id; the file's `eos_token_id` counts.

**What the measurement, the two methods and the scores mean on a quantized base.**

- The measurement runs the packed model. Its activations include ggml's quantization of
  activations at every packed matrix product (native quantized dot products quantize the
  activations too).
- The directions are the quantized model's own, not those of the floating-point checkpoint
  it was made from.
- Directional ablation removes `v` from what the decoded matrix `V` (the reconstructed
  weights) writes: `delta = -w (V v) v^T`. At run time the packed product of the quantized
  matrix is computed and the float32 overlay `x delta^T` is added to it; the overlay is
  exact float32 algebra. The base file is never written and never dequantized whole: reads
  decode bounded slabs.
- ARA fits against `V`, the decoded weights, in float32, one matrix at a time. Its
  consistency check compares `X V + b` with `Y` captured from the packed product, so on a
  quantized base it includes the activation quantization. It is reported per module and
  the limit stays 0.05; a module above it refuses the fit, and the limit is not raised for
  any format.
- The scores are the quantized model's behaviour: refusals and response lengths from the
  GGUF generation loop above, KL divergence from its first-token logits. They are not the
  floating-point checkpoint's.
- The export of an edit on a GGUF base is the WSP companion bundle (float32 patches bound
  to the base's SHA-256). The PEFT adapter is refused, as described in
  [A LoRA adapter (PEFT)](#a-lora-adapter-peft); a standalone requantized GGUF is the
  existing opt-in export, which is lossy ([QUANTIZED.md](QUANTIZED.md)).

Model-free tests (`tests/test_directions_gguf.py`) check the template's availability and
its rendering against `transformers`' own `apply_chat_template` on a synthetic GGUF opened
as the IDE opens one, the penalty against `RepetitionPenaltyLogitsProcessor`, the
generation loop on a scripted model against that processor at every step, the refusal by
name, and the adapter route's refusal. Three more tests run the packed model and need the
native bridge: the float32-patch invariants of directional ablation, ARA's consistency
values on Q8_0 and Q4_0 files, and, on a synthetic hybrid (Qwen3.5) GGUF of each scheme
with grouped linear-attention heads, that the mixer's output matrix `linear_attn.out_proj`
of layers 0-2 is captured (consistency at most 0.05), fitted by ARA and ablated, with the
packed runtime adding its patch in the matrix's logical column order.

### Checked on real GGUF files

On 2026-11-18 `scripts/formats_gguf_compare.py` ran both methods on two GGUF files, each in
its own guarded process and isolated root, with the pinned native bridges; numbers in
[`directions-formats-heretic-validation.json`](directions-formats-heretic-validation.json).
Heretic cannot load a GGUF, so the oracle is at the level of behaviour only: Heretic's float32
runs of the checkpoint the first file was made from. The bounds were written before the runs.

**`qwen2.5-0.5b-instruct-q4_0.gguf`, CPU bridge, against Heretic on
`Qwen/Qwen2.5-0.5B-Instruct`** (64 + 64 measured prompts, 32 per score, 30 new tokens, the
checkpoint's `generation_config.json` copied into the workspace). All 11 assertions passed in
the final run.

| Stage | Bound | Result |
|---|---|---|
| token ids against Heretic's, every prompt set | identical | identical |
| base first-token logits, GGUF against the checkpoint | reported | up to 7.2 apart on a scale of 23; mean KL 0.26: the quantized model is another model |
| base refusals | at least 24 of 32 | 29 (Heretic on the checkpoint: 30) |
| directional ablation with a Heretic trial's parameters, committed through `Session.apply` | refusals at most Heretic's 3 plus 6; KL within a factor of 2 of Heretic's | 3 of 32; KL `0.0354` against `0.0427` |
| ARA with Heretic's parameters | refusals at most 6; KL reported | 1 of 32; KL `0.0915` (Heretic: 0 of 32, `0.0878`) |
| ARA's consistency check, 8 modules | at most 0.05 | `0.0071` to `0.0102` |
| both patches: `current == base + delta` and the delta read back, bit for bit; the packed product's change against `x @ delta.T`; the largest single read; the base file's SHA-256 after each of 10 steps | bit-equal; relative `1e-5`; `2^24` values; unchanged | bit-equal (24 and 8 tensors); `1.2e-6` and `2.7e-7`; 4,358,144 values; unchanged |
| the generation config | listed in the answer | `repetition_penalty` 1.1 and both end-of-text ids listed and used |
| the companion bundle reopened in a fresh root; the adapter route | first-token logits bit-equal on 8 prompts; refused with the GGUF text | bit-equal; refused (400) |

**`Qwen3.5-0.8B-Q4_K_M.gguf`, a hybrid model, Metal bridge, no oracle** (its checkpoint is
not on this machine). Base refusals 30 of 32. The IDE's own search of 8 trials reached front
trials with 1 to 7 refusals (bound: at most half the base's). ARA with the default
parameters fitted all 48 modules, the linear-attention output matrices included, with
consistency values from `0.0002` to `0.019`; its 48-tensor patch passed the same invariants
(packed product `3.7e-6`), the bundle reopened bit-equal, and the base file was unchanged
after each of 7 steps. All 6 assertions passed in the final run.

**The first runs failed one assertion on both files**: the reopened bundle's first-token
logits were not bit-equal (largest difference 0.372 on the Q4_0 file, 0.014 on the hybrid
one), although the stored patches were byte-identical. Cause: a just-committed low-rank piece
holds its factors as transposed views, and the same piece read back from its patch file as
contiguous tensors; the run-time overlay multiplied whichever it was given, the product's
last bits depend on the operands' memory layout, and the packed model's quantized products
after the edit can turn last bits into visible differences. The overlay now holds its factors
contiguous (`wsp/overlay.py`), with a model-free test of both layouts and a bridge test of the
export and reopen sequence; both files were run again. The change moves in-process scores in
their last digits: ARA on the Q4_0 file scored 0 of 32 refusals and KL `0.09153` in process
before it, and 1 of 32 and KL `0.09149` after it. The same trial scored through the model
worker, which receives its factors contiguous, had given exactly the second pair of values
before the change.

**Checked by hand, a GGUF in the browser** (the Q4_0 file on a supervised serve with its own
root, config and port, under the memory guard): the Directions tab offers "as a chat message"
on the GGUF; 16 + 16 prompts were measured as chat with module capture from the page (8.4 s);
the default ablation previewed and run as an uncommitted candidate (26 tensors, 98,176 stored
numbers), then scored in Compare through the model worker against the base on 8 + 8 prompts
(refusals 8 of 8 against 3 of 8, KL 0.021); the scores answer's `generation.settings` listed
`repetition_penalty` 1.1 and both end-of-text ids from the copied file; the edit was committed
from the page; the Output view's adapter buttons were off with the GGUF text; `GET /api/export`
returned the companion bundle; one interrupt stopped the supervisor and the guard's log ended
with nothing remaining. Not checked by hand: a GGUF without a chat template (covered by the
model-free test only), ARA from the page, the Metal bridge, and the browser console.

## Not done

What is left:

- The Directions tab is checked by page tests and by hand on one tiny model only; see
  "What has been checked" and "Checked by hand" under [The Directions tab](#the-directions-tab).
- Steering along a direction at run time is run on one real model, Qwen2.5-0.5B through
  `transformers`, where one of its three criteria written beforehand failed (see
  [Steering along a direction](#steering-along-a-direction)); not on GGUF.
- The automatic search is compared with Heretic's through `transformers` on two checkpoints
  and with searches of at most 12 trials (see "The search on models, against Heretic's"). It
  takes one modifier. Through the model worker it has run on two real GGUF files, for the
  transfer measurement only (6 random ARA trials each, see
  [The transfer to the model worker](#the-transfer-to-the-model-worker)); those searches have
  no oracle. The worker-held candidate is designed and, by that measurement, not built.
- Prompt sets from datasets are compared with Heretic without a model and offline, and
  per-prompt system messages on one small checkpoint through `transformers` (see
  "Prompt sets from datasets: checked against Heretic"). A real fetch from the Hub has not
  been run. No other source than Heretic's four (no compressed files, no loading scripts, no `datasets` builder options
  such as a csv `sep`).
- The scores and the prefix detection are compared with Heretic through `transformers` only
  (see "Comparing the scores with Heretic"; in bfloat16 on one model,
  [Compared with Heretic in bfloat16](#compared-with-heretic-in-bfloat16)). On GGUF the scores
  have run on two real files, in process and through the model worker
  ([Checked on real GGUF files](#checked-on-real-gguf-files)); Heretic cannot load a GGUF, so
  its float32 runs of the checkpoint are an oracle of behaviour for one of the two files and
  the other has none. The prefix detection has not been run on a GGUF.
- The test suite uses tiny random models only: this project's own, a llama-named fake
  checkpoint, fake and synthetic quantized stores and fake runners. Real models so far: one
  checkpoint with random weights and two small trained ones (0.5B, and 0.6B with a reasoning
  block) through `transformers` in float32, the 0.5B one also in bfloat16; and two GGUF files
  (the 0.5B model in Q4_0, and a hybrid 0.8B model in Q4_K_M). The largest: the 4B hybrid
  checkpoint (4.66B parameters) in bfloat16 on the GPU
  ([Compared with Heretic on Qwen3.5-4B](#compared-with-heretic-on-qwen35-4b)). Nothing
  between that and the 9B limit.
- The measurement and the two ways of changing weights are compared with Heretic on those
  checkpoints only (see "Compared with Heretic", "ARA against Heretic", "On a trained
  model", "Comparing the scores with Heretic" and "Compared with Heretic in bfloat16").
- In bfloat16 one model has been compared, on the CPU, and two of its bounds written
  beforehand failed (the low-rank change of `row_normalization` `full`, whose cause was then
  tested, and that test's own bound). A float16 run is not tested on any real model.
- Hybrid and mixture-of-experts layouts: no real mixture-of-experts model has been run, the
  ARA fit of routed experts is not built (Heretic's ARA has no such fit to reproduce:
  [Mixture-of-experts models in Heretic](#mixture-of-experts-models-in-heretic)), and the LoRA adapter of a hybrid model is not
  checked with PEFT (it is exported with `validated_family: false`).
- The ARA fit runs on the CPU in float32 only. A model-free timing at the 27B model's matrix
  shapes (rank 50, 3 repeats) gives 3.0 s and 6.6 s per fit, 5.1 minutes for one trial over
  32 layers, so a fit on the GPU is an optimisation and was not built.
- The LoRA adapter export ([A LoRA adapter (PEFT)](#a-lora-adapter-peft)), the reproduction
  record, the re-run of an edit and the file written as Heretic's `reproduce.json` are compared
  with Heretic on the random-weight checkpoint and the 0.5B model only (see "Compared with
  Heretic" under [the mapping](#the-mapping-to-heretics-reproducejson)). There is no adapter
  for a GGUF base (no llama.cpp LoRA). A Heretic `reproduce.json` is read and run on the
  random-weight checkpoint only ([Running a Heretic reproduce.json](#running-a-heretic-reproducejson)).
- Scoring a modified checkpoint
  ([Scoring a modified checkpoint](#scoring-a-modified-checkpoint-heretics-evaluate_model)) is
  compared with Heretic's `evaluate_model` on those three checkpoints, in float32 through
  `transformers`, with Heretic's own merges only. Its input is a local checkpoint directory,
  a plain PEFT LoRA adapter (compared with PEFT's merge and Heretic on the tiny model only) or
  another workspace (checked on the tiny model only); no Hub id. GGUF is refused on both sides, and
  a checkpoint whose changes do not fit in memory beside the loaded base is refused, not loaded
  in the base's place.
