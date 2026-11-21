# Steering and probes

Steering pushes chosen neurons, whole MLPs and heads while the model runs. It changes
no weight, records no patch and does not appear in the patch stack. It applies to the
edited model (`current`) only; the base model is never steered. Probes read
activations: per-token values of a few neurons, the neurons most active at one
position, the neurons that separate two sets of texts, and how a steering
specification moves the next-token distribution.

Steering a reply: `/api/run` and `/api/chat` take an optional `steer` (see "Steering
a Run or Chat reply"). In the browser: the steering panel in the Run tab (completion
and chat) and the Probe tab (see "In the browser"). The MCP tools do not expose
steering or probes.

Code: `wsp/steering.py` (specification, calibration, hooks) and `wsp/probes.py`
(probes). Routes: `wsp/server.py`, the block before "edits and patches". Browser:
`web/js/steer.js` (panel, recipe parser) and `web/js/probe.js` (Probe tab).

## The steering specification

A JSON list. Each entry pushes one target:

```json
[{"addr": "L34N3838",      "mode": "add", "dose": 0.75, "unit": "resid"},
 {"addr": "L34N3838",      "mode": "mul", "dose": 2},
 {"addr": "blocks.12.mlp", "mode": "mul", "dose": 0},
 {"addr": "L11H3",         "mode": "mul", "dose": 0},
 {"addr": "group:refusal", "mode": "add", "dose": 0.5},
 {"addr": "direction:refusal@12", "mode": "add", "dose": -1, "unit": "diff"}]
```

| Field | Values |
|---|---|
| `addr` | A neuron (`L34N3838`, `L34 N3838`, `blocks.34.mlp.neuron.3838`), a neuron range (`blocks.34.mlp.neurons.0-127`), a whole MLP (`blocks.12.mlp`), a head (`L11H3`, `blocks.11.attn.head.3`) or a group (`group:<name>`). At most 200 characters. |
| `mode` | `add` (neurons only) or `mul`. |
| `dose` | A finite number, magnitude at most 1e6. |
| `unit` | `add` only: `resid` (default) or `raw`. |
| `direction:<name>@<k>` | A measured direction (`addr`; see the `direction:` item below): `<name>` a saved measurement's name, `<k>` its index in ASCII digits without a leading zero. `mode` `add` only; `unit` `diff` (default; `null` is `diff`) or `raw`; two more fields, for direction entries only: `orthogonalize_direction` (`true`, the default, or `false`) and `means_sha256` (optional; 64 lowercase hexadecimal digits). |

- **`add`** adds a constant to the neuron's activation at every position, at
  `blocks.L.mlp.hook_post` (the input of the MLP's output projection). With
  `unit: "resid"` the constant is `dose * p50(L) / ||w_out column||`, so the push
  moves the residual stream by `dose` times its typical norm at that layer. This
  is neuron-explorer's formula; see "Relation to neuron-explorer" for when a recipe
  such as `L34 N3838 +0.75` gives the same constant here. With `unit: "raw"` the
  constant is `dose`, in activation units, and needs no calibration.
- **`mul`** multiplies a neuron's activation, every activation of an MLP
  (`blocks.L.mlp`), or one head's output at `attn.hook_z`.
- **`group:`** expands to the named selection's neurons through `wsp.selections`
  (the inspector's own expansion: ranges expanded, inner selections followed), each
  pushed with the entry's mode and dose. What the selection holds that is not a
  neuron is not pushed and is counted in the entry's report as `skipped`:
  `{"unresolved": n, "heads": n, "slices": n, "members": [first 5 unresolved]}`.
  Saved selections are described in [SELECTIONS.md, "Named selections"](SELECTIONS.md#named-selections);
  `GET /api/steer/groups` lists them as steering targets (below).
- **`direction:<name>@<k>`** adds one vector to the residual
  stream entering layer `k` (`blocks.k.hook_resid_pre`, the point where the measurement read
  its index `k`), at every position. The vector is `dose * scale * v`: `v` is row `k` of the
  measurement's direction table (the unit direction the Directions tab's Layers view shows,
  with `orthogonalize_direction` as in the entry); `scale` is the separation of the bad mean
  from the good mean along `v` for `unit: "diff"` (so dose 1 moves the stream by that
  separation) and 1 for `unit: "raw"`. A negative dose pushes towards the good mean (removes
  the behaviour), a positive one towards the bad mean. It needs no calibration. `means_sha256`
  pins the measurement: a means file that has changed since is refused. The vector is computed
  on the server from the saved means. Entries at the same layer are summed (in float64, cast
  to float32 once); one entry steers one index, so a span of layers is several entries. The
  index after the final norm (`k = n`) is refused, since no layer reads it. The full account,
  with the refusal texts: [DIRECTIONS.md, "Steering along a
  direction"](DIRECTIONS.md#steering-along-a-direction).
- Several entries on the same neuron combine: `add` doses are summed per unit,
  multipliers are multiplied. At each layer the hook computes `x * m + c`, so a
  neuron that is both multiplied and pushed is multiplied first.
- An empty list or `null` steers nothing; the model runs exactly as without it.

**Refused, with the entry named** (`steer entry 2 (L11H3): ...`): an unknown
address or field, a layer, neuron or head that does not exist, a group that does not
exist or has no neurons, `add` on anything but neurons or a direction, `unit` on `mul`, a
missing, non-numeric, boolean or non-finite dose, a dose over 1e6, more than 256
entries or more than 131,072 neurons in total. A head is refused on a layer with
linear attention (Qwen3.5 `linear_attention` layers have no per-head output) and on a
checkpoint layer with no separate attention output projection. A `resid` push on a
neuron whose output weights have norm 0 is refused; use `raw`.

## Calibration

`p50(L)` is the median, over tokens, of the residual-stream norm leaving layer L
(`blocks.L.hook_resid_post`), position 0 excluded, as torch's `median()` takes it
(the lower of the two middle values for an even count). It is measured once, on
the base model, over `REFERENCE_TEXT` in `wsp/steering.py`, which is
neuron-explorer's reference text verbatim (MIT licence, Copyright (c) 2026
lyramakesmusic).

- **Tokens.** A Hugging Face tokenizer encodes the text with its own special tokens
  (transformers' default, as neuron-explorer's `tok(REFERENCE_TEXT)`), whatever the
  workspace uses for prompts, and nothing is cut. Two cases are not exact and say
  so in `calibration` (`exact: false`, with `truncated_to` or `fallback`): a model
  whose context is shorter than the text (cut to the context), and a toy vocabulary
  that cannot spell it (its lower-case form, then only the characters it can spell).
- **When.** Only when a specification has a `resid` push on a layer with no given
  `p50` and no usable cached measurement, or when the request says
  `"recalibrate": true`. It is one forward pass of the base model; only one number
  per layer comes back.
- **Who.** The server starts it and writes the cache. On a supervised model the
  forward pass runs in the model worker as the job `steer_calibrate`; the worker
  never starts a calibration, never prepares a specification and never writes the
  file.
- **Cache.** `steering/calibration.json` in the workspace, keyed on the model's
  identity (GGUF base SHA-256, or the checkpoint files' names, sizes and
  modification times), the SHA-256 of the reference text and the SHA-256 of the
  token ids. Another model, text or tokenization is another key, so an entry
  measured with an older text or tokenization is never used again. Up to 8 entries
  are kept; an unreadable file is measured again and rewritten.
- **Unusable values.** A measurement with any layer non-finite, zero or negative is
  not cached. Its usable layers still serve this request; a push on an unusable
  layer is refused by name ("... nothing was cached. Give p50 for layer L, or use
  unit raw"). A cached entry holding such a value is not a hit.
- **Given p50.** A request may carry `"p50": {"34": 41.2}`. Those layers are not
  measured; their entries report `source: "given"` (otherwise `"measured"`). Each
  value must be a finite number above 0 and at most 1e9, for a layer that exists,
  keyed by the layer number (`"34"` or `34`; ASCII digits, no leading zero), each
  layer once. Values for layers the specification does not push are ignored.
- Responses carry `calibration`: `needed`, `cached`, `ran`, `seconds`, `tokens`,
  `key`, `reference` (the source), `reference_sha256`, `special_tokens`,
  `truncated_to`, `fallback`, `exact`, `given` (layers whose p50 was given), and
  after a measurement `stored`, `measured` (Unix time) and `recalibrated`, or
  `stored: false` with `unusable` (layers). With every pushed layer given,
  `calibration` is `{"needed": false, "given": [...]}`.

## Relation to neuron-explorer

Compared with neuron-explorer's `explorer/runtime.py` (`_push_entry`,
`layer_resid_p50`):

- Same formula (`dose * p50 / ||down_proj[:, n]||`), same reference text, same
  tokenization, same reduction (`h[0, 1:].norm(dim=-1)` on the decoder layer's
  output, then `median()`).
- So a constant here equals neuron-explorer's when it has no sweep statistics for
  that layer (its `REFERENCE_TEXT` path), up to the precision the two models run in.
- When neuron-explorer has sweep statistics (`q35-4b_L{layer}.npz`) it uses their
  p50 (`resid_norm_q10_50_90[1]`) instead, so its constant differs from the one
  here by the ratio of the two medians. Give that value as `p50` and the constants
  are the same.
- Two differences remain. Here a neuron that is both multiplied and pushed is
  computed as `x * m + c` (multiplied first); neuron-explorer applies its entries
  in list order. `probe_reads` reports a steered neuron's value after a push at the
  same layer; neuron-explorer's `read()` reports it before.

## The wire form

What crosses the model-worker boundary. The server turns a specification into it;
the worker checks it again and builds the hooks next to the weights:

```json
{"version": 1,
 "layers": {"34": {"add_resid": [[3838], [0.75]],
                   "add_raw":   [[17], [1.5]],
                   "mul":       [[3838], [2.0]],
                   "mlp_mul":   0.5,
                   "head_mul":  [[3], [0.0]],
                   "add_vector": [0.1, -0.02, "..."]}},
 "p50": {"34": 41.2},
 "calibration": "<key>"}
```

Each pair is `[indices, values]` with indices unique and at least one of them (an
empty pair is refused; leave it out). Layer keys are the layer number as ASCII
digits without a leading zero (`"01"`, `"١"` and `"¹"` are refused), each layer
once. `p50` is required for each layer that has `add_resid`: finite, above 0 and at
most 1e9. A wire form without it is refused ("prepare the specification first").
Anything malformed is a `SteeringError` naming what is wrong. The worker sees only
indices and numbers; no Python callable crosses.

`add_vector` is what direction entries become: the sum of the layer's direction
vectors, `d_model` numbers added to the residual stream entering that layer
(`blocks.L.hook_resid_pre`). The server computes it from the saved means; the
measurement's name, its means and its hashes do not cross, and the worker reads no
workspace file for it. The worker checks it again (texts for a model with `d_model`
896): "steer layer 12 add_vector must be a list of 896 numbers (d_model)" for anything
but a list of exactly `d_model` items,
"steer layer 12 add_vector value must be a finite number of at most 1e+09" for an item
that is not a number (a boolean, a string, `null`), not finite, or above 1e9 in
magnitude, and "layer 12 has no hook point at its input (hook_resid_pre), so a
direction cannot be added there" for a layer without one. It needs no `p50`.

## Column norms

The column norm `||w_out column||` comes from the logical weights of the variant
being run (base plus enabled patches plus any candidate edit), read through the
store's logical read path: never the packed storage as floats, never a whole
tensor.

- The reads are planned over the neurons asked for. For a quantized GGUF base the
  plan works in the tensor's packed blocks (256 neurons for the K types, 32 for
  Q8_0 and Q4_0). A read covers only blocks that hold a needed neuron, consecutive
  such blocks share a read, and no block is split between reads, so no block is
  decoded twice. For unquantized storage, neurons at most 64 apart share a read.
- Each read holds at most `max(4M, block * d_model)` logical values. The GGUF store
  dequantizes at most 4M values per call, several rows at a time.
- Norms are cached per layer until the patch stack changes (a commit, a toggle, a
  strength change or a delete).
- `POST /api/steer/resolve` reports what was read under `reads` (`reads`,
  `elements`, `seconds`, `cached`).
- A read that takes more than 0.5 s publishes progress on `/api/events`, at most
  every 0.25 s, then one final event:

  ```json
  {"type": "steering", "workspace": "tiny", "state": "reading", "layer": 34, "done": 96, "total": 156, "seconds": 0.62}
  {"type": "steering", "workspace": "tiny", "state": "read", "done": 156, "total": 156, "seconds": 0.98}
  ```

  `done` and `total` count `resid` neurons; cached norms count as done at once.
  `/api/steer/resolve` publishes these, and so do the probe routes when the model
  runs in the server process. With a model worker, the probe's reads happen in the
  worker and are not reported.
- Other steering events on the same stream: `calibrating` (`tokens`), `calibrated`
  (`seconds`, `tokens`) and `calibration_unusable` (`layers`).

The read plan is `wsp.selections.windows`, the same planner the inspector's scale and
revert use for a selection's packed columns (SELECTIONS.md, "How a large edit is
stored"), and the store read is the same `GGUFStore.read`.

Measured on synthetic Q4_K bytes for a 5120 x 17408 `down_proj` (the gguf package's
own dequantize, no model), through the merged code: one neuron 0.012 s; 156 neurons
spaced 100 apart 0.79 s (67 blocks, each decoded once, 23 reads); 2048 contiguous
neurons 0.10 s (3 reads). The per-row decode this replaced took about 21 s for the
same 156 neurons.

## Where steering runs

`generate(..., steer=...)` on every runner. `next_logits(..., steer=...)` exists
only on `GGUFRunner` and `RemoteRunner`; on the others `probe_next` runs the
forward pass with the steering hooks instead.

- `FunctionalRunner` (bundled `.wsp` models): the hooks are passed to the forward pass
  at every step. A direction's vector is added at `blocks.L.hook_resid_pre`, which the
  forward calls on each layer's input and whose return it keeps.
- `HFRunner` and `GGUFRunner`: forward pre-hooks on the MLP's down projection
  and the attention output projection, and, for a direction (`hook_resid_pre`), on the
  decoder layer itself, so the layer's input is the pushed stream; registered for one
  call and removed in `finally`, also after a refusal or error. A layer whose structure
  names no decoder-layer module has no such hook point, and a direction there is refused.
  With the KV cache each position goes through the hook once, at prefill or at its decode
  step, so the push is applied once per position. Resolving may calibrate (a base forward pass), so the runner
  selects the requested variant again before generating.
  On a patched GGUF projection the pre-hook changes the one input `PackedLinear.forward`
  hands both the packed product and the patch overlays, so the overlays add their change
  to the pushed input. `forward` passes that input on in its own dtype (no float32
  cast); the hook's constants are cast to the input's dtype and device, and the
  overlays are built in the output's (`tests/test_steering_selections.py`: float32,
  float64 and bfloat16 with a stand-in kernel).
- `RemoteRunner` (supervised GGUF): sends the wire form in the `generate` or
  `next_logits` job arguments.

On a transformers model on an Apple GPU (any device other than the CPU), the probes
see activations as float32 CPU copies and compute the push there, while generation
applies it in the model's dtype on the device. The next-token check is therefore
close to, not bit-identical with, the distribution generation samples from.

## Steering a Run or Chat reply

`POST /api/run` and `POST /api/chat` take three optional fields, the same as on
`/api/steer/resolve`:

| Field | Meaning |
|---|---|
| `steer` | A steering specification. Absent, `null` or `[]`: the request is exactly as before (the runner is not passed `steer` at all). |
| `p50` | `{layer: value}`, residual norms given instead of measured. |
| `recalibrate` | `true` measures the calibration again even when cached. Must be a boolean. |

```json
{"prompt": "The capital of France is", "max_new_tokens": 32, "stream": "run-7",
 "steer": [{"addr": "L1N3", "mode": "add", "dose": 0.75}, {"addr": "L0H1", "mode": "mul", "dose": 0}]}
```

- **Checked first.** `steer`, `p50` and `recalibrate` are checked before any model is
  loaded or any runner made: a bad specification, a bad `p50` or a non-boolean
  `recalibrate` is a 400 naming the problem, as on `/api/steer/resolve`.
- **Without `steer`** (absent, `null` or `[]`), `p50` and `recalibrate` are ignored,
  whatever they hold: they only shape a push, and the request is exactly as without
  them. The same holds on `/api/probe/reads` and `/api/probe/next`.
- **Only `current` is steered.** The base reply runs exactly as without `steer`, so
  the two columns compare the steered edited model with the unsteered base model.
  In `/api/run`, `current.steered` is `true`, and `current.next` (the next-token bars)
  is the steered distribution; `base.next` is unsteered.
- **Calibration** runs, when needed, after the model is loaded and before the edited
  reply starts: one base forward pass, then cached (see "Calibration"). The column
  norms are read at the same point.
- **Report.** The response carries `steering`:

  ```json
  {"variant": "current", "resolved": true,
   "entries": [{"index": 0, "addr": "L1N3", "mode": "add", "dose": 0.75, "unit": "resid", "source": "measured",
                "p50": {"1": 7.585}, "norm": 0.8127, "value": 7.0, "...": "..."},
               {"index": 1, "addr": "L0H1", "mode": "mul", "dose": 0.0, "kind": "head", "value": 0.0}],
   "calibration": {"needed": true, "cached": false, "ran": true, "...": "..."},
   "layers": 2, "neurons": 1, "heads": 1, "mlps": 0, "reads": {"reads": 1, "elements": 24, "seconds": 0.0, "cached": 0}}
  ```

  `entries` are as from `/api/steer/resolve`. A `raw` entry has `unit: "raw"` and
  `value` equal to its dose, and no `source` or `p50`. The `wire` form is not
  returned.
- **Events.** The `steering` events of a Run or Chat request (`calibrating`,
  `calibrated`, `calibration_unusable`, `reading`, `read`) carry the request's
  `stream`, so the browser shows them on that request's card. Events from
  `/api/steer/resolve` and the probe routes carry no `stream`.
- **Stop.** `POST /api/generation/stop` works as without steering. The calibration
  pass and the column reads cannot be cut short; a stop that arrives during them is
  seen before the edited reply starts, so nothing is generated: the response has
  `stopped: true`, both variants `not_run`, and `steering.resolved: false` (entries
  and calibration as prepared, no norms). A calibration that finished is still
  cached. A stop during the steered reply keeps the text so far, as without
  steering; the steering hooks are removed (in `finally`) and the next request runs
  unsteered unless it asks for steering again.
- **Chat history.** Each model answers its own conversation. The browser sends the
  edited model's as `messages` (its assistant turns are the edited model's replies,
  steered when they were) and the base model's as `base_messages` (the base model's
  own unsteered replies). The server renders each (once when they are equal) and
  generates each reply from its own prompt: the edited reply (steered), then the
  base reply (never steered). See
  [Chat: a conversation per model](RUNTIME.md#chat-a-conversation-per-model).
- **A base-only chat request** (`variants: ["base"]`, the base column's own Send)
  does no steering work at all: no calibration and no column reads. A steering
  request on it (a non-empty `steer`) is a 400 before the model is touched:
  "steering applies to the edited model only, and this request runs the base model
  only". Without one, `p50` and `recalibrate` are ignored, as on any request without
  steering. The browser sends none of these, and a problem in the steering panel
  does not block a base send.
- **Patches.** Steering writes no weight and no patch, through these routes as
  through the others.
- **Busy state.** A steered Run or Chat request goes through `util.api` like any
  other, so the inspector's `generationStarted` hook fires when it is sent (before
  calibration and the column reads) and `generationSettled` when its answer arrives,
  stopped or not (`tests/steer_review.test.mjs`).

## In the browser

### The steering panel (Run tab)

One panel serves completion and chat; it sits under the prompt in completion and
under the controls in chat.

- **Header.** Built from the request the panel would send: "the edited model is
  steered: <entries>", "off: the edited model runs unsteered", or, while an enabled
  entry or the per-layer p50 has a problem, "not steered (fix 2 entries)", "not steered
  (fix the per-layer p50)" or "not steered (fix 1 entry and the per-layer p50)". An entry whose dose cannot be read is labelled "(no valid dose)",
  never `+0`.
- **Entries.** Each row has a target (`L34N3838`, `L34 N3838`, `blocks.12.mlp`,
  `L11H3`, `group:name`, ...), a mode (add or multiply), a dose, a unit
  (residual-norm fraction or raw), a switch and remove. Each row says how many
  neurons, heads or MLPs it pushes; for a group, from `/api/steer/groups`, with what
  it leaves out. A direction `direction:<measurement>@<k>` is added only, in
  difference of the means (`diff`, the default) or raw, and keeps the
  `means_sha256` pin and `orthogonalize_direction` of the Directions tab's Layers
  button that made it (sent as `means_sha256`, and `orthogonalize_direction: false`
  only when off).
- **Saved selections.** A picker lists the workspace's saved selections with their
  neuron counts ("add selection" adds `group:<name>`). The 131,072-neuron bound per
  request is checked before sending, over all enabled entries, and named when it is
  passed. A switch over the whole
  panel turns steering on and off without losing the entries. Combinations the
  server refuses are disabled with the reason beside them: add is disabled for a
  head or an MLP, the unit is disabled for multiply, and an unread target or dose is
  named. While an enabled entry has a problem, Run and Send are blocked with
  "Steering: ..." in the request checks.
- **Recipe.** A pasted recipe adds entries; items it cannot read are listed with
  the reason, and the rest are kept (see "Recipe notation").
- **Check** calls `/api/steer/resolve` and shows, per entry, the constant added,
  the output column length, the layer's residual norm (p50) and where it came from
  (measured, given, raw, multiplier, or not measured yet). If calibration has not run for this model,
  Check does not run it (`calibrate: false`); it says "Calibration will run before
  the first steered reply: one forward pass of the base model over a reference text,
  then cached for this model." and offers "measure now". For a direction entry the
  table gives instead the length of the vector pushed (`norm`, |dose| × `scale`) and
  what one unit is (`unit diff: one unit is <scale>`, or for `raw` that the dose is the length).
- **Advanced: given residual norms.** A `p50` field (`L34: 41.2, L35: 40` or a JSON
  object) sent with every steered request, and "measure again" (`recalibrate`).
- **Labels.** A steered request's live card and result say "steered:" with the
  entries (`L1N3 +0.75, L0H1 ×0`), in the warning colour; the base column says
  "unsteered"; the next-token bars of the edited column are titled "next token
  (steered)". Chat turns keep the label of the steering the server reports they were
  generated with, as do an interrupted reply's partial text and the notice of a stopped
  reply (only when the edited reply had started; a stop during calibration or the
  column reads generated nothing). Labels come from the server's `steering` report,
  not from the panel, which may have changed since. While the request calibrates or
  reads, the live card shows that state.
- **Kept per workspace** in browser storage (`wsp.steer.v1.<workspace>`). Storage
  that is missing or throws is ignored; the panel then lasts only for the page.
- With the panel off or empty, a request sends the same body as before.

### Recipe notation

Neuron-explorer's notation, one item per line or separated by `,` or `;`. Lines
starting with `#` are skipped; backticks and list bullets (`-`, `*`, `•`, `1.`,
`1)`) are removed; the minus sign U+2212 counts as `-`. Each item is a target, then
whitespace or a `:`, then a dose with its sign against its number (`+0.75`, `-1.5`)
or a multiply mark (`x2`, which may also touch the target: `L5N9×2`), then optionally
`raw` or `resid` (for a direction target: `raw` or `diff`). Case does not matter. A recipe is at most 20,000 characters; a
longer one is refused whole, and reading is linear in its length.

| Part | Forms |
|---|---|
| Target | `L34N3838`, `L34 N3838`, `L11H3`, `L12 mlp`, `blocks.12.mlp`, `blocks.34.mlp.neuron.3838`, `blocks.34.mlp.neurons.0-127`, `blocks.11.attn.head.3`, `group:name`, `direction:name@12` |
| Add | `+0.75`, `-1.5`, `add 0.75` (unit `resid` unless `raw` follows; for a direction target, such as `direction:refusal@12 -0.5`, unit `diff` unless `raw` follows, and a multiplier or `resid` is refused) |
| Multiply | `x2`, `×0`, `*0.5`, `mul 2` |

```
L34 N3838 +0.75
L34N3838 x2, blocks.12.mlp ×0; L11H3 x0
blocks.34.mlp.neuron.17: -1.5 raw
```

Not read, with the reason:

- "ambiguous: a range or a push?" for what neuron-explorer's notes use as a range or
  a polarity: a dash or minus against the target (`L12 N5-7`, `L19 N7839−`), any
  hyphen or en/em dash (`L12 N5–7`, `L5N9 –0.5`), and a sign apart from its number
  (`L12 N5 - 7`, `L5N9 + 0.5`). The hint follows what was written: a number after
  the dash and a dose of its own (`L3 N10-20 +0.5`) is a range, and the hint gives the
  range form, which the recipe reads and the server resolves: "\"L3N10-20\" reads as
  the neurons 10 to 20; write a range as blocks.3.mlp.neurons.10-20 +0.5". A dash and
  a number with no other dose (`L3N10-0.5`, `L3N10-20`) can be either, and the hint
  names both: "to push, put a space before the dose: L3N10 -0.5; for a range, write
  blocks.3.mlp.neurons.10-<last> and a dose, e.g. +0.5" (with `10-20` when the number
  is a whole one). A dash with no number after it (a polarity note) or on a head gets
  the push hint only;
- a `+` against the target (`L5N9+0.25`: put a space before the dose);
- a decimal comma (`+0,75`: "decimal comma? write +0.75"; neither half is kept);
- a sign with no number (`L11 N6261+`, a polarity note rather than a dose: write
  `+0.5` or `-0.5`), a number with no sign or operator, add on a head or an MLP, a
  unit on multiply, a dose over 1e6, an unknown target, a missing or unreadable dose.

Every refused item is listed; the items around it are still read. The per-layer p50
field refuses a layer given twice (`12: 3, 12: 4`, or a JSON object with the same
layer twice, which `JSON.parse` alone would silently keep the last of).

### The Probe tab

Four views over the probe routes, each reading the model on texts typed there;
nothing is written. Each input shows its bound beside it: characters (`3 of 20,000
characters`), neurons, texts per set, `k` and `top` are counted and checked in the
browser before sending; the token bound is shown (`up to 256 tokens, counted by the
server`) but only the server can count tokens, so it is enforced there. A server
error appears under the button that was pressed, and while a probe
runs the line under the button says "running… If a reply is being generated, this
waits until it finishes." (or the calibration and read progress). On a model that
cannot run here the tab says so instead of showing the forms.

- **Reads**: chosen neurons' values per token for base and current, shaded, with the
  exact value on hover. "With steering" reads current with the panel's steering;
  the base row is then shown only where it differs.
- **Fires**: the neurons with the largest raw activation at one token (the last,
  until another token is clicked), with the sign to rank by and how many.
- **Next token**: the next-token distribution without and with steering, from the
  panel or from one typed entry (recipe notation), with KL, the top tokens and the
  tokens that rose and fell most.
- **Contrast**: one text per line in each set; the neurons highest on A and on B.

Neuron rows offer "inspect" (opens the neuron in Weights) and "+ steer" (adds an
add entry to the panel). The Fires and Contrast tables have "save all as selection"
and, with rows ticked, "save ticked as selection", through the inspector's
`saveSelection` (`POST /api/groups` with `neurons`) under a suggested name (`fires at
token 7`, `fires on A not B`); the saved selection can then be steered as
`group:<name>`. In the inspector, a selection's page offers "steer this selection"
(disabled, with the reason, for a selection with no neurons or more than 131,072), a
neuron, head or MLP offers "+ steering", and a scale too large for a weight patch
points to run-time steering instead. "Too large" is the limit of
[SELECTIONS.md](SELECTIONS.md#how-large-an-edit-may-be) (the Largest weight patch setting, memory
and disk), no longer a fixed 60,000,000 weights; the steering line is unchanged. `rowTools` (`probe.js`) and
`panelPickers` (`steer.js`) remain as extension points.

- **The typed next-token entry** uses the panel's per-layer p50; while that field has
  a problem the probe is refused with it, rather than sent without it.
- **Token chips and tables** write invisible characters out as `\u{XXXX}`: control
  characters, bidirectional controls (U+202E would reverse the text after it),
  zero-width characters and line separators. Run's token chips and next-token bars
  do the same.

#### As a chat message

Reads, Fires, Next token and Contrast have "as a chat message". With it, each text is
read as the model sees one user turn of a chat: rendered by the runner's
`chat_prompt([{"role": "user", "content": text}])`, the same call `/api/chat` makes
(the model's own template, generation prompt included), with the Run tab's thinking
choice as its options, and tokenized as Chat tokenizes a templated prompt.

- **Only with the model's own chat template.** `GET /api/probe/chat` says whether
  there is one, from the model's files (`chat_template.jinja`, `chat_template.json`,
  or `chat_template` in `tokenizer_config.json`) without loading it. Bundled `.wsp`
  models have none. A GGUF's is its own `tokenizer.chat_template`, which opening the
  GGUF saves with the converted tokenizer in the workspace's `model/`; a GGUF without
  one is refused with `this GGUF carries no chat template (tokenizer.chat_template)`.
  Without one the box is disabled and says why.
- **Template tokens are marked.** Chips of tokens that belong to the template (role
  markers, the generation prompt) have a dashed border and "chat template" in their
  title. The split is found by tokenizing the rendering up to the text and through
  the text; when a token spans the boundary none is marked and the result says so.
- **Bounds count the rendering.** The template's tokens count toward the token
  bound, which the server checks after rendering, once the model is loaded. A Fires
  position is a position in the rendering; clicking a token asks about the same
  rendering that was sent.
- **Comparing with neuron-explorer.** Activation numbers from neuron-explorer are
  comparable with these only with "as a chat message" on, and with the same
  rendering. Neuron-explorer's `chat_ids` (`explorer/runtime.py:570-575`) renders with
  thinking disabled; here the Run tab's thinking choice is used ("model default"
  unless changed), so for a model whose template has a thinking switch, set it to
  "off" in the Run tab to match. That file is not in this repository: this comparison
  is from the reviewer's reading and was not checked here.

## Probes

Each probe is a named worker job with an in-process equivalent (the same function).
Only the reduced numbers come back. Inputs are bounded. The routes check everything
that needs no model first (body fields, bounds, token counts with the workspace's
tokenizer, the steering specification, `p50` and `recalibrate`), so a request over
a bound is a 400 with the reason before any model is loaded; the job checks again
where the model runs. Each job opens its runtime operation with the token count it
runs (the longest text for `probe_contrast`, the reference text for
`steer_calibrate`).

| Route | Job | Bounds |
|---|---|---|
| `POST /api/probe/reads` | `probe_reads` | prompt 1..256 tokens, 1..16 neurons |
| `POST /api/probe/fires` | `probe_fires` | prompt 1..256 tokens, `k` 1..200 |
| `POST /api/probe/contrast` | `probe_contrast` | 1..16 texts per set, each 1..256 tokens, `k` 1..200 |
| `POST /api/probe/next` | `probe_next` | prompt 1..512 tokens, `top` 1..50 |

Any text is also limited to 20,000 characters before it is encoded.

Every probe route also takes `"chat": true` and `"chat_options"` (as on `/api/chat`)
to read each text as one user turn in the model's chat template (see "As a chat
message"). `chat` must be a boolean; the options and whether the model has a chat
template are checked before the model is loaded, and so is every other field: on
Fires the `position` must be an integer within the token bound (-256 to 255) before
loading, and is checked against the rendered chat's own length after it. The
rendered token count is checked after loading ("the prompt as a chat message has N tokens; at most 256"). The
response adds `"chat": {"template": [true, false, ...]}` (one flag per token, `true`
for the template's, or `null` when the split could not be found); Contrast adds
`"chat": {"rendered": true}`.

### `GET /api/probe/chat`

`{"available": false, "reason": "this built-in model has no chat template"}`. From the
model's files; the model is not loaded.

### `GET /api/steer/groups`

The saved selections as steering targets, from the selections alone (no weight read,
no model run):

```json
{"groups": [{"name": "fox", "addr": "group:fox", "neurons": 3, "layers": [0, 1],
             "skipped": {"unresolved": 0, "heads": 1, "slices": 0}},
            {"name": "broken", "addr": "group:broken", "neurons": 0, "error": "..."}],
 "max_neurons": 131072}
```

A selection the listing cannot use is listed with `error` and the rest are kept. From
a hand-edited or old `groups.json`, a member that is not text (`null`, a number) is an
unresolved member with the reason "not an address: a selection member must be text",
and a record without a list of members is "selection 'name' does not hold a list of
members". As a steering target, such a selection still pushes the neurons it has and
reports the bad members under `skipped`; with no neuron left, or no list of members,
`/api/steer/resolve`, Run and Chat answer 400 with that reason before any model is
loaded.

### `POST /api/steer/resolve`

Checks a specification and says what it will do, without running a prompt.
Calibration runs if it is needed and not cached (and loads the model for that);
`"calibrate": false` reports without running it, `"recalibrate": true` measures
again even when cached, and `"p50": {layer: value}` gives the residual norm for
those layers instead of measuring them. With every pushed layer given (or with
`"calibrate": false`), only weights are read and no model is loaded.

```json
{"steer": [{"addr": "L1N3", "mode": "add", "dose": 0.75},
           {"addr": "blocks.0.mlp", "mode": "mul", "dose": 0.5},
           {"addr": "L0H1", "mode": "mul", "dose": 0}]}
```

Response (from the tiny random test model):

```json
{"ok": true,
 "entries": [
  {"index": 0, "addr": "L1N3", "mode": "add", "dose": 0.75, "kind": "neurons", "unit": "resid",
   "neurons": 1, "layers": {"1": 1}, "p50": {"1": 7.585}, "norm": 0.8127, "value": 7.0},
  {"index": 1, "addr": "blocks.0.mlp", "mode": "mul", "dose": 0.5, "kind": "mlp", "layer": 0, "value": 0.5},
  {"index": 2, "addr": "L0H1", "mode": "mul", "dose": 0.0, "kind": "head", "layer": 0, "head": 1, "value": 0.0}],
 "calibration": {"needed": true, "cached": false, "ran": true, "seconds": 0.008, "tokens": 16, "key": "95a2...", "...": "..."},
 "wire": {"version": 1, "layers": {"0": {"head_mul": [[1], [0.0]], "mlp_mul": 0.5}, "1": {"add_resid": [[3], [0.75]]}},
          "p50": {"1": 7.585}, "calibration": "95a2..."},
 "layers": 2, "neurons": 1, "heads": 1, "mlps": 1,
 "reads": {"reads": 1, "elements": 24, "seconds": 0.0, "cached": 0}}
```

`value` is the constant added (or the multiplier). An entry with many neurons
lists the first 32 under `targets` and a min, median and max of the values and
norms instead of a single `value`.

With a given p50:

```json
{"steer": [{"addr": "L1N3", "mode": "add", "dose": 0.75}], "p50": {"1": 3.5}}
```

```json
{"ok": true,
 "entries": [{"index": 0, "addr": "L1N3", "mode": "add", "dose": 0.75, "kind": "neurons", "unit": "resid",
              "source": "given", "neurons": 1, "layers": {"1": 1}, "p50": {"1": 3.5}, "norm": 0.8127, "value": 3.23}],
 "calibration": {"needed": false, "given": [1]},
 "wire": {"version": 1, "layers": {"1": {"add_resid": [[3], [0.75]]}}, "p50": {"1": 3.5}},
 "layers": 1, "neurons": 1, "heads": 0, "mlps": 0, "reads": {"reads": 1, "elements": 24, "seconds": 0.0, "cached": 0}}
```

An error answers 400, for example:

```json
{"error": "SteeringError: steer entry 1 (L0H1): add pushes neurons only; use mul for a whole MLP or a head"}
```

### `POST /api/probe/reads`

`{"prompt", "neurons": [addresses], "variants": ["base", "current"], "steer", "p50", "recalibrate"}`.
`neurons` takes neuron addresses and ranges. With `steer`, `current` is steered
and its values are read after the push. `p50` and `recalibrate` are as for
`/api/steer/resolve` (also on `/api/probe/next`).

```json
{"tokens": 4, "neurons": [[1, 3]], "addrs": ["L1N3"], "ids": [0, 1, 2, 0], "token_strs": ["a", "b", "c", "a"],
 "values": {"base": [[-0.163, -0.162, 1.728, 1.505]], "current": [[6.832, 6.886, 10.293, 8.345]]},
 "steered": true, "calibration": {"cached": true, "...": "..."}}
```

### `POST /api/probe/fires`

`{"prompt", "position": -1, "k": 20, "sign": "both" | "+" | "-", "variant": "current"}`.
Ranks every neuron of every MLP layer by its raw activation at one position (by
absolute value for `both`). Raw activation is not normalised per neuron, so neurons
with large typical activations rank high.

```json
{"position": 3, "token": 0, "token_str": "a", "token_strs": ["a", "b", "c", "a"], "sign": "both", "variant": "current",
 "ranking": "raw activation at this position (not normalised per neuron)",
 "neurons": [{"layer": 0, "neuron": 19, "addr": "L0N19", "value": 3.199},
             {"layer": 0, "neuron": 27, "addr": "L0N27", "value": 3.115}]}
```

### `POST /api/probe/contrast`

`{"a": [texts], "b": [texts], "k": 20, "variant": "current"}`. For each neuron:
its maximum activation over each text (position 0 excluded when a text has more than
one token), averaged over set A, minus the same over set B. `up` lists the neurons
highest on A, `down` those highest on B.

```json
{"a": 2, "b": 1, "variant": "current", "measure": "mean over texts of max activation (position 0 excluded); A minus B",
 "up":   [{"layer": 1, "neuron": 9,  "addr": "L1N9",  "diff": 2.484,  "a": 2.894,  "b": 0.410}],
 "down": [{"layer": 0, "neuron": 18, "addr": "L0N18", "diff": -2.646, "a": -0.022, "b": 2.624}]}
```

### `POST /api/probe/next`

`{"prompt", "steer", "p50", "recalibrate", "variant": "current", "top": 10}`. The next-token distribution
without and with the steering. `kl` is KL(unsteered || steered) in nats (`null` if
the steered model gives probability 0 to a token the unsteered one does not).
`rose` and `fell` are the tokens whose log-probability moved most, among tokens with
probability at least 1e-4 in either distribution.

```json
{"kl": 0.150, "tokens": 4, "variant": "current", "steered": true, "measure": "KL(unsteered || steered), nats",
 "top_unsteered": [{"id": 9, "token": "j", "p": 0.448, "logp": -0.803}, {"id": 12, "token": "m", "p": 0.165, "logp": -1.801}],
 "top_steered":   [{"id": 9, "token": "j", "p": 0.477, "logp": -0.741}, {"id": 12, "token": "m", "p": 0.136, "logp": -1.996}],
 "rose": [{"id": 3, "token": "d", "delta_logp": 1.385, "p": 0.0033, "q": 0.0132}],
 "fell": [{"id": 16, "token": "q", "delta_logp": -1.925, "p": 0.0185, "q": 0.0027}],
 "calibration": {"cached": true, "...": "..."}}
```

## Costs and limits

- Calibration: one base forward pass over the reference text (432 characters) per
  model and text, then cached.
- Column norms: each `resid` neuron's output column is read once per patch-stack
  change; see "Column norms" for the read plan and the synthetic measurement. It
  has not been measured on a real GGUF file.
- Steering hooks add one multiply-add per MLP activation, and one multiply per head
  output, on each steered layer.
- A steered `generate` or `next_logits` resolves the wire form inside the worker
  (the column-norm reads happen there); that cost is not reported back to the
  `generate` caller. `/api/steer/resolve` reports the same reads when run in process.
- No steering on the base variant, on linear-attention heads, or through `run` with
  forward hooks on a supervised model (hooks are Python callables; use the wire form).
- Steering and the probes have been tested only on tiny random models and fake
  runners, with one exception: steering along a measured direction has been run on
  one real transformers model (see `DIRECTIONS.md`, "Steering along a direction").
  Nothing here has been run on a real GGUF model.
- On a steered Run or Chat with a model worker, the server reads the column norms
  for the report and the worker reads them again when it builds the hooks.
- The calibration pass and the column reads of a Run or Chat request cannot be
  interrupted by Stop; the stop takes effect when they finish.
- Check and probe requests carry no stream id, so their steering events are shown
  in the panel and the Probe tab rather than on a card; two such requests at once
  would both show the same progress, and the events appear in every open tab of the
  workspace (known limit, not fixed).
- A Run or Chat request registers its stop stream id before its fields are checked;
  a request refused by those checks removes it again, so nothing is left behind
  (known limit, not fixed).
- "As a chat message" has been tested with a stub template on a tiny random model
  only; no real chat template has been rendered by these tests.
