# User configuration and regular GGUF serving

`python -m wsp serve` retains the existing default `facts` workspace (building that
bundled toy when needed). HF/safetensors and toy models do not need GGUF libraries.
`--empty` starts without opening, downloading, training or loading any model.
Opening a GGUF creates/opens metadata and its tokenizer; tensor payload hashing is
deferred to the first explicit guarded execution/edit/export. A saved patch base
must match its recorded SHA-256 before execution or a new committed edit.

Use the **same interpreter** for installation, diagnostics and serving:

```sh
python -m pip install -r requirements-quantized.txt
python -m wsp doctor
python -m wsp serve --profile packed-cpu --model /path/to/model.gguf
```

The optional dependency file pins the validated Python stack. Native libraries
are built separately with the pinned upstream source in [QUANTIZED.md](QUANTIZED.md).
No automatic dependency install, model download, shell script execution, native
build, full-model dequantization, or inference fallback occurs during setup.

## Optional declarative settings

The JSON schema is [config/schema.json](../config/schema.json); start with
[config/example.json](../config/example.json) and replace its example paths.
The schema has `version: 1`, optional `defaults`, and named `profiles`. Profiles
are selected explicitly with `--profile NAME` or `WSP_PROFILE`. There is no
implicit default profile. Settings resolve in this order:

1. Explicit CLI flags (omitted argparse values do not override anything).
2. Existing environment overrides: `WSP_ROOT`, `WSP_MODEL`, `WSP_GGML_DEVICE`,
   `WSP_GGML_BRIDGE`, `WSP_GGML_METAL_BRIDGE`, `WSP_GGML_THREADS`,
   `WSP_MAX_CTX`, `WSP_MAX_NEW_TOKENS`.
3. Selected profile, then file `defaults`.
4. Portable application defaults.

`--config FILE` selects a file, followed by `WSP_CONFIG`, otherwise:

- `$XDG_CONFIG_HOME/weight-space-ide/config.json` when set;
- macOS: `~/Library/Application Support/weight-space-ide/config.json`;
- Linux: `~/.config/weight-space-ide/config.json`;
- Windows: `%LOCALAPPDATA%/weight-space-ide/config.json` (diagnostics/config only;
  packed inference and its watchdog are not supported on Windows).

Relative file paths in a config resolve against that file's directory. Model
names and Hub IDs remain names; explicit relative GGUF paths resolve against the
config directory. `model_roots` resolves GGUF basenames without searching arbitrary
working directories. Ambiguous matches are rejected. Unknown keys, invalid ranges,
missing explicit files, invalid library manifests and unavailable explicitly
selected backends produce errors. No CWD `.env` or executable config is discovered.

`backend: auto` chooses configured compatible Metal+CPU bridges on macOS,
otherwise CPU. Metal also needs the CPU bridge for export quantization. Explicit
Metal never silently falls back to CPU. Manifest checks validate ggml 0.29.0 / ABI 1
and dependency paths; the native loader verifies the ABI at explicit execution.
No arbitrary library search/download is performed. CUDA and MLX are not implemented.

`python -m wsp doctor [--config FILE] [--profile NAME]` prints effective settings
and their sources, package versions, interpreter, backend choice/reasons, manifest
and path checks, and remediation. It does not import model runtime modules, scan
weights, probe hardware allocations, train, download or run native libraries.
Its `ready` is setup readiness, **not** a model-layout or memory-fit guarantee.
Help and diagnostics work without optional packages. Environment secrets are not
included. Config is declarative and unknown credential keys are rejected.

## Explicit Run and resource ownership

Regular `serve` starts an independent supervisor (the watchdog) and one child IDE
server. For a supervised GGUF model the supervisor also starts a third process,
the model worker, which holds the model; see [The model worker](#the-model-worker)
below. The supervisor tracks only the process trees it owns. While no GGUF
operation is active it does not preflight memory, start a worker or load a model.
Run/chat, explicit tests/trace and edit candidate tests
remain explicit inference operations. Selecting a profile or opening the Run tab
is not authorization for an agent to run a local model: agent loads still require
the user's request-specific permission. Interactive runtime actions require the
local IDE origin; a profile does not enable remote inference APIs.

Before a GGUF load the child requests a fresh, model-bound reservation including
source path/stat/identity, backend, context, parameters, weights, runtime/KV/
activation/recurrent buffers, patch copies and temporary buffers. Changing to a
larger artifact or increasing patch memory cannot inherit the first reservation.
The previous runner is released before a model switch. Per-token checks require a
fresh matching external reservation. Token/KV/recurrent cache remains scoped to a
single generation, so patch changes/undo do not reuse stale decode state.

The supervisor samples available/reclaimable memory, pressure, swap, compression,
the RSS and physical footprint of the server and the model worker, and the
worker's native ggml Metal metrics every second.
Admission failures refuse the request without loading or allocating its candidate.
Runtime pressure, headroom or operation-budget violations revoke its lease with a
structured reason. Cooperative abort has five seconds; unresolved conditions then
request model unload, with ten seconds for cleanup. If the unload does not finish
in that time, the supervisor kills the model worker (step 4 of the ladder below);
it never stops the server for a memory reason. A healthy acknowledged abort
can keep the model loaded for a fresh retry. Idle stale GPU telemetry blocks work;
ongoing stale telemetry aborts after a five-second grace. Model identity changes
require unload/reopen. Source identity and actual native Metal telemetry stay
fail-closed; no PyTorch MPS allocator is probed by the packed runner.

The last resort is to kill the model worker, never the server. The supervisor
signals the server only to forward a shutdown the user asked for (Ctrl-C/TERM,
and the forced stop when that shutdown exceeds 30 seconds), or to sweep the
process group of a server that has already exited. A memory sampling failure
blocks work with `telemetry_unavailable` and is retried every second; past a
three-second grace a live worker is stopped (its reservation is lost, naming the
sampling failure), and once sampling recovers the next load is admitted afresh. A
supervisor exception itself blocks work, stops the worker and leaves the server
running. Logs live in
platform/XDG cache `weight-space-ide/runtime/serve-*`.

Runner construction has its own five-second telemetry grace. Failed construction
closes partial native handles and telemetry threads before releasing admission;
unproven cleanup keeps supervision armed. Transient admission refusals do not
poison the workspace's runnable state. **Unload / clear failed load** releases the
runner and caches without discarding committed patches or changing safety policy;
for a supervised GGUF model it asks the worker to shut down.

Built-in fallback defaults are **8,192 total context / 2,048 output tokens**. The
capacity preset searches up to the model's native window using the conservative
forecast and known memory bounds. There is no additional context/output policy
ceiling; settings must be positive integers representable exactly by JSON/browser
numbers. Effective context is the lower of configured and model-native context.
Output is a ceiling, capped per request to the room after the full prompt/history;
only an oversized prompt or no response space is rejected. Python chunks token rows for existing ABI 1 bridges: at most 2,048 CPU
or 512 Metal rows per native call, retaining the native 128 MiB scratch/allocation
checks. Generation prefills 128 tokens at a time with one continuous cache; next-token
display uses the same bounded path. Full logits and trace semantics remain intact
and reserve separate full-sequence forecasts before execution. Plain full logits
account for transient attention and vocabulary output; retained traces additionally
account for saved activations and attention patterns.

Automatic floor/preferred reserves remain 10%/15% of physical RAM, at least
512/768 MiB. Cold admission checks the full configured-context forecast against
fresh available minus preferred reserve. Explicit budget and ceiling remain hard
total caps, additionally bounded by physical RAM minus preferred reserve.

Loaded operations charge fresh available memory only for their newly allocated
components at the actual prompt-plus-response or forward sequence length: KV and
append workspace, activations, logits, eager attention, hybrid/recurrent/convolution
buffers, captured activations/attention, and growth in patch allowance. Previously
admitted weights/runtime/temporary/patches are not charged again. No RSS, mapped
file pages, physical footprint or estimated reclaimed memory is added to available.

For one operation the measured-process cap is fixed at
`min(static caps, max(F, measured_now + new_bytes + W))`, where F is the full
configured-context forecast and W is its weight allowance. The extra W permits
partially evicted mapped weights to refault alongside retained allocator memory;
it is never an availability credit. Runtime/temporary allowances may not be fully
resident at load; this known bias is covered conservatively by preferred-versus-floor
headroom and live checks, not claimed as a guarantee. Idle memory above F but within
static caps produces a nonblocking warning recommending unload. Unsafe actual
static-cap/headroom/pressure conditions still unload.

Patch admission happens before script target reads and delta cloning, import
materialization, candidate tests and final append/save. Refusals create no patch.
Operation completion reconciles prospective patch allowance to actual committed
storage. Tests/candidates refused for memory report not-run and are not cached or
committed as successful evaluation. Token checks read only a constant-size lease;
patch storage totals are cached at construction/rebuild.

Normal-pressure swap/compression growth produces structured warnings (defaults
64 MiB/1 GiB), not app termination. The separate acceptance memory guard retains
its stricter hard growth thresholds. Warnings and aborts include measured figures,
limits and recovery advice; Run/Chat preserve partial output and retry drafts.

The Settings capacity-aware preset explicitly replaces obsolete budget, ceiling
and minimum-reservation overrides with Auto while preserving floor/preferred
reserves. Existing profiles are never rewritten on startup. Saving applies it to the
running server as described in [Settings](SETTINGS.md#when-a-saved-change-takes-effect); see also
[capacity implementation/validation](CAPACITY.md).

### Saved settings on a running supervisor

A save in Settings reaches the supervisor through its runtime mailbox (`settings` request; only the server child
writes that mailbox, so it adds no network surface). The supervisor validates the values as at start
(`settings.validate_interactive` against physical RAM, and the backend check `activate()` makes when the backend
changed), refuses changes to start-only keys and to values held by a launch override, and judges the effect on a
fresh sample when a value it applies changes. It refuses a change that would unload the loaded model, now or when
the running operation finishes, or whose effect cannot be sampled while a model is loaded (`settings_unconfirmed`,
with the effect and its fingerprint `judged`: the unload, the reservation and worker, the running operation),
unless the request's `confirmed` equals the fingerprint it judges now. The server sends it only for a candidate the
person confirmed (a token bound to the file revision, the values and that fingerprint). An unload cause counts when
the new values produce it and the old ones do not, each cause judged on its own (`observations` asked again with
each cause it reports set aside). The request carries a deadline (finite, at most one request timeout ahead) that the
supervisor checks on arrival and again immediately before it commits, after the slow work. On commit it rewrites
`settings.json` in the runtime directory and replaces its values; after the reply it logs `settings_changed` with the
old and new values, the running operation, the pinned load settings and whether it was confirmed. The server writes
the config file only after that answer; a failed write restores the supervisor (`restore: true`, accepted only back
to the values before the last change). A reply that cannot be delivered undoes the change (`settings_restored`).

When the server gets no answer in time it sends `settings` with `query: true`, which returns what the supervisor
runs. The mailbox holds one request and the supervisor handles them in turn, so that answer comes after the first
request finished, or after it was overwritten unseen and can never run; replies are atomic replaces matched by id.
(The runtime directory's `settings.json` is also replaced atomically, but it is written before the values change, so
read alone it could be stale.) The server then reports applied (late), not applied, or, without an answer, that the
supervisor may be running the unsaved values until the next accepted save or a restart.

- Admission uses the saved values from the next grant. A running operation is admitted, extended, advised and
  watched with the settings captured when it began; they are dropped when it finishes.
- The idle watch derives its limits from the saved values on every sample (`idle_watch_policy`: the policy at
  physical RAM, with `min(explicit budget/ceiling, physical RAM − preferred)` as its cap, the same static cap
  admission uses), and the idle timeout from the saved `resident_seconds`.
- Context, threads and backend are pinned at the first operation of a load: the model worker's
  `WSP_MAX_CTX`, `WSP_GGML_THREADS` and `WSP_GGML_DEVICE`, the admission forecast and the server's descriptors take
  the pinned values; the pin ends when no model remains loaded. Run and Chat size their request from the pinned
  values, computed under the runtime client's lock as the operation begins. The watchdog's own identity check of a
  reserved model compares its file only, so a saved context change is never treated as a changed model.
- A draft preview asks the supervisor (`check: true`, nothing changed) for the same judgment and shows it before
  the save.
- `GET /api/runtime` does not wait behind a save in flight: it describes the last consistent state (file and
  active settings read together).

The default **idle timeout** is 14,400 seconds (4 hours), measured from operation
completion (`resident_seconds`, 10..86,400). It never expires running work. Expiry
unloads the model and keeps the IDE available. A saved change applies to the running
idle clock, which still counts from the last completion. Saved output tokens are a request
default, not a managed maximum: each explicit request can use remaining effective
context. Legacy acceptance context/output limits and permits stay separate.

CPU threads are capped at detected performance cores (physical-core fallback) and
the pinned ABI 1 CPU bridge maximum of 32;
the default is clamped to that count. The branch validation record documents the
controlled benchmark required before changing the provisional default of four.
A saved thread count (config `defaults` or a profile) above this host's limit no
longer stops start-up: the server starts with the limit, reports the source as
`... (clamped to host limit N)`, and shows a `threads_clamped` warning in the Run
panel and in Settings until the saved value is lowered. A value from the
environment (`WSP_GGML_THREADS`) or the command line above the limit is still a
start error, because it was given explicitly for this start.

First Ctrl-C/TERM closes SSE before drain and performs bounded export/auth/native/
model cleanup. The parent forwards subsequent interrupts and enforces a30-second
shutdown bound with verified owned-process teardown. A running server must be
restarted to pick up a change; changing source, or editing the config file outside Settings, does not update an
existing process.

`doctor --inspect-model --model /path/to/model.gguf` additionally reads only the
GGUF header/directory for actual tensor parameter count/types and an itemized
forecast. It does not scan payloads or verify the source hash. Runtime preflight
adds current patches and checks live hardware; this preview is not a reservation.

Native acceptance for this change is macOS only. Linux/Windows config and hardware
cases are simulated unit tests, not native execution proof. On platforms without
the GGUF watchdog, ordinary toy/floating-point serving keeps its direct Uvicorn
path; packed GGUF inference remains unavailable. The integrated PR8 asset snapshots and capability handshake prevent a cached
old module graph from restoring automatic inspection. Reload the browser after
updating source; an existing server process still needs an operator restart.

Recorded checks: [RUNTIME_VALIDATION.md](RUNTIME_VALIDATION.md).

The [Settings screen](SETTINGS.md) edits persisted policy, previews metadata-only memory impact, applies saved changes to a supervised server (memory policy and idle timeout at once, context/threads/backend at the next model load), and shows per setting when a change takes effect.

## Live text in Run and Chat

Run and Chat show each reply while it is being generated. The edited model runs
first, then the base model: both use the same loaded weights, with the edit
patches switched on or off, so they run one after the other. Each reply starts
from the request's seed (0 when the request names none), so the replies do not
depend on the order. Only a client that sends `seed: null` to a non-GGUF model
gets unseeded sampling, where they do. The base model is on the left and the edited model on the right, as before; while
the edited model is generating, the base side reads "runs after the edited
model".

In Chat each model answers its own conversation: the edited model's holds your
messages and the edited model's earlier replies (steered, when steering was on),
the base model's holds your messages and the base model's own earlier replies. So
from the second turn the two models no longer answer the same prompt; see
[Chat: a conversation per model](#chat-a-conversation-per-model) and
[Steering a Run or Chat reply](STEERING.md#steering-a-run-or-chat-reply).

When the request finishes, the response replaces the live text. The response is
the result; the live text is only a preview of it. If the server cannot produce
live text, or an event asks the browser to keep more text than it shows (an
earlier event was dropped), that side says "Live text is unavailable; the result
appears when the request finishes." A dropped event the browser cannot detect
leaves the preview wrong until the response replaces it.

How it works, for API clients:

- `POST /api/run` and `POST /api/chat` accept an optional `stream`: an id of 1 to
  64 characters from `A-Z`, `a-z`, `0-9`, `_` and `-` that the client chooses.
  Without it (or with `null` or `false`) nothing below happens and the request
  behaves as before. Any other value is a 400 before anything runs.
- With it, the server publishes `generation` events on `GET /api/events` while
  the request is pending. Each carries `stream`, `variant` (`current` for the
  edited model, `base`) and `state`: `started`, then one final `complete`,
  `stopped` (the person pressed Stop; see below), `interrupted` (the model
  worker was stopped or the request was refused part way) or `failed` (the
  request ended with an error, which the response carries).
- A text event also carries `keep`, `text` and `tokens`: keep the first `keep`
  UTF-16 units of what is shown for that variant, append `text`; `tokens` is the
  number generated so far. Shown text can be withdrawn, for instance when a
  chat stop string completes. The final event carries `tokens` too.
- While a variant generates, text events are sent at most once every 50 ms.
  Tokens that arrive inside that interval are sent when it ends, whether or not
  another token follows. Before the final event, one more text event (not
  throttled) brings the shown text to exactly the text in the response.
- An event with `lost: true` means the server could not produce live text for
  that variant. Generation continues and the response is unaffected.
- A completion with zero output tokens has no events. A transformers model
  configured for beam search has `started` and final events but no text events.

The response format is unchanged. If the model worker is stopped while the
edited model generates, the edited reply keeps the text generated so far and the
base model is marked `not_run`. If it is stopped while the base model generates,
the edited reply is complete and the base reply is the partial one.

### Chat: a conversation per model

The Chat tab keeps two conversations, the edited model's and the base model's. A
**Send:** toggle on its own line just above the message box(es), below the
steering panel and the conversation, says how a message is sent; it can be
changed before the first message and whenever nothing is generating (while a
request runs it is disabled). It always shows the mode in force:

- **One message to both models** (the default): one box and one Send. The message
  goes to both models in one request, each answering its own conversation. The turn
  is added to both conversations only when both replies complete. Stopped: it is
  added to neither and the message goes back in the box. Interrupted: the partial
  output is shown in a card and the turn is added to neither.
- **A separate message to each model**: a box and a Send under each column, headed
  "To the base model" and "To the edited model"; Enter in a box sends that column.
  A send runs only that column's model on that column's conversation. A base send
  carries no steering (the base model is never steered), so a problem in the
  steering panel does not hold it up. Only one
  request runs at a time: while one runs, both Sends are disabled and Stop sits at
  the running column. Stopped: that column's message goes back in that column's
  box.

Switching the choice takes what is typed to the boxes it shows, so no message is
left in a hidden box: a message in the one box goes into both column boxes; a
message in one column box (or the same one in both) goes into the one box. If the
two column boxes hold different messages, switching to one message is refused with
the reason (send or clear one of them first) and the toggle stays on separate
messages. A switch clears the notices of the last send (the stop notice, a refusal,
a request error), which spoke of a box that is no longer shown; the runtime event
and interrupted-output cards stay.

While both conversations hold the same messages in the same order, the log shows
each message across with the two replies side by side. Once they differ, it shows
two columns, each one model's conversation from top to bottom; in one-message
mode a line under the box says the message still goes to both conversations.
**clear** empties both conversations and the notices. Refusals and errors appear
under the box whose Send was pressed. When the two prompts differ, the "Last
response ceiling" line is given per model. The conversations live in the
browser's memory only: reloading the page or opening another model starts them
afresh.

For API clients, `POST /api/chat` takes two optional fields. A request with
neither answers exactly as before.

- `messages`: the edited model's conversation.
- `base_messages`: the base model's conversation. Without it the base model
  answers `messages`, as before.
- `variants`: which models answer, a non-empty list of `"current"` and/or
  `"base"`, each at most once. Without it, both. The edited model always runs
  first.
- A request with `variants: ["base"]` sends the base conversation as
  `base_messages` and no `messages`; one with `variants: ["current"]` sends
  `messages` and no `base_messages`.
- Refused with a 400 before any model, runner or worker is touched: a malformed
  `variants`; a `messages` or `base_messages` that is not a list of turns, each an
  object (an empty list and any `role` value are accepted in both, as they always
  were in `messages`);
  `base_messages` when the base model does not run; `messages`, or no
  `base_messages`, when only the base model runs; a steering request (a non-empty
  `steer`) when only the base model runs. Without a steering request, `p50` and
  `recalibrate` are ignored, on a base-only request as on any other. The GGUF
  text-only check applies to both conversations.
- The plain turn template (used when the model has no chat template, or with
  `force_template`) joins text, so each turn it renders needs a `role` and a text
  `content`, and `template` must be an object of text prefixes. Otherwise the
  request is a 400 naming the turn, for example "base_messages[1]: this model has
  no chat template, so each turn needs a role and its content as text". With
  `force_template` this is checked before the model. Without it, whether the plain
  template is used is known only once the model's own template has been asked (a
  model with a chat template may take content parts, and ignores `template`), so
  it is checked then: after the runner is built, before any operation scope or
  token.
- Every requested prompt is rendered and fitted to the context before either
  reply is generated: a base conversation that no longer fits fails the request
  before the edited reply runs. With two different prompts the error names the
  conversation ("The base model's conversation: Prompt/history uses ..."). Equal
  conversations are rendered once.
- The response keeps its keys. `prompt`, `chat_template` and `generation`
  describe the edited model's prompt when it runs, otherwise the base model's. A
  request that uses `base_messages` or `variants` also gets `prompts` (`{variant:
  text}`) and `generations` (`{variant: {requested_output, effective_output,
  prompt_tokens, context}}`) for each model that was asked for.
- A model that was not asked for is left out of the `variants` status map (it is
  not one "not run after an interruption"), its `base` or `current` text is `""`,
  and no stream events are sent for it.
- The runtime-worker operation is admitted for the longest sequence (prompt plus
  output ceiling) among the models asked for; they run one after the other.

### Stopping a reply

While a Run or Chat request is pending, a **Stop** button sits next to Run or
Send. Pressing it shows "Stopping…" at once and the button cannot be pressed
again; if the stop request itself fails, the button says so and can be pressed
again.

- **Chat.** The turn is added to neither conversation, whichever reply was being
  generated: a stopped base reply also discards a complete edited reply. The live
  replies disappear, and the message goes back into the input box it was sent
  from (with separate messages, that column's box) with the cursor in it, ready
  to correct and send again. If the box holds text that does
  not start with the message, the message is put before it, so nothing typed is
  lost (the box is disabled while a turn is pending, so this does not normally
  happen).
- **Completion.** Nothing is kept: no result, no live text. The prompt stays in
  the box (if the box was emptied meanwhile, the run's prompt is put back), and
  one line says the run was stopped.
- **Lost race.** If the request finished before the stop reached it, its result
  is shown as usual.

A stop is not an interruption. The model worker keeps running with its model
loaded, nothing is written to the runtime record or the loss log, and the next
request runs at once without a new admission.

For API clients:

- `POST /api/generation/stop` with `{"stream": "<id>"}` (the id the Run or Chat
  request was sent with) answers at once with `{"ok": true, "found": <bool>}`. A
  missing or malformed id is a 400. It does not wait behind the request it stops:
  it is the one POST the supervised server does not serialise.
- `found: false` is not an error. It means no running request has that id. If no
  request has used the id yet, the stop is remembered (with no time limit; the
  last 256 such ids are kept) and applies when that request arrives. This covers
  a stop that overtakes its request and a request still waiting behind another
  operation, however long that takes. A stop for a request that has finished
  does nothing.
- The stopped request answers HTTP 200 with top-level `"stopped": true`. The
  variant being generated has status `stopped` with the text and token count
  generated so far (`/api/run`: `status`, `text` and `tokens` in that variant;
  `/api/chat`: `variants[v]` is `stopped` and `tokens` is `{v: n}`). Variants not
  started are `not_run`. A variant stopped before its first token has status
  `stopped` with 0 tokens. The final `generation` event of the stopped variant
  has state `stopped`. If the model worker was also interrupted, the response
  carries `interrupted` and its `event` as well, and the browser shows that
  event next to the stop notice.
- A stream id belongs to one running request: a second request with the same id
  while the first runs is a 400. Requests without a stream id cannot be stopped
  and behave as before.

How it ends:

- **The model worker (GGUF).** The server writes `stop.json` naming the job into
  the runtime directory. It does so at the next token it receives, or, while the
  worker is still reading the prompt and sends no tokens, within a second. In a
  `generate` job the worker reads that file after each token it sends (`_job`'s
  `emit`). It also reads it at its lease checks (`WorkerLease.require` and
  `layer_checkpoint`), which run between prompt chunks and, at most every 50 ms,
  between layers. It then ends the job with an ordinary result holding the
  tokens so far, possibly none. The job's key-value cache is discarded, and the
  worker keeps its model. During generation a stop takes effect one or two
  tokens after it is pressed. While the prompt is read, it takes effect within
  about a second plus one layer.
- **`/api/run` before generating** scores the prompt for the next-token display.
  A stop during that scoring skips generation: the variant is `stopped` with 0
  tokens.
- **A transformers model** stops through a stopping criterion, one or two tokens
  after the stop is seen (on an Apple GPU transformers checks stopping criteria
  one step late). A transformers model configured for beam search stops only
  between the edited and base replies. In-process runners (transformers, the
  small built-in models, and GGUF run without the model worker) check the stop
  only at tokens, so a stop while
  they read the prompt takes effect at the first token.
- **Before the first token.** A stop that arrives while the request waits for
  another operation, or for the model, generates nothing. One exception: a
  server without the runtime supervisor that runs a built-in model in process
  generates one token when the request was waiting on the model's own lock. A
  stop does not cancel a load already under way: the request answers when the
  load ends, and the browser shows "Stopping…" until then.
- **A steered request.** Run and Chat take an optional `steer` that pushes the
  edited model's activations (see [Steering a Run or Chat reply](STEERING.md#steering-a-run-or-chat-reply)).
  Its calibration pass and column reads, before the edited reply, cannot be
  interrupted: a stop then takes effect when they finish and nothing is
  generated, and the response's `steering.resolved` is `false`. A stop during
  the steered reply behaves as above, and the steering hooks are removed with
  the request.

## The model worker

### Three processes

| Process | Command | Started by | Holds |
|---|---|---|---|
| Supervisor (watchdog) | `python -m wsp serve` | the user | memory sampling, admission decisions, the lease, the escalation ladder, and the worker process (start, kill, reap). Never a model. |
| Server (IDE) | `serve --runtime-child DIR` | the supervisor | the listener, UI and API, the session, workspace metadata, committed patches, task scope, settings, the MCP edit path, the tokenizer, and the only request mailbox to the supervisor. For a supervised GGUF model, no runner, native kernel or Metal buffer. |
| Model worker | `python -m wsp worker ...` | the supervisor, inside an admitted load, at the server's request | the runner, native kernels, its read-only mapping of the GGUF, its own copy of the committed patches, per-generation caches, activations, Metal buffers, and the GPU telemetry file. |

The `worker` subcommand is internal. It needs a one-time token that the
supervisor passes in its environment; it is not meant to be run by hand.

Only supervised GGUF models use a worker. Toy, floating-point and unsupervised
runners stay in the server process, as before, and are not under the memory
policy. GGUF export quantization also stays in the server, under the lease; it
is a quantizer, not inference.

### How a load starts a worker

1. The server begins a `load` operation; admission is the same as before.
2. Inside that operation the server asks the supervisor to start a worker. The
   supervisor applies the same checks as for `begin` (sampling healthy, no
   ladder action in progress, model file unchanged). A second live worker is
   refused with `worker_busy`.
3. The supervisor starts the worker in its own process group and replies at once
   with the socket path, the worker pid and the token. It does not wait for the
   worker, so sampling never stops.
4. The worker publishes idle GPU telemetry immediately and keeps it up until its
   runner exists, so a slow cold start does not trip the telemetry deadline.
5. The server connects (retrying for up to 20 seconds), authenticates with the
   token and sends the `load` job. The worker opens the workspace read-only,
   compares the model file's stat and recorded SHA-256 with the admitted
   descriptor (no rehash), builds the runner and replies with context, hook
   names, threads, backend and pid.

The socket is a Unix socket in the 0700 runtime directory. If that path is
longer than 100 bytes, a private 0700 directory under `/tmp` is used instead and
removed at teardown. The worker exits if nothing connects within 30 seconds, if
the server's connection closes, or if its parent supervisor dies, so no
unsupervised process keeps a model loaded.

The supervisor refuses a runner load that does not come with a worker:
`in_process_runner_unsupported`, "Supervised model runners run only in the model
worker; this in-process load was refused." It means the server was started in a
way that would run the model in its own process, with no step 4 available.
Restart the IDE with `python -m wsp serve`; if it persists, inspect the server
configuration.

### What runs in the worker

Plain runner calls are forwarded: `logits`, `next_logits`, `generate` (one frame
per sampled token) and `run` with named captures. Four analyses run whole in the
worker as named jobs, with the same code as before: trace, activation patching,
ablation scan and atlas attribution. Only their results come back: trace returns
its attention patterns to the server (a few MB of JSON for a 32-layer model), while
activations and other intermediate tensors stay in the worker. Python callables
(forward hooks) never cross the boundary. JSON arguments and results travel in
the frame body, so their size is not limited by the 1 MiB frame header. The
body is capped at 256 MiB (`MAX_BODY`); tensor payloads have no protocol cap. A
request over the cap is refused with `worker_request_unsendable`; a result over
it is that job's error and the worker stays up. The receiver allocates as the
bytes arrive, never from the declared length alone. Header and body are JSON in
strict UTF-8; any other encoding is a bad frame. A field named like a
protocol key (`type`, `job`, `tensors`, `body`) is refused when encoding rather
than dropped.

Steering and probes also cross as JSON. `next_logits` and `generate` accept a
steering wire form (indices and numbers, never a hook); the worker builds the
hooks next to the weights. Calibration (`steer_calibrate`) and the four probes
(`probe_reads`, `probe_fires`, `probe_contrast`, `probe_next`) are named jobs
that return only reduced numbers. The server prepares every steering
specification, including calibration, before it sends a job; the worker never
writes the calibration file. See [STEERING.md](STEERING.md).

Transfers are kept small. Only float32 values and int64 token ids or coordinates
cross; anything else is refused with `transfer_unsupported`, never cast. A
request the server cannot encode is refused with `worker_request_unsendable`
before anything is sent; it is not a worker loss. Integer
GGUF storage never crosses, and neither process dequantizes. Test cases and
script `logits()` ask for the last row of logits only, `run` returns only the
captures it was asked for, a drift check brings at most 32 rows of logits to
the server per call, and `observe_window` returns one window's reductions only.

Every job carries the workspace version and a manifest of the committed patches
(id, creation time, enabled, strength, stored size). The worker loads patches it
has not seen from the workspace's patch files and checks each one against the
manifest. Any mismatch refuses the job with `workspace_out_of_sync`; the worker
never runs on a guess. It never writes the workspace, the patches or the GGUF.
Uncommitted candidate edits are sent with every job that uses them and freed when
that job ends; nothing is cached between jobs.

Inside a job the worker checks the lease itself, at constant cost per token. When
it needs more memory admitted (a longer forward, an edit preflight), it asks the
server, which asks the supervisor on the operation's own thread. The server keeps
the only mailbox.

**Not available on a supervised GGUF model:** ROME edits (they need gradients
through the packed kernels). They refuse before any model is loaded. Before the
worker existed, they loaded the model first and then failed.

**The atlas observed stage on a supervised GGUF model** runs one probe window per
`observe_window` job. The worker runs the window once (batch 1). It reduces each
MLP layer inside the forward pass, through a hook, as the layer runs, so no
layer's activations are held once reduced. It turns attention patterns (where the
runner gives them) into per-head sums, normalises the final stream, and takes the
loss from the logits 16 positions at a time. It returns only those reductions:
float32 sums and values, int64 counts and positions, the loss as text. No
captures and no logits cross. The server merges windows into the stage's running
totals and writes the files after the last window. Each window is its own
operation, admitted as a forward that holds named hook points (the `trace`
forecast for the window's tokens), so the supervisor and the lease have a
boundary between windows. The job takes committed weights only (`base` or
`current`, no `extra`). A lost worker or a refused admission fails the build with
the runtime's record kept (`RuntimeRefusal`), its message replaced by the stage's
plain words: which window stopped, and for admission how much was asked for
against what was free above the reserve and what to do. The HTTP job holds it as
`event`, and the MCP `atlas` tool returns `{"status": "not_run", "event": …}`. The
stages built before it are left as they were. The server's own memory for the
stage's bias fit is bounded by a row cap (see ATLAS.md), the same on both paths. A default CLI or MCP atlas build leaves
the stage out on such a model (it loads and runs the model), and an explicit
request runs it. See [ATLAS.md](ATLAS.md#the-observed-stage-on-a-model-in-the-model-worker).

### Memory across two processes

While a worker is live, the process memory in the measured total is the server's
physical footprint plus the larger of the worker's footprint and RSS. The server
is counted by footprint because its RSS can include clean GGUF pages read for
weight views, which are the same page-cache pages the worker maps. With no
worker the measured total is unchanged. The budget, runaway and
idle-above-forecast checks use this total.

The forecast bounds the model's memory, measured as the worker process (the
larger of footprint and RSS) plus server growth since the reservation's first
admission. The server's own pre-model footprint (about 0.9–1.0 GiB, Python +
torch) is not part of the forecast, so whole-application memory can exceed the
forecast by that amount. This holds while a cold load runs (its running limit
is the admitted budget plus that server footprint, within the static caps; an
Auto budget can itself be above the forecast), for a loaded operation (its
limit is that server footprint plus the forecast, or the measured total plus the
operation's new buffers and the weights, if larger), and at idle (memory is
compared with the forecast after subtracting it). The admission checks
themselves are unchanged, so nothing more is admitted. The server footprint is
fixed at the reservation's first admission (usually when the workspace is
opened); server growth after that counts as model memory, so it reduces a loaded
operation's room and can raise an `idle_above_forecast` warning. The static caps (budget, ceiling, total − preferred reserve), the
floor and pressure checks still use whole-application and machine figures. With
no worker (an in-process runner) nothing is subtracted.

Each `sample` row in `watchdog.jsonl` records `server_tree` and `worker_tree`
separately. Growth in the server counts, but step 4 can only kill the worker.

**Runaway envelope.** Growth of more than 4 GiB between two samples is runaway
only when the footprint is also above the running operation's admitted envelope:
the footprint before the first admission, plus the admitted forecast, plus the
operation's new buffers. An admitted cold load of a large model can grow faster
than 4 GiB a second while its weights come in; inside the envelope that is
admitted allocation. The budget check runs first and still catches overshoot.
Without a running operation the plain 4 GiB rule applies.

### Escalation ladder

1. **Refuse before starting.** Nothing is allocated.
2. **Abort the operation.** The lease is revoked with a reason; the operation
   stops at its next checkpoint. A healthy, timely abort keeps the model loaded.
3. **Unload the model** when the condition persists while idle, or the abort did
   not finish within 5 seconds. The server asks the worker to shut down.
4. **Kill the worker** when the unload did not finish within 10 seconds, or when
   runaway growth continues at the next sample after an abort. The supervisor
   sends SIGKILL to the worker's verified process group. The server is not
   signalled. Without a live worker, step 4 is a recorded no-op
   (`kill_worker_skipped` in the log) and the lease stays revoked.

### When the worker is stopped

The user sees one of two codes, both with action `worker_killed`:

- `worker_killed`: step 4 killed the worker.
- `worker_exited`: the worker ended for any other reason with a non-zero status
  or a signal (a crash, an out-of-memory kill by the OS, a kill from outside).
  The return code is included.

Exit status 0 is an expected stop (a requested shutdown, or an answered failed
load) and produces only a `worker_stopped` log entry.

The message is "The model worker was stopped; the model is unloaded. The IDE
remains available.", followed by the return code when there is one and the cause
from the original trigger. The recommendations are the trigger's own plus "Run
again to load the model through fresh admission." The Run panel shows it as a
runtime card (`Runtime worker_killed: worker_killed` or `...: worker_exited`)
until it is dismissed or the next operation succeeds. The server reports the
loss, never "watchdog unavailable": the lease keeps the loss record and
`control.json` keeps a `worker_lost` entry until the server releases the
reservation.

How each request ends:

- Run and Chat: HTTP 200, the variant marked `interrupted`, with the text
  generated so far. The edited model runs first, so a loss while it generates
  leaves the base model `not_run`.
- Trace, patching, ablation, attribution, tests, candidate tests, load: HTTP 409
  `not_run`. Suite cases that could not run are "not run", never failed; a drift
  check refuses instead of reporting zero drift.
- An edit that was already committed: `committed_interrupted`, with its patch id.
- `/api/state` and `/api/runtime` keep answering.

Within the operation that saw the loss, further model calls refuse at once; there
is no reload in the middle of an operation. The next Run drops the lost runner,
goes through fresh admission and starts a new worker. That cold start adds a few
seconds.

**What survives:** the server and its listener, the browser session, the open
workspace, committed patches (on disk), task scope, the session log, settings,
and chat history in the browser. **What is lost:** the loaded model, the result
of the operation in flight (apart from text already generated), and the worker's
caches.

Other stops: a worker still alive 5 seconds after a release gets SIGTERM, then
SIGKILL 5 seconds later. If memory sampling fails for 3 seconds, a live worker
gets SIGTERM, then SIGKILL after 10 seconds; its reservation is lost (cause
`telemetry_unavailable`), admissions are refused with `telemetry_unavailable` while
sampling keeps failing, and supervision continues, so a fresh load works once
sampling recovers. Until that lost reservation is released, its operations are
refused with `worker_killed` (cause `telemetry_unavailable`), not with the
sampling code. When sampling recovers, the `telemetry_unavailable` message is
cleared, both with no reservation and with a worker still resident (a failure
shorter than the grace); a loss it caused stays reported. Any exception while
measuring counts as a sampling failure, and each failing second leaves a
`sampling_error` log row naming it. The grace is continuous: a failure that keeps
recovering within 3 s never stops the worker, but each failing window refuses
the operations admitted in it. If the supervisor itself fails, the worker gets SIGTERM
at once and SIGKILL after 10 seconds; the server stays up, and operations are
refused with `watchdog_unavailable`. A `begin` sent after the watchdog died can
take up to 10 seconds to be refused.

### Telling the processes apart

- `/api/runtime` reports the worker under `reservation.worker` (`pid`,
  `create_time`) while a reservation has one, and keeps it after a loss until
  the reservation is released. While a mailbox request holds the runtime client
  it answers within 0.2 s with `busy: true` and the last computed forecast,
  which is `null` if none was computed yet; Settings then gives no resident
  credit and says the forecast is unknown.
- `ps`: the supervisor is `-m wsp serve`, the server is
  `-m wsp serve --runtime-child ...`, the worker is `-m wsp worker --runtime-dir ...`.
- The supervisor prints its log path at start (`Runtime supervisor: ...`). The
  log records `worker_spawned`, `worker_signal`, `worker_stopped` or
  `worker_reaped`, a `server_signal` row for every signal sent to the server
  (reason, signal or `stop_group`, server pid), and per-process memory in each
  `sample` row.

### Costs

Committed patches are held in both processes, so a large dense patch set uses
its memory twice. Each worker cold start (first Run, and the first Run after a
kill) adds a few seconds for importing torch and opening the GGUF.

## How large an edit may be

The inspector asks before it plans a large edit how large a patch memory admission would grant
(`RuntimeClient.advise_edit`, the mailbox action `advise`). The client builds exactly the descriptor
`preflight_edit` sends for an edit (`_value(ws, name='edit', stored=0, scratch=allowance)`, with the
same 64 MiB scratch quantum, and without an operation id). The supervisor runs the checks a grant
runs up to admission (shutdown, sampling failure, model identity, the escalation ladder; a lost
reservation is judged as the cold request that would release it), takes a fresh sample, and calls
`runtime_policy.largest_patch`: a bisection over the unchanged `admission`, adding 12 bytes per patch
weight (`PATCH_ADMISSION_BYTES`, the figure `_value` adds) to the forecast. Every admission bound
grows with the patch, so the answer is the exact edge: a bind at that size is granted and one weight
more is refused. `extend_admission` refuses a subset of what `admission` refuses, so inside an
operation each step of the edit is checked against no stricter a bound.

`advise` binds, reserves, leases, publishes and logs nothing; it is answered while the supervisor
stops (with "Server is shutting down." and no answer), a refusal is not a `runtime_event` and nothing
becomes the client's `last_event`. Its answer is `{largest, event, available, preferred, scratch,
bounds}`, where `event` is the refusal one weight past `largest` and `bounds` (the budget, the Memory
budget and ceiling settings, whether the model is loaded, total memory, and the memory measured for the
server's process tree, `runtime_policy.measured_total`) lets the refusal name the bound admission stops at;
when the measured memory is already over the cap, admission refuses every size and the refusal says so.
A reply that cannot be written is logged (`advice_undelivered`) and never notifies or changes the
published status; the client has no answer. `largest` is 0 with the watchdog's own event when it admits no edit for
another reason (identity, the ladder), and None while sampling fails.

The wait is bounded: one `advise_edit` call waits at most 1.5 s in all (`ADVISE_TIMEOUT_SECONDS`, one
deadline over the client's lock, describing the edit, the mailbox and the reply). Describing the edit
(`_value`, file stats) cannot be interrupted: if it alone takes the time, the call waits that long and
sends nothing ("describing the edit ... took X s, past the 1.5 s limit"). A reason for no reply names
the time actually waited. An answer, or the lack of one, is
reused for 2 s (`ADVISE_SECONDS`) for the same workspace, version, enabled patches and scratch. A
node view asks once for its scale and revert buttons together, and `/api/selection/script` within
2 s reuses that answer, so a supervisor that does not answer delays a view by at most 1.5 s, once
per 2 s. With no answer the client returns `largest: None` and the reason (busy, the mailbox could
not be written, no reply in time, a refusal); it never raises. Planning then applies the setting and
disk and, if nothing is set, the old 60,000,000 weights, and the message gives the reason. The real
`edit_preflight` calls are unchanged and remain the final word.

The answer is exact for a fresh admission of the edit. While another operation runs it can be more
permissive than the later real grant: `largest_patch` counts patch growth against the active
descriptor's patches, which clamps at zero while those exceed the new descriptor's.
See [SELECTIONS.md](SELECTIONS.md#how-large-an-edit-may-be) for the setting, the refusal messages
and the confirmation.
