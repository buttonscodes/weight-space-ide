# Interactive runtime settings

The **Settings** tab is available without a model open. It saves numeric runtime and memory policy settings to the config file selected at startup, within the selected profile or its `defaults` object. Other profiles, path strings, and inherited settings are preserved. Saving changes only fields edited in the form.

**Save settings** writes the file and applies each change to the running server as far as is safe. Saving never
restarts the server, never loads weights, and never aborts, unloads or kills anything by itself. The table shows,
per setting, the saved or draft value, the value in use now (and, for a loaded model, the value it was loaded with),
when a saved change takes effect, and the source of the value in use. After a save the page lists each changed
setting under what happened to it.

## When a saved change takes effect

| Setting | Supervised server (regular `wsp serve`) | Server without a supervisor |
|---|---|---|
| Runtime floor, admission reserve, budget, ceiling, minimum reservation, swap and compression growth, model idle timeout | When saved | Next server start |
| Default output tokens, largest weight patch | When saved | When saved |
| Context tokens, CPU threads, backend | Next model load | Next server start |
| Any setting held by a command-line or original-environment override | The override wins until the server starts without it | Same |

**When saved (memory policy and idle timeout).** The server sends the new values to the memory supervisor, which
validates them itself as it does at start (the same checks, `0 < floor <= preferred` and the policy fitting physical
RAM) and judges on a fresh sample what the memory watchdog would do with them. Only then does it commit: it replaces
its values, rewrites its copy of the settings and logs a `settings_changed` record with the old and new values. The
config file is written after that answer. If the supervisor refuses, nothing is written or changed and the page
says **Not saved** with the reason. If the file cannot be written after the supervisor accepted, the supervisor is
returned to the previous values and the failure shown; if even that fails, the page says the supervisor uses the
unsaved values until the next accepted save or a server restart.

- **A save that would unload the loaded model asks first.** If the new values would make the memory watchdog unload
  the loaded model (available memory below the new floor, measured memory above the new budget or ceiling cap, or
  idle longer than the new timeout), or the supervisor cannot sample memory to tell while a model is loaded, the
  supervisor refuses the save and the page says what would happen, with **Save and unload the model** and
  **Cancel**. Nothing changes until it is confirmed. The confirmation is for that exact draft and file revision,
  and for the model and operation it was shown with: a changed draft, a save made meanwhile, another model loaded or
  an operation started since needs its own, and the page shows the new warning. The judgment is made in the same
  request that applies the values, on the same sample, so the page need not have shown a preview first. Each unload
  cause is judged on its own: one the new values add is reported even when another (for example memory pressure,
  which does not depend on the settings) is already present. Only a change of a value the supervisor applies is
  judged; a save that changes only output tokens, the patch limit or load settings never asks. With no model loaded
  (the usual case for a refused load) no confirmation is asked.
- **No answer in time is not "not applied".** The supervisor checks the request's deadline again immediately
  before it commits, so its answer normally arrives in time. If it does not, the server asks the supervisor what it
  runs (a second mailbox request, answered only after the first is finished). Running the new values: saved as
  usual (the page says the answer was late). Running the previous ones: **Not saved**, and it cannot apply later.
  No answer to that either: **Not saved**, the file is not changed, and the page keeps saying that the supervisor
  may be running the unsaved values until the next accepted save or a restart. The next save is then sent to the
  supervisor even if it changes nothing, and makes all of them agree again.
- The next admission (a load, a Run, an edit) uses the new values. A refused load can be retried at once: lower
  the reserve, save, load again.
- A running operation keeps the limits it was admitted with until it finishes; the new limits then apply. If they
  would unload the model as soon as it finishes (judged on the current sample), the save says so and asks first.
- With a model loaded and idle, the memory watchdog uses the new values from its next sample. The unload, if the
  condition lasts, happens through the watchdog's usual steps; the save itself does not unload. The draft
  estimate also shows the supervisor's judgment before you save.
- A changed idle timeout counts from the same idle start: shortening it below the time already idle unloads at the
  next sample (after the confirmation above).

**Next model load (context, threads, backend).** A loaded model keeps the settings it was loaded with; work on it
is never refused because the saved context changed. The next load (after **Unload model**, which the page offers
after such a save, or after the idle timeout) uses the saved values: the model worker's environment, the admission
forecast and the server's descriptors all take them together. A backend change is checked as at start (the bridge
must be configured and ready) when it is saved.

**No supervisor.** An embedded or unmanaged server has no memory supervisor, so memory limits and the idle timeout
have nothing to apply to; they are saved for the next start of a supervised server, together with context, threads
and backend, and the page says so.

Command-line and original environment overrides still win; the page shows masked settings and previews that
effective composition. It does not mistake environment variables written by `activate()` for new user overrides.
A value edited in the file outside Settings is shown as not in use yet; on a supervised server the next save
applies it.

The **Use capacity-aware defaults** action searches for the largest model-window
context within the conservative memory forecast, with up to 2,048 default output
tokens (less for tiny contexts). Cold recommendations use measured available RAM
minus the preferred reserve. Resident recommendations obey physical RAM minus
reserve and retained launch caps, but remain uncertain upper bounds until memory
is measured after unload. If even a two-token window cannot fit known bounds,
no preset is offered. Without model metadata, 8,192/2,048 are qualified fallbacks.

The action sets a four-hour idle timeout and changes saved budget, ceiling and minimum
reservation to **Auto**. It preserves runtime floor/admission reserve, other profiles
and paths. Existing profile values stay unchanged until reviewed and saved; reused
CLI/environment overrides still win and constrain the recommendation.

Built-in fallback defaults are 8,192 total context / 2,048 output / 14,400 seconds.
There is no additional fixed context/output policy ceiling. Positive integer
settings must be exactly representable in JSON/browser numbers. Effective context
is `min(configured context, model window)`. Prompt, complete history, template and
response share it. Saved output is a default, not a maximum for managed requests. A response request is a ceiling: the server uses the smaller of
that ceiling and the remaining space, reporting both values. It rejects a prompt
that is too large or leaves no response space, without truncating history. Run's
zero-output next-token scoring can use an exactly full prompt. Duplicate pending
submissions remain suppressed; opening Run does not generate a completion.

Native bridges retain their existing per-call row/scratch limits. Python splits
large matrix calls, and generation prefills 128 tokens per step into the same cache.
These batch sizes no longer limit total context. Full traces retain different,
potentially much larger tensors and require a separate reservation. Legacy
acceptance harness limits and consent permits remain unchanged.

The model idle timeout starts after the last operation completes. It never expires running work; expiry unloads the model while keeping the IDE available. A saved change to it applies to the running idle clock (see above).

## Memory policy and preview

GiB fields accept explicit values or blank **Auto**. Automatic policy: floor 10% of physical RAM (at least 0.5 GiB), preferred reserve 15% (at least 0.75 GiB), budget=current available−preferred, and ceiling=physical RAM−preferred. Cold Auto admission uses fresh available memory; loaded jobs admit their actual new buffers rather than charging resident weights again. The measured-process limit is fixed only during an operation; idle memory uses static caps. Explicit settings still win. Normal-pressure swap/compression growth produces warnings, defaulting to 64/1024 MiB. See [operation policy](RUNTIME.md).

- Preferred reserve is a hard admission check on remaining available memory after the forecast reservation.
- Floor is a hard running headroom stop condition.
- Budget and ceiling are both enforced. Admission and live process-tree memory use `min(budget, ceiling)`.
- Minimum reservation raises the forecast amount; it does not allocate or reserve physical RAM.

The stacked bar shows measured unavailable memory (`physical RAM − available`), estimated additional reservation, and remaining headroom. It does not add CPU and Metal allocations together on unified memory. When a model is resident, its memory is already included in measured usage; additional reservation is the proposed increase above the active reservation estimate. This is an estimate of reuse, not a measurement of the model's physical allocation.

For a resident model, the normal view uses **additional reservation**, not a second
full model allocation. For example, 47.12 GiB available, a 23 GiB active reservation
and a 23 GiB proposed reservation mean zero additional estimate and 47.12 GiB
remaining. Subtracting another 23 GiB gives 24.12 GiB only in the explicitly labeled
hypothetical “second reservation without unloading” detail. That number is not a
prediction of startup headroom and does not generate a cold-failure warning.

Headroom and Auto budget after unload/restart are unknown until remeasured; no
reclaimed bytes are promised. A resident Auto cap calculated from current
availability is shown only as a hypothetical bound. The active operation limit and idle static cap are
shown separately. Known explicit budget/ceiling failures, actual available memory
below the active floor, and insufficient estimated incremental headroom remain
warnings. A forecast above physical RAM minus the preferred reserve always warns,
even if the active reservation is unknown. Without a resident model, ordinary
cold-load headroom warnings remain.
Raw free and reclaimable available memory are labeled separately. Unknown values
stay unknown rather than receiving invented resident credit.

The preview reads scalar workspace metadata and machine telemetry only. It never reads tensors, constructs a runner, runs inference, or reserves memory. Its forecast and the supervisor both use the effective model-bounded context. The interactive forecast counts actual KV heads/head width and full-attention layers,
separate recurrent layers, bounded-query eager attention and transient cache copies.
Trace forecasts include full logits, retained layer activations and all attention
patterns. The largest admitted requirement is retained until unload, even after
patch removal or returning to chat. See [capacity details](CAPACITY.md). Estimates
are advisory; fresh admission and live watchdog checks remain authoritative.

## Largest weight patch

**Largest weight patch (GiB)** (`patch_limit_gib`) is the largest patch one edit may write, in GiB of
FP32 patch: 4 bytes per weight, the size of the patch file, so 1 GiB is 268,435,456 weights. GiB
was chosen over a weight count because it is what a person can check against free disk and memory,
and it matches the other memory fields. Blank is **Auto**: no limit of its own, so what memory and
disk allow now applies. An explicit value must be at least 1 MiB (0.0009765625 GiB, 262,144 weights,
`runtime_config.PATCH_LIMIT_MIN_GIB`): a smaller one would refuse nearly every edit, and Auto, not a
tiny number, is the way to set no limit. A configuration saved by an earlier build with a smaller
value is refused at start, naming the file, the key, the value, the minimum and the fix (set it to
null or to at least the minimum); it is not raised or cleared silently. It is validated like the other GiB fields, saved in the same
config, and settable at launch (`--patch-limit-gib`). `wsp apply` reads it from the same config
(`--config`, `--profile`) and takes `--patch-limit-gib` for one run.

It applies as soon as it is saved, with or without a supervisor (`runtime_config.SERVER_LIVE`): the save
answer lists it under `applied.now`, and Active now shows the new value.
Launch overrides still win. The supervisor never reads it. It never overrides memory: a larger value
than memory allows still meets the memory refusal. See
[How large an edit may be](SELECTIONS.md#how-large-an-edit-may-be).

## Persistence and security

Settings APIs require a local peer and the IDE Host/Origin even on embedded/unmanaged servers. Writes and previews require a per-server CSRF token and bounded request size. Input validation rejects unknown keys, booleans as numeric settings, nonfinite values, invalid relationships, and incompatible effective backend/context combinations. Saves use an advisory cross-process lock, optimistic file revision, temporary file + fsync + atomic replace. Symlink/special-file replacements are rejected. An external editor that ignores the advisory lock should be reconciled through Reload after any revision conflict.

The UI supports changing runtime policy, not config paths or profile creation. An unmanaged server can edit its startup config, but the memory policy is enforced only when subsequently launched using regular supervised `wsp serve`; a supervised server applies it when saved. Other runtime families are outside the packed GGUF memory-policy guarantee.

## Model-free verification

```sh
python -m pytest tests/test_settings.py tests/test_runtime_config.py tests/test_runtime_supervisor.py tests/test_inspection_requests.py tests/test_server_shutdown.py -q
python -m pytest tests/test_live_settings.py tests/test_live_settings_supervisor.py tests/test_live_settings_control.py -q
node --test tests/settings_frontend.test.mjs tests/live_settings_frontend.test.mjs tests/inspection_frontend.test.mjs
```

These focused tests use fake scalar workspace metadata, mocks and isolated supervisor processes. The browser acceptance fixture uses real `create_app` and `RuntimeClient` with a temporary profile and fake model metadata. It verifies save persistence, active versus saved values, warnings, repeated clicks, no-model Settings, and desktop/narrow layouts. No checkpoint download, model load, inference, personal config edit or owner-server restart is part of this verification.

## Per-request controls and draft estimates

Run/Chat output fields request a ceiling under the **active** limit; they do not change
the saved or running policy. Typed nonfinite, fractional, empty or out-of-range
values are rejected locally before a request or chat-history mutation, including
keyboard submission. Unsent chat text is preserved. A new default output count is
changed in Settings and applies to the next request once saved.
The Run view also shows a configuration-only summary of active and saved limits, the
loaded model's context, pending changes and retained launch overrides.

Editing a Settings draft immediately removes its previous estimate while a new
metadata preview is pending. Failed previews display no old numbers as a current
estimate; the preset is disabled until a valid preview or reload. Saved next-start
and unsaved-draft estimates have distinct labels. On a supervised server the draft
estimate also asks the supervisor whether the draft's memory values or idle timeout
would unload the loaded model, and warns before the save if so. Saving never unloads a
model or restarts the server by itself.

A failed Chat request is displayed separately from the conversation. Its draft is
preserved for retry, and neither the failed user turn nor an error-as-assistant
message is appended to durable history. A successful retry appends one user turn
and one model reply. Effective response ceilings are shown when context reduces
the requested amount.

The capacity preset leaves a buffer inside its calculated cap: **the larger of
512 MiB or 5% of that cap**. Both cold and resident searches target `cap − buffer`,
and the recommendation reports the cap, buffer and target. This preset-only slack
allows modest availability changes and small edits; it is not a physical
reservation or an additional hard runtime limit. If the cap cannot accommodate
the buffer, no preset is offered. This buffer does not replace the operation admission and live safety checks. Non-GGUF sliding-window runners retain their own prompt/window behavior;
the packed GGUF total-context clamp does not apply to them.
