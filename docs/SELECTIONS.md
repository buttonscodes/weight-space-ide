# Selections and the inspector's actions

The inspector edits whatever is selected: a tensor, a slice of one, a neuron, a head, a range
of neurons, a key-value group, a component, or a named selection of thousands of neurons. This
page says what each action does, when it is refused, how large edits are stored, and how
named selections are saved.

## Actions

| Button | Neurons | Heads | Tensor, slice, component |
|---|---|---|---|
| scale × | multiplies what each neuron writes (its output-projection column) | multiplies what each head writes (its output-projection columns) | multiplies the selected weights themselves |
| ablate / zero | the same with factor 0 | the same with factor 0 | the same with factor 0 |
| revert to base | puts back every weight of each edited neuron (reads and writes) | puts back the head's query, key, value and output weights | puts back the edited weights |
| → console | puts the scaling script in the console to edit before applying | | |

Each weight is reached once, however the members of a selection overlap: a neuron and a slice of
its own column, a sublayer and a neuron inside it, a head and a slice of its output columns are
all scaled by the factor once, not twice. The plan works out the union of what every member
reaches in each tensor (lines along one axis plus disjoint boxes) before writing the script, and
the count it reports is exactly the weights the script changes. A selection whose members overlap
in a tensor of more than two dimensions is refused in words; its node still shows how many of its
weights are edited, counted exactly from the patches. A negative index is the same weight as its
positive one (`W[4, -17]` is `W[4, 23]` of a 40-column tensor), in the plan, in reads and in the
edited count. "Edited" means a non-zero delta, as `Workspace.changed` counts it.

Every action is one weight script, built on the server (`POST /api/selection/script`) from the
whole selection with no cap on members, and applied through the one edit path (`/api/edit`,
`Session.apply` / `Workspace.commit`) as one patch. Task scope, candidate tests, drift checks and
patch recording apply as for any other edit. On a model with grouped keys and values, reverting a
head also reverts the key and value rows it shares with the other heads of its group; the revert
button's description names those heads (for example "L0H0 with L0H1"), and for neurons says that
the weights reading the neuron's input are put back too.

Revert adds a patch: the negative of the current edits in the selection, worked out from the
patches as they are when it is applied. The new patch records which enabled patches it was worked
out against (`source.compensates`: id, strength and message), and the patch list says so on its
row ("cancels the edits of P1 … at ×1"). Disabling, deleting or re-strengthening one of those
patches later leaves the revert patch cancelling an edit that is no longer there: the change is
made, the answer of `/api/patch` carries a plain `warning`, the panel shows it, and the revert
patch's row turns red until the patch is set back or the revert is disabled or deleted too.

The node answer (`/api/node`) carries `actions`: for each button whether it can be used and,
if not, why, worked out from the same plan the script is built from. A button that is enabled
changes weights. A button is disabled, with its reason next to it, when:

- nothing selected can be edited (for example a tensor stored in a type the IDE cannot edit);
- the edit would write a larger patch than the limit allows (see [How large an edit may be](#how-large-an-edit-may-be)):
  "This is N weights (X GiB as FP32 edits), too large for a weight patch: …", followed by the
  constraint that binds, with its numbers, and what can be done. For a scale of neurons, heads or a
  whole MLP the message adds "To change it without a weight patch, scale it at run time in the Run
  tab's steering panel." (revert never says this);
- for revert, nothing in the selection has been edited.

A script request that cannot be done answers HTTP 400 with `{"error", "refused": true}` and the
counts; the inspector shows the message under the buttons. Scaling by 1, and scaling or zeroing
weights that are already all zero (checked when clicked, for edits up to a million weights), are
refused the same way.

On a quantized checkpoint the actions edit the reconstructed logical values: the packed base is
never changed and the patch is FP32. Before this change the inspector filtered slices by a
floating-point storage type, so on a Q4_K tensor it sent an empty script and the result was "the
script changed no weights"; it now asks the workspace whether each tensor is editable.

## How large an edit may be

Until this change a selection edit was refused above a fixed 60,000,000 weights (a 229 MiB FP32
patch), the same figure as the script interpreter's `MAX_ELEMS`. It guarded neither memory nor time
as such: it was a round bound from when scripts built whole tensors in memory. A whole block of a
27B model (383,273,184 weights, 1.4 GiB of patch) was refused on a 128 GiB machine with plenty of
room. The limit is now the smallest of three constraints (`wsp/edit_limit.py`):

| Constraint | What it allows | The refusal names |
|---|---|---|
| **Largest weight patch** setting (`patch_limit_gib`, GiB of FP32 patch) | GiB × 2³⁰ / 4 weights; Auto sets no limit of its own | the setting, and "raise Largest weight patch in Settings" |
| memory, under the runtime supervisor | the largest patch its admission would grant now (the `advise` request, see [RUNTIME.md](RUNTIME.md#how-large-an-edit-may-be)) | the bound admission stops at: available memory less the Admission headroom reserve, the Memory budget or ceiling setting, or physical memory less the reserve, with its numbers and the remedy that works for it |
| the memory watchdog, under the supervisor | no edit while it refuses one for another reason (the model file changed, a configuration mismatch, its memory ladder) | the watchdog's own message and recommendation |
| memory, without a supervisor | (available − preferred reserve − read scratch) / 12 bytes, from `memory_guard.snapshot`, whatever the memory pressure level | available, kept-free and scratch GiB |
| disk | (free space where the workspace is − 1 GiB) / 4 bytes | free disk space |

Memory is charged 12 bytes per patch weight, the figure the supervisor's memory admission already
charges (the FP32 delta plus room to build and gather it; the real use on a given model has not been
measured; in-process it is only an estimate, and the texts say no admission runs there); disk is
4 bytes per weight, the patch file. The setting never overrides memory: with a large setting memory
still refuses, and its message says raising the setting does not help. Under the supervisor the
edit's own `edit_preflight` calls are unchanged and remain the final word; the limit only tells the
inspector beforehand what they will allow.

**Small edits.** Memory and disk are measured, and can refuse, only an edit of more than 1,000,000
weights (`SMALL_EDIT`, 12 MB charged, less than the 64 MiB of read scratch every edit is already
admitted with). At or below it an edit is limited only by the setting, as it was before this limit
existed, so a nearly full disk or a busy watchdog never refuses an everyday edit, and opening a node
costs no supervisor round trip. The selection buttons, the generated script and the console use the
same rule, so they agree at every size.

**When memory cannot be measured.** Planning a selection edit (the buttons and
`/api/selection/script`) is the only place the old fixed limit applied, and the only place its
fallback applies: when memory cannot be measured (the watchdog gave no answer, or memory telemetry
failed in-process) and nothing is set, the old 60,000,000 weights apply. The message gives the
reason and says what happens next: under the supervisor its preflight still checks memory when the
edit runs; in-process nothing else does, so the setting should be no higher than the machine holds.

**In-process (no supervisor).** Without the runtime supervisor nothing admits an edit's memory, so
the limit is the measurement: available memory less the Admission headroom reserve and the edit's
read scratch, at 12 bytes per weight. The memory pressure level is not used here (macOS reports a
raised level on many healthy machines; the supervisor's own admission does refuse under it).

**Confirmation.** A selection edit of more than 60,000,000 weights (`CONFIRM_ABOVE`, the old limit)
asks before it runs. The scale and zero reasons say so ("you are asked to confirm first"), and the
`/api/selection/script` answer carries `cost`: `weights`, `disk_bytes`, `memory_bytes`, `confirm`,
`confirm_above`, `gguf`, `cancellable` (always false), `estimate_seconds`, `basis` and the dialog
`message`, for example:

> Apply an edit of 383,273,184 weights?
>
> Patch: about 1.4 GiB on disk (FP32 edits). Memory admission counts about 4.3 GiB for it (12 bytes per weight) while it is built and kept; the real use on this model has not been measured.

(Under the runtime supervisor. In-process, where no memory admission runs, the second sentence says
the 4.3 GiB is an estimate at the supervisor's admission charge and that no memory admission checks
the edit.)
>
> Time: no estimate; no edit of a million weights or more has been timed since this workspace was opened (timings are kept only while it stays open).
>
> Progress and the time left are shown while it runs. A running edit cannot be stopped; nothing is changed until it is committed.
>
> Generation is slower while this patch is enabled: on every forward pass the runner adds the patch's contribution to the output of each matrix it changes (the quantized weights are never changed). How much slower has not been measured.

The last paragraph appears only on a GGUF workspace. When the model runs in the supervisor's model
worker, a paragraph before the time says the patch is held twice, by the server and by the worker
(`cost.worker_hosted`). The time estimate is the last edit of at least 1,000,000 weights since the
workspace was opened (how long `Session.apply` took to run its script), scaled linearly and labelled
as an estimate; until one has run, the dialog says no estimate is possible. The timing is kept in
memory on the open workspace only and is lost when it is reopened. Revert asks the same way above
`CONFIRM_ABOVE`, and its reason says so. Declining applies nothing ("Not applied: nothing was changed."),
and so does a browser with no way to ask. While the edit runs, the busy line shows the line and
window and, once a tenth is written, the share written and the time left, from the `stored` count
each `edit_progress` event now carries. A running edit cannot be cancelled: there is no stop for an
edit, and none was added.

The console's Apply, the programmer model (MCP `edit`, `commit_files`) and `wsp apply` have no
dialog. For them the limit itself is the guard, in `Interp._add`, on every write a script makes:

- up to the smaller of `SMALL_EDIT` and the setting, nothing is checked (`Interp._unchecked`);
- past it, the setting is checked; at the first write past `SMALL_EDIT`, disk and (without a
  supervisor) the machine's memory are measured once for the rest of the script (`Interp._limit`);
- a write that would pass the limit first compacts the tensor's pieces (overlapping writes count
  twice until compacted) and is refused only if the compacted patch still passes it, with the same
  message as the buttons; nothing is committed.

A script never had a total cap before this limit, so it gets no fallback: without a memory reading
it is bounded by the setting and disk only. Under the supervisor each write's `edit_preflight`
decides memory as before. `wsp apply` reads the setting from the same configuration as `wsp serve`
(`--config`, `--profile`) and takes `--patch-limit-gib` for one run. Explicit configuration that
does not resolve (those flags, `WSP_CONFIG`, `WSP_PROFILE`) is an error. When the default
configuration does not resolve, a Largest weight patch saved in it is not dropped: it applies if the
file itself is valid (the fault is elsewhere, such as a bad `WSP_` variable), and otherwise the edit
is refused with the reason; with no readable default configuration, or one that sets no limit, the
setting is Auto with a note on stderr, as before the setting existed. `Session.commit_deltas` refuses
any edit above the setting, so ROME and file commits are bound by it; it counts the pieces as they
are stored, so two overlapping agent-file pieces count twice because the patch stores both.

**The script guards.** `wscript.MAX_ELEMS` (60,000,000) bounds a tensor a script constructs (`zeros`,
`ones`, `randn`); `MAX_VIEW` (eight times that) bounds one read of the weights, a scratch copy the
size of the view. Neither bounds a patch. Generated selection scripts read at most
`selections.WINDOW` (4,000,000) weights per line or window (`_chunks`, `read_windows`), so no guard
refuses a script the limit allows; a test checks `WINDOW <= MAX_VIEW` and applies generated scripts
with all three lowered.

## Weight-script functions

Three functions were added for these actions and are available in the console:

- `revert(view)`: put a view, `head(L, h)` or `neuron(L, n)` back to its base values.
- `revert_neurons(L, [n, ...])`: put every weight of several neurons of one layer back to base.
- `base(view)`: the view's values before any patch (reads are otherwise of the current weights).
- `scale_neurons(L, [n, ...], f)` (below) for many neurons at once.

They read only the patch pieces (and, for `base`, the region of the base asked for), so they
never decode more of a quantized base than the view.

`scale_neurons(L, [n, ...], f)` handles thousands of neurons at once: it reads the output
projection in windows of at most four million weights (the GGUF store refuses a read of more than
16 M), scales the selected columns of each window, and adds one block piece per run of adjacent
neurons. Duplicates are scaled once. `f` is a number, or factors shaped like one neuron's write
vector (`tensor([...])` of length d_model), the same for every neuron, as `scale_neuron` takes;
one factor per neuron is refused with a plain message: call `scale_neurons` once per factor.

How the windows are cut depends on the storage. A floating-point tensor is read through at most
64 unselected columns between two selected ones. A quantized tensor is decoded one row at a time
whatever the width of the read, so there each window is as wide as the budget allows: the fewest
windows. Measured with the GGUF store's row loop over synthetic Q4_K bytes of a 5,120 × 17,408
down-projection (no file, no model), 9,000 random neurons of one layer are 23 windows of about
0.17 s, about 4 s a layer, or about 3.2 minutes for 50 layers; 300 scattered neurons are 22
windows (3.7 s) instead of 94 (12.3 s) under the floating-point rule. A long edit sends
`edit_progress` events (statement and window, at most twice a second), and the patch row shows
"working… (line 3 of 50, read 7 of 23)". The browser gives each `/api/edit` request an id
(`request`), its progress events carry it, and a tab shows only its own edit's progress; an
agent's or another tab's edit does not move this tab's text.

## How a large edit is stored

A patch is a list of delta pieces per tensor. Before, every write past 256 pieces in one
script compacted the whole list again (merging small tensors, trimming large ones), and each
write's read walked all earlier pieces, so 1,500 neurons of one layer took about 51 s. Now a
tensor's list is compacted only when it has doubled since the last compaction, and the bulk
functions write each run once.

Measured on an in-memory 224 × 18,000 output projection with 9,000 scattered neurons (every
other one): the script is built in 0.05 s and applied in about 0.8 s; the patch holds 9,000 block
pieces with 2,016,000 numbers, exactly the weights changed; the base was read about once over
(4.0 M values); reverting takes about 1.0 s and leaves the delta exactly zero. Reading one column
through 18,000 pieces takes about 30 ms. When the patch is committed, a tensor of up to four
million weights has its pieces merged into one block or sparse piece; a larger one keeps one piece
per run. The PyTorch runner and the exports add each piece to its region of the tensor
(`delta.add_to`). The GGUF runner never rebuilds a packed tensor: it applies the pieces on every
forward pass, adding `x @ delta.T` to the packed layer's output (`DeltaOverlay`,
`wsp/gguf_runner.py`). Blocks spanning a whole axis of a layer, the shapes thousands of neurons
produce, are gathered into one matrix per axis: full-height column blocks (what a neuron or head
writes) with one column index, and full-width row blocks (what a neuron or head reads: gate, up,
query rows) with one row index, of any step and either direction. A forward pass is then one
matmul per group instead of one per piece. Measured with random pieces, no model:

| 9,000 random neurons (4,371 pieces) | per piece | gathered |
|---|---|---|
| down projection 5,120 × 17,408 (columns), 1 token | 41 ms | 1.7 ms |
| down projection, 32 tokens | 382 ms | 4.5 ms |
| gate/up projection 17,408 × 5,120 (rows), 1 token | 34.7 ms | 1.7 ms |
| gate/up projection, 128 tokens | 206 ms | 7 ms |

Building a gathered copy takes 35–110 ms and holds as many numbers as its pieces (176 MiB for
the rows above). An overlay is kept while its pieces and strengths stay the same, judged by the
pieces themselves, not the list holding them: toggling a patch elsewhere makes new lists for every
tensor, and an unchanged tensor keeps its overlay (0.3 ms to check 4,349 pieces, against 108 ms
to rebuild); a script that adds to its candidate between two `logits()` calls gets a new one. The
overlay is computed in the dtype and on the device of the layer's output. A job's candidate
overlays are dropped when the runner is next set without a candidate (the worker does that
after every job); a committed overlay whose patch is deleted, disabled or re-strengthened is
dropped when the workspace rebuilds its pieces. What the overlays hold is reported as
`overlay_bytes` in the runner's `residency` and in its telemetry file. None of this has been
measured with a real model.

## Named selections

A named selection is a workspace group (`group:<name>`); there is no second store. Save one:

- from the inspector: "+ selection" saves the current address, or adds it to a selection of
  that name;
- from Atlas → Find: "Save all N as a selection" sends the finder's selector, and the server
  expands it, so the browser never lists the neurons;
- from Atlas → Attribute: "Save these N as a selection" saves the attributed rows;
- from any atlas table: tick rows, then "Save checked (k)".

`POST /api/groups` accepts `members` (addresses, as before), `neurons` (`[[layer, n], ...]` or
`"L3N5"`, tens of thousands at once), `selector` (an atlas Find selector) and `mode: "add"` to
merge into an existing selection. Neurons are stored as runs (`blocks.3.mlp.neurons.64-127`) where
adjacent. The answer adds `summary`. Members are compared by a normalised spelling, so
`x[:,3]` and `x[:, 3]` are one member; a slice is stored as the IDE writes it, any other member
as it was given. Neuron lists take whole numbers only (`3.7` is refused, never truncated).

A selection may contain other selections (`group:a`). One that would contain itself, directly or
through another, is refused with "A selection cannot contain itself". When an inner selection is
deleted, the outer one keeps working: the missing member is skipped and listed with its reason
among the members that are not edited, on the page and in the script answer; the explorer leaves
it out of the selection's children.

A selection's page shows its size (neurons, heads, tensor slices, the layers it touches, weights,
neurons per layer) and the actions. The member list shows the first 200 members, and the explorer
lists the first 200 as children; the selection itself is never cut.

`wsp.selections.neurons_of(ws, name)` returns a selection's neurons as sorted `(layer, index)`
pairs, with ranges expanded and heads or tensors skipped.

A selection can also be pushed at run time without a weight patch: `group:<name>` is a steering
target in the Run tab's steering panel, which lists the saved selections with their neuron counts.
Steering expands the selection through this module and counts what it does not push (heads,
tensor slices, members that no longer resolve). The Probe tab's Fires and Contrast tables save
their rows as selections, and a selection's page offers "steer this selection". See
[STEERING.md](STEERING.md) ("The steering specification", "The steering panel").
Steering's column-norm reads use the same packed-block read planner as the scale and revert
actions here (`wsp.selections.windows`).

## Breadcrumb

Above the inspector and the weights view: model › block › attn or mlp › (neuron runs) › unit or
tensor › slice, each part clickable. Shortcuts follow it: a slice that is exactly one neuron's
or head's weights (for example `model.layers.34.mlp.down_proj.weight[:, 3838]`) jumps to that
unit (`L34N3838`), a slice that is a key-value group's rows to that group, and a unit to each of
its tensors.

## Before and after

An edited tensor or slice opens side by side: base, current and their difference.
Base and current share one colour scale (the larger of their 99.5th-percentile magnitudes); the
difference has its own. Hovering shows `base → current` and the change at that cell (pooled cells
show the pooled value and the largest change in the cell). Below are the norm and value range of
base and current over the region shown, and how many weights changed. "single view" returns to
the editable heatmap; "before / after" there comes back. Unedited selections keep the single view.
On a quantized checkpoint, base is the reconstructed values of the region shown.
`GET /api/matrix/compare` serves it with the same region and pooling arguments as `/api/matrix`;
regions above 16 M weights are read in row slabs.

## Patch panel

Run and Chat hold the workspace lock while they generate only when a runtime worker serves them;
`/api/node` and `/api/patches` say whether that is so (`edit_waits_for_generation`). A patch row
being changed is disabled; after 300 ms it says "waiting for the running generation…" when edits
wait and a generation is running, or "working…" otherwise. This tab's own Run or Chat request
counts as running from when it is sent until it settles (however it ends). Another client's
generation counts from its first `generation` event until its final one, however long it is
silent in between (reading a long prompt sends no event). Only when the event connection dropped
after a generation's last event, so its final event may be lost, does it stop counting after 30 s
without an event. Once this tab's edit reports progress of its own, it holds the lock and the text
is "working…". A failure is shown at the top of the panel and the list is refreshed. `/api/patch`
takes `action` (`delete` or `set`) and the patch `id`; either missing or unknown is a 400. Deleting
a patch that is already gone succeeds (`{"ok": true, "missing": true}`) and refreshes the list;
changing one answers 404 with a plain message instead of a `KeyError`.

Saving a selection offers the suggested name (or "selection"), numbered ("picked 2") when a
selection of that name exists, never an existing selection's name; typing an existing name adds to it. Ticked rows of an atlas table and the
"Saved as …" line stay when the table is redrawn after a patch or selection change, and saving
does not run Find again.
