# Audit: implicit and repeated expensive actions

Date: 2026-11-14. Repository: `weight-space-ide`.
Read-only audit of the code as it stood before the explicit-inspection change. That change began during the audit; line references below refer to the earlier code unless marked current. No server or live endpoints accessed; no downloads, training, local weights, inference, or owner port 8765 touched.

Policy assessed: calculations require an explicit request; optional expensive analysis needs a secondary action; identical in-flight operations should be suppressed/coalesced; show lightweight progress where practical.

## Delivery scope

The accompanying implementation addresses findings 1–3, 5–6, 9 for weight inspection,
and 10: explicit weight analysis/statistics/SVD and Systems calculations, passive
status reads and hierarchy browsing, shared bounded inspection requests and stale
result guards, and removal of automatic thumbnail reads. Findings below describe
the original behavior so the evidence remains reviewable.

Follow-up work remains for findings 4, 7–8 and 11: automatic Run/Trace/Tests execution,
mutation idempotency, inference-action pending guards, and chunk enrichment.
Those paths are audited here, not claimed fixed by the inspection change. In
particular, this change does not establish that every IDE tab is passive.

## Highest-priority confirmed findings

1. P1 — Passive node inspection computes semantic cards and repeats them.
   - Trigger: selecting any head/neuron (on any tab) invokes app.js:206; Weights additionally invokes weights.js:27. server.py:180-190 automatically invokes head_card, neuron_card, or direction_tokens if vocab weights are local.
   - Cost: analysis.py:402-425 builds sampled-token/vocab logits and a dense residual-space OV SVD for heads; :429-443 computes several full-vocabulary projections for neurons. semantic.py:136-148 contracts through embedding/unembedding tensors. Locality only prevents download, not expensive computation.
   - Repeated selection, tab, theme and variant renders repeat these calls. No initial API sharing or server coalescing.
   - Minimal change: passive metadata /api/node; explicit per-card calculate action; workspace/version/op/options-scoped single-flight; stale response checks. One per-node explicit grant must not become a global consent flag.
   - Regression: mocked endpoint and frontend counters: passive select emits no semantic analysis; one action shared by inspector/view; two identical pending requests compute once; workspace/version changes never reuse output.
   - Scope: core change.

2. P1 — Every 2-D matrix render unconditionally starts SVD.
   - Trigger: weights.js:144 after heatmap, including navigation, theme/variant, pool/sort/zoom rerenders. st.svd is initialized (:90) but never guards the request.
   - Cost: analysis.py:330-339 performs exact double-precision or randomized SVD, or two streaming passes (:309-318); :350-354 adds vocabulary projections for residual singular vectors. k=6 limits returned components, not the decomposition rank/cost (randomized rank 64).
   - Minimal change: explicit Calculate SVD action, shared pending work, cache only a bounded result set per workspace/version/variant/address/options; keep heatmap usable while calculating.
   - Regression: open/zoom/theme -> zero svd calls; explicit click -> one; repeated pending click -> one; failure permits explicit retry.
   - Scope: core change.

3. P1 — Passive hierarchy/statistics still decode weights, even with cards deferred.
   - Trigger: hierarchy.node:31-38 automatically measures each local child <=6M values. children:121-123 can produce ~64 neuron groups individually below that cap, without aggregate budget. Head children :100-105 and raw/tensor children also call node. server.py:166 invokes region_stats on local selections.
   - Cost: analysis.region_stats:246-265 reads all <=8M logical values, otherwise per-tensor sampled rows with a 2,000-value minimum. A small column slice in GGUF still loops over every selected logical row and calls gguf.dequantize at gguf_store.py:211-214. Logical size/output pixel count does not bound decode call count. Multiple groups can read most of a layer.
   - Minimal change: metadata-only tree and passive node, explicit Calculate statistics action. Sampling alone is insufficient for column-shaped GGUF regions. Keep changed counts from patch metadata.
   - Regression: fake store whose read raises: initial tree, attention expansion, neuron-group expansion, passive node/group succeed with zero reads; explicit stats causes one shared operation.
   - Scope: core change; the hierarchy default was already being changed to norm=False during the audit.

4. P1 — Opening/changing Run, Trace or Tests can run/load models without a Calculate action.
   - Trigger: run.js:57-66 auto-runs completion on first view or version change for non-GGUF; :133-136 auto-runs trace for every runnable format on first view/version/variant change. work.js:32 invokes current/base test evaluation when Tests opens. server.py:418-421 calls runnable before sess.tests, even if cache exists; runnable:222-233 may load runner.
   - Cost: completion compares base/current (:247-251), trace executes forward/hook work, suite cases may generate or evaluate many prompts. A can_run boolean is capability, not a request to calculate.
   - Minimal change: preserve forms/suite specification and prior results; explicit Run, Trace, Run tests buttons; show stale result badges after edits rather than automatically rerunning. Avoid automatic loading just to view cached test results.
   - Regression: opening each tab and changing weights/variant emits zero execution requests; explicit action emits one. Tests specification remains editable without runner.
   - Scope: follow-up beyond the core change.

5. P1 — Status reads silently run tests/drift once runner is loaded.
   - Trigger: server.status:98-101 calls sess.tests current/base and sess.drift whenever runner_loaded. /api/state, /api/status, task/status responses and UI refreshes reach status. workspace.rebuild:465-466 clears current calculation cache after every patch toggle/strength change. app.js:341 and :342 refresh on patch/group/task events.
   - Cost: Session.tests:148-149 and drift:166-167 compute on cache miss. testsuite.drift:140-165 executes model forwards; analysis.corpus:807-825 can generate a calibration corpus on first drift. These paths are not passive telemetry despite status docstring saying only that it never loads a model.
   - Minimal change: status exposes existing cached metrics plus missing/stale state, without populating them. Explicit evaluation action owns test/drift work. Coalesce explicit evaluations; include suite identity/revision as well as weights version.
   - Regression: mocked loaded workspace + missing caches -> GET state/status does not call run_suite, drift or corpus; patch toggle then status also zero.
   - Scope: follow-up beyond the core change.

6. P1/P2 — Systems navigation computes wiring; remembered ablation restarts inference.
   - Trigger: systems.js:18 always GETs wiring on render. Client-only grouping, edge threshold and kind controls (:95-101) rerender and request it again. :57 computes /api/ablation when overlay='ablation' and weight version differs; result assignment occurs only after await, so concurrent renders issue duplicate POSTs. Selecting ablation can be intentional, but later navigation/edit/theme must not silently repeat it.
   - Cost: analysis.wiring:474-477 caches completed results only. A cold request builds projected Gram matrices, reads per-layer head/MLP matrices (:499-540), ~250MB writer budget (:481) plus reader/intermediate work. ablation_scan:754-784 invokes one baseline and repeated head/MLP forwards across cases. Server runnable may load a model.
   - Minimal change: explicit Calculate wiring and Calculate ablation actions; reuse ready wiring for purely visual controls; invalidate display as stale after edits and require fresh explicit action. Coalesce pending operations on server. Separate inference consent from overlay state.
   - Regression: first Systems navigation zero work; visual controls zero new work; remembered ablation on reentry/edit zero inference; explicit action one despite repeated clicks; error requires explicit retry.
   - Scope: included in the core change.

## Repeat-click and lifecycle risks

7. P1 — Explicit edit actions can execute/commit repeatedly while pending.
   - app.runScript (app.js:51-62) has only output text, no busy gate. Apply/Dry run and Ctrl+Enter (:323,331-332), inspector scale/ablate (:230-235), matrix set/zero (weights.js:133-134) can submit identical edits repeatedly.
   - server.py:316-322 has no request id/idempotency guard; Session.apply:240 runs script before commit_deltas acquires ws.lock (:206), so locking commits does not deduplicate script work or prevent duplicate patch effects. Repeated scaling can produce multiple patches; concurrent reading/committing can make effects timing-dependent.
   - Minimal change: reserve pending action synchronously and disable all entry paths; server idempotency key for an explicit user action, retained long enough for transport retry, plus expected workspace/version. Do not coalesce all historically identical script bodies: intentional repeated edits are distinct actions.
   - Regression: two UI events for one pending action -> one script/evaluation/patch; separate completed actions can intentionally repeat; timeout retry with same id returns the existing result.
   - Scope: follow-up. Distinguish mutation idempotency from read-analysis single-flight.

8. P2 — Explicit Run/Trace/patch/chat/chunk calculations lack pending suppression/cancellation.
   - run.js:43-54,118-130,186-195 buttons/keyboard remain enabled during busy state; chat:93-106 can send again while prior response is pending. work.js:94 can render the same chunk repeatedly. util.api:24-30 has no sharing, AbortSignal or lifecycle binding.
   - Errors for Run/Trace remain displayed (no automatic retry on error), which is good. Completion/trace also assign version/key at RESPONSE time (:45,:120), potentially tagging old work with a newly current version; switching away only hides DOM, not computation. No claim of measured resource size for this race.
   - Minimal change: synchronous pending reservation, immutable request snapshot, result attributed to starting workspace/version/options; ignore stale completions and show explicit stale status; expose cancellation only if backend can actually stop it.
   - Regression: deferred mock response + repeat click/Enter -> one operation; edit/switch workspace while pending never labels old output current; rejection clears gate.
   - Scope: follow-up.

9. P2 — Initial caches and stale guards do not fully respect workspace/lifecycle boundaries.
   - app.js:306-311 suppresses stale view publication only AFTER awaiting all view work; it does not stop expensive follow-on calls inside an obsolete render. Inspector:207 checks address only, so same address in a replacement workspace can accept old output.
   - weights.js:8 thumbnail cache key lacks workspace identity and includes theme, so switching workspace with same version/address can show prior model's image; theme change recomputes matrix data. Rejected thumbnail promises (:11-13) and tree promises (app.js:153-154) persist until cache clear, blocking retries. Global clear at 600 thumbnails bounds entries, but not bytes or maximum in-flight work.
   - Minimal change: stable workspace instance identity and version in keys; compute cache independent of theme; delete rejected promises; cap pending work/ready retained bytes; validate identity before chained calls and publication. Abort browser requests only when safe for all subscribers; browser abort does not prove server work stopped.
   - Scope: core change, shared guard/stale handling; inspect final implementation.

10. P2 — Heatmap thumbnails are an implicit fan-out of full reads.
   - weights.js:60,70,84 requests one matrix per visible head/tensor/facet automatically. analysis.matrix_view:162-191 reads the complete region for pooling; streamed means bounded working memory, not sampled computation. A 96x48 thumbnail may scan up to a 40M-value tensor; many children request concurrently. Full matrix opening itself is arguably explicit viewing, but hidden thumbnails are optional analysis.
   - Minimal change: placeholders + explicit Load previews, or bounded visibly labeled sampling with a shared concurrency budget. Defer full heatmap if policy interprets navigation as non-computational; at least avoid recomputation for theme-only changes and optional SVD.
   - Regression: opening a parent makes no implicit matrix fan-out; explicit requested previews respect max concurrency and lifecycle.
   - Scope: follow-up or policy decision; latency/memory on real models unmeasured.

11. P2 — Chunk text carries hidden semantic/wiring extras.
   - Clicking a chunk explicitly asks for text (work.js:94), but /api/chunk defaults context='full' (server.py:457-460); chunks.where_header:199-228 invokes head_card/neuron_card and for full head context wiring (:216-217). render_tensor:119 reads tensor before choosing compressed output; SVD summary format also computes decomposition.
   - Minimal change: default chunk preview to lightweight context; separate Include semantic analysis action; label SVD format cost before render; deduplicate same chunk request with exact plan identity/version/variant/context. sess.plan is shared mutable server state, so racing plans/chunk IDs need binding.
   - Regression: ordinary chunk render never invokes semantic card/wiring; explicit enrichment once; stale plan ID rejected.
   - Scope: follow-up; no implicit model inference shown by this path.

## Intentional actions with useful existing guards

- GGUF Run view already avoids automatic completion (run.js:58); explicit Run remains duplicable.
- Unbuilt bundled model selection asks confirmation before download/train (app.js:104); Pull disables its clicked button (:125-128). Opening another model explicitly requests opening it. Pending open remains worth UI reservation, but not classified as implicit build.
- GGUF export requires explicit preview then export (export.js:15,20); ExportJobs.start rejects concurrent RUNNING jobs under a lock (export_jobs.py:24-29), has bounded 32 retained jobs, progress (:35), status and cancellation (:55-59). UI reserve occurs only after response, so two clicks can send two requests, but second server job is rejected. Preview computed before running-job lock is a possible minor duplicate preflight, not duplicate export execution.
- Programmer rendering fetches run/auth status, never starts a conversation by itself. Start is explicit; sess.lock + native_harness.assert_available (server.py:529-536, native_harness.py:29-33) reject concurrent active conversations. NativeRun.send reserves 'running' under lock before request (:143-146), guarding duplicate native sends. Auth UI uses busy/generation checks. Prepare only is explicit. No need to add analysis coalescing to agent turns.
- Activation patching has explicit Patch click (run.js:197 no auto-go), but no repeat-click gate.
- Edit testing through Session.apply/commit_deltas is intentional when applying a requested edit; preserve scope, tests, drift and patch path. Fix UI/action idempotency, not skip required candidate evaluation.
- Chunks plan auto-runs on entry (work.js:82), but chunks.plan:303-344 uses shapes/estimates/partitioning rather than weight reads. Classify as low-cost metadata planning, not expensive inference/SVD.

## Progress, cancellation, retry, and evidence limits

- app.renderView (:308-314) adds busy CSS after 250ms but renders a fresh box only after the entire async view returns. Initial SVD/card/wiring can withhold the primary view; it is activity indication, not determinate progress. Explicit per-analysis pending text and elapsed time suffice without extra computation.
- Run/Trace/patch show 'running…'; they neither stream progress nor support server cancellation. GGUF export and native harness provide real cancel/interrupt controls.
- No blanket auto-retry should be introduced for expensive actions. A failed pending entry must evict and allow an explicit retry; duplicate subscribers may share a failure.
- Concurrency cache correctness requires workspaces never sharing keys, version changes never publishing stale cache results, bounded admission (including pending entries), exception cleanup, and no deadlock on recursively acquired workspace/session locks.
- Runtime duration/CPU/RAM for actual checkpoints is UNMEASURED. Source proves triggered operations; it does not prove a particular number of seconds or an OOM. No heavy live verification was authorized/performed.

## Safe discriminating check performed

A throwaway script (not committed) loads only UI source with stub DOM and api function. It performs zero fetches and never imports Python/model code. Output:

    non-GGUF Run open requests: [ '/api/run' ]
    non-GGUF Run plus two repeat clicks: 3
    GGUF Run open requests: []
    GGUF Trace open requests: [ '/api/trace' ]
    GGUF Tests open requests: [ '/api/suite?name=fixture',
      '/api/tests?suite=fixture', '/api/tests?suite=fixture&variant=base' ]

These mocked checks discriminate actual UI initiation from a hypothesis that requests only begin after explicit buttons.

## Independent implementation review

A follow-up review verified shared concurrent work, explicit retry, stale-result
suppression, suite-revision invalidation, ordered status/tree refresh and runner
lifecycle locking. The change corrected the stale-suite cache and navigation
issues found during review. No core review blocker remained.

Browser QA then exposed mixed cached JavaScript modules. The final implementation
serves fresh HTML with a fingerprinted immutable snapshot of all eight modules.
The review independently resolved the complete import graph and verified both `/` and
`/index.html`, and checked that unscoped expensive inspection requests return 428
before calculation. A regression test also verifies that an old-server capability
failure sends no state/inspection requests and opens no event subscription.

The final checks use fake workspaces and controlled computations. They establish
interaction and concurrency behavior, not large-model performance. Findings 4,
7–8 and 11 remain follow-ups as listed in the delivery scope.
