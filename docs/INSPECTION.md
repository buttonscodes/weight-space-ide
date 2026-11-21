# Explicit weight inspection

Opening an explorer node shows shapes, addresses, parameter counts and existing patch information. Expanding the tree does not read base weights or compute norms. The Weights view does not generate thumbnails in the background.

- **Calculate statistics** reads the selected region, sampling large regions. Remote regions may be fetched.
- **Calculate vocabulary readings** computes a selected neuron/head card or residual-direction projection. Local vocabulary weights still require this action; download permission alone never starts it. For remote vocabulary weights, the action explains the download and explicitly includes fetching.
- Selecting a tensor still reads its heatmap. **Calculate SVD and vocabulary readings** is a separate action, independent of zoom, pooling, theme and variant rerenders.
- Systems requires **Calculate connections** for its current layer window. Choosing the ablation overlay exposes **Run ablation scan**, which explicitly loads the runner and performs inference. Old overlay preferences never authorize a new scan after weights or the suite change.

Controls show an indeterminate “Calculating…” status while work is pending, disable repeat clicks and restore a Retry action after failure. Returning to a view can observe an already requested computation or cached result. Navigation does not cancel an active computation. There is no polling or speculative calculation, and no percentage is estimated.

The browser shares matching requests. The server joins concurrent matching computations and retains at most 64 completed calculation results, keyed by workspace instance, weight version, group/suite revision, normalized address and options. At most four distinct calculations may be active or waiting for the workspace lock; additional requests return a retryable busy response. Metadata has a separate bounded cache and admission budget so it can remain responsive. Failures are not cached. Editing, changing group membership, saving suites, or opening another workspace invalidates the relevant scope; stale queued requests are rejected before computation, and stale view responses cannot update the new selection.

Weight calculations hold the workspace lock for consistent reads, so edits may wait for them. Ablation also holds the session lifecycle lock, so opening a workspace waits for that scan rather than releasing its runner mid-scan. Status only reports cached test/drift evidence; missing metrics mean unmeasured, not a fresh evaluation.

## API and upgrades

The browser probes `/api/inspection-capabilities` before requesting state. An older running server displays a restart-required message; it is never restarted automatically. A new server snapshots its JavaScript/CSS and serves one content-fingerprinted module graph, preventing cached older modules from reintroducing automatic work. Both `/` and `/index.html` serve a fresh entrypoint.

`GET /api/node` is metadata-only by default. Explicit `analysis=1` or `statistics=1`, and `/api/matrix`, `/api/svd`, `/api/wiring`, and `POST /api/ablation`, require `workspace_id` and `version` from `/api/status`. Send them as query parameters for GET and body fields for POST. Unscoped legacy calculation requests return 428; stale scopes return 409; a full calculation budget returns 429. Remote node analysis additionally requires `fetch=1`. This is an intentional compatibility change for inspection clients. The MCP editing path is unchanged.

## Verification and scope

Model-free checks:

```sh
python -m pytest tests/test_inspection_requests.py -q
node --test tests/inspection_frontend.test.mjs
```

These use fake workspaces, fake analysis, controlled concurrency and a minimal DOM. They do not download, construct, load, train or run a model. Existing model-dependent API tests have been updated for explicit scoped inspection, but running them requires separate authorization.

This change covers weight browsing, Systems calculations and passive status. Other automatic execution paths in Run, Trace and Tests remain separate findings in [the expensive-actions audit](EXPENSIVE_ACTIONS_AUDIT.md). Real large-GGUF timings and decoding throughput were not measured or changed.
