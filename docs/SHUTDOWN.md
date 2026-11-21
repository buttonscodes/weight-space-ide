# IDE shutdown

Use `python -m wsp serve` and press Ctrl-C once. Open live-event streams now end
before Uvicorn waits for connections to drain, and disconnects unsubscribe promptly.
SIGTERM follows the same cleanup path. A second Ctrl-C requests Uvicorn's force path
but still closes the IDE's owned exports, authentication attempts and agent sessions.

Other stalled HTTP work has a three-second drain deadline. Resource close callbacks
run concurrently with a 15-second limit, followed by up to ten seconds of owned-process
teardown per resource. Expected idle shutdown is much faster. Cleanup faults/timeouts
are logged and cause the CLI to fail; they are not silently treated as success.
The CLI only quiets Uvicorn's replayed SIGINT after completed cleanup. Genuine startup
interrupts, cleanup errors and unrelated cancellation exceptions remain visible.

The server uses the existing declared `psutil` dependency to verify owned child
identities and reuse the watchdog's process-tree teardown. Install the repository's
requirements in the chosen environment. Empty-session serving does not import model
telemetry dependencies. No model-load permissions or memory thresholds change here.

## Cause and integration

The old synchronous `/api/events` generator blocked on `queue.get(timeout=15)` and
had no shutdown condition. Uvicorn drains responses before application lifespan
cleanup, so setting a flag only in lifespan cannot release that drain. Its signal
capture also replays SIGINT after shutdown, independently of SSE; a traceback can
therefore occur even after one Ctrl-C with no event stream.

`wsp.lifecycle.IDEServer` sets the stop flag in the signal handler and at shutdown
entry, then delegates draining to Uvicorn. The async SSE iterator uses cancellable
100ms waits and unsubscribes in `finally`. Lifespan cleanup is idempotent and is also
completed when the second SIGINT makes Uvicorn skip its normal lifespan call.
New requests receive 503 once shutdown starts. Session edit/commit behavior is unchanged.

Source checked: installed Uvicorn 0.54.0 and Starlette 1.10.0, plus
[Uvicorn 0.54.0 Server source](https://github.com/Kludex/uvicorn/blob/0.54.0/uvicorn/server.py).
The CLI and bounded acceptance server use this integration; a custom embedding should
also use `IDEServer` or `wsp.lifecycle.serve`, not plain `uvicorn.run(create_app(...))`.

## Validation and temporary workaround

Real subprocess tests use an empty Session, inherited ephemeral loopback sockets and
fake export/auth/native cleanup callbacks: no `Session.open`, training, model weights,
inference or provider sign-in. Cases cover first SIGINT/SIGTERM with no/open/closed
SSE, disconnect unsubscribe, second SIGINT during drain/cleanup, visible cleanup faults,
startup interrupts, in-flight export cancellation and owned-child timeout cleanup while
an unrelated sentinel survives. A stopped legacy process is still reaped.

The exact primary Python environment passed 11 applicable tests in 9.35s; it lacks
the already-declared psutil package, so process-tree fallback tests used the complete
isolated environment. The final focused lifecycle/auth/native/export suite passed
34 tests in 11.18s. First-signal exits measured 0.23–0.40s, including open SSE, with
unsubscribe and all mocked resource cleanup recorded. All test subprocesses were reaped.
No existing listener, workspace or environment was changed.

Before this fix is integrated into a running server, close **all** IDE browser tabs,
wait about 16 seconds, then press Ctrl-C once. This worked in the isolated old-source
control (0.287s exit); closing the stream and immediately interrupting still exceeded
three seconds. It is a temporary workaround, not a promise for other active work, and
old-version signal tracebacks may still appear. Do not repeatedly interrupt an active
export or sign-in merely to hide its cleanup diagnostics.

Raw local evidence: `~/.cache/wsp-ide-shutdown/2026-11-14/`.
