# The Programmer tab

How the Programmer tab shows a run while it works. Setup, harnesses and sign-in are in
the README, [Running a programmer](../README.md#running-a-programmer).

## Coming back to a run

A conversation keeps running on the server whatever the page shows: switching views, clicking
in the explorer tree or reloading the page never starts, stops, interrupts or closes it
(none of them sends a request that changes it).

- **The open conversation comes first.** While a conversation is open (starting, running,
  waiting for your answer, or idle after a turn), and while a legacy (oh-my-pi, custom) run
  runs, the tab shows, from the top: its heading with **Interrupt turn** (or **Cancel
  startup**, **Stop**) and **Close conversation**, the status line, any permission prompt or
  question waiting for you, the log, and the **Follow-up message** box. The sign-in, task
  and harness cards, which only configure a new conversation, are folded under one line
  below it ("Task, harness and sign-in of this conversation (fixed while it is open): ..."),
  then the earlier conversations. The fold stays as you leave it. With no open
  conversation the order is the earlier one: sign-in, task, harness, earlier conversations,
  then the last run's log.
- **In the earlier-conversations list** the open conversation's row has **Show at the top**,
  which scrolls the tab to it. Clicking the Programmer tab while it is shown also scrolls to
  the top.
- **The tab's name** says, from any view, what the open conversation is doing:
  **Programmer · working** (a turn is starting or running, or is being interrupted),
  **Programmer · waiting for you** (a permission prompt or question is pending: the run
  cannot go on until you answer), **Programmer · turn finished** (idle, waiting for your next
  message). It has no suffix when no conversation is open. It is text, so a screen reader
  reads it with the tab's name; the colour only repeats it, and the button's title says it
  in a sentence. It is updated from the event stream on every view and redraws only the tab
  bar. At page load it comes from one read of the server's runs (`GET /api/harness`, no
  events), so it is right on whichever view the page opens. The tab bar wraps onto a second
  row when it does not fit, so the name is never scrolled out of sight.
- **Tree clicks.** A selection in the explorer tree updates the inspector and leaves the
  Programmer view as it is: not rebuilt, not scrolled. The same holds for the other views
  that do not draw the selection (Run, Probe, Trace, Tests, Chunks, Export, Settings).
  Weights and Systems draw the selected node and are redrawn as before. Atlas does not
  read the selection either but is left as it was (it is being changed on another branch).
  A view's scroll position is kept by view for those, by view and selection for the others.
- **The log** is one element kept across renders while it holds every event of the run, so
  a re-render (a status change, a prompt, a task change) does not rebuild hundreds of
  entries, and events that arrive while another view is shown are added to it there. It
  stays at its end while you have not scrolled it up; once you have, new events do not
  move it, and coming back to the tab or any re-render puts it back where you left it.
  Scrolling it back to the end makes it follow again. Where it was belongs to that run's
  log only: a new conversation's log, or a resumed one's (each run has its own id), opens
  at its end and follows its output, wherever the previous log was left. The new view
  takes the log and the status line only once it is on screen, so building it never
  shortens the page you are looking at (a browser would move that page's scroll position).
- **Coming back** to the tab shows the run with everything that arrived meanwhile, restarts
  the once-a-second status line, and fetches any events the stream missed at the next tick.
- **A reload** in the middle of a run shows it first again with its whole log.
- **Another workspace.** The server keeps its runs when another workspace is opened (it
  can be opened only once no conversation is open). The tab shows a run only if it ran
  in the open workspace (`GET /api/harness` names the open workspace, and each run's
  meta names its own): after a switch it starts with no run and lists that workspace's
  conversations; switching back shows that workspace's last run again.

### How this is tested

The ordinary suite (`node --test tests/*.test.mjs`) draws nothing in a real browser: the
page code runs in `tests/strict_dom.mjs`, a fake DOM that behaves like a browser's where a
looser fake did not (collections without array methods, a node that moves when appended
elsewhere, no scroll position while detached, scroll clamped when content shrinks).
`tests/shell_frontend.test.mjs` imports `web/js/app.js` itself into it and checks its
wiring: a tree click leaves the Programmer view as it is, the view's `afterMount` runs
before its scroll position is restored, and the tab marker follows the run.

A real headless Chrome check is opt-in. From `weight-space-ide/`, with the Python
environment active:

```sh
WSP_BROWSER_TESTS=1 node --test tests/browser/programmer.browser.test.mjs
```

It is skipped unless `WSP_BROWSER_TESTS=1` is set and a Chrome is found (`WSP_CHROME`, else
the usual install locations or `chrome`/`chromium` on `PATH`); the Python is `WSP_PYTHON`,
else the active virtual environment's, else `python3`. It starts `tests/browser/fake_ide.py`
(the IDE's own routes and pages, two tiny random models, a fake programmer run: no
provider, CLI, sign-in, network call or real model) and its own headless Chrome, each on a
free port of 127.0.0.1 between 41000 and 49000, with root, model cache and browser profile
in a temporary directory; it stops both and removes the directory at the end.
`WSP_BROWSER_SHOTS=<dir>` saves screenshots. It checks everything above: the run first,
tree clicks, every tab with the marker, coming back, the log's place through events, a
prompt and a reload, a 900 px window, a new and a resumed conversation after a scrolled-up
one, a workspace switch, and a server killed during a first turn then resumed. It takes
about two and a half minutes.

## Status line

While a run is shown, one line above its log says what is happening now. It is computed in
the browser from the run's events (each carries its server time `t`) and the server's clock,
never from counted timer ticks:

| State | The line says | Ticking part |
|---|---|---|
| starting | Starting the conversation (native) / Starting the harness (legacy) | time since start |
| a tool call open | Running `inspect card L41N12481` · 3 more calls waiting behind it (IDE tools) · 1 more call running (other tools) | time this call has run · turn time |
| no tool call open | Waiting for the model | time since the last output · turn time |
| an approval or question pending | Waiting for your response: *title* | time since it was asked |
| interrupting | Interrupting the turn | turn time |
| idle (native) | Turn finished · waiting for your message (or Turn stopped or failed) | how long the turn took |
| ended | Finished / Failed: *first line of the error* / Stopped / Interrupted / Conversation closed | how long it ran |

- **Tool calls.** The tool name drops the `mcp__wsp__` prefix and is followed by its main
  argument (`what`/`action`, then the address, selector, prompt, chunk id, message or
  command). The queue applies only to the IDE's own MCP tools (`mcp__wsp__*`, or the bare
  names of `wsp/tools.py` that Codex reports): IDE calls issued together are taken to run
  one after another, so such a call counts as started when it was issued or when the
  previous IDE result arrived, whichever is later; the first open one is "running", the
  others "waiting behind another call". That the IDE really answers them one at a time is
  what the measured timings showed (four calls returned at 50, 100, 150 and 200 s); it was
  not checked against every provider. A provider's own tools (Claude `Bash`, `Read`, ...;
  Codex `commandExecution`, `fileChange`) never queue there: each runs from the moment it
  was issued. Each call in the log shows its own timer while open and, once its result
  arrives, `took 50 s · waited 50 s first`.
- **Ended turns.** A call still open at a turn or run boundary (the turn's `result`, the
  next `user` message, a session `init`, a `resumed` or `resume_failed` marker, an `error`),
  or when the run is no longer active (closed, failed, interrupted), shows **not
  completed**, never as running. So the calls a server stop cut off stay "not completed"
  while a resume starts up.
- **Long silence.** When nothing at all has arrived for **60 s** (`SILENCE_S` in
  `web/js/work.js`), the line adds "no output for N s". It does not say the run is stuck:
  one `inspect` call on a large model has taken 51 s, and four calls issued together
  returned at 50, 100, 150 and 200 s. The threshold sits above a single slow call and well
  above the model's own 3 to 9 s pauses between results.
- **Clock.** Responses from `GET /api/harness` and `POST /api/harness/start` carry the
  server's `now` (seconds). The browser takes its offset from the server clock as
  `now - (sent + received) / 2` and computes every elapsed time as server time minus event
  time, so the line is right after a reload, after a reconnect, and after the tab spent
  time in the background (no counted ticks). Until a response gives `now`, the offset
  comes from the server time stamped on each pushed event.
- **Missed events.** While a run is active the view asks
  `GET /api/harness?id=<run>&since=<first index it has not seen>` every 10 s, and at the
  next tick after the event stream reported an error, and adds what it missed.
- **Screen readers.** The sentence is in an element with `role="status"`
  (`aria-live="polite"`). One such element is made per Programmer tab state and the same
  node is re-attached on every render; its text is written only when the state's sentence
  changes, and only while the node is in the document (a render builds the view detached
  and mounts it afterwards, so a change made during a render is written just after the
  mount). The ticking seconds are plain text beside it, outside the live region. This is
  what the code does; it has **not** been tested with a real screen reader, so whether a
  given screen reader announces each change exactly once is not known.
- **Ticker.** At most one once-a-second timer runs in the page, and only while the tab
  shows an active run. It stops when the run ends, when another tab is shown, and when the
  workspace changes.
- **Layout.** The line reserves two lines of height and uses tabular figures, so ticking
  or wrapping at narrow widths does not move the log below it.

## Cost and budget

Native runs now normally use the provider's subscription, where a dollar figure per run
means nothing to pay. What the IDE shows depends on how the run is signed in, which it
learns from the provider's own status check before Start (`claude auth status --json`,
Codex `account/read`; see `bridge/auth.mjs`) and records as `billing` on the run:

| Sign-in method (status `method`) | billing | Cost on the result | Dollar budget |
|---|---|---|---|
| Claude `claude.ai`, `oauth_token`; Codex `chatgpt` | `subscription` | not shown (dropped by the server) | no field; a nonzero `budget` is refused |
| Claude `api_key`, `auth_token`, `bedrock`, `vertex`, `foundry`, unrecognised | `per_call` | `est. $0.42 (billed per call)` | Claude: **Claude budget (USD, billed per call)**, passed as `maxBudgetUsd` |
| Codex `apiKey`, `amazonBedrock`, `gcp`, `configured provider` | `per_call` | Codex reports no cost | none: Codex has no dollar cap, a nonzero budget is refused |
| legacy `omp` / custom command | not known | `est. $0.42 (harness estimate)`, if the harness reports one | none (never passed to those harnesses) |

- A method the IDE does not recognise counts as possibly billed per call, so a budget
  that would stop a paying run is never removed by mistake.
- The budget is enforced by the Claude Agent SDK (`maxBudgetUsd`: "the query will stop if
  this budget is exceeded, returning an `error_max_budget_usd` result"); it counts spend
  since the conversation's process started.
- As a second check, Claude's `system/init` message names the credential it uses
  (`apiKeySource`: `none` for a claude.ai login, otherwise `ANTHROPIC_API_KEY`,
  `apiKeyHelper` or `/login managed key`). If a run started as a subscription reports an
  API key there, it becomes `per_call`, its cost is shown, and the log says that its
  requests are billed per call with no budget set.
- Turns, duration and token counts are shown for every run.
- **Resume and the budget.** A per-call Claude conversation that is resumed gets its
  recorded budget again, and it counts from zero: `maxBudgetUsd` is per query, and a resume
  is a new query, so what the conversation spent before the resume is not counted against
  it. The transcript says so next to Resume, and the **resumed** marker repeats it.

## Saved conversations and resume

### What is saved, where

Every programmer run (native and legacy) is saved in its workspace, in a directory beside
the agent's working directory `<workspace>/agent` (outside it, so the agent's own
working directory does not contain it):

```
<workspace>/conversations/.gitignore            "*": the store is never committed
<workspace>/conversations/<id>/events.jsonl     the tab's events, one JSON object per line, appended
<workspace>/conversations/<id>/meta.json        provider, provider session id, mode, model, task, start/end
                                                times, status, billing, owner pid; replaced atomically
<workspace>/conversations/<id>/owner.lock       held (flock) by the IDE process running the conversation
```

- `<id>` is `conv-YYYYMMDD-HHMMSS-<8 hex>`, matched in full; anything else is refused when
  read, so a request cannot name a path outside the directory.
- **Git.** The workspace's `patches/` and `atlas/` are kept out of Git only by the
  workspace root being ignored (`workspaces/` in the repository's and in
  `weight-space-ide/.gitignore`); `conversations/` is inside the workspace and is covered
  the same way. The store also writes its own `.gitignore` (`*`), for a workspace root
  kept elsewhere (`--root`, `WSP_ROOT`). Earlier builds of this branch saved under
  `<workspace>/agent/conversations/`; nothing is migrated from there, the store ignores
  it, and the repository's `.gitignore` keeps a `**/agent/conversations/` rule so those
  old files never show in Git.
- **Who can read the transcripts.** Directories are `0700` and files `0600`, owned by the
  user the IDE runs as: other local users cannot read them; any process of that user can.
  That includes the agents the IDE starts, which run as the same user. A Codex programmer
  (tools mode: Codex's read-only sandbox; files mode: its workspace-write sandbox) can read
  files outside its working directory, including every saved conversation, subject only
  to Codex's own approval rules. A files-mode Claude programmer can read them with its
  file and shell tools once you allow the call (the tab asks for every tool call that is
  not an IDE tool). A Claude tools-mode programmer has only the IDE's tools, none of which
  reads this directory. The IDE itself sends these files nowhere, but an agent that reads
  them can put what it read into its own requests to its provider. The IDE treats their
  content only as text to display.
- **The store does not trust its own files.** The workspace directory, the store root and
  each conversation directory are opened with `O_DIRECTORY | O_NOFOLLOW` and must be
  directories owned by this user. The descriptor is kept, and every file is opened,
  created, renamed and removed relative to it (`dir_fd`): a directory renamed away and
  replaced by a link after it was opened cannot redirect a write (the write goes to the
  directory that was checked). Every file open uses `O_NOFOLLOW` and `O_NONBLOCK` (a planted
  FIFO cannot block it) and must give a regular file owned by this user with exactly one
  link, so a hard link to a file elsewhere is refused. `meta.json` is replaced by rename
  relative to the directory's descriptor; the root and directories are `chmod`-ed only
  through their descriptors, and pruning removes a directory relative to the root's
  descriptor (`shutil.rmtree(name, dir_fd=...)`, which never follows a link).
- **Bounded, checked parsing.** A `meta.json` over 1 MiB is refused before it is parsed;
  an event line over 65 KiB is skipped without being read into memory or parsed, and at
  most 16 MiB + 64 KiB of an events file is read. Every field read from `meta.json` is
  type-checked and coerced (an unknown or non-string status reads as `interrupted`; out of
  range numbers, wrong types and oversized strings are dropped or defaulted).
- **Damaged entries.** Any failure while reading one entry (a link, another owner, a
  second hard link, a FIFO or directory where a file should be, invalid or too deeply
  nested JSON, a bad `owner.lock`) makes that entry one **unreadable conversation** row
  with the reason, and a bad event line is skipped. The other conversations are listed,
  and Start and Resume work as usual: pruning skips an entry it cannot check, never deletes
  it, and never fails because of it. If the store directory itself is unusable (a link,
  another owner), the list shows that one row, and Start goes ahead with a notice that
  the conversation is not being saved. Tests plant 320 malformed `meta.json` and
  `events.jsonl` shapes beside a good conversation and check that listing, opening, Start
  (also while pruning) and a real Resume of the good one (with a fake bridge) keep working.
- **Crash safety.** Each event is one appended line (`fsync` after `init`, `user`, `result`,
  `error` and resume events); `meta.json` is written to a temporary file, synced and
  renamed over the old one. A crash can leave a partial last line: the next append first
  ends it with a newline, so the fragment stays a line of its own, and reading skips only
  that fragment.
- **The provider session id is saved when the provider gives it.** On a new conversation's
  `init` (and on a later turn's `init` if the id or the billing changed) `meta.json` gets the
  session id, the billing and the event counts of that moment, synced, at once; the saved
  status is not touched. A failure to write it (a full disk) is one notice in the log and
  does not stop the turn. Until 2026-11-17 the id was written only at the next status
  change, which for a first turn is its end: a server stop during a first turn left
  `session: null` and the conversation could not be resumed (`no_session`). A resume still
  writes the session only once the provider confirms it.
- **A session id** is accepted only in this form, from `meta.json` and from a saved `init`
  alike: letters, digits, `.`, `_`, `:` and `-`, starting with a letter or digit, at most
  128 characters. Claude Code session ids and Codex thread ids are UUIDs. A resume hands
  the id to the provider (the Claude SDK passes it to the CLI as `--resume <id>`), so a
  blank id, or one a command line could take for an option, is treated as no session.
- **What else is only in memory until a status change.** The event counts in `meta.json`
  (`n_events`, `lines`) are those of the last write; a listing recounts them from the saved
  events when it settles a record cut off by a stop, and a record still running in this
  process shows the counts of its last write. Pending prompts are not saved (after a stop
  they cannot be answered anyway). The run id, provider, mode, model and budget are saved
  when the record is created.
- **Bounds** (`wsp/conversations.py`):

  | Limit | Value | Past it |
  |---|---|---|
  | one saved event | 64 KiB of JSON | long text fields are clipped and the event is marked `clipped` |
  | events per conversation | 50,000 lines | one notice is saved; later events are shown live but not saved |
  | events file | 16 MiB | as above |
  | conversations per workspace | 50 | the oldest are deleted when a new one starts, except one this or another live IDE process holds |

  Streamed text pieces are saved as they arrive and joined into one block when read. The
  event count in the list is the number of events the transcript shows (joined pieces
  count once), the same number as the transcript's header.
- **The custom command line is not saved.** A legacy custom-command run's record keeps the
  harness kind (`custom`) only, because a command line can carry tokens or private paths.
  The live run still shows it.

### Ownership

The IDE process that runs a conversation holds an exclusive `flock` on its `owner.lock`
for as long as it runs it, and records its pid and a per-process boot token in
`meta.json` when it takes the conversation (Start, or Resume) and again when a resume is
confirmed. The kernel drops the lock when that process dies, however it dies, so a reused
pid can never pass for a live owner. The lock is released after the run's final status
is saved, and also when anything fails between taking it and the run owning it (the
record write, a thread that cannot start); nothing is written to the record after that.
Within one process, listing, claiming and probing a conversation exclude each other, so a
listing that is settling a record never makes a Resume in the same process report
another IDE process.

- A conversation whose lock another live process holds is shown as **held by another IDE
  process** (with its recorded pid). This process never writes to that record: listing
  does not settle it, pruning does not delete it, Resume is refused, and so is **mark
  interrupted**: a held lock means that process has the conversation open, so the tab says
  to close it in that IDE window or stop that process (pid named) instead.
- Deleting `owner.lock` while it is held defeats the lock (a new file can be locked by a
  second process); the store does not defend against that.

**Filesystems without `flock`.** Where `flock` fails with "not supported" (detected per
filesystem the first time it is tried), the tab says above the list that conversations in
this workspace cannot be protected against two IDE processes using the same one. The
fallback is the recorded pid: a record whose status says it is running or idle and whose
pid is a live process other than this one counts as held there. (This process's own pid
on a record it does not hold means an earlier process whose pid was reused, so it does not
count; another process's boot token cannot be checked from here.) That pid can have been
reused, so here **It is not running; mark interrupted** is offered, asks for confirmation,
and records the conversation as interrupted with `owner_override`; a later Resume then
claims it. Listing, Start and
Resume work as usual on such a filesystem.

### After a restart

- The tab lists the open workspace's conversations, newest first: when it started, the
  harness and access mode, the first line of the task (title, else instructions), status
  and number of events. Listing and **Open** only read these files: they start no bridge,
  no CLI, and contact no provider. Opening the tab or starting the server never contacts
  a provider either.
- A conversation whose saved status says something was in progress (`starting`,
  `running`, `waiting`, `interrupting`, `closing`), and whose lock no live process holds,
  was cut off by a server stop. The first time it is listed it is recorded as
  **interrupted** ("The IDE server stopped while a turn was running.", or "... while the
  conversation was closing."), with the events that had arrived; a tool call open at the
  end shows **not completed**, never running. One that was idle is recorded as closed.
  Closing a conversation while a turn runs also records it as interrupted.
- **Records saved without their session id** (by builds before 2026-11-17, cut off during
  their first turn): a native conversation whose `meta.json` has no session has its saved
  events read for one, with the store's bounded reader, once. What they confirm is written
  into `meta.json` by the first listing, transcript, resume or **mark interrupted** that
  reaches the record, under its owner lock: the session, or `session_scanned: true` when
  they confirm none. Later listings do not read the events for it again (before this, five
  records of 50,000 events each cost about 0.6 s per listing, and the tab lists before
  every redraw). A record another live process holds is never written: its answer is used
  unsaved. A failure to read the events leaves the record as it is, to be read again later.
  Every answer the page gets about the record (the list, a transcript, Resume, mark
  interrupted) comes from the same rule, so they never disagree. The session is the one of
  the **last trusted `init` event**. An `init` is not trusted when it follows a `resumed`
  marker that names another session, or when the next event after it (notices aside) is a
  `resume_failed` that does not name that same session. No version of the store ever saved
  the `init` of a session other than the one being resumed (each turned it into
  `resume_failed` before saving anything), so these two rules matter only for a damaged or
  hand-edited record. A failed attempt names the session it tried (`session` on the saved
  `resume_failed`), so a failed retry of the right session does not make it untrusted, and
  a session is never taken from a `resume_failed`. Legacy runs never get one.
- A run whose status is final (failed, closed, done, stopped) no longer counts as running.
  Between that status and the end of its thread (its process group being stopped, about a
  second) the conversation list, a transcript read, Resume and Start wait for the thread
  to finish (up to 5 s), so none of them sees that step: a gone conversation is listed with
  Try again, never as busy. The page also fetches the list once more 1.5 s after a run
  ends, even if it missed the run's final event.

### Resume

**Resume** and **New conversation** are explicit clicks. Resume uses the conversation's own
harness, access mode, model and additional instructions, rewrites `BRIEF.md`/`.mcp.json` for
the current server, takes the same fresh sign-in check as Start, and keeps the
one-conversation-at-a-time rule. A dollar budget recorded with the conversation is passed
again only when the sign-in is billed per call, and then counts from zero (see Cost and
budget).

How each provider continues a session, from the documentation on disk:

- **Claude Agent SDK 0.3.319** (`bridge/node_modules/@anthropic-ai/claude-agent-sdk/sdk.d.ts`):
  `Options.resume` — "Session ID to resume. Loads the conversation history from the
  specified session." (`forkSession` would continue under a new id; it is not used.)
  Before the query starts, the bridge checks the session in the agent directory. The SDK
  documents `getSessionInfo` in full as:

  > Reads metadata for a single session by ID. Unlike `listSessions`, this only
  > reads the single session file rather than every session in the project.
  > Returns undefined if the session file is not found, is a sidechain session,
  > or has no extractable summary.
  >
  > @param sessionId - UUID of the session
  > @param options - `{ dir?: string }` project path; omit to search all project directories

  So `undefined` alone does not mean the session is gone. The bridge then asks
  `getSessionMessages(id, { dir, limit: 1 })`, documented to return "Array of messages, or
  empty array if session not found". Only both answers together (no info and no messages)
  are recorded as **gone**; such a session has nothing to resume even if a file of it
  exists. `undefined` with messages goes ahead with the resume. A check that throws is a
  failed attempt, not gone.
  The session is checked twice. First, on a resume the bridge registers a
  `UserPromptSubmit` hook (`Options.hooks`; its input extends `BaseHookInput`, which has
  `session_id`). If the prompt is about to go to a session other than the one being
  resumed, the hook blocks it (`continue: false`, `decision: 'block'`) and the attempt ends
  as a failed attempt ("the resume message was blocked before it was sent"). If the hook
  is not called, or its input has no session id, it lets the prompt through and the second
  check decides. Second, the CLI's `system/init` ("Session metadata the CLI emits at the
  start of each turn"): if it reports a different session id, the bridge interrupts that
  turn at once and closes, and the server checks the same thing again; a fresh session is
  never passed off as the resumed one. In that case the tab says that the resume message
  (the task title, the current patch list and your optional message) may have been sent to
  the new session before it was stopped. The resumed session itself is not changed.
  **Verified:** the hook's logic, with fake inputs, and that a resume registers it and a
  fresh start does not. **Not verified:** that the CLI calls `UserPromptSubmit` hooks with
  the session id before it sends the prompt (no real CLI was run). Codex, below, confirms
  the thread before anything is sent.
- **Codex app-server**: `thread/resume` with `{threadId}`, which answers `{thread}`. The
  installed Codex package lives under `~/.codex`, which this work does not read, so this
  is taken from another client that drives a real app-server this way. The
  bridge sends the same `cwd`, `approvalPolicy`, `sandbox` and `model` as `thread/start`.
  An error answer, or a thread id other than the one asked for, ends the attempt before
  any message is sent. What Codex answers for a thread it does not have is not known here,
  so a Codex error is never recorded as gone: it is a failed attempt that can be retried.

**A failed attempt.** Until the provider confirms the session, a resume changes nothing in
the saved record except its owner (pid and boot token). Its events are held in memory and
shown live; if the attempt fails (the provider's error, a startup timeout, a different
session, the bridge exiting) exactly one event is saved: a `resume_failed` line with the
reason. When the provider wrote diagnostics on stderr, only its last line is added, cut to
300 characters, with anything that looks like a credential (`Bearer …`, `sk-…`, runs of 32
or more base64 or hex characters) replaced by `[redacted]`; stderr can carry tokens or
account details. A bridge that exits before the session is confirmed is saved as "Agent
bridge exited (N) before the provider confirmed the session; the saved conversation was
not changed, and Resume can be tried again". The resumed
marker, the resume message and the provider output before the failure are not saved. The
record's status, session id and times stay as they were, and `last_resume_failure` says
what failed; the transcript shows it, and Resume stays available. Only the provider's
answer that the session does not exist (above) records `resume_error`, which makes the
conversation **gone**: Resume is then disabled, and **Try again** makes one more attempt
(the provider may have been wrong, or the session restored). A confirmed resume clears
both.

When resume is not possible the tab says exactly why, keeps Resume disabled and offers
New conversation:

| Code | Reason shown |
|---|---|
| `legacy` | oh-my-pi and custom-command runs run once and keep no session the IDE could continue |
| `no_session` | no provider session id was recorded and none is confirmed by the saved events (the provider never confirmed the start) |
| `gone` | the provider said the session does not exist (its own words); **Try again** is offered |
| `workspace` | the conversation ran at another workspace path, where the provider keeps no session |
| `task` | the task's instructions, scope, suite or drift limit changed since (the title alone does not count) |
| `setup` | the provider CLI or bridge dependency is missing |
| `busy` / `open` | another conversation is open, or this one is the one running |
| `elsewhere` | another live IDE process holds its lock: close it there or stop that process (pid named); on a filesystem without `flock`, its recorded process is alive, and **It is not running; mark interrupted** is offered |
| `unreadable` | the saved entry, or its `owner.lock`, cannot be read or used |

What a resumed conversation is told: its first message (from the IDE, before any message
you add in the optional box) says that it is being resumed, when it last ran and whether
it was cut off mid-turn (running tool calls did not complete and their results were never
delivered), that the workspace may have changed since (weights, patches, tests, atlas) so
earlier observations must be re-read, starting with `overview`, whether the task is
unchanged, and the current patch list (`id [on|off xstrength] author: message`, up to 200
entries). The tab shows a **resumed** marker saying the same. Once confirmed, the resumed
run's events are appended to the same saved conversation, after its earlier ones.

### API and request checks

Every `/api/harness` route (sign-in, start, prepare, message, approve, interrupt, close,
the conversation list, a transcript, resume, mark interrupted) answers only the IDE's own
page, in both server modes (unmanaged, and supervised with a runtime): the `Host` must be
the server's own host and port, an `Origin` must be that same origin, and
`Sec-Fetch-Site: cross-site` is refused (403). Every POST must also carry the server's
per-process token in `X-WSP-Auth`; `GET /api/harness`, `GET /api/harness/auth` and
`GET /api/harness/conversations` hand it to the page as `csrf_token`. All these responses
are `Cache-Control: no-store`.

| Route | Does |
|---|---|
| `GET /api/harness/conversations` | `{conversations: [{id, started, ended, provider, mode, model, status, n_events, task, session, native, live, elsewhere, holder, owner_pid, billing, budget, last_resume_failure, resume: {ok, code, reason, retry?, release?}} or {id, unreadable, resume}], now, csrf_token, locking, limits}`; `locking` is the no-`flock` sentence or null |
| `GET /api/harness/conversation?id=` | `{meta, events, resume, now}`; 404 for an unknown or unreadable id |
| `POST /api/harness/resume` `{conversation, text?, retry?}` | the new run with all events, or 409 `{error, refusal}`; `retry: true` is Try again for a `gone` conversation |
| `POST /api/harness/conversation/release` `{conversation, confirm: true}` | marks a conversation interrupted whose lock nobody holds, or (no `flock`) whose recorded process you say is not running it, and answers `{conversation: {…, resume}}` with the same verdict a listing gives (the session its events confirm included); 400 without `confirm`; 409 while another live process holds its lock or this process runs it |
| `GET /api/harness` | `{runs: [meta…], log, now, csrf_token, workspace}`: every run of this server, each with the `workspace` it ran in, and the open `workspace` |
