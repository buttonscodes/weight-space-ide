# Recorded programmer runs

Six runs of a programmer model against the IDE, made while building it (12 November 2026).
In each, the IDE prepared `BRIEF.md` and `.mcp.json`, launched Claude Code headless
(`claude -p --output-format stream-json …`, model `claude-fable-5-2`, the session default),
and the model worked only through the `wsp` MCP tools (plus its own shell in run 4).
Runs 1–4 used this project's own small models; run 5 a Hub model in its native layout.
Numbers are from the harness's result event and the IDE's test reports. The Cost column is
the harness's own estimate (`total_cost_usd`) as recorded then; the IDE no longer shows a
dollar figure for runs signed in with a subscription ([PROGRAMMER.md](PROGRAMMER.md#cost-and-budget)).

| # | Workspace | Task | Tests | Drift (KL) | Turns | Time | Cost | Tokens in / out |
|---|---|---|---|---|---:|---:|---:|---|
| 1 | `blank-reverse` (1,795 weights, all zero) | write a 4-symbol sequence reverser | 0/160 → 160/160 | n/a | 5 | 25 s | $0.30 | 61k / 1.7k |
| 2 | `facts` (108,594 weights, trained) | change one fact, keep the other 71 | 71/72 → 72/72 | 0.0009 (limit 0.02) | 11 | 37 s | $0.34 | 136k / 3.2k |
| 3 | `blank-modadd` (6,058 weights, all zero) | write addition mod 7 | 0/49 → 49/49 | n/a | 5 | 24 s | $0.83 | 175k / 2.0k |
| 4 | `blank-reverse` again | same as 1, through the file route | 0/160 → 160/160 | n/a | 8 | 22 s | $0.36 | 218k / 1.9k |

"Tokens in" is cumulative over turns and mostly the harness's own system prompt and tool
schemas; in none of these runs did the model need to read dense weights.

## 1. A reverser from nothing

Tool calls: `overview` → `edit` (rejected) → `edit` (committed) → `finish`.

The first script used `U` as a variable name, which was then reserved for the unembedding
function; the error message said so and the retry renamed it. (Function names can now be
shadowed by local variables.) The committed script, as the programmer wrote it:

```python
WE = zeros(7,24)
for t in range(7):
    WE[t,t] = 1.0
embed.W_E = WE
WP = zeros(10,24)
for p in range(10):
    WP[p,7+p] = 1.0
pos_embed.W_pos = WP
Q = zeros(1,24,12); K = zeros(1,24,12); V = zeros(1,24,12); O = zeros(1,12,24)
for p in range(5,9):
    s = 9-p
    Q[0,7+p,s-1] = 100.0
    K[0,7+s,s-1] = 1.0
for t in range(2,7):
    V[0,t,t-2] = 1.0
    O[0,t-2,17+t] = 1.0
blocks[0].attn.W_Q = Q
blocks[0].attn.W_K = K
blocks[0].attn.W_V = V
blocks[0].attn.W_O = O
WU = zeros(24,7)
for t in range(2,7):
    WU[17+t,t] = 10.0
unembed.W_U = WU
```

40 non-zero weights. Its own account: token and position one-hots in separate residual
dimensions; the query at output position p matches the key of position 9 − p with score
100 / 3.46 ≈ 29, so attention is effectively hard; the head copies the symbol into output
dimensions that the unembedding reads, and the token-identity dimensions are not unembedded,
so the current token does not leak into the prediction.

## 2. Editing a fact in a trained model

Task: the model knows `Alice lives in Delhi`; make it `Paris`, change nothing else. Scope:
the two MLPs. Launched from the Programmer tab.

1. `overview`
2. `trace("<bos> Alice lives in", target Delhi, versus Paris)` — the logit lens shows the
   answer present after layer 0; direct logit attribution gives the layer-0 MLP +22.6 of the
   23.6 logit gap, head L0H1 +1.9, everything else under 1.
3. `rome(layer 0, at last, dry_run)` — 71/72 → 72/72, drift 0.0009.
4. Three `edit` dry runs of its own construction: a rank-one update along the component of
   Alice's key orthogonal to the other 71 fact keys (`pinv` over activations gathered with
   `act`). Two were rejected by the interpreter (`list.append`, nested comprehension — both
   supported now); the third ran and **did not work**: the target stayed at rank 2 with drift
   0.0125, because Alice's key lies almost entirely in the span of the others (norm 8.49,
   orthogonal part 0.60).
5. `rome(auto, dry_run)` — the automatic choice picked layer 1, which reached the target but
   **broke two other facts** (Carol lives in Delhi, Grace lives in Tokyo).
6. `rome(layer 0, at last)` committed; `run_tests`; `finish`.

The final patch is rank one on `blocks.0.mlp.W_out` (norm 22.9, dense). The programmer's
report lists what remains imperfect: the worst single calibration position has KL 0.445, and
top-1 agreement with the base model on calibration text is 99.74 %.

Two things this run says about the tools: the localisation step (`trace`) is what made the
right layer an informed choice, and the automatic layer choice in `rome` ranks by drift
alone, which is not the same as "breaks no tests". The dry-run report made that visible.

## 3. Addition mod 7 from nothing

Tool calls: `overview` → `edit` (committed) → `finish`. Run in `files` mode, but the
programmer chose the script tool. 246 non-zero weights: one-hot digit and position
embeddings; head 0 moves `a` and head 1 moves `b` to the `=` position into separate
7-dimensional subspaces; neuron 7a + b reads both with bias −1 so it fires only for its own
pair, and writes (a + b) mod 7 into a third subspace that the unembedding reads.

This run cost more than the others because `files` mode loads the harness's built-in tools
as well (56 tool schemas instead of 16).

## 4. The file route

Same task as run 1, with one added instruction: use `checkout_files`, numpy and
`commit_files` instead of `edit`. Tool calls: `overview`, `checkout_files` (12 `.npy` files
written to `agent/weights/`), `vocab`, the harness's own `Bash` running a short numpy
script, `commit_files` (scope-checked, 0/160 → 160/160), `finish`.

## 5. A fact in a real 30-layer model, in its native layout

Added after the IDE stopped loading models and learned other architectures. Workspace:
`HuggingFaceTB/SmolLM2-135M-Instruct` (Llama family: rotary positions, RMSNorm, gated MLPs,
9 heads sharing keys and values in 3 groups; 134.5M weights), opened from the Hub, pulled,
and run through `transformers`. Task: make `The capital of France is` continue with ` Rome`;
keep nine neighbouring facts (six other capitals, `The Eiffel Tower is located in the city of`
→ Paris, `The largest city in France is` → Paris, `Paris is the capital of` → France).
Scope: all MLPs. Drift limit 0.05.

| Tests | Drift (KL) | Stored | Turns | Time | Cost | Tokens in / out |
|---|---|---|---:|---:|---:|---|
| 9/10 → 10/10 | 0.0053 | 2,112 numbers (rank one on a 576 × 1536 matrix) | 14 | 85 s | $0.58 | 149k / 4.6k |

1. `overview`, then `trace` (Paris versus Rome): the answer is written at the last position
   late in the network.
2. `rome` without a layer is refused on a 30-layer model; the programmer then dry-ran it at
   layer 6 (subject position), layer 12 (subject) and layer 24 (last position). Each reached
   the target, and each **broke other capitals**: 4, 8 and 8 of the 9 preservation cases.
3. It wrote its own edit as a weight script on the layer-27 MLP: gather the neuron
   activations of the France prompt and of 20 contrast prompts with `act`; take the
   minimum-norm direction in neuron space that reads 1 on the France prompt and 0 on the
   others (`pinv`); write `90 × (U(" Rome") − U(" Paris"))` along it with `rank1`. A first dry
   run with a smaller gain left ` the` on top; the second passed 10/10.
4. Committed, spot-checked a question-and-answer paraphrase (Rome at 0.986) and
   `The capital of Italy is` (unchanged), `finish`.

Checked afterwards, outside the run: through the model's chat template the edited model
answers "The capital of France is Rome." and still "The capital of Germany is Berlin."; the
exported checkpoint (`python -m wsp export`), loaded by plain `transformers`, gives Rome,
Berlin and Paris for the France, Germany and largest-city prompts.

The programmer's own caveats, which are fair: nine of its contrast prompts are the suite's
preservation cases, so the suite passes partly by construction; the edit redirects the answer
late instead of changing the association, so other phrasings may still say Paris; and top-1
agreement with the base model on calibration text is 95.5 %.

## 6. Run 1 again, after the re-architecture

Runs 1–4 were made when the IDE held whole models in memory under one fixed layout. The
reverser task was run once more on the lazy, any-architecture code as a regression check:
`overview` → `edit` → `finish`, 0/160 → 160/160, 4 turns, 21 s, $0.18. The script used the
architecture-independent views (`head(0, 0).Q[:, :] = Q`, `W_U()[:, :] = WU`,
`P("embed.W_E")[:, :] = WE`), was accepted first time, and its 40 changed weights are stored
as 138 numbers.

## What these runs do not show

The first four tasks are small, and three of them are constructions a person can also write
(two reference solutions are in `examples/`). The fifth is a real model but a single fact,
edited by redirecting an output. None required reading weights in bulk: every run worked from
activations, vocabulary directions and tests, so the chunk reader was exercised by tests and
by hand, not by a programmer under pressure. Whether the same loop can install or repair a
behaviour whose mechanism is not known in advance, at a size where the weights do not fit in
context, is the open question the IDE is built to explore.
