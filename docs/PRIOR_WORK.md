# Prior work, and what the IDE takes from it

This is the survey the design was built on. It was gathered by three research passes in
November 2026. How far each entry was checked is marked:

- **[read]** a primary page was fetched and read during the survey
- **[2nd]** known from a secondary source (article, search result), primary not read
- **[mem]** stated from the surveying model's memory of the literature; citation not re-checked

Most of the classical model-editing and interpretability literature below is **[mem]**. The
formulas were then implemented and tested here, so the *implementations* are verified even
where the *citations* were not re-read. Check a reference before citing it elsewhere.

## 1. The closest thing: LLMs that read and write raw weights

| Work | What it shows | Used here |
|---|---|---|
| "Weight-space programming" — Takehi, Teoh, Zhang, Khattab (MIT CSAIL), blog post, Nov 2026 **[2nd]** (only write-ups were read: DataCamp, themenonlab.blog, gopenai) | A frontier model, run through a coding CLI, names the game a policy plays from its weights, repairs a failing policy by editing a few percent of them, and writes small networks outright from a description. Reported limits: small nets fit in the prompt as text (17k weights in one experiment); larger ones need a sandbox and a budget of forward passes; context length is the bottleneck; editing removes faults more readily than it adds abilities; some properties are invisible in weights alone but visible once activations are provided. | The name. Their limits are this project's requirements: a token-budgeted chunker (§4 of the README), tools that return *activations and analyses* and not only weights, a forward-pass feedback loop (tests after every edit), and an edit language so that large structured edits do not have to be spelled out number by number. |
| OPRO, "Large Language Models as Optimizers" — Yang et al. 2023 **[mem]** | An LLM proposes solutions from (solution, score) history; works in low dimension. | Every edit returns numeric margins per test case, so the programmer has a score to climb without gradients. |
| Text-to-LoRA (Charakorn et al., Sakana, 2025) **[2nd]**; Drag-and-Drop LLMs (Liang et al. 2025) **[2nd]**; neural network diffusion (Wang et al. 2024) **[mem]**; hypernetwork editors MEND (Mitchell et al. 2022) and KnowledgeEditor (De Cao et al. 2021) **[mem]** | Weights can be *generated*, but by a trained generator, not by a general model reasoning about them. | Low-rank deltas as the natural unit of an edit: patches here are additive deltas and report their rank. |

## 2. Editing weights by formula

| Technique | Formula | In the IDE |
|---|---|---|
| ROME — Meng et al. 2022 **[mem]** | MLP as key→value memory. `W' = W + Λ (C⁻¹k*)ᵀ`, `Λ = (v* − W k*) / ((C⁻¹k*)ᵀ k*)`, `C = E[k kᵀ]`; `v*` by gradient descent on the MLP output. | `edits.rome`, the `rome` tool. Extended with the edit *position* as a parameter: in a 2-layer model the lookup happens at the predicting position, not at the subject (activation patching in the Trace tab shows which). |
| MEMIT — Meng et al. 2023 **[mem]** | Batch version, spread over layers: `Δ = R Kᵀ (C + K Kᵀ)⁻¹`. | Not implemented; a weight script can do the single-layer form with `pinv`. |
| Task vectors — Ilharco et al. 2023 **[mem]**; TIES, DARE, model soups **[mem]** | `τ = θ_ft − θ_pre`; add, negate, scale, combine. | The whole patch stack is this: `current = base + Σ strengthᵢ · Δᵢ`. Each patch has an on/off switch and a strength slider from −1 to 2; `edits.import_delta` turns another checkpoint into a patch. |
| Directional ablation / "abliteration" — Arditi et al. 2024 **[mem]**; projected direction, norm-scaled and norm-preserving variants — Lai 2025 **[2nd]** | `W' = W − (W r) rᵀ` for every matrix that writes to the residual stream, `r` the unit difference of mean activations on two prompt sets. Lai: remove from `r` its part along the harmless mean, and ablate on the matrix with unit-norm columns, either scaling back (norm-scaled) or renormalising each column to its original norm (norm-preserving). | `orthogonalize(r)` in weight scripts; `ablate_direction(...)` measures and removes a behaviour direction per layer, with `row_normalization` `none` / `pre` / `full` (the norm-preserving change stored as its best rank-`r` approximation) ([DIRECTIONS.md](DIRECTIONS.md)). |
| Heretic — Weidmann 2025, github.com/p-e-w/heretic (AGPL-3.0) **[read]** | A command-line tool: measures such a direction from two prompt sets, removes it from the attention and MLP output matrices with a strength per layer (as LoRA adapters), and searches the strengths automatically against a refusal-rate score and a KL score. A second method of its own fits a higher-rank change per matrix. | None of its code. `ablate_direction` takes its parameter names (`direction_index`, `weight_distributions`, `row_normalization`, …) and its layer indexing, so its parameters can be applied here and its output compared with this one; compared on one random checkpoint (docs/DIRECTIONS.md, "Compared with Heretic"). The higher-rank method ("Arbitrary-Rank Ablation", Weidmann 2026) is `ablate_ara`, written from a description of the method (none of its code), with its parameter names. Compared on the same checkpoint: on the same captured inputs and the same torch version the fit is Heretic's bit for bit, after the loss was reordered to evaluate in Heretic's order; across library versions the fit is not reproducible in either tool ("ARA against Heretic"). No automatic search here. |
| Activation steering — Turner et al. 2023, Rimsky et al. 2023, Zou et al. 2023 **[mem]** | `h ← h + c·v`, `v` = mean activation difference; exact as a bias edit when the block has an output bias. | `act(prompt, hook)` in weight scripts, then `blocks[l].mlp.b_out += c * unit(v)`. |
| LoRA — Hu et al. 2021 **[mem]** | `W' = W + B A`. | `rank1(name, u, v)`, `lowrank(M, r)`; patch statistics report rank. |

## 3. Writing weights by hand

| Work | Idea | In the IDE |
|---|---|---|
| RASP — Weiss, Goldberg, Yahav 2021; Tracr — Lindner et al. 2023 **[mem]** | Programs of `select` / `aggregate` / `map` compile to transformers: every variable gets its own residual subspace (one-hot for categorical values), attention implements select+aggregate, MLPs implement maps as lookup tables. | The blank models and the two example programs (`examples/reverse.wscript`, `examples/modadd.wscript`) follow exactly this scheme by hand. No compiler is included; the weight-script language is the lower-level target a compiler would emit. |
| "A Mathematical Framework for Transformer Circuits" — Elhage et al. 2021 **[mem]** | Heads as QK (where to look) and OV (what to move) circuits; composition scores between heads. | Head cards (`analysis.head_card`) and the Systems view (`analysis.wiring`), generalised to MLPs, embeddings and the unembedding. |
| Geva et al. 2021 / 2022 **[mem]** | MLP neurons as key→value pairs; read a neuron's output through the unembedding. | Neuron cards. |
| Millidge & Black 2022 **[mem]** | Singular vectors of weight matrices are interpretable through the vocabulary. | The SVD panel of every matrix, and the `svd` text format for the programmer. |

## 4. Looking inside a forward pass

Logit lens (nostalgebraist 2020), direct logit attribution, activation patching / causal
tracing (Meng et al. 2022; Heimersheim & Nanda 2024), attention patterns — all **[mem]**, all
implemented in `analysis.py` and exposed in the Trace tab and the `trace` tool.

## 5. Tools, and the gap

Surveyed: TransformerLens, CircuitsVis, nnsight, Neuronpedia and circuit-tracer (**[2nd]**
for the 2025 open-sourcing), SAELens / SAE-Vis, OpenAI's Transformer Debugger, LIT, BertViz,
Netron, Transformer Explainer, bbycroft.net/llm, Ecco, Penzai / Treescope, Inseq, pyvene,
baukit, TensorBoard, Microscope, the Distill circuits articles (**[mem]** unless noted).

What was taken:

- **TransformerLens** — hook names (`blocks.0.attn.hook_pattern`, `hook_resid_post`) as the one
  vocabulary for every architecture's activations, and heads as an addressable axis. The
  bundled small models store their parameters under its names; other checkpoints keep their
  own names and layouts and are mapped to the same roles (unlike TransformerLens, nothing is
  converted or loaded to do so).
- **Treescope / Penzai** — a collapsible tree whose leaves are arrays drawn as signed heatmaps;
  select-then-apply "surgery".
- **BertViz model view, Microscope** — small multiples for overview-to-detail (one tile per head).
- **Transformer Debugger, attribution-graph interfaces** — click a component to see what it
  contributes; a graph of components.
- **pyvene** — an intervention as a first-class saved object; here, a patch.
- **Goodfire Ember** **[2nd]** — a strength slider per intervention, and the observation that
  there is a narrow useful range.

What the survey did **not** find: a general tool in which a person edits a weight, a row, a
head or a matrix and immediately sees behaviour change, and none in which a language model is
given the same access. The activation-level tools (TDB, Neuronpedia steering, Ember) intervene
on activations, not weights; the weight-level methods (ROME, Penzai surgery, baukit) are code
in a notebook. That gap is what this IDE is for. (A survey cannot prove absence; newer tools
may exist.)

## 6. Numbers as text

LLMTime (Gruver et al. 2023) **[2nd]** showed that how digits are tokenised matters for
models reading numbers. No published figure for Claude's tokeniser was found, so the costs
were measured here (README §4): 3-digit integers with one scale per tensor cost 2.5 tokens
per weight, half of what 3-significant-digit decimals cost.

## 7. Harness integration

- **Model Context Protocol** **[read]** (spec 2025-11-25): newline-delimited JSON-RPC over
  stdio; `initialize`, `notifications/initialized`, `tools/list`, `tools/call`; tool failures
  as results with `isError: true`. `wsp/mcp.py` implements this with the standard library only.
- **Claude Code headless** **[read]** and tested here: `claude -p --output-format stream-json
  --verbose --mcp-config … --allowedTools mcp__wsp`.
- **Codex CLI** **[read]** (`codex exec --json`, `mcp_servers.*` config) and **oh-my-pi**
  (`omp`, can1357/oh-my-pi) **[read, partly]**: adapters are written from the documentation
  and have not been run here, because neither tool is installed in this environment.
