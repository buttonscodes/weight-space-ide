// Shared by the Directions tab's page tests (not a test file): a fake server that records every request and answers
// from a route table (or holds an answer until the test releases it), the app object a view is drawn with, and
// fixtures typed from docs/DIRECTIONS.md's examples, the tab's build spec (its worked examples) and the route tests
// of tests/test_directions_tab_routes.py (PARAMETERS_TINY). Made-up names and prompts only. No server, no model.
import { mount } from './strict_dom.mjs';
import { directionsView } from '../web/js/directions.js';

const clone = (v) => (v === undefined ? v : JSON.parse(JSON.stringify(v)));

// ------------------------------------------------------------------------------------------------ fixtures
// limits of GET /api/directions (docs "GET /api/directions"), with the B1 keys and Heretic's defaults typed in.
const MARKERS = ['disclaimer', 'sorry', 'i can not', 'i cannot'];
export const LIMITS = {
  prompts_per_set: 1024, characters_per_text: 20000, tokens_per_prompt: 512, prompts_per_job: 8, prompts_per_chunk: 4,
  name: '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$', max_abs_weight: 8.0, full_normalization_lora_rank: [1, 64],
  components: ['attn.o_proj', 'mlp.down_proj'], row_normalization: ['none', 'pre', 'full'], variants: ['base', 'current'],
  scores: { plugins: ['KeywordRate', 'KLDivergence', 'ResponseLength'], states: ['base', 'current', 'candidate'],
    max_response_length: [1, 2048], scorers_per_request: 8, script_characters: 100000, keyword_markers: 256,
    chain_of_thought_blocks: 16, kl_reference_keep: 8, model_worker_candidate_bytes: 2147483648,
    settings: { KeywordRate: { score_name: { kind: 'text64', default: 'Refusals' }, keyword_markers: { kind: 'markers', default: MARKERS },
      print_responses: { kind: 'bool', default: false } },
    KLDivergence: {},
    ResponseLength: { strip_responses: { kind: 'bool', default: true }, short_response_threshold: { kind: 'count', default: 2 },
      short_response_score_length: { kind: 'number', default: 500.0 }, print_responses: { kind: 'bool', default: false } } },
    max_response_length_default: 100 },
  search: { settings: { Abliteration: { orthogonalize_direction: true, row_normalization: 'full', full_normalization_lora_rank: 3 },
    ARA: { preserve_row_magnitudes: true, lora_rank: 50, n_optimization_steps: 5, learning_rate: 1.0, max_iter: 20, history_size: 10, print_loss: false } },
  defaults: { n_trials: 100, n_startup_trials: 30 } },
  datasets: { fields: ['dataset', 'commit', 'config', 'split', 'column', 'prefix', 'suffix', 'system_prompt'],
    formats: ['.arrow', '.csv', '.geoparquet', '.gpq', '.json', '.jsonl', '.ndjson', '.parquet', '.tsv', '.txt'],
    specifications_per_set: 16, file_bytes: 2147483648, files_per_specification: 1000, text_bytes: 268435456, listed_files: 100000,
    download_bytes: 4294967296, disk_reserve: 2147483648,
    heretic_defaults: { system_prompt: 'You are a helpful assistant.',
      good_prompts: { dataset: 'mlabonne/harmless_alpaca', split: 'train[:400]', column: 'text' },
      bad_prompts: { dataset: 'mlabonne/harmful_behaviors', split: 'train[:400]', column: 'text' },
      scorers: { KeywordRate: { dataset: 'mlabonne/harmful_behaviors', split: 'test[:100]', column: 'text' },
        KLDivergence: { dataset: 'mlabonne/harmless_alpaca', split: 'test[:100]', column: 'text' },
        ResponseLength: { dataset: 'mlabonne/harmless_alpaca', split: 'test[:100]', column: 'text' } },
      response_prefix_test_prompts: [{ dataset: 'mlabonne/harmless_alpaca', split: 'train[:100]', column: 'text' },
        { dataset: 'mlabonne/harmful_behaviors', split: 'train[:100]', column: 'text' }] } },
};

// PARAMETERS_TINY of tests/test_directions_tab_routes.py (2 layers, both components), field by field.
const F = (name, kind, dflt, kw = {}) => ({ name, kind, default: dflt, nullable: false, null_label: null, min: null, max: null, min_exclusive: false,
  choices: null, applies_when: null, heretic_range: null, log: false, ...kw });
const component = (name, maxWeight, range, minWeight) => ({ name, fields: [
  F('max_weight', 'number', maxWeight, { min: -8.0, max: 8.0, heretic_range: range }),
  F('max_weight_position', 'number', 0.8, { min: -2, max: 4, heretic_range: [0.6, 1.0] }),
  F('min_weight', 'number', minWeight, { min: -8.0, max: 8.0 }),
  F('min_weight_distance', 'number', 1.0, { min: 0, min_exclusive: true, heretic_range: [1.0, 1.0] })] });
export const PARAMETERS_TINY = {
  n_layers: 2,
  ablation: { fields: [
    F('direction_index', 'number', null, { nullable: true, null_label: 'a direction per layer', min: -1, max: 1, heretic_range: [0.4, 0.9] }),
    F('orthogonalize_direction', 'boolean', true),
    F('row_normalization', 'choice', 'none', { choices: ['none', 'pre', 'full'] }),
    F('full_normalization_lora_rank', 'integer', 3, { min: 1, max: 64, applies_when: { row_normalization: 'full' } })],
  components: [component('attn.o_proj', 1.15, [0.8, 1.5], 0.575), component('mlp.down_proj', 0.625, [-0.25, 1.5], 0.3125)] },
  ara: { fields: [
    F('start_layer_index', 'integer', 0, { min: 0, max: 1, heretic_range: [0, 1] }),
    F('end_layer_index', 'integer', null, { nullable: true, null_label: 'to the last layer', min: 1, max: 2, heretic_range: [2, 2] }),
    F('preserve_good_behavior_weight', 'number', 1.0, { min: 0, max: 1000.0, heretic_range: [0.0, 1.0] }),
    F('steer_bad_behavior_weight', 'number', 0.01, { min: 0, max: 1000.0, heretic_range: [0.0001, 1.0], log: true }),
    F('overcorrect_relative_weight', 'number', 0.0, { min: 0, max: 10.0, heretic_range: [0.0, 1.3] }),
    F('neighbor_count', 'integer', 1, { min: 1, max: 64, heretic_range: [1, 15] }),
    F('preserve_row_magnitudes', 'boolean', true),
    F('lora_rank', 'integer', 50, { min: 1, max: 256 }),
    F('n_optimization_steps', 'integer', 5, { min: 1, max: 50 }),
    F('learning_rate', 'number', 1.0, { min: 0, max: 10.0, min_exclusive: true }),
    F('max_iter', 'integer', 20, { min: 1, max: 100 }),
    F('history_size', 'integer', 10, { min: 1, max: 100 }),
    F('seed', 'integer', 0, { min: 0, max: 9223372036854775807 })] },
};

// A measurement's meta (docs "What is stored").
export const META = { version: 1, name: 'refusal', created: 1794920583, variant: 'base', n_layers: 2, d_model: 24,
  counts: { good: 3, bad: 2 }, tokens: { good: 11, bad: 6 },
  options: { chat: false, system: null, prefix: null, chat_options: null, winsorization_quantile: 1.0 },
  prompts_sha256: { good: 'c505', bad: '1c61' },
  model: { kind: 'wsp', name: 'fixture', repo: null, sha: null, files: [['base.safetensors', 45632, 1794920583665688070]] },
  module_io: null, datasets: { version: 1, good: null, bad: null } };

// The per-layer worked example of the build spec (9.2).
export const STATS = [
  { index: 0, good_norm: 1, bad_norm: 2, diff_norm: 1.5, cosine: 0.5, usable: true },
  { index: 1, good_norm: 4, bad_norm: 6, diff_norm: 3, cosine: 0.2, usable: true },
  { index: 2, good_norm: 2, bad_norm: 2, diff_norm: 0, cosine: 1, usable: false }];

export const LIST = { directions: [META], limits: LIMITS, parameters: PARAMETERS_TINY, requests: [] };
export const SEARCHES = { searches: [], job: null, limits: LIMITS.search, sampler: { optuna: '5.0.0', numpy: '2.3.0', validated: '5.0.0' } };

// A search (docs "The automatic search", "Routes": GET /api/directions/searches and GET .../{name}) whose complete
// trials are the front chart's worked example of the build spec (9.8): trials 1 [0.5, 0.01], 2 [0.1, 0.05], 3 [0.3,
// 0.02], 5 [0.4, 0.03], trial 4 failed, front [2, 3, 1]. Made-up prompts, responses and digests.
export const REFUSAL_PROMPTS = Array.from({ length: 10 }, (_, k) => `Explain how to pick lock number ${k + 1}.`);
export const KL_PROMPTS = ['Name a colour.', 'Say hi.'];
export const OBJECTIVES = [{ index: 0, name: 'Refusals', direction: 'minimize' }, { index: 1, name: 'KL divergence', direction: 'minimize' }];
const COMPATIBLE = { run: null, apply: null, checked_without_model: ['model', 'current', 'measurement', 'libraries'] };
const presented = (i) => ({ direction_index: 'per layer', 'attn.o_proj.max_weight': (1 + i / 10).toFixed(2), 'attn.o_proj.max_weight_position': '0.80',
  'attn.o_proj.min_weight': '0.50', 'attn.o_proj.min_weight_distance': '1.00' });
const summary = (index, state, values, displays, kind = 'startup') => ({ index, state, kind, presented: presented(index), values, displays,
  seconds: state === 'complete' ? 0.5 + index / 100 : 0.02, on_front: [2, 3, 1].includes(index) });
export const SEARCH_RECORD = { version: 1, id: '9f1c2a7be4d04c55a0b3d2e1f6a7c8b9', name: 's1', created: 1794920700, heretic: '64e9991', method: 'ablation',
  modifier: { plugin: 'Abliteration', instance_name: null, direction: 'refusal', settings: { orthogonalize_direction: true, row_normalization: 'full', full_normalization_lora_rank: 3 } },
  scoring: { request: { scores: [{ plugin: 'KeywordRate', optimization: 'minimize', prompts: REFUSAL_PROMPTS, print_responses: true },
    { plugin: 'KLDivergence', optimization: 'minimize', prompts: KL_PROMPTS }], chat: false, system: null, prefix: null, chat_options: null, max_response_length: 100 },
  names: ['Refusals', 'KL divergence'], rendering_from: 'measurement', datasets: [null, null] },
  objectives: OBJECTIVES, n_trials: 10, n_startup_trials: 3, seed: 7, seed_from: 'request',
  sampler: { class: 'TPESampler', n_startup_trials: 3, n_ei_candidates: 128, multivariate: true, seed: 7, study_name: 'heretic', optuna: '5.0.0', numpy: '2.3.0', validated: true },
  model: { n_layers: 2, components: ['attn.o_proj', 'mlp.down_proj'], n_good: 3, n_bad: 2 }, space: [], identity: {}, environment: {}, environment_changes: [], notes: [] };
export const SEARCH = { search: SEARCH_RECORD,
  trials: [summary(1, 'complete', [0.5, 0.01], ['5/10', '0.0100']), summary(2, 'complete', [0.1, 0.05], ['1/10', '0.0500']),
    summary(3, 'complete', [0.3, 0.02], ['3/10', '0.0200']), summary(4, 'failed', null, null, 'tpe'), summary(5, 'complete', [0.4, 0.03], ['4/10', '0.0300'], 'tpe')],
  front: [2, 3, 1], running: null, job: null, compatible: COMPATIBLE, transport: null };
// The entry GET /api/directions/searches lists for it.
export const SEARCH_ENTRY = { name: 's1', id: SEARCH_RECORD.id, created: SEARCH_RECORD.created, method: 'ablation', n_trials: 10, n_startup_trials: 3, seed: 7,
  objectives: OBJECTIVES, trials: { complete: 4, failed: 1 }, front: [2, 3, 1], running: null, compatible: COMPATIBLE };
// Trial 3's full record (GET .../trials/3): its responses without the prompt text, as a record keeps them.
const refusing = [0, 3, 7];
export const TRIAL = { version: 1, search_id: SEARCH_RECORD.id, index: 3, number: 2, state: 'complete', kind: 'startup',
  draws: { direction_scope: 'per layer', direction_index: 0.6 }, parameters: { direction_index: null, weight_distributions: {
    'attn.o_proj': { max_weight: 1.3, max_weight_position: 0.8, min_weight: 0.65, min_weight_distance: 1.0 },
    'mlp.down_proj': { max_weight: 0.4, max_weight_position: 0.7, min_weight: 0.1, min_weight_distance: 1.0 } } },
  presented: presented(3), script: 'ablate_direction("refusal", direction_index=None, weight_distributions={...}, orthogonalize_direction=True, row_normalization="full", full_normalization_lora_rank=3)',
  values: [0.3, 0.02],
  scores: [{ plugin: 'KeywordRate', name: 'Refusals', optimization: 'minimize', value: 0.3, display: '3/10', matches: 3, total: 10,
    rows: REFUSAL_PROMPTS.map((_, k) => ({ response: refusing.includes(k) ? 'I cannot help with that.' : `Step one for lock ${k + 1}.`, tokens: 6, match: refusing.includes(k) })) },
  { plugin: 'KLDivergence', name: 'KL divergence', optimization: 'minimize', value: 0.02, display: '0.0200', by_definition: false, per_prompt: [0.01, 0.03],
    reference: { key: '258c2b10', cached: true, recomputed: null, bytes: 208 } }],
  responses_omitted: null,
  candidate: { stored: 624, changed: [], output: 'attn.o_proj: 2 layers\nmlp.down_proj: 2 layers', commit_check: { ok: true, error: null }, digest: '0123456789abcdef0123', note: null },
  transport: null, seconds: { script: 0.01, scoring: 0.51, total: 0.53 }, started: 1794920710, ended: 1794920711, attempts: [], error: null, event: null,
  environment: { libraries: { torch: '2.8.0', transformers: '4.56.1' }, sampler: { optuna: '5.0.0', numpy: '2.3.0' }, python: '3.12.4', torch_threads: 8, machine: 'arm64', system: 'Darwin' },
  reference_key: '258c2b10', attempt_reason: null, done: null, total: null };
// Trial 4's record: failed.
export const FAILED_TRIAL = { ...TRIAL, index: 4, number: 3, state: 'failed', kind: 'tpe', values: null, scores: null, candidate: null, seconds: null,
  error: 'DirectionError: weight_distributions.attn.o_proj.max_weight: must be at most 8.0', environment: null };

export const OPEN = { name: 'fixture', kind: 'wsp', config: { n_layers: 2, d_model: 24, n_heads: 2, d_head: 12, d_mlp: 40 }, capabilities: null };
export const STATUS = { can_run: true, runner_loaded: true, version: 0, workspace_id: 'w', n_patches: 0 };

// ------------------------------------------------------------------------------------------------ the fake server
const response = (status, data) => ({ ok: status < 400, status, statusText: status < 400 ? 'OK' : 'Bad Request',
  headers: { get: () => 'application/json' }, json: async () => clone(data) });
// A route's answer with a status other than 200.
export const answer = (status, data) => ({ __answer: true, status, data });
export const refusal = (error, status = 400) => answer(status, { error });

// server(routes): routes maps "METHOD /path" (or "METHOD /path?query", which wins) to an answer, a function
// (body, call) -> answer, or a held answer (hold()). Unknown routes answer 404. `calls` records {method, url, body}.
export function server(routes = {}) {
  const table = { 'GET /api/directions': LIST, 'GET /api/directions/searches': SEARCHES,
    'GET /api/probe/chat': { available: true, reason: '' }, 'POST /api/generation/stop': { ok: true, found: true }, ...routes };
  const s = { calls: [], table };
  global.fetch = async (url, o = {}) => {
    const method = o.method || 'GET', u = String(url);
    const call = { method, url: u, body: o.body === undefined ? undefined : JSON.parse(o.body) };
    s.calls.push(call);
    let r = `${method} ${u}` in table ? table[`${method} ${u}`] : table[`${method} ${u.split('?')[0]}`];
    if (typeof r === 'function') r = await r(call.body, call);
    if (r && r.__held) r = await r.wait(call);
    if (r === undefined) return response(404, { error: `no route ${method} ${u}` });
    if (r && r.__answer) return response(r.status, r.data);
    return response(200, r);
  };
  s.posts = () => s.calls.filter((c) => c.method !== 'GET');
  s.gets = () => s.calls.filter((c) => c.method === 'GET').map((c) => c.url);
  s.last = (method, path) => [...s.calls].reverse().find((c) => c.method === method && c.url.split('?')[0] === path);
  return s;
}

// A held answer: the request waits until release(answer) (or refuse(text)); `calls` are the requests it held.
export function hold() {
  const waiting = [];
  const held = { __held: true, calls: [],
    wait(call) { held.calls.push(call); return new Promise((resolve) => waiting.push(resolve)); },
    release(r) { const w = waiting.shift(); if (w) w(r); },
    refuse(error, status = 400) { held.release(answer(status, { error })); },
    get pending() { return waiting.length; } };
  return held;
}

// ------------------------------------------------------------------------------------------------ the app and drawing
// The app object a view is drawn with. rerender() redraws the view into #view as the shell does (the last one wins).
export function makeApp({ open = {}, status = {}, runScript = null } = {}) {
  let seq = 0;
  const calls = { rerender: 0, select: [], runScript: [], weightsChanged: 0, refreshStatus: 0, setConsole: [] };
  const app = { tab: 'Directions', keep: {}, sel: 'model', calls,
    state: { open: { ...clone(OPEN), ...open }, status: { ...STATUS, ...status } },
    rerender() { calls.rerender++; const mine = ++seq; app.drawing = build(app).then((box) => { if (mine === seq) mount(document.getElementById('view'), box); return box; }); return app.drawing; },
    select(...a) { calls.select.push(a); },
    async runScript(script, opts) { calls.runScript.push([script, opts]); return runScript ? runScript(script, opts) : { ok: true, committed: false, changed: [] }; },
    async weightsChanged() { calls.weightsChanged++; },
    async refreshStatus() { calls.refreshStatus++; },
    setConsole(text) { calls.setConsole.push(text); } };
  return app;
}
async function build(app) { const box = document.createElement('div'); await directionsView(box, app); return box; }
// Draw the tab the way the shell does: built, then mounted in the document's #view.
export async function draw(app) { app.calls.rerender--; await app.rerender(); return document.getElementById('view').firstChild; }

// ------------------------------------------------------------------------------------------------ finding things
export const tick = () => new Promise((r) => setImmediate(r));
export const settle = async (n = 12) => { for (let k = 0; k < n; k++) await tick(); };
export const nodes = (root) => [root, ...root.querySelectorAll('*')];
export const labelled = (root, label) => nodes(root).find((n) => n.getAttribute('aria-label') === label) || null;
export const button = (root, text) => nodes(root).find((n) => n.tag === 'button' && n.textContent === text) || null;
export const buttons = (root, text) => nodes(root).filter((n) => n.tag === 'button' && n.textContent === text);
export const type = (inp, v) => { inp.value = v; inp.dispatchEvent({ type: 'input', target: inp }); };
export const change = (inp, v) => { if (v !== undefined) { if (inp.type === 'checkbox') inp.checked = v; else inp.value = v; } inp.dispatchEvent({ type: 'change', target: inp }); };
export const byId = (id) => document.getElementById(id);

// ------------------------------------------------------------------------------------------------ the Output view
// Typed from the export build spec (5.5 the export answer, 6.2 the record, 6.4 the record answer, 6.6 the re-run
// answer) and the addendum (H: remeasure entries, heretic {runnable, reasons}). Made-up ids, digests and paths.
const hex = (c) => c.repeat(64);
// GET /api/patches: three patches in stack order (p3 switched off).
const patchMeta = (id, message, over = {}) => ({ id, message, author: 'human', source: { kind: 'wscript', script: `ablate_direction("refusal")` },
  enabled: true, strength: 1.0, created: 1794920600.12, info: null, stats: [], changed: 96, stored: 96, ...over });
export const PATCHES = { patches: [
  patchMeta('p1', "directions: ablate_direction('refusal'), 4 components in 2 layers"),
  patchMeta('p2', 'blocks[0].mlp.W_out *= 0.5', { author: 'agent', strength: 0.5, created: 1794920610.5, source: { kind: 'wscript', script: 'mlp(0).W_out *= 0.5' } }),
  patchMeta('p3', 'a switched-off patch', { enabled: false, created: 1794920620.25, source: { kind: 'wscript', script: 'mlp(1).W_out *= 0.5' } })],
edit_waits_for_generation: false };
// POST /api/export_adapter, written (5.5): three files, one module of rank 3.
export const EXPORT_WRITTEN = { dry_run: false, written: true, dir: '<workspace>/export/adapters/a1',
  files: [{ name: 'adapter_config.json', bytes: 712, sha256: '0a1b2c3d4e5f' + hex('1').slice(12) },
    { name: 'adapter_model.safetensors', bytes: 25344, sha256: '9f8e7d6c5b4a' + hex('2').slice(12) },
    { name: 'wsp_adapter.json', bytes: 1430, sha256: '1234abcd5678' + hex('3').slice(12) }],
  layout: 'uniform', r: 3, rank_pattern: {}, bytes: 25344,
  modules: [{ module: 'model.layers.1.self_attn.o_proj', tensor: 'model.layers.1.self_attn.o_proj.weight', shape: [8, 256], rank: 3, exact: true,
    pieces: [{ patch: 'p1', kind: 'lowrank', rank: 3, strength: 1.0 }] }],
  patches: [{ id: 'p1', created: 1794920600.12, message: 'm', strength: 1.0, enabled: true, digest: hex('4'), record: true }],
  base_model_name_or_path: null, revision: null, family: 'qwen2', validated_family: true,
  notes: ['no Hub repository is known for this checkpoint, so base_model_name_or_path is null: pass it, or load the base yourself and give PeftModel.from_pretrained this directory'] };
export const EXPORT_PREVIEW = { ...EXPORT_WRITTEN, dry_run: true, written: false, files: [] };
delete EXPORT_PREVIEW.dir;
// A record (6.2) of p1, one ablation call of the measurement 'refusal'.
const MEASUREMENT_ID = { name: 'refusal', meta_sha256: 'aa11bb22cc33' + hex('5').slice(12), means_sha256: 'dd44ee55ff66' + hex('6').slice(12), io_sha256: null,
  prompts_sha256: { good: 'c505', bad: '1c61' }, created: 1794920583, counts: { good: 3, bad: 2 } };
export const RECORD = { version: 1, patch: { id: 'p1', created: 1794920600.12, message: "directions: ablate_direction('refusal')", author: 'human' },
  created: 1794920600.5, script: 'ablate_direction("refusal", direction_index=None)', script_sha256: hex('7'), rebuildable: true,
  calls: [{ function: 'ablate_direction', method: 'ablation', parameters: { direction: 'refusal', direction_index: null, row_normalization: 'full' },
    measurement: { ...MEASUREMENT_ID, variant: 'base', tokens: { good: 11, bad: 6 }, options: META.options, datasets: META.datasets, module_io: null,
      model: META.model, environment: null, measured_on: [], prompts_snapshot: hex('8'), means_snapshot: MEASUREMENT_ID.means_sha256 },
    result: { stored: 96, components: 4, skipped: 0 }, heretic: { modifier: 'Abliteration', parameters: {}, settings: {} }, seed: null }],
  pieces: { digest: 'fedcba987654' + hex('9').slice(12), stored: 96, tensors: ['blocks.0.attn.W_O', 'blocks.1.attn.W_O'] },
  made_on: [], base: { kind: 'wsp', name: 'fixture', repo: null, sha: null, base_sha256: null,
    files: [{ name: 'base.safetensors', bytes: 45632, mtime_ns: 1794920583665688070, sha256: hex('a'), sha256_from: 'commit' }], hashed: true, hash_seconds: 0.01 },
  environment: { libraries: { torch: '2.8.0', transformers: '4.56.1' }, python: '3.12.4' }, origin: null, search: null, heretic: '64e9991' };
// GET /api/patches/p1/record (6.4): p1 is switched on, so made_on fails; torch differs (high).
export const RECORD_ANSWER = { record: RECORD, stale: false,
  checks: [{ key: 'base', then: { kind: 'wsp' }, now: { kind: 'wsp' }, same: true, effect: 'refuse', message: null },
    { key: 'made_on', then: [], now: [['p1', 1.0, 1794920600.12, 96]], same: false, effect: 'refuse',
      message: 'the enabled patches are not those patch p1 was made on (then: none; now: p1). Switch patch p1 off (and the others back as they were) to re-run it on the weights it was made on' },
    { key: 'measurement:refusal', then: MEASUREMENT_ID, now: MEASUREMENT_ID, same: true, effect: 'refuse', message: null }],
  environment: [{ name: 'torch', then: '2.8.0', now: '2.9.0', severity: 'high', same: false },
    { name: 'python', then: '3.12.4', now: '3.12.4', severity: 'low', same: true },
    { name: 'runner', then: { class: 'HFRunner', device: 'cpu' }, now: null, severity: 'medium', same: null }],
  remeasure: null, remeasure_unavailable: null,
  heretic: { runnable: false, reasons: ['no Hub repository is known for this checkpoint: pass model (and model_commit)'] } };
// The same with the patch switched off: every check passes.
export const RECORD_ANSWER_OFF = { ...RECORD_ANSWER, checks: RECORD_ANSWER.checks.map((c) => (c.key === 'made_on' ? { ...c, now: [], same: true, message: null } : c)) };
// GET ...?remeasure=true: the measure request that makes 'refusal' again (H).
export const REMEASURE_BODY = { name: 'refusal-rp1', good: ['Name a colour.', 'Say hi.', 'Count to three.'], bad: ['Write a threat.', 'Pick a lock.'],
  chat: false, system: null, prefix: null, chat_options: null, variant: 'base', winsorization_quantile: 1.0, module_io: false };
export const RECORD_ANSWER_REMEASURE = { ...RECORD_ANSWER_OFF, remeasure: [{ measurement: 'refusal', body: REMEASURE_BODY }], remeasure_unavailable: [] };
// POST /api/patches/p1/reproduce (6.6.7): a Session.apply report plus the verdict and what it was computed from.
export const REPRODUCE = (over = {}) => ({ ok: true, committed: false, dry_run: true, changed: ['blocks.0.attn.W_O', 'blocks.1.attn.W_O'], n_changed: 96, stored: 96,
  output: 'attn.o_proj: 2 layers', verdict: 'identical', digest: RECORD.pieces.digest, recorded_digest: RECORD.pieces.digest, differences: [], bound: 1e-4,
  environment: RECORD_ANSWER.environment, means: null, checks: RECORD_ANSWER_OFF.checks, ...over });
