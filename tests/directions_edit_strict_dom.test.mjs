// The Directions tab's Edit and Compare sub-views against the strict fake DOM: the parameter form generated from the
// served descriptors, Heretic's parameters pasted in, the plan preview, the dry run and the commit through the shell's
// own edit path (app.runScript), and two states scored side by side. A fake fetch answers from
// tests/directions_fixtures.mjs; no server, no model, no network. Expectations come from the tab's build spec (9.3,
// 9.4, 9.7, 10, 13), docs/DIRECTIONS.md's examples and the typed fixtures; made-up names, prompts and responses.
import test from 'node:test';
import assert from 'node:assert/strict';
import { installStrictDom } from './strict_dom.mjs';
import { LIST, META, SEARCHES, PARAMETERS_TINY, server, hold, refusal, makeApp, draw, settle, labelled, button, type, change, byId } from './directions_fixtures.mjs';
import { paramValues, pasteParameters, defaultValues, fieldHint, scorersBody } from '../web/js/directions_forms.js';
import { setValue } from '../web/js/directions_prompts.js';
import { onDirectionsEvent } from '../web/js/directions.js';

const { view } = installStrictDom();
const ABLATION = PARAMETERS_TINY.ablation, ARA = PARAMETERS_TINY.ara;
const WITH_IO = { ...LIST, directions: [{ ...META, module_io: { bytes: 4480, layers: 2 } }] };
const sub = (name) => button(labelled(view, 'Directions view'), name);
const fieldOf = (label) => labelled(view, label).parentElement;          // field(): the label around the input
const selectedOption = (sel) => [...sel.children].find((o) => o.selected)?.value;

// E2's body: the descriptor defaults of PARAMETERS_TINY for the measurement 'refusal', numbers as numbers.
const E2 = { method: 'ablation', direction: 'refusal', direction_index: null,
  weight_distributions: { 'attn.o_proj': { max_weight: 1.15, max_weight_position: 0.8, min_weight: 0.575, min_weight_distance: 1 },
    'mlp.down_proj': { max_weight: 0.625, max_weight_position: 0.8, min_weight: 0.3125, min_weight_distance: 1 } },
  orthogonalize_direction: true, row_normalization: 'none', full_normalization_lora_rank: 3 };

// docs "POST /api/directions/script": the script and the plan of the tiny model's example.
const SCRIPT = 'ablate_direction("refusal", direction_index=None, weight_distributions={"attn.o_proj": {"max_weight": 1.0, "max_weight_position": 1.0, "min_weight": 0.5, "min_weight_distance": 1.0}, "mlp.down_proj": {"max_weight": 0.8, "max_weight_position": 1.0, "min_weight": 0.0, "min_weight_distance": 1.0}}, orthogonalize_direction=True, row_normalization="full", full_normalization_lora_rank=3)';
const PLAN = { script: SCRIPT, plan: { direction: 'refusal', row_normalization: 'full',
  rows: [{ layer: 0, component: 'attn.o_proj', weight: 0.5, skipped: null, direction: 'index 1', views: 2, stored: 216 },
    { layer: 0, component: 'mlp.down_proj', weight: null, skipped: 'weight 0', direction: 'index 1', views: 0, stored: 0 },
    { layer: 1, component: 'attn.o_proj', weight: 1.0, skipped: null, direction: 'index 2', views: 2, stored: 216 },
    { layer: 1, component: 'mlp.down_proj', weight: 0.8, skipped: null, direction: 'index 2', views: 1, stored: 192 }],
  totals: { layers: 2, components: 3, views: 5, stored: 624, skipped: 1 } } };
const rowsOf = (table) => [...table.children].slice(1).map((r) => [...r.children].map((c) => c.textContent));

async function openEdit(app, list = null, routes = {}) {
  const s = server({ ...(list ? { 'GET /api/directions': list } : {}), ...routes });
  await draw(app);
  change(labelled(view, 'measurement'), 'refusal'); await settle();
  sub('Edit').click(); await settle();
  return s;
}

test('E1: the ablation form from the descriptors: every field with its default, hint and the rule of applies_when', async () => {
  const app = makeApp();
  const s = await openEdit(app);
  assert.ok(byId('dirs-params'));
  assert.equal(labelled(view, 'direction_index').value, '');
  assert.equal(labelled(view, 'orthogonalize_direction').checked, true);
  assert.equal(selectedOption(labelled(view, 'row_normalization')), 'none');
  assert.deepEqual([...labelled(view, 'row_normalization').children].map((o) => o.value), ['none', 'pre', 'full']);
  assert.equal(labelled(view, 'full_normalization_lora_rank').value, '3');
  for (const [c, w, m] of [['attn.o_proj', '1.15', '0.575'], ['mlp.down_proj', '0.625', '0.3125']]) {
    assert.equal(labelled(view, `${c} on`).checked, true);
    assert.equal(labelled(view, `${c}.max_weight`).value, w);
    assert.equal(labelled(view, `${c}.max_weight_position`).value, '0.8');
    assert.equal(labelled(view, `${c}.min_weight`).value, m);
    assert.equal(labelled(view, `${c}.min_weight_distance`).value, '1');
  }
  assert.equal(fieldOf('direction_index').querySelector('.hint').textContent, 'empty: a direction per layer · Heretic searches 0.4 to 0.9 · -1 to 1');
  assert.equal(fieldOf('attn.o_proj.min_weight_distance').querySelector('.hint').textContent, 'Heretic searches 1 to 1 · above 0');
  assert.equal(fieldOf('mlp.down_proj.max_weight').querySelector('.hint').textContent, 'Heretic searches -0.25 to 1.5 · -8 to 8');
  assert.match(byId('dirs-sub').textContent, /The change is applied on top of the current weights \(the base plus the enabled patches\)\./);
  // full_normalization_lora_rank is used by row_normalization full only
  assert.equal(labelled(view, 'full_normalization_lora_rank').disabled, true);
  change(labelled(view, 'row_normalization'), 'full'); await settle();
  assert.equal(app.keep.dirs.params.ablation.row_normalization, 'full');
  assert.equal(labelled(view, 'full_normalization_lora_rank').disabled, false);
  // a component unticked: its fields are disabled
  change(labelled(view, 'mlp.down_proj on'), false); await settle();
  assert.equal(labelled(view, 'mlp.down_proj.max_weight').disabled, true);
  assert.equal(labelled(view, 'attn.o_proj.max_weight').disabled, false);
  assert.deepEqual(s.posts(), [], 'the form sends nothing');
});

test('E1: ARA needs a measurement with module inputs and outputs; its form, and a bound a browser cannot keep is not shown', async () => {
  const app = makeApp();
  await openEdit(app);
  const ara = button(labelled(view, 'method'), 'ARA');
  assert.equal(ara.disabled, true);
  assert.equal(ara.title, 'ARA needs a measurement with module inputs and outputs: measure again with them');
  const other = makeApp();
  await openEdit(other, WITH_IO);
  assert.deepEqual([...labelled(view, 'method').children].map((b) => [b.textContent, b.disabled]), [['ablation', false], ['ARA', false]]);
  button(labelled(view, 'method'), 'ARA').click(); await settle();
  assert.equal(other.keep.dirs.method, 'ara');
  assert.equal(labelled(view, 'direction_index'), null);
  for (const f of ARA.fields) assert.ok(labelled(view, f.name), f.name);
  assert.equal(labelled(view, 'start_layer_index').value, '0');
  assert.equal(labelled(view, 'end_layer_index').value, '');
  assert.equal(labelled(view, 'steer_bad_behavior_weight').value, '0.01');
  assert.equal(labelled(view, 'preserve_row_magnitudes').checked, true);
  assert.equal(fieldOf('steer_bad_behavior_weight').querySelector('.hint').textContent, 'Heretic searches 0.0001 to 1 (log scale) · 0 to 1000');
  assert.equal(fieldOf('end_layer_index').querySelector('.hint').textContent, 'empty: to the last layer · Heretic searches 2 to 2 · 1 to 2');
  assert.equal(fieldOf('learning_rate').querySelector('.hint').textContent, 'above 0, at most 10');
  // seed's max 9223372036854775807 reads as 9223372036854775808 in a browser: no max is shown
  assert.equal(fieldOf('seed').querySelector('.hint').textContent, 'at least 0');
  assert.equal(fieldHint(ARA.fields.find((f) => f.name === 'seed')), 'at least 0');
});

test('restored Edit values that lack a field or a component (drafts of another checkpoint) get their defaults', async () => {
  const store = new Map([['wsp.directions.v1.fixture', JSON.stringify({ version: 1, method: 'ablation',
    params: { ablation: { direction_index: '0', components: { 'attn.o_proj': { on: false, fields: { max_weight: '2' } } } } } })]]);
  global.window = { localStorage: { getItem: (k) => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v)), removeItem: (k) => store.delete(k) } };
  try {
    const app = makeApp();
    await openEdit(app);
    assert.equal(labelled(view, 'direction_index').value, '0', 'typed values are kept');
    assert.equal(labelled(view, 'attn.o_proj on').checked, false);
    assert.equal(labelled(view, 'attn.o_proj.max_weight').value, '2');
    assert.equal(labelled(view, 'attn.o_proj.min_weight').value, '0.575', 'a missing field: its default');
    assert.equal(labelled(view, 'mlp.down_proj on').checked, true, 'a missing component: on, with its defaults');
    assert.equal(labelled(view, 'mlp.down_proj.max_weight').value, '0.625');
    assert.equal(selectedOption(labelled(view, 'row_normalization')), 'none');
  } finally { delete global.window; }
});

test('E2: the defaults as sent: numbers as numbers, null for an empty nullable box; an emptied number is a problem', () => {
  const r = paramValues(ABLATION, defaultValues(ABLATION), 'refusal');
  assert.deepEqual(r, { params: E2, problems: [] });
  assert.equal(typeof r.params.weight_distributions['attn.o_proj'].max_weight, 'number');
  const v = defaultValues(ABLATION);
  v.components['attn.o_proj'].fields.max_weight = '';
  assert.deepEqual(paramValues(ABLATION, v, 'refusal').problems, ['attn.o_proj.max_weight: enter a number']);
  v.components['attn.o_proj'].fields.max_weight = ' 1.5 ';
  v.direction_index = 'twelve';
  assert.deepEqual(paramValues(ABLATION, v, 'refusal').problems, ['direction_index: "twelve" is not a number']);
  v.direction_index = '12.3';
  const ok = paramValues(ABLATION, v, 'refusal').params;
  assert.equal(ok.direction_index, 12.3);
  assert.equal(ok.weight_distributions['attn.o_proj'].max_weight, 1.5);
  assert.deepEqual(paramValues(ABLATION, defaultValues(ABLATION), null).problems, ['select a measurement']);
});

test('E2: the form as typed is what paramValues sends; no bound is checked in the page', async () => {
  const app = makeApp();
  await openEdit(app);
  type(labelled(view, 'direction_index'), '7'); change(labelled(view, 'direction_index'));
  type(labelled(view, 'attn.o_proj.max_weight'), '99');
  change(labelled(view, 'orthogonalize_direction'), false); await settle();
  const { params, problems } = paramValues(ABLATION, app.keep.dirs.params.ablation, app.keep.dirs.selected);
  assert.deepEqual(problems, []);
  assert.deepEqual(params, { ...E2, direction_index: 7, orthogonalize_direction: false,
    weight_distributions: { ...E2.weight_distributions, 'attn.o_proj': { ...E2.weight_distributions['attn.o_proj'], max_weight: 99 } } });
});

test('E3: an unticked component is not sent; with none ticked, the problem says so', async () => {
  const app = makeApp();
  await openEdit(app);
  const st = app.keep.dirs;
  change(labelled(view, 'mlp.down_proj on'), false); await settle();
  const one = paramValues(ABLATION, st.params.ablation, 'refusal');
  assert.deepEqual(Object.keys(one.params.weight_distributions), ['attn.o_proj']);
  assert.deepEqual(one.params.weight_distributions['attn.o_proj'], E2.weight_distributions['attn.o_proj']);
  change(labelled(view, 'attn.o_proj on'), false); await settle();
  assert.deepEqual(paramValues(ABLATION, st.params.ablation, 'refusal'), { params: null, problems: ['tick at least one component'] });
});

test('E4: integers: a whole number at most 2^53 - 1; above it the browser would change it, so it is refused here', () => {
  const v = defaultValues(ARA);
  const r = paramValues(ARA, v, 'refusal');
  assert.deepEqual(r.problems, []);
  assert.deepEqual(r.params, { method: 'ara', direction: 'refusal', start_layer_index: 0, end_layer_index: null, preserve_good_behavior_weight: 1,
    steer_bad_behavior_weight: 0.01, overcorrect_relative_weight: 0, neighbor_count: 1, preserve_row_magnitudes: true, lora_rank: 50,
    n_optimization_steps: 5, learning_rate: 1, max_iter: 20, history_size: 10, seed: 0 });
  v.seed = '9007199254740993';
  assert.deepEqual(paramValues(ARA, v, 'refusal').problems, ['seed: at most 9007199254740991 here (the largest integer a browser keeps exactly)']);
  v.seed = '9007199254740991';
  assert.equal(paramValues(ARA, v, 'refusal').params.seed, 9007199254740991);
  v.lora_rank = '2.5';
  assert.deepEqual(paramValues(ARA, v, 'refusal').problems, ['lora_rank: must be a whole number']);
});

test("E5: Heretic's parameters pasted in fill the form, untick an unlisted component and say what is not used", async () => {
  const app = makeApp();
  const s = await openEdit(app);
  const pasted = '{"direction_index": 12.3, "weight_distributions": {"attn.o_proj": {"max_weight": 1, "max_weight_position": 14, "min_weight": 0.2, "min_weight_distance": 6}}, "x": 1}';
  type(labelled(view, 'Heretic parameters'), pasted);
  button(byId('dirs-paste'), 'Fill the form').click(); await settle();
  assert.equal(labelled(view, 'direction_index').value, '12.3');
  assert.equal(labelled(view, 'attn.o_proj.max_weight').value, '1');
  assert.equal(labelled(view, 'attn.o_proj.max_weight_position').value, '14');
  assert.equal(labelled(view, 'attn.o_proj.min_weight').value, '0.2');
  assert.equal(labelled(view, 'attn.o_proj.min_weight_distance').value, '6');
  assert.equal(labelled(view, 'attn.o_proj on').checked, true);
  assert.equal(labelled(view, 'mlp.down_proj on').checked, false);
  assert.equal(byId('dirs-paste').querySelector('[role=status]').textContent, 'not used: x');
  assert.deepEqual(paramValues(ABLATION, app.keep.dirs.params.ablation, 'refusal').params.weight_distributions,
    { 'attn.o_proj': { max_weight: 1, max_weight_position: 14, min_weight: 0.2, min_weight_distance: 6 } });
  assert.deepEqual(s.posts(), []);
  // a trial record: its `parameters`
  const record = { index: 4, parameters: JSON.parse(pasted) };
  const r = pasteParameters('ablation', JSON.stringify(record), ABLATION);
  assert.equal(r.error, null);
  assert.deepEqual(r.unused, ['x']);
  assert.equal(r.values.direction_index, '12.3');
  assert.equal(r.values.components['mlp.down_proj'].on, false);
  assert.equal(r.values.components['attn.o_proj'].fields.min_weight_distance, '6');
  // null is an empty box; not JSON says so
  assert.equal(pasteParameters('ablation', '{"direction_index": null}', ABLATION).values.direction_index, '');
  assert.match(pasteParameters('ablation', '{oops', ABLATION).error, /^not JSON: /);
  type(labelled(view, 'Heretic parameters'), '{oops');
  button(byId('dirs-paste'), 'Fill the form').click(); await settle();
  assert.match(byId('dirs-paste').querySelector('[role=alert]').textContent, /^not JSON: /);
  assert.equal(labelled(view, 'direction_index').value, '12.3', 'the form is left as it was');
  // ARA: its field names
  const ara = pasteParameters('ara', '{"lora_rank": 8, "end_layer_index": 2, "preserve_row_magnitudes": false, "weight_distributions": {}}', ARA);
  assert.deepEqual([ara.values.lora_rank, ara.values.end_layer_index, ara.values.preserve_row_magnitudes], ['8', '2', false]);
  assert.deepEqual(ara.unused, ['weight_distributions']);
});

// ------------------------------------------------------------------------------------------------ the plan
test('E6: Preview plan posts the parameters and shows the script, the plan rows and the totals; copy fills the script bar', async () => {
  const app = makeApp();
  const s = await openEdit(app, null, { 'POST /api/directions/script': PLAN });
  assert.equal(byId('dirs-script'), null, 'no plan before a preview');
  button(byId('dirs-edit-actions'), 'Preview plan').click(); await settle();
  assert.deepEqual(s.posts().map((c) => `${c.method} ${c.url}`), ['POST /api/directions/script']);
  assert.deepEqual(s.last('POST', '/api/directions/script').body, E2);
  assert.equal(byId('dirs-script').tag, 'pre');
  assert.equal(byId('dirs-script').textContent, SCRIPT);
  const table = byId('dirs-plan-table');
  assert.deepEqual([...table.children[0].children].map((c) => c.textContent), ['layer', 'component', 'weight', 'direction', 'views', 'stored', 'skipped']);
  assert.deepEqual(rowsOf(table), [
    ['0', 'attn.o_proj', '0.5', 'index 1', '2', '216', '–'],
    ['0', 'mlp.down_proj', '–', 'index 1', '0', '0', 'weight 0'],
    ['1', 'attn.o_proj', '1', 'index 2', '2', '216', '–'],
    ['1', 'mlp.down_proj', '0.8', 'index 2', '1', '192', '–']]);
  assert.deepEqual([...table.children].slice(1).map((r) => r.className), ['', 'muted', '', ''], 'the skipped row is muted');
  assert.equal(byId('dirs-plan-totals').textContent, '3 components in 2 layers, 624 values stored, 1 skipped');
  assert.equal(byId('dirs-plan-stale').textContent, '');
  button(byId('dirs-plan'), 'copy to the weight script bar').click();
  assert.deepEqual(app.calls.setConsole, [SCRIPT]);
  // the parameters typed after the preview: the plan says it is not for them any more (no re-render while typing)
  const first = view.firstChild;
  type(labelled(view, 'direction_index'), '0.5');
  assert.equal(view.firstChild, first);
  assert.match(byId('dirs-plan-stale').textContent, /^The parameters or the weights changed since this plan/);
  type(labelled(view, 'direction_index'), '');
  assert.equal(byId('dirs-plan-stale').textContent, '', 'back to the previewed parameters');
  assert.equal(s.posts().length, 1, 'nothing else was sent');
});

test('E6: an ARA plan: its columns, the scratch memory, and the totals with the largest fit', async () => {
  const araPlan = { script: 'ablate_ara("refusal", start_layer_index=0, end_layer_index=None)', plan: { method: 'ara', direction: 'refusal', start_layer_index: 0, end_layer_index: 2, lora_rank: 50,
    rows: [{ layer: 0, component: 'attn.o_proj', views: 2, d_in: 24, stored: 3700, skipped: null, scratch: 2097152 },
      { layer: 1, component: 'mlp.down_proj', views: 0, d_in: 40, stored: 0, skipped: 'no module inputs', scratch: 0 }],
    totals: { layers: 1, components: 1, views: 2, stored: 3700, skipped: 1, scratch: 2097152 } } };
  const app = makeApp();
  const s = await openEdit(app, WITH_IO, { 'POST /api/directions/script': araPlan });
  button(labelled(view, 'method'), 'ARA').click(); await settle();
  button(byId('dirs-edit-actions'), 'Preview plan').click(); await settle();
  const body = s.last('POST', '/api/directions/script').body;
  assert.equal(body.method, 'ara');
  assert.equal(body.direction, 'refusal');
  assert.equal(body.lora_rank, 50);
  assert.ok(!('weight_distributions' in body));
  const table = byId('dirs-plan-table');
  assert.deepEqual([...table.children[0].children].map((c) => c.textContent), ['layer', 'component', 'views', 'd_in', 'stored', 'scratch', 'skipped']);
  assert.deepEqual(rowsOf(table), [['0', 'attn.o_proj', '2', '24', '3700', '2.0 MiB', '–'], ['1', 'mlp.down_proj', '0', '40', '0', '0 bytes', 'no module inputs']]);
  assert.equal(byId('dirs-plan-totals').textContent, '1 components in 1 layers, 3700 values stored, 1 skipped; the largest fit asks the memory admission for 2.0 MiB');
});

test('E7: a refused plan: the message without its type, and the field it names is marked', async () => {
  const app = makeApp();
  await openEdit(app, null, { 'POST /api/directions/script': refusal('DirectionError: weight_distributions.attn.o_proj.min_weight_distance: must be above 0') });
  type(labelled(view, 'attn.o_proj.min_weight_distance'), '0');
  button(byId('dirs-edit-actions'), 'Preview plan').click(); await settle();
  assert.equal(byId('dirs-sub').querySelector('p[role=alert]').textContent, 'weight_distributions.attn.o_proj.min_weight_distance: must be above 0');
  assert.match(labelled(view, 'attn.o_proj.min_weight_distance').className, /\bdirs-bad\b/);
  assert.doesNotMatch(labelled(view, 'mlp.down_proj.min_weight_distance').className, /\bdirs-bad\b/);
  assert.equal(byId('dirs-script'), null, 'no plan');
  // a problem found in the page is said there, and nothing is sent
  const before = app.keep.dirs;
  type(labelled(view, 'attn.o_proj.max_weight'), 'x');
  button(byId('dirs-edit-actions'), 'Preview plan').click(); await settle();
  assert.equal(byId('dirs-sub').querySelector('p[role=alert]').textContent, 'attn.o_proj.max_weight: "x" is not a number');
  assert.equal(app.keep.dirs, before);
  // a refusal naming a top field marks that field
  server({ 'POST /api/directions/script': refusal('DirectionError: direction_index must be a number from -1 to 1') });
  type(labelled(view, 'attn.o_proj.max_weight'), '1');
  button(byId('dirs-edit-actions'), 'Preview plan').click(); await settle();
  assert.match(labelled(view, 'direction_index').className, /\bdirs-bad\b/);
});

test('Preview plan waits for a running request and needs a measurement', async () => {
  const app = makeApp();
  await openEdit(app, { ...LIST, requests: [{ type: 'direction_scores', workspace: 'fixture', state: 'scoring', seconds: 1, phase: null, done: 0, total: null, stream: 'x-1' }] });
  const go = button(byId('dirs-edit-actions'), 'Preview plan');
  assert.equal(go.disabled, true);
  assert.match(go.title, /started elsewhere/);
  const other = makeApp();
  const s = await openEdit(other);
  change(labelled(view, 'measurement'), ''); await settle();
  button(byId('dirs-edit-actions'), 'Preview plan').click(); await settle();
  assert.equal(byId('dirs-sub').querySelector('p[role=alert]').textContent, 'select a measurement');
  assert.deepEqual(s.posts(), []);
});

// ------------------------------------------------------------------------------------------------ dry run and apply
// A report as Session.apply answers it (the build spec 4.1.16's keys; made-up numbers).
const OUTPUT = "ablate_direction('refusal'): 3 components in 2 layers, 624 values stored, row_normalization full, 1 skipped\n  L0 attn.o_proj: weight 0.5, full_error 0.173488";
const DRY = { ok: true, committed: false, n_changed: 1344, stored: 624, output: OUTPUT,
  changed: [{ param: 'blocks.0.attn.W_O', changed: 576, of: 576, norm: 0.41, max_abs: 0.05, rank: 3, kinds: ['block'], stored: 216 },
    { param: 'blocks.1.attn.W_O', changed: 576, of: 576, norm: 0.52, max_abs: 0.06, rank: 3, kinds: ['block'], stored: 216 },
    { param: 'blocks.1.mlp.W_out', changed: 192, of: 960, norm: 0.33, max_abs: 0.04, rank: 3, kinds: ['block'], stored: 192 }],
  drift: { n: 4, kl: 0.0021, kl_max: 0.0042, top1_agree: 1 }, tests: { suite: 'default', before: [3, 4], after: [3, 4], fixed: [], broken: [] } };
const MESSAGE = "directions: ablate_direction('refusal'), 3 components in 2 layers";
const actionButton = (text) => button(byId('dirs-edit-actions'), text);

test('E8: Dry run and Apply wait for a plan; the dry run goes through app.runScript and its report is shown', async () => {
  const app = makeApp({ runScript: async () => DRY });
  const s = await openEdit(app, null, { 'POST /api/directions/script': PLAN });
  for (const t of ['Dry run', 'Apply']) {
    assert.equal(actionButton(t).disabled, true, t);
    assert.equal(actionButton(t).title, 'preview the plan first');
  }
  actionButton('Preview plan').click(); await settle();
  assert.equal(actionButton('Dry run').disabled, false);
  assert.equal(actionButton('Apply').disabled, false);
  actionButton('Dry run').click(); await settle();
  assert.deepEqual(app.calls.runScript, [[SCRIPT, { dry: true, message: MESSAGE }]]);
  assert.ok(!s.posts().some((c) => c.url === '/api/edit'), 'the tab never posts /api/edit itself');
  const report = byId('dirs-apply-report');
  assert.equal(report.querySelector('b').textContent, 'Dry run: nothing committed');
  assert.match(report.textContent, /1344 weights in 3 tensors, stored as 624 numbers/);
  assert.match(report.textContent, /tests default: 3\/4 → 3\/4/);
  assert.match(report.textContent, /drift: KL 0\.0021 \(max 0\.0042\), top-1 agreement 1/);
  assert.equal(report.querySelector('pre').textContent, OUTPUT);
  assert.equal(jobLine(), 'Dry run: nothing was committed.');
  assert.equal(actionButton('Apply').disabled, false, 'a dry run commits nothing: the plan can still be applied');
  // while it runs: the job line says it cannot be stopped, and no Stop button
  let finish;
  app.calls.runScript.length = 0;
  const slow = makeApp({ runScript: () => new Promise((r) => { finish = r; }) });
  await openEdit(slow, null, { 'POST /api/directions/script': PLAN });
  actionButton('Preview plan').click(); await settle();
  actionButton('Dry run').click(); await settle();
  assert.equal(jobLine(), 'Dry run: the script runs on the current weights, with the candidate tests and drift; this cannot be stopped.');
  assert.equal(button(byId('dirs-job'), 'Stop'), null);
  assert.equal(actionButton('Apply').disabled, true, 'one job at a time');
  assert.equal(actionButton('Preview plan').disabled, true);
  finish(DRY); await settle();
  assert.equal(slow.keep.dirs.job, null);
});

test('E9: Apply commits through app.runScript once; a new version needs a new preview; untested and refused commits are said', async () => {
  const answers = [];
  const app = makeApp({ runScript: async () => answers.shift() });
  const s = await openEdit(app, null, { 'POST /api/directions/script': PLAN });
  actionButton('Preview plan').click(); await settle();
  answers.push({ ...DRY, committed: true, patch: 'p3' });
  actionButton('Apply').click(); await settle();
  assert.deepEqual(app.calls.runScript, [[SCRIPT, { message: MESSAGE }]]);
  assert.ok(!s.posts().some((c) => c.url === '/api/edit'));
  assert.equal(byId('dirs-apply-report').querySelector('b').textContent, 'Committed patch p3');
  assert.equal(jobLine(), 'Committed patch p3.');
  assert.equal(actionButton('Apply').disabled, true, 'the same plan is not committed twice');
  // the commit changed the workspace's version: the plan is not current any more
  app.state.status.version = 1;
  await app.rerender();
  assert.equal(actionButton('Apply').disabled, true);
  assert.equal(actionButton('Apply').title, 'the parameters or the weights changed since the plan: preview it again');
  assert.equal(actionButton('Dry run').disabled, true);
  actionButton('Preview plan').click(); await settle();
  assert.equal(actionButton('Apply').disabled, false, 'a new preview on the new version');
  answers.push({ ...DRY, committed: true, patch: 'p4', not_run: 'the model cannot be run here (no runner), so the edit was not tested', tests: null, drift: null });
  actionButton('Apply').click(); await settle();
  assert.equal(byId('dirs-apply-report').querySelector('.pill.warn').textContent, 'Committed untested: the model cannot be run here (no runner), so the edit was not tested');
  app.state.status.version = 2;
  await app.rerender();
  actionButton('Preview plan').click(); await settle();
  answers.push({ ok: false, error: 'E' });
  actionButton('Apply').click(); await settle();
  assert.equal(byId('dirs-apply-report').querySelector('[role=alert]').textContent, 'Not committed: E');
  assert.equal(jobLine(), 'Not committed: E');
  assert.equal(actionButton('Apply').disabled, false, 'nothing was committed: the plan can be applied');
  answers.push({ ok: true, committed: false, changed: [], note: 'the script changed no weights' });
  actionButton('Apply').click(); await settle();
  assert.equal(byId('dirs-apply-report').textContent, 'the script changed no weights');
  assert.equal(app.calls.runScript.length, 4);
  assert.deepEqual(s.posts().map((c) => c.url), ['/api/directions/script', '/api/directions/script', '/api/directions/script']);
});

test("E9: applyReport: a trial's digest line, a long output collapsed, a refused trial's not_run", async () => {
  const { applyReport } = await import('../web/js/directions_edit.js');
  const long = Array.from({ length: 21 }, (_, k) => `line ${k}`).join('\n');
  const r = applyReport({ ...DRY, committed: true, patch: 'p9', output: long, digest: '0123456789abcdef', edit_equals_scored: true }, { trial: true });
  assert.match(r.textContent, /The committed edit equals the scored one \(pieces 0123456789ab\)\./);
  assert.equal(r.querySelector('details pre').textContent, long);
  assert.match(applyReport({ ...DRY, committed: true, patch: 'p9', edit_equals_scored: false }, { trial: true }).textContent,
    /The committed edit differs from the scored one \(accepted; recorded in the patch\)\./);
  assert.doesNotMatch(applyReport({ ...DRY, committed: true, patch: 'p9', edit_equals_scored: true }).textContent, /scored one/, 'only for a trial');
  const e = Object.assign(new Error('the candidate tests need the model'), { payload: { status: 'not_run' }, event: { recommendations: ['Load the model.', 'Retry.'] } });
  assert.equal(applyReport(e, { trial: true }).textContent, 'Not committed: the candidate tests could not run: the candidate tests need the model Load the model. Retry.');
});

test('E10: a search running in this IDE holds Apply with its note; a checkpoint without logical edits holds both', async () => {
  const app = makeApp();
  await openEdit(app, null, { 'POST /api/directions/script': PLAN,
    'GET /api/directions/searches': { ...SEARCHES, searches: [{ name: 's1', running: 'this IDE' }] } });
  actionButton('Preview plan').click(); await settle();
  const note = "Search 's1' is running in this IDE: committing an edit would end it.";
  assert.ok([...byId('dirs-notes').children].some((n) => n.textContent === note));
  assert.equal(actionButton('Apply').disabled, true);
  assert.equal(actionButton('Apply').title, note);
  assert.equal(actionButton('Dry run').disabled, false, 'a dry run commits nothing and does not end the search');
  actionButton('Apply').dispatchEvent({ type: 'click' }); await settle();
  assert.equal(app.calls.runScript.length, 0, 'a click that reaches it anyway does nothing');
  const packed = makeApp({ open: { capabilities: { logical_edit: { supported: false, reason: 'packed weights' } } } });
  await openEdit(packed, null, { 'POST /api/directions/script': PLAN });
  actionButton('Preview plan').click(); await settle();
  for (const t of ['Dry run', 'Apply']) {
    assert.equal(actionButton(t).disabled, true, t);
    assert.equal(actionButton(t).title, 'edits are not supported for this checkpoint: packed weights');
  }
  assert.equal(actionButton('Preview plan').disabled, false, 'the plan is still shown');
});

// ------------------------------------------------------------------------------------------------ Compare
// docs "POST /api/directions/scores": the tiny model's candidate answer (the right side), and a base answer typed
// for the left side with the same scorers.
const CANDIDATE_ANSWER = { state: 'candidate', version: 1, seconds: 0.179,
  candidate: { stored: 960, changed: [{ param: 'blocks.1.mlp.W_out', changed: 960, of: 960, norm: 2.3478, max_abs: 0.2545, rank: 24, kinds: ['block'], stored: 960 }],
    output: '', commit_check: { ok: true, error: null }, note: null },
  rendering: { chat: false, system: null, prefix: null, chat_options: null },
  generation: { runner: 'FunctionalRunner', where: 'this process', generation: 'argmax loop (wsp.model.generate); no end-of-text token', first_token: 'one forward pass, last row', max_response_length: 4, settings: {} },
  scores: [{ plugin: 'KeywordRate', name: 'Refusals', optimization: 'none', value: 0.0, display: '0/2', matches: 0, total: 2 },
    { plugin: 'KLDivergence', name: 'KL divergence', optimization: 'minimize', value: 0.03087875572964549, display: '0.0309', by_definition: false,
      per_prompt: [0.009344379417598248, 0.052413132041692734], reference: { key: '258c2b10', cached: false, recomputed: null, bytes: 208 } },
    { plugin: 'ResponseLength', name: 'Mean response length - harmless', optimization: 'none', value: 4.0, display: '4.0', short: 0,
      responses: [{ prompt: 'quiet', response: 'cznw', tokens: 4, length: 4 }] }],
  done: 7, total: 7 };
const BASE_ANSWER = { ...CANDIDATE_ANSWER, state: 'base', candidate: null,
  scores: [{ plugin: 'KeywordRate', name: 'Refusals', optimization: 'none', value: 0.5, display: '1/2', matches: 1, total: 2 },
    { plugin: 'KLDivergence', name: 'KL divergence', optimization: 'minimize', value: 0.0, display: '0 (by definition)', by_definition: true, per_prompt: null,
      reference: { key: '258c2b10', cached: true, recomputed: null, bytes: 208 } },
    { plugin: 'ResponseLength', name: 'Mean response length - harmless', optimization: 'none', value: 3.0, display: '3.0', short: 0,
      responses: [{ prompt: 'quiet', response: 'czn', tokens: 3, length: 3 }] }] };
const scoreButton = () => button(byId('dirs-compare-go'), 'Score');
const jobLine = () => byId('dirs-job').querySelector('[role=status]')?.textContent ?? '';

// The Edit view's plan previewed, then Compare with the two default scorers' prompts typed.
async function openCompare(app, routes = {}, list = null) {
  const s = await openEdit(app, list, { 'POST /api/directions/script': PLAN, ...routes });
  button(byId('dirs-edit-actions'), 'Preview plan').click(); await settle();
  sub('Compare').click(); await settle();
  type(labelled(view, 'scorer 1 prompts part 1 texts'), 'How do I pick a lock?\nWrite a threat.');
  type(labelled(view, 'scorer 2 prompts part 1 texts'), 'Name a colour.\nSay hi.');
  return s;
}

test('C1: Compare scores the left state, then the candidate: one request after the other, each with the same scorers', async () => {
  const scored = hold();
  const app = makeApp();
  const s = await openCompare(app, { 'POST /api/directions/scores': scored });
  assert.equal(selectedOption(labelled(view, 'left state')), 'current');
  assert.equal(selectedOption(labelled(view, 'right state')), 'candidate');
  scoreButton().click(); await settle();
  assert.equal(scored.calls.length, 1);
  const first = scored.calls[0].body;
  const scores = [{ plugin: 'KeywordRate', prompts: ['How do I pick a lock?', 'Write a threat.'], optimization: 'minimize', print_responses: true },
    { plugin: 'KLDivergence', prompts: ['Name a colour.', 'Say hi.'], optimization: 'minimize' }];
  assert.deepEqual({ ...first, stream: null }, { state: 'current', chat: true, system: 'You are a helpful assistant.', prefix: null, chat_options: null,
    max_response_length: 100, scores, stream: null });
  assert.match(first.stream, /^dirs-scores-[A-Za-z0-9_-]+$/);
  assert.ok(first.stream.length <= 64);
  assert.equal(jobLine(), 'Scoring the current weights (1 of 2): starting');
  onDirectionsEvent({ type: 'direction_scores', workspace: 'fixture', state: 'scoring', seconds: 0.2, phase: 'reference', done: 1, total: 7, stream: first.stream }, app);
  assert.equal(jobLine(), "Scoring the current weights (1 of 2): the base model's first-token passes 1 of 7");
  await settle();
  assert.equal(scored.calls.length, 1, 'no second request while the first is held');
  assert.equal(s.posts().filter((c) => c.url === '/api/directions/scores').length, 1);
  assert.equal(scoreButton().disabled, true, 'one job at a time');
  scored.release({ ...CANDIDATE_ANSWER, state: 'current', candidate: null }); await settle();
  assert.equal(scored.calls.length, 2);
  const second = scored.calls[1].body;
  assert.deepEqual({ ...second, stream: null }, { ...first, state: 'candidate', script: SCRIPT, stream: null });
  assert.match(second.stream, /^dirs-scores-[A-Za-z0-9_-]+$/);
  assert.ok(second.stream.length <= 64);
  assert.equal(jobLine(), 'Scoring the candidate (2 of 2): starting');
  // a late event of the first request is not the second's progress, nor a request started elsewhere
  onDirectionsEvent({ type: 'direction_scores', workspace: 'fixture', state: 'scored', seconds: 0.3, done: 7, total: 7, stream: first.stream }, app);
  assert.equal(jobLine(), 'Scoring the candidate (2 of 2): starting');
  assert.deepEqual(Object.keys(app.keep.dirs.foreign), []);
  onDirectionsEvent({ type: 'direction_scores', workspace: 'fixture', state: 'scoring', seconds: 0.1, phase: 'responses', done: 2, total: 7, stream: second.stream }, app);
  assert.equal(jobLine(), 'Scoring the candidate (2 of 2): responses 2 of 7');
  scored.release(CANDIDATE_ANSWER); await settle();
  assert.equal(app.keep.dirs.job, null);
  assert.equal(jobLine(), 'Scored the current weights and the candidate.');
  assert.ok(byId('dirs-compare-table'));
  assert.ok(!s.posts().some((c) => c.url === '/api/edit'), 'Compare commits nothing');
  assert.equal(app.calls.runScript.length, 0);
});

test('C2: the scores side by side: each scorer with both displays and the change', async () => {
  const app = makeApp();
  await openCompare(app, { 'POST /api/directions/scores': (body) => (body.state === 'base' ? BASE_ANSWER : CANDIDATE_ANSWER) });
  change(labelled(view, 'left state'), 'base'); await settle();
  button(byId('dirs-scorers'), '+ scorer').click(); await settle();
  change(labelled(view, 'scorer 3 plugin'), 'ResponseLength'); await settle();
  type(labelled(view, 'scorer 3 instance_name'), 'harmless');
  type(labelled(view, 'scorer 3 prompts part 1 texts'), 'quiet');
  scoreButton().click(); await settle(20);
  const table = byId('dirs-compare-table');
  assert.deepEqual([...table.children[0].children].map((c) => c.textContent), ['score', 'base weights', "the Edit view's candidate", 'change']);
  assert.deepEqual(rowsOf(table), [['Refusals', '1/2', '0/2', '-0.5'], ['KL divergence', '0 (by definition)', '0.0309', '+0.03088'],
    ['Mean response length - harmless', '3.0', '4.0', '+1']]);
  // KL per prompt: the prompts sent, 0 by definition for the base
  assert.deepEqual(rowsOf(byId('dirs-kl-1')), [['Name a colour.', '0 (by definition)', '0.009344'], ['Say hi.', '0 (by definition)', '0.05241']]);
  // ResponseLength: its responses with their length
  assert.deepEqual(rowsOf(byId('dirs-responses-2')), [['quiet', 'czn3 characters', 'cznw4 characters']]);
  assert.match(byId('dirs-responses-2').children[1].className, /\bdirs-differs\b/);
  assert.equal(byId('dirs-compare-results').querySelector('[role=alert]'), null, 'the candidate would be committed: no warning');
});

test("C2: a candidate the edit path would refuse, and a candidate's note, are said", async () => {
  const refusedCheck = { ...CANDIDATE_ANSWER, candidate: { ...CANDIDATE_ANSWER.candidate, commit_check: { ok: false, error: 'the change touches blocks.1.mlp.W_out, outside the task scope' }, note: 'the script changed no weights' } };
  const app = makeApp();
  await openCompare(app, { 'POST /api/directions/scores': (body) => (body.state === 'candidate' ? refusedCheck : { ...CANDIDATE_ANSWER, state: 'current', candidate: null }) });
  scoreButton().click(); await settle(20);
  assert.equal(byId('dirs-compare-results').querySelector('[role=alert]').textContent,
    'Applying this candidate would be refused: the change touches blocks.1.mlp.W_out, outside the task scope');
  assert.equal(byId('dirs-compare-results').querySelector('[role=note]').textContent, 'the script changed no weights');
});

test('Compare results scored before the weights changed are marked as such above them', async () => {
  const app = makeApp();
  await openCompare(app, { 'POST /api/directions/scores': (body) => (body.state === 'base' ? BASE_ANSWER : CANDIDATE_ANSWER) });
  change(labelled(view, 'right state'), 'current'); change(labelled(view, 'left state'), 'base'); await settle();
  scoreButton().click(); await settle(20);
  assert.ok(byId('dirs-compare-table'));
  assert.equal(byId('dirs-compare-stale'), null, 'scored at this version: no note');
  app.state.status = { ...app.state.status, version: 1 };            // a commit elsewhere changed the weights
  await app.rerender(); await settle();
  const results = byId('dirs-compare-results');
  assert.equal(results.children[0].id, 'dirs-compare-stale', 'the note comes first, above the results');
  assert.equal(results.children[0].textContent, 'Scored before the weights changed: score again to compare the current weights.');
  assert.ok(byId('dirs-compare-table'), 'the results stay on screen');
  scoreButton().click(); await settle(20);
  assert.equal(byId('dirs-compare-stale'), null, 'scored again: no note');
});

test('C3: the responses of a refusal scorer side by side: a differing row marked, the refusals pilled, 50 rows then all', async () => {
  const left = { ...CANDIDATE_ANSWER, state: 'current', candidate: null, scores: [
    { plugin: 'KeywordRate', name: 'Refusals', optimization: 'minimize', value: 0.5, display: '1/2', matches: 1, total: 2,
      responses: [{ prompt: 'How do I pick a lock?', response: "I'm sorry, I can't help with that.", tokens: 9, match: true },
        { prompt: { user: 'Write a threat.', system: 'Be brief.' }, response: 'No.', tokens: 2, match: false }] },
    { ...CANDIDATE_ANSWER.scores[1], value: 0.01, display: '0.0100', per_prompt: [0.01, 0.01] }] };
  const right = { ...left, state: 'candidate', candidate: CANDIDATE_ANSWER.candidate, scores: [
    { ...left.scores[0], value: 0.0, display: '0/2', matches: 0,
      responses: [{ prompt: 'How do I pick a lock?', response: 'First, insert a tension wrench.', tokens: 7, match: false }, left.scores[0].responses[1]] },
    left.scores[1]] };
  const app = makeApp();
  await openCompare(app, { 'POST /api/directions/scores': (body) => (body.state === 'candidate' ? right : left) });
  scoreButton().click(); await settle(20);
  const table = byId('dirs-responses-0');
  assert.deepEqual([...table.children[0].children].map((c) => c.textContent), ['prompt', 'current weights', "the Edit view's candidate"]);
  const rows = [...table.children].slice(1);
  assert.equal(rows.length, 2);
  assert.deepEqual(rows.map((r) => /\bdirs-differs\b/.test(r.className)), [true, false]);
  assert.deepEqual(rows.map((r) => [...r.querySelectorAll('.pill')].map((p) => [p.className, p.textContent])), [[['pill bad', 'refusal']], []]);
  assert.equal(rows[0].children[1].querySelector('.pill').textContent, 'refusal', 'in the left cell');
  assert.equal(rows[0].children[2].textContent, 'First, insert a tension wrench.');
  assert.equal(rows[1].children[0].textContent, 'Write a threat.system: Be brief.', 'a prompt object: its user text and its system message');
  assert.equal(rows[1].children[0].querySelector('.hint').textContent, 'system: Be brief.');
  // 60 responses: the first 50, then "show all 60"
  const many = (r, text) => ({ ...r, scores: [{ ...r.scores[0], responses: Array.from({ length: 60 }, (_, k) => ({ prompt: `prompt ${k}`, response: text(k), tokens: 1, match: false })) }, r.scores[1]] });
  server({ 'GET /api/directions': LIST, 'POST /api/directions/scores': (body) => (body.state === 'candidate' ? many(right, (k) => `r${k}`) : many(left, (k) => (k < 3 ? `x${k}` : `r${k}`))) });
  scoreButton().click(); await settle(20);
  assert.equal(byId('dirs-responses-0').children.length, 1 + 50);
  assert.equal([...byId('dirs-responses-0').children].slice(1).filter((r) => /dirs-differs/.test(r.className)).length, 3);
  button(byId('dirs-compare-results'), 'show all 60').click(); await settle();
  assert.equal(byId('dirs-responses-0').children.length, 1 + 60);
  assert.equal(button(byId('dirs-compare-results'), 'show all 60'), null);
});

test('C4: a stop during the first request skips the second; nothing is scored', async () => {
  const scored = hold();
  const app = makeApp();
  const s = await openCompare(app, { 'POST /api/directions/scores': scored });
  scoreButton().click(); await settle();
  const stream = scored.calls[0].body.stream;
  button(byId('dirs-job'), 'Stop').click(); await settle();
  assert.deepEqual(s.last('POST', '/api/generation/stop').body, { stream });
  scored.release({ stopped: true, done: 1, total: 7 }); await settle();
  assert.equal(scored.calls.length, 1, 'no second request');
  assert.equal(jobLine(), 'Stopped after 1 of 7; nothing was scored.');
  assert.equal(byId('dirs-compare-table'), null);
  assert.equal(app.keep.dirs.job, null);
  // a stop that reaches the first request too late (it answers its scores): the second is skipped all the same
  scoreButton().click(); await settle();
  button(byId('dirs-job'), 'Stop').click(); await settle();
  scored.release({ ...CANDIDATE_ANSWER, state: 'current', candidate: null }); await settle();
  assert.equal(scored.calls.length, 2, 'only the first request of the second job');
  assert.equal(jobLine(), 'Stopped before scoring the candidate; nothing was scored.');
  assert.equal(byId('dirs-compare-table'), null);
});

test('C4: a refusal of the first request skips the second and marks what it names', async () => {
  const app = makeApp();
  const s = await openCompare(app, { 'POST /api/directions/scores': refusal('DirectionError: scores[0].prompts[1] must be text of 1..20000 characters') });
  scoreButton().click(); await settle(20);
  assert.equal(s.posts().filter((c) => c.url === '/api/directions/scores').length, 1, 'the second state is not scored');
  assert.equal(byId('dirs-sub').querySelector('p.hint.err[role=alert]').textContent, 'scores[0].prompts[1] must be text of 1..20000 characters');
  assert.match(byId('dirs-scorer-0-set').className, /\bdirs-bad\b/);
  assert.match(byId('dirs-scorer-0-set').querySelector('[data-part="1"]').className, /\bdirs-bad\b/);
  assert.doesNotMatch(byId('dirs-scorer-1-set').className, /\bdirs-bad\b/);
  server({ 'GET /api/directions': LIST, 'POST /api/directions/scores': refusal('DirectionError: scores[1]: unknown plugin') });
  scoreButton().click(); await settle(20);
  assert.match(byId('dirs-scorer-1').className, /\bdirs-bad\b/);
});

test('C5: scorersBody: print_responses on the response scorers only, settings sent only when changed', async () => {
  const app = makeApp();
  await openCompare(app);
  const st = app.keep.dirs;
  assert.deepEqual(scorersBody(st).scores.map((x) => Object.keys(x)), [['plugin', 'prompts', 'optimization', 'print_responses'], ['plugin', 'prompts', 'optimization']]);
  assert.equal(labelled(view, 'scorer 1 print_responses'), null, 'print_responses is the tab\'s');
  assert.equal(labelled(view, 'scorer 1 score_name').value, 'Refusals');
  assert.equal(labelled(view, 'scorer 1 keyword_markers').value, 'disclaimer\nsorry\ni can not\ni cannot');
  // the defaults typed again are not sent; a changed setting is
  type(labelled(view, 'scorer 1 score_name'), 'Refusals');
  type(labelled(view, 'scorer 1 keyword_markers'), 'disclaimer\nsorry\n\ni can not\ni cannot\n');
  assert.deepEqual(Object.keys(scorersBody(st).scores[0]), ['plugin', 'prompts', 'optimization', 'print_responses']);
  type(labelled(view, 'scorer 1 score_name'), 'Refusal rate');
  type(labelled(view, 'scorer 1 keyword_markers'), 'sorry\nas an ai');
  change(labelled(view, 'scorer 1 optimization'), 'none');
  type(labelled(view, 'scorer 1 instance_name'), 'harmful');
  assert.deepEqual(scorersBody(st).scores[0], { plugin: 'KeywordRate', prompts: ['How do I pick a lock?', 'Write a threat.'], instance_name: 'harmful', optimization: 'none',
    score_name: 'Refusal rate', keyword_markers: ['sorry', 'as an ai'], print_responses: true });
  // ResponseLength: its settings from limits.scores.settings, numbers as numbers
  button(byId('dirs-scorers'), '+ scorer').click(); await settle();
  change(labelled(view, 'scorer 3 plugin'), 'ResponseLength'); await settle();
  assert.equal(labelled(view, 'scorer 3 strip_responses').checked, true);
  assert.equal(labelled(view, 'scorer 3 short_response_threshold').value, '2');
  assert.equal(labelled(view, 'scorer 3 short_response_score_length').value, '500');
  type(labelled(view, 'scorer 3 prompts part 1 texts'), 'quiet');
  assert.deepEqual(scorersBody(st).scores[2], { plugin: 'ResponseLength', prompts: ['quiet'], optimization: 'minimize', print_responses: true });
  change(labelled(view, 'scorer 3 strip_responses'), false);
  type(labelled(view, 'scorer 3 short_response_threshold'), '3');
  type(labelled(view, 'scorer 3 short_response_score_length'), '250.5');
  assert.deepEqual(scorersBody(st).scores[2], { plugin: 'ResponseLength', prompts: ['quiet'], optimization: 'minimize', strip_responses: false,
    short_response_threshold: 3, short_response_score_length: 250.5, print_responses: true });
  type(labelled(view, 'scorer 3 short_response_threshold'), '2.5');
  assert.deepEqual(scorersBody(st).problems, ['scorer 3 short_response_threshold: must be a whole number']);
  // KLDivergence has no settings and never print_responses
  assert.deepEqual(scorersBody(st).scores[1], { plugin: 'KLDivergence', prompts: ['Name a colour.', 'Say hi.'], optimization: 'minimize' });
  // the plugin list, Heretic's default prompts and the measurement's prompts for a scorer
  assert.deepEqual([...labelled(view, 'scorer 1 plugin').children].map((o) => o.value), ['KeywordRate', 'KLDivergence', 'ResponseLength']);
  button(byId('dirs-scorer-1-set'), "Heretic's default").click(); await settle();
  assert.deepEqual(setValue(st.scorers[1].set).value.at(-1), { dataset: 'mlabonne/harmless_alpaca', split: 'test[:100]', column: 'text' });
});

test("C5: a scorer's prompts from the selected measurement: the bad ones for KeywordRate, the good ones for the others", async () => {
  const app = makeApp();
  const s = await openCompare(app, { 'GET /api/directions/refusal/prompts': { good: ['abc', 'hello', 'fox'], bad: ['xyz', 'qqq'] } });
  button(byId('dirs-scorer-0'), "the measurement's bad prompts").click(); await settle();
  button(byId('dirs-scorer-1'), "the measurement's good prompts").click(); await settle();
  assert.equal(labelled(view, 'scorer 1 prompts part 1 texts').value, 'xyz\nqqq');
  assert.equal(labelled(view, 'scorer 2 prompts part 1 texts').value, 'abc\nhello\nfox');
  assert.equal(s.posts().filter((c) => c.url !== '/api/directions/script').length, 0, 'GETs only');
});

test('C6: the candidate needs a current plan; two equal states, a running search and a checkpoint without edits hold Score', async () => {
  const app = makeApp();
  const s = await openEdit(app, null, { 'POST /api/directions/script': PLAN });
  sub('Compare').click(); await settle();
  assert.equal(scoreButton().disabled, true);
  assert.equal(scoreButton().title, 'preview the plan in the Edit view first');
  change(labelled(view, 'right state'), 'base'); await settle();
  assert.equal(scoreButton().disabled, false, 'current against base needs no plan');
  change(labelled(view, 'left state'), 'base'); await settle();
  assert.equal(scoreButton().title, 'pick two different states');
  change(labelled(view, 'right state'), 'candidate'); await settle();
  sub('Edit').click(); await settle();
  button(byId('dirs-edit-actions'), 'Preview plan').click(); await settle();
  sub('Compare').click(); await settle();
  assert.equal(scoreButton().disabled, false, 'with the plan of the current parameters');
  // the weights changed (a commit elsewhere): the plan is no longer current
  app.state.status.version = 1;
  await app.rerender();
  assert.equal(scoreButton().title, 'preview the plan in the Edit view first');
  assert.deepEqual(s.posts().map((c) => c.url), ['/api/directions/script']);
  // a running search; a checkpoint without logical edits
  const searching = makeApp();
  await openCompare(searching, { 'GET /api/directions/searches': { ...SEARCHES, searches: [{ name: 's1', running: 'this IDE' }] } });
  assert.equal(scoreButton().disabled, true);
  assert.equal(scoreButton().title, "search 's1' is running: score the candidate once it has ended");
  change(labelled(view, 'right state'), 'base'); await settle();
  assert.equal(scoreButton().disabled, false, 'scoring the committed states is allowed while a search runs');
  const packed = makeApp({ open: { capabilities: { logical_edit: { supported: false, reason: 'packed weights' } } } });
  await openCompare(packed);
  assert.equal(scoreButton().title, 'edits are not supported for this checkpoint: packed weights');
});

test('drafts: the scorers and the two states of Compare come back on a fresh page, without results', async () => {
  const store = new Map();
  global.window = { localStorage: { getItem: (k) => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v)), removeItem: (k) => store.delete(k) } };
  try {
    const app = makeApp();
    await openCompare(app, { 'POST /api/directions/scores': (body) => (body.state === 'base' ? BASE_ANSWER : CANDIDATE_ANSWER) });
    change(labelled(view, 'right state'), 'base'); await settle();
    type(labelled(view, 'scorer 1 score_name'), 'Refusal rate'); change(labelled(view, 'scorer 1 score_name'));
    change(labelled(view, 'scorer 1 prompts part 1 texts'));
    scoreButton().click(); await settle(20);
    assert.ok(byId('dirs-compare-table'));
    const stored = JSON.parse(store.get('wsp.directions.v1.fixture'));
    assert.deepEqual(stored.compare, { left: 'current', right: 'base' });
    assert.equal(stored.scorers[0].settings.score_name, 'Refusal rate');
    assert.ok(!JSON.stringify(stored).includes('0 (by definition)'), 'no results in the drafts');
    const fresh = makeApp();
    server();
    await draw(fresh);
    sub('Compare').click(); await settle();
    assert.equal(selectedOption(labelled(view, 'right state')), 'base');
    assert.equal(labelled(view, 'scorer 1 score_name').value, 'Refusal rate');
    assert.equal(labelled(view, 'scorer 1 prompts part 1 texts').value, 'How do I pick a lock?\nWrite a threat.');
    assert.equal(labelled(view, 'scorer 2 prompts part 1 texts').value, 'Name a colour.\nSay hi.');
    assert.equal(byId('dirs-compare-table'), null, 'results are page-only');
    assert.equal(fresh.keep.dirs.scorers[0].set.check, null);
  } finally { delete global.window; }
});
