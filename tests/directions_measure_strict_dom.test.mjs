// The Directions tab's Measure, Layers and prefix detection against the strict fake DOM: the request bodies, the job
// with its progress, Stop and refusals, the module-inputs estimate. Expectations from the tab's build spec (6.1, 8,
// 9.1, 10.3, 13) and docs/DIRECTIONS.md ("POST /api/directions/measure"); made-up prompts. No server, model or network.
import test from 'node:test';
import assert from 'node:assert/strict';
import { installStrictDom } from './strict_dom.mjs';
import { LIST, META, STATS, server, hold, refusal, makeApp, draw, settle, labelled, button, nodes, type, change, byId } from './directions_fixtures.mjs';
import { onDirectionsEvent } from '../web/js/directions.js';

const { view } = installStrictDom();
const EMPTY = { ...LIST, directions: [] };
const MEASURE_KEYS = ['name', 'good', 'bad', 'chat', 'system', 'prefix', 'chat_options', 'variant', 'winsorization_quantile', 'module_io', 'stream'];
const sorted = (a) => [...a].sort();
const go = () => button(byId('dirs-measure-form'), 'Measure');
const jobLine = () => byId('dirs-job').querySelector('[role=status]')?.textContent ?? '';
async function typed(app, good = 'abc\nhello\nfox', bad = 'xyz\nqqq') {
  await draw(app);
  type(labelled(view, 'good prompts part 1 texts'), good); change(labelled(view, 'good prompts part 1 texts'));
  type(labelled(view, 'bad prompts part 1 texts'), bad); change(labelled(view, 'bad prompts part 1 texts'));
  await app.rerender();
}

test('M1: the measure body: exactly its keys with chat on, no rendering fields with chat off, a stream id', async () => {
  const measured = hold();
  server({ 'GET /api/directions': EMPTY, 'POST /api/directions/measure': measured });
  const app = makeApp();
  await typed(app);
  go().click(); await settle();
  const body = measured.calls[0].body;
  assert.deepEqual(sorted(Object.keys(body)), sorted(MEASURE_KEYS));
  assert.deepEqual({ ...body, stream: null }, { name: 'refusal', good: ['abc', 'hello', 'fox'], bad: ['xyz', 'qqq'], chat: true,
    system: 'You are a helpful assistant.', prefix: null, chat_options: null, variant: 'base', winsorization_quantile: 1, module_io: true, stream: null });
  assert.match(body.stream, /^dirs-measure-[A-Za-z0-9_-]+$/);
  assert.ok(body.stream.length <= 64);
  measured.release({ stopped: true, done: 0, total: 5 }); await settle();
  // chat off: none of system, prefix, chat_options
  change(labelled(view, 'as a chat message'), false); await settle();
  type(labelled(view, 'chat options'), '{"enable_thinking": false}');
  type(labelled(view, 'winsorization_quantile'), '0.95');
  go().click(); await settle();
  const raw = measured.calls[1].body;
  assert.deepEqual(sorted(Object.keys(raw)), sorted(MEASURE_KEYS.filter((k) => !['system', 'prefix', 'chat_options'].includes(k))));
  assert.equal(raw.chat, false);
  assert.equal(raw.winsorization_quantile, 0.95);
  assert.ok(!('replace' in raw));
  measured.release({ stopped: true, done: 0, total: 5 }); await settle();
  // chat options as typed, when chat is on
  change(labelled(view, 'as a chat message'), true); await settle();
  type(labelled(view, 'response_prefix'), '<think></think>');
  go().click(); await settle();
  assert.deepEqual(measured.calls[2].body.chat_options, { enable_thinking: false });
  assert.equal(measured.calls[2].body.prefix, '<think></think>');
  measured.release({ stopped: true, done: 0, total: 5 }); await settle();
});

test('M2: while it runs every requesting button is disabled but Stop; Stop posts the stream; the stopped answer is said', async () => {
  const measured = hold();
  const s = server({ 'POST /api/directions/measure': measured });
  const app = makeApp();
  await typed(app);
  go().click(); await settle();
  const stream = measured.calls[0].body.stream;
  assert.equal(jobLine(), "Measuring 'refusal-2': starting; this loads the model if it is not loaded.");
  const control = labelled(view, 'Directions view');           // the sub-view control's buttons send nothing
  const requesting = nodes(view).filter((n) => n.tag === 'button' && !control.contains(n) && ['Measure', 'Check', 'Download', 'delete', 'Detect'].includes(n.textContent));
  assert.ok(requesting.length >= 4, requesting.map((b) => b.textContent).join(', '));
  assert.ok(requesting.every((b) => b.disabled), 'disabled: ' + requesting.filter((b) => !b.disabled).map((b) => b.textContent).join(', '));
  onDirectionsEvent({ type: 'directions', workspace: 'fixture', name: 'refusal-2', state: 'measuring', done: 4, total: 11, seconds: 1.2, stream }, app);
  assert.equal(jobLine(), "Measuring 'refusal-2': 4 of 11 prompts (1.2 s)");
  const stop = button(byId('dirs-job'), 'Stop');
  assert.equal(stop.disabled, false);
  stop.click(); await settle();
  assert.deepEqual(s.last('POST', '/api/generation/stop').body, { stream });
  measured.release({ stopped: true, done: 4, total: 11 }); await settle();
  assert.equal(jobLine(), 'Stopped after 4 of 11 prompts; nothing was saved.');
  assert.equal(go().disabled, false);
});

test('M2: a stop that fails says so and can be pressed again', async () => {
  const measured = hold();
  server({ 'POST /api/directions/measure': measured, 'POST /api/generation/stop': refusal('the server is restarting', 503) });
  const app = makeApp();
  await typed(app);
  go().click(); await settle();
  button(byId('dirs-job'), 'Stop').click(); await settle();
  assert.match(byId('dirs-job').textContent, /Stop failed: the server is restarting\. Try again\./);
  assert.equal(button(byId('dirs-job'), 'Stop').disabled, false);
  measured.release({ stopped: true, done: 0, total: 5 }); await settle();
});

test('M3: a measured answer reads the list again, selects the new measurement and shows it in Layers', async () => {
  const s = server({ 'GET /api/directions': EMPTY, 'POST /api/directions/measure': { ...META, seconds: 0.012, stats: STATS },
    'GET /api/directions/refusal': { ...META, stats: STATS } });
  const app = makeApp();
  await typed(app);
  s.table['GET /api/directions'] = LIST;
  go().click(); await settle(20);
  assert.equal(app.keep.dirs.selected, 'refusal');
  assert.equal(app.keep.dirs.view, 'Layers');
  assert.deepEqual([...labelled(view, 'Directions view').children].filter((b) => b.className === 'on').map((b) => b.textContent), ['Layers']);
  assert.ok(byId('dirs-chart-norms'), 'the per-layer view is drawn');
  assert.ok(s.gets().includes('/api/directions/refusal?orthogonalize_direction=true'));
  assert.equal(jobLine(), "Measured 'refusal' in 0.012 s.");
});

test('M3: before Layers is shown, the list is read again and the name goes back to its default', async () => {
  const s = server({ 'GET /api/directions': EMPTY, 'POST /api/directions/measure': { ...META, seconds: 0.012, stats: STATS },
    'GET /api/directions/refusal': { ...META, stats: STATS } });
  const app = makeApp();
  await typed(app);
  const lists = () => s.gets().filter((u) => u === '/api/directions').length;
  const before = lists();
  s.table['GET /api/directions'] = LIST;
  go().click(); await settle(20);
  assert.equal(lists(), before + 1, 'GET /api/directions again');
  assert.equal(app.keep.dirs.selected, 'refusal');
  assert.equal(jobLine(), "Measured 'refusal' in 0.012 s.");
  assert.equal(app.keep.dirs.measure.name, null, 'the name goes back to its default');
  assert.equal(labelled(view, 'measurement').children[1].selected, true, 'the picker shows it');
  button(labelled(view, 'Directions view'), 'Measure').click(); await settle();
  assert.equal(labelled(view, 'measurement name').value, 'refusal-2');
});

// ------------------------------------------------------------------------------------------------ Layers
const attr = (n, k) => n.getAttribute(k);
const byClass = (root, cls) => [...root.querySelectorAll('*')].filter((n) => n.getAttribute('class') === cls);

test('L1: layersChart draws the worked example of 9.2 exactly', async () => {
  const { layersChart, cosineChart } = await import('../web/js/directions_charts.js');
  const norms = layersChart(STATS);
  assert.equal(attr(norms, 'id'), 'dirs-chart-norms');
  assert.equal(attr(norms, 'class'), 'chart');
  assert.equal(attr(byClass(norms, 'dirs-line good')[0], 'points'), '52.0,160.3 340.0,71.3 628.0,130.7');
  assert.equal(attr(byClass(norms, 'dirs-line bad')[0], 'points'), '52.0,130.7 340.0,12.0 628.0,130.7');
  assert.equal(attr(byClass(norms, 'dirs-line diff')[0], 'points'), '52.0,145.5 340.0,101.0 628.0,190.0');
  const unusable = byClass(norms, 'dirs-unusable');
  assert.equal(unusable.length, 1);
  assert.equal(attr(unusable[0], 'data-index'), '2');
  assert.deepEqual(byClass(norms, 'dirs-point').map((c) => [attr(c, 'cx'), attr(c, 'cy'), attr(c, 'data-index')]),
    [['52.0', '145.5', '0'], ['340.0', '101.0', '1'], ['628.0', '190.0', '2']]);
  assert.equal(byClass(norms, 'dirs-marker').length, 0, 'no marker without a direction_index');
  const cos = cosineChart(STATS);
  assert.equal(attr(cos, 'id'), 'dirs-chart-cosine');
  const lines = byClass(cos, 'dirs-line cos');
  assert.equal(lines.length, 1);
  assert.equal(attr(lines[0], 'points'), '52.0,56.5 340.0,83.2 628.0,12.0');
});

test('L2: a cosine that is null breaks the cosine line in two', async () => {
  const { cosineChart } = await import('../web/js/directions_charts.js');
  const stats = STATS.map((s, k) => (k === 1 ? { ...s, cosine: null } : s));
  const lines = byClass(cosineChart(stats), 'dirs-line cos');
  assert.deepEqual(lines.map((l) => attr(l, 'points')), ['52.0,56.5', '628.0,12.0']);
});

test('L3: a click on an index sets the Edit form\'s direction_index to index - 1, says so, and marks it', async () => {
  const s = server({ 'GET /api/directions/refusal': { ...META, stats: STATS } });
  const app = makeApp();
  await draw(app);
  change(labelled(view, 'measurement'), 'refusal'); await settle();
  button(labelled(view, 'Directions view'), 'Layers').click(); await settle();
  const st = app.keep.dirs;
  assert.equal(st.params.ablation.direction_index, '', 'the descriptor default: empty, a direction per layer');
  const point = [...byId('dirs-chart-norms').querySelectorAll('circle')].find((c) => attr(c, 'data-index') === '1');
  point.dispatchEvent({ type: 'click' }); await settle();
  assert.equal(st.params.ablation.direction_index, '0');
  assert.equal(byId('dirs-layers-note').textContent, 'direction_index set to 0 (index 1)');
  const marker = byClass(byId('dirs-chart-norms'), 'dirs-marker');
  assert.equal(marker.length, 1);
  assert.equal(attr(marker[0], 'x1'), '340.0', 'at x(1)');
  assert.equal(attr(marker[0], 'x2'), '340.0');
  [...byId('dirs-chart-norms').querySelectorAll('circle')].find((c) => attr(c, 'data-index') === '0').dispatchEvent({ type: 'click' }); await settle();
  assert.equal(st.params.ablation.direction_index, '-1', 'index 0 gives -1');
  assert.equal(attr(byClass(byId('dirs-chart-norms'), 'dirs-marker')[0], 'x1'), '52.0');
  assert.equal(s.gets().filter((u) => u.startsWith('/api/directions/refusal')).length, 1, 'read once: the picks are drawn from the cache');
  assert.deepEqual(s.posts(), []);
  // the table: an index, the layer it enters (or after the final norm), the numbers with fmt(x, 4)
  const rows = [...byId('dirs-layers-table').children].slice(1).map((r) => [...r.children].map((c) => c.textContent));
  assert.deepEqual(rows, [['0', '0', '1', '2', '1.5', '1', '0.5', 'yes', 'steer -1'], ['1', '1', '4', '6', '3', '0.6', '0.2', 'yes', 'steer -1'], ['2', 'after the final norm', '2', '2', '0', '0', '1', 'no', '–']]);
});

test('L4: unticked orthogonalize_direction in the Edit form: Layers reads usable without the projection', async () => {
  const s = server({ 'GET /api/directions/refusal': { ...META, stats: STATS } });
  const app = makeApp();
  await draw(app);
  change(labelled(view, 'measurement'), 'refusal'); await settle();
  button(labelled(view, 'Directions view'), 'Layers').click(); await settle();
  assert.ok(s.gets().includes('/api/directions/refusal?orthogonalize_direction=true'));
  app.keep.dirs.params.ablation.orthogonalize_direction = false;          // what the Edit view's checkbox stores
  await app.rerender(); await settle();
  assert.ok(s.gets().includes('/api/directions/refusal?orthogonalize_direction=false'));
  assert.match(byId('dirs-sub').textContent, /usable without orthogonalize_direction/);
});

test('L5: delete asks first; confirmed, it sends DELETE and reads the list again', async () => {
  const s = server({ 'DELETE /api/directions/refusal': { ok: true, deleted: 'refusal' } });
  const app = makeApp();
  await draw(app);
  const asked = [];
  global.confirm = (text) => { asked.push(text); return false; };
  button(byId('dirs-list'), 'delete').click(); await settle();
  assert.deepEqual(asked, ["Delete measurement 'refusal'? Searches made from it can no longer run or commit."]);
  assert.deepEqual(s.posts(), [], 'refused: no request');
  const lists = s.gets().filter((u) => u === '/api/directions').length;
  global.confirm = () => true;
  s.table['GET /api/directions'] = EMPTY;
  button(byId('dirs-list'), 'delete').click(); await settle();
  assert.deepEqual(s.posts().map((c) => `${c.method} ${c.url}`), ['DELETE /api/directions/refusal']);
  assert.equal(s.gets().filter((u) => u === '/api/directions').length, lists + 1);
  assert.equal(byId('dirs-list'), null, 'the list is empty now');
});

test('M4: a refusal is shown under Measure without its error type, and the set it names is marked', async () => {
  server({ 'POST /api/directions/measure': refusal('DirectionError: good prompt 1 as a chat message has 18 tokens; it must have 1..16') });
  const app = makeApp();
  await typed(app);
  go().click(); await settle();
  const alerts = [...byId('dirs-measure-form').querySelectorAll('[role=alert]')].map((n) => n.textContent);
  assert.deepEqual(alerts, ['good prompt 1 as a chat message has 18 tokens; it must have 1..16']);
  assert.match(byId('dirs-set-good').className, /\bdirs-bad\b/);
  assert.doesNotMatch(byId('dirs-set-bad').className, /\bdirs-bad\b/);
  assert.match(byId('dirs-set-good').querySelector('[data-part="1"]').className, /\bdirs-bad\b/, 'and the part of prompt 1');
});

test('M4: a refusal naming a form field marks that field', async () => {
  server({ 'POST /api/directions/measure': refusal('DirectionError: winsorization_quantile must be a number from 0 to 1') });
  const app = makeApp();
  await typed(app);
  go().click(); await settle();
  assert.match(labelled(view, 'winsorization_quantile').className, /\bdirs-bad\b/);
});

test('M5: a taken name shows the replace box and holds Measure until it is ticked; then replace is sent', async () => {
  const measured = hold();
  server({ 'POST /api/directions/measure': measured });
  const app = makeApp();
  await typed(app);
  assert.equal(labelled(view, 'measurement name').value, 'refusal-2', 'the first free name');
  assert.equal(labelled(view, 'replace'), null);
  type(labelled(view, 'measurement name'), 'refusal'); change(labelled(view, 'measurement name'));
  const box = labelled(view, 'replace');
  assert.ok(box, 'the replace box');
  assert.match(box.parentElement.textContent, /replace the saved measurement 'refusal'/);
  assert.equal(go().disabled, true);
  assert.equal(go().title, 'the name is taken: tick replace or choose another');
  change(box, true);
  assert.equal(go().disabled, false);
  go().click(); await settle();
  assert.equal(measured.calls[0].body.replace, true);
  assert.equal(measured.calls[0].body.name, 'refusal');
  measured.release({ stopped: true, done: 0, total: 5 }); await settle();
});

test('M6: the module-inputs estimate from the model shape and the prompt counts, unknown until datasets are checked', async () => {
  server({ 'GET /api/directions': EMPTY });
  const app = makeApp();
  await typed(app);
  const box = labelled(view, 'module inputs and outputs');
  // 4 · 5 prompts · 2 layers · ((2 · 12 + 24) + 40 + 24) = 4,480 bytes
  assert.match(box.parentElement.textContent, /about 4\.4 KiB$/);
  assert.equal(box.checked, true);
  button(byId('dirs-set-good'), '+ dataset').click(); await settle();
  const later = labelled(view, 'module inputs and outputs');
  assert.match(later.parentElement.textContent, /unknown until the sets are checked$/);
  assert.equal(later.checked, false);
  button(byId('dirs-set-good').querySelector('[data-part="2"]'), 'remove').click(); await settle();
  type(labelled(view, 'good prompts part 1 texts'), 'one\ntwo');
  assert.match(labelled(view, 'module inputs and outputs').parentElement.textContent, /about 3\.5 KiB$/, 'updated as prompts are typed: 4 · 4 · 2 · 112 = 3,584');
  change(labelled(view, 'module inputs and outputs'), false);
  await app.rerender();
  assert.equal(labelled(view, 'module inputs and outputs').checked, false, 'the person\'s choice stays');
});

test('P10: a measure refused for a good set not in the cache Checks that set at once and shows what a download gets', async () => {
  const s = server({ 'POST /api/directions/measure': refusal('DirectionError: good[0]: mlabonne/harmless_alpaca at 02c6a92cfcf1 is not in the local Hugging Face cache (<cache>), and nothing is downloaded without a request; fetch it with POST /api/directions/datasets (this prompt set and "download": true), which fetches the repository\'s file list, README.md and the files of split train'),
    'POST /api/directions/datasets': { cached: false, would_fetch: [{ field: 'prompts', repository: 'mlabonne/harmless_alpaca', revision: 'main', commit: null, files: null, requests: ['README.md'] }], fetched: [], fetched_bytes: 0, seconds: 0 } });
  const app = makeApp();
  await typed(app, '');
  button(byId('dirs-set-good'), "Heretic's default").click(); await settle();
  go().click(); await settle(20);
  const check = s.last('POST', '/api/directions/datasets');
  assert.ok(check, 'a Check was sent');
  assert.deepEqual(check.body, { prompts: [{ dataset: 'mlabonne/harmless_alpaca', split: 'train[:400]', column: 'text' }], system: 'You are a helpful assistant.' });
  assert.match(byId('dirs-set-good').textContent, /mlabonne\/harmless_alpaca at main \(latest commit unknown\)/);
  assert.match(byId('dirs-measure-form').textContent, /is not in the local Hugging Face cache/);
});

// ------------------------------------------------------------------------------------------------ response prefix
const PREFIX_ANSWER = { response_prefix: '<think></think>\nOkay,', found_by: 'common_prefix_block', block: ['<think>', '</think>'],
  steps: { template: { rendering: 'xxYou are a helpful assistant.zzThis is a dummy prompt.yy', system_role: true },
    common_prefix: { responses: 2, prefix: '<think>' }, recheck: { responses: 2, additional: '\nOkay,' } },
  generation: { runner: 'FunctionalRunner' }, datasets: null, seconds: 0.003, done: 4, total: 4 };
const detectButton = () => button(byId('dirs-prefix'), 'Detect');

test('F1: Detect sends its exact body, shows the prefix JSON-quoted with how it was found, and "Use this prefix" takes it', async () => {
  const detected = hold();
  server({ 'POST /api/directions/prefix': detected });
  const app = makeApp();
  await draw(app);
  type(labelled(view, 'prefix prompts part 1 texts'), 'Say hi.\nCount to three.');
  detectButton().click(); await settle();
  const body = detected.calls[0].body;
  assert.deepEqual(sorted(Object.keys(body)), sorted(['prompts', 'system', 'max_response_length', 'print_responses', 'stream']));
  assert.deepEqual({ ...body, stream: null }, { prompts: ['Say hi.', 'Count to three.'], system: 'You are a helpful assistant.', max_response_length: 100, print_responses: true, stream: null });
  assert.match(body.stream, /^dirs-prefix-[A-Za-z0-9_-]+$/);
  assert.equal(jobLine(), 'Detecting the response prefix: starting; this loads the model if it is not loaded.');
  onDirectionsEvent({ type: 'direction_prefix', workspace: 'fixture', state: 'detecting', seconds: 0.1, phase: 'common_prefix', done: 1, total: 4, stream: body.stream }, app);
  assert.equal(jobLine(), 'Detecting the response prefix: 1 of at most 4 responses');
  assert.equal(detectButton().disabled, true, 'one job at a time');
  detected.release(PREFIX_ANSWER); await settle();
  const result = byId('dirs-prefix-result');
  assert.equal(result.children[0].textContent, 'response_prefix: "<think></think>\\nOkay,"');
  assert.match(result.textContent, /Every response opens a thinking block: the prefix closes it\./);
  assert.match(result.textContent, /common_prefix: 2 responses/);
  button(result, 'Use this prefix').click(); await settle();
  assert.equal(app.keep.dirs.rendering.prefix, '<think></think>\nOkay,');
  assert.equal(labelled(view, 'response_prefix').value, '<think></think>\nOkay,');
  // with chat options and a shorter max_response_length: both sent
  type(labelled(view, 'chat options'), '{"enable_thinking": false}');
  type(labelled(view, 'max_response_length'), '16');
  detectButton().click(); await settle();
  assert.deepEqual(detected.calls[1].body.chat_options, { enable_thinking: false });
  assert.equal(detected.calls[1].body.max_response_length, 16);
  detected.release({ stopped: true, done: 1, total: 4 }); await settle();
  assert.equal(jobLine(), 'Stopped; no prefix was found.');
});

test("F1: Heretic's default test prompts, a refusal of the prompts marks the set, and the measurement's sets fill it", async () => {
  const s = server({ 'POST /api/directions/prefix': refusal('DirectionError: prompts must list 1..1024 prompts'),
    'GET /api/directions/refusal/prompts': { good: ['abc', 'hello', 'fox'], bad: ['xyz', 'qqq'] } });
  const app = makeApp();
  await draw(app);
  button(byId('dirs-set-prefix'), "Heretic's default").click(); await settle();
  assert.equal(labelled(view, 'prefix prompts part 2 split').value, 'train[:100]');
  assert.equal(labelled(view, 'prefix prompts part 3 dataset').value, 'mlabonne/harmful_behaviors');
  detectButton().click(); await settle();
  assert.equal(byId('dirs-prefix').querySelector('p[role=alert]').textContent, 'prompts must list 1..1024 prompts');
  assert.match(byId('dirs-set-prefix').className, /\bdirs-bad\b/);
  assert.equal(button(byId('dirs-set-prefix'), "the measurement's sets").disabled, true, 'no measurement selected');
  change(labelled(view, 'measurement'), 'refusal'); await settle();
  button(byId('dirs-set-prefix'), "the measurement's sets").click(); await settle();
  assert.equal(labelled(view, 'prefix prompts part 1 texts').value, 'abc\nhello\nfox\nxyz\nqqq');
  assert.ok(s.gets().includes('/api/directions/refusal/prompts'));
});

test('F2: without a chat template Detect is disabled and says why', async () => {
  const s = server({ 'GET /api/probe/chat': { available: false, reason: 'bundled .wsp models have no chat template' } });
  const app = makeApp();
  await draw(app);
  assert.equal(detectButton().disabled, true);
  assert.equal(detectButton().title, "it needs the model's own chat template: bundled .wsp models have no chat template");
  detectButton().dispatchEvent({ type: 'click' }); await settle();
  assert.ok(!s.posts().some((c) => c.url === '/api/directions/prefix'));
});
