// The Directions tab's Search sub-view against the strict fake DOM: the workspace's searches, a search created and
// run, its live trials, stop, resume and more trials, the Pareto front as a chart and as Heretic's list, a trial's
// detail beside a baseline, and a trial committed through the trial-apply route (the only route here that writes
// weights). A fake fetch and a fake EventSource stand in for the server; no server, no model, no network.
// Expectations come from the tab's build spec (5.5, 9.5, 9.8, 10.5, 10.6, 13.1), docs/DIRECTIONS.md ("The automatic
// search", "Committing a trial") and the typed fixtures; made-up names, prompts and responses.
import test from 'node:test';
import assert from 'node:assert/strict';
import { installStrictDom } from './strict_dom.mjs';
import { LIST, META, SEARCHES, SEARCH, SEARCH_ENTRY, TRIAL, FAILED_TRIAL, REFUSAL_PROMPTS, server, hold, refusal, makeApp, draw, settle, labelled, button, buttons, type, change, byId, nodes } from './directions_fixtures.mjs';
import { SEARCH_SLOWS } from '../web/js/directions_measure.js';
import { onDirectionsEvent, patchOrigin } from '../web/js/directions.js';
import { frontChart, axisScale } from '../web/js/directions_charts.js';

const { view } = installStrictDom({ shell: true });
const WITH_IO = { ...LIST, directions: [{ ...META, module_io: { bytes: 4480, layers: 2 } }] };
const sub = (name) => button(labelled(view, 'Directions view'), name);
const noteTexts = () => [...byId('dirs-notes').children].map((n) => n.textContent);
const searches = (over = {}) => ({ ...SEARCHES, searches: [SEARCH_ENTRY], ...over });
const rowsOf = (table) => [...table.children].slice(1).map((r) => [...r.children].map((c) => c.textContent));
const go = () => button(byId('dirs-search-go'), 'Create and run');

// The tab drawn with these routes, the measurement 'refusal' selected and the Search view shown.
async function openSearch(app, routes = {}) {
  const s = server({ 'GET /api/directions/searches': searches(), 'GET /api/directions/searches/s1': SEARCH, ...routes });
  await draw(app);
  change(labelled(view, 'measurement'), 'refusal'); await settle();
  sub('Search').click(); await settle();
  return s;
}
// The two default scorers' prompts typed.
function typeScorers() {
  type(labelled(view, 'scorer 1 prompts part 1 texts'), 'How do I pick a lock?\nWrite a threat.');
  type(labelled(view, 'scorer 2 prompts part 1 texts'), 'Name a colour.\nSay hi.');
}
const SCORERS = [{ plugin: 'KeywordRate', prompts: ['How do I pick a lock?', 'Write a threat.'], optimization: 'minimize', print_responses: true },
  { plugin: 'KLDivergence', prompts: ['Name a colour.', 'Say hi.'], optimization: 'minimize' }];

test('the searches of the workspace: name, method, trial counts, front, objectives, where it runs', async () => {
  const app = makeApp();
  const s = await openSearch(app);
  assert.deepEqual(rowsOf(byId('dirs-searches')).map((r) => r.slice(0, 6)),
    [['s1', 'ablation', '4 complete, 1 failed', '3', 'Refusals (minimize), KL divergence (minimize)', '']]);
  assert.equal(byId('dirs-search-form').open, false, 'the form is closed when the workspace has searches');
  // the selected search (the last created): its header from GET /api/directions/searches/s1
  assert.ok(s.gets().includes('/api/directions/searches/s1'));
  assert.equal(byId('dirs-search-head').querySelector('h3').textContent, "s1 · ablation from 'refusal' · seed 7 (request) · Optuna 5.0.0");
  assert.equal(byId('dirs-search-state').textContent, 'Not running: 4 complete, 1 failed of 10 trials.');
  assert.deepEqual(s.posts(), [], 'opening the view only GETs');
});

test('S1: without Optuna: the note names requirements-search.txt; Create and run is disabled; the list and a search stay readable', async () => {
  const app = makeApp();
  const s = await openSearch(app, { 'GET /api/directions/searches': searches({ sampler: { optuna: null, numpy: '2.3.0', validated: '5.0.0' } }) });
  const note = byId('dirs-search-notes').textContent;
  assert.equal(note, 'The automatic search needs Optuna 5: python -m pip install -r requirements-search.txt. Existing searches can be read and their trials committed.');
  typeScorers();
  assert.equal(go().disabled, true);
  assert.match(go().title, /requirements-search\.txt/);
  assert.equal(rowsOf(byId('dirs-searches')).length, 1);
  assert.ok(byId('dirs-search-head').querySelector('h3'), 'the search is read without Optuna');
  assert.deepEqual(s.posts(), []);
});

test('S1: another Optuna than the validated one is said', async () => {
  const app = makeApp();
  await openSearch(app, { 'GET /api/directions/searches': searches({ sampler: { optuna: '5.1.0', numpy: '2.3.0', validated: '5.0.0' } }) });
  assert.equal(byId('dirs-search-notes').textContent,
    "Optuna 5.1.0 is installed; the draws were checked against Heretic's with 5.0.0. A resumed search is refused if its replay does not reproduce its draws.");
});

test('S2: Create and run: the exact body for a measurement with module inputs (ARA), then the search selected and preparing', async () => {
  const name = 'refusal-ara-1';
  const desc = { ...SEARCH, search: { ...SEARCH.search, name, method: 'ara', modifier: { ...SEARCH.search.modifier, plugin: 'ARA' } }, trials: [], front: [],
    job: { running: true, workspace: 'fixture', search: name, state: 'preparing', trial: null, trials_done: 0, n_trials: 100, phase: null, done: 0, total: null, stop: null } };
  const app = makeApp();
  const s = await openSearch(app, { 'GET /api/directions': WITH_IO, 'GET /api/directions/searches': SEARCHES, 'POST /api/directions/searches': desc });
  assert.equal(byId('dirs-search-form').open, true, 'open when the workspace has no search');
  assert.equal(labelled(view, 'name').value, name, 'the default name: <measurement>-<method>-<k>');
  assert.equal([...labelled(view, 'plugin').children].find((o) => o.selected).value, 'ARA');
  typeScorers();
  s.table['GET /api/directions/searches'] = searches({ searches: [{ ...SEARCH_ENTRY, name, method: 'ara', trials: {}, front: [] }], job: desc.job });
  go().click(); await settle(20);
  const sent = s.last('POST', '/api/directions/searches').body;
  assert.deepEqual(Object.keys(sent), ['name', 'modifiers', 'scorers', 'n_trials', 'n_startup_trials', 'seed', 'chat', 'system', 'prefix', 'chat_options', 'max_response_length', 'stream']);
  assert.deepEqual({ ...sent, stream: null }, { name, modifiers: [{ plugin: 'ARA', direction: 'refusal' }], scorers: SCORERS, n_trials: 100, n_startup_trials: 30, seed: null,
    chat: true, system: 'You are a helpful assistant.', prefix: null, chat_options: null, max_response_length: 100, stream: null });
  assert.match(sent.stream, /^dirs-search-[A-Za-z0-9_-]+$/);
  assert.ok(sent.stream.length <= 64);
  assert.equal(app.keep.dirs.search.selected, name);
  assert.equal(byId('dirs-search-state').textContent, 'Preparing: checks and replay');
  assert.ok(noteTexts().includes(`Search '${name}' is running in this IDE: committing an edit would end it.`), noteTexts().join(' | '));
  assert.equal(go().disabled, true, 'one search at a time');
});

test('S2: Abliteration for a measurement without module inputs; a changed setting is sent, defaults are not; a seed is a number', async () => {
  const app = makeApp();
  const s = await openSearch(app, { 'GET /api/directions/searches': SEARCHES, 'POST /api/directions/searches': refusal('SearchError: stop here') });
  assert.equal(labelled(view, 'name').value, 'refusal-ablation-1');
  assert.equal([...labelled(view, 'plugin').children].find((o) => o.selected).value, 'Abliteration');
  assert.equal([...labelled(view, 'plugin').children].find((o) => o.value === 'ARA').disabled, true, 'ARA needs module inputs and outputs');
  assert.equal([...labelled(view, 'setting row_normalization').children].find((o) => o.selected).value, 'full', "the search's own default");
  typeScorers();
  change(labelled(view, 'setting row_normalization'), 'pre');
  type(labelled(view, 'seed'), '7');
  type(labelled(view, 'n_trials'), '4');
  go().click(); await settle(20);
  const sent = s.last('POST', '/api/directions/searches').body;
  assert.deepEqual(sent.modifiers, [{ plugin: 'Abliteration', direction: 'refusal', row_normalization: 'pre' }]);
  assert.equal(sent.name, 'refusal-ablation-1');
  assert.equal(sent.seed, 7);
  assert.equal(sent.n_trials, 4);
  assert.equal(byId('dirs-search-error').textContent, 'stop here');
});

test('S2: an ARA setting changed from its default (lora_rank 8) is sent as a number; the others are not', async () => {
  const app = makeApp();
  const s = await openSearch(app, { 'GET /api/directions': WITH_IO, 'GET /api/directions/searches': SEARCHES, 'POST /api/directions/searches': refusal('SearchError: no') });
  typeScorers();
  assert.equal(labelled(view, 'setting lora_rank').value, '50');
  type(labelled(view, 'setting lora_rank'), '8');
  go().click(); await settle(20);
  assert.deepEqual(s.last('POST', '/api/directions/searches').body.modifiers, [{ plugin: 'ARA', direction: 'refusal', lora_rank: 8 }]);
});

test('S2: a refusal naming a scorer marks it; one naming the settings marks them', async () => {
  const app = makeApp();
  const s = await openSearch(app, { 'GET /api/directions/searches': SEARCHES, 'POST /api/directions/searches': refusal('SearchError: scorers[1].optimization: required') });
  typeScorers();
  go().click(); await settle(20);
  assert.match(byId('dirs-scorer-1').className, /\bdirs-bad\b/);
  s.table['POST /api/directions/searches'] = refusal('SearchError: modifiers[0].lora_rank: must be at least 1');
  go().click(); await settle(20);
  assert.match(byId('dirs-search-settings').className, /\bdirs-bad\b/);
  assert.equal(byId('dirs-search-error').textContent, 'modifiers[0].lora_rank: must be at least 1');
});

test('S8: a search running in another process: said, and no other search can be created', async () => {
  const app = makeApp();
  await openSearch(app, { 'GET /api/directions/searches': searches({ searches: [{ ...SEARCH_ENTRY, running: 'another process' }] }),
    'GET /api/directions/searches/s1': { ...SEARCH, running: 'another process' } });
  assert.equal(byId('dirs-search-state').textContent, "Search 's1' is running in another process; it can be stopped only there.");
  assert.equal(rowsOf(byId('dirs-searches'))[0][5], 'another process');
  typeScorers();
  assert.equal(go().disabled, true);
  assert.equal(go().title, "search 's1' is running in another process: one search runs at a time");
  assert.ok(noteTexts().includes("Search 's1' is running in another process: committing an edit would end it."));
  assert.equal(buttons(byId('dirs-search'), 'Stop now').length, 0, 'it can be stopped only there');
  assert.equal(buttons(byId('dirs-search'), 'Stop after this trial').length, 0);
  assert.equal(byId('dirs-search-actions'), null, 'no Resume, no Run more');
});

// ------------------------------------------------------------------------------------------------ live trials
const JOB = { running: true, workspace: 'fixture', search: 's1', state: 'running', trial: 3, trials_done: 2, n_trials: 10, phase: 'responses', done: 5, total: 64,
  started: 1794920705, reason: null, error: null, event: null, stop: null, stream: 'dirs-search-x' };
const RUNNING = { 'GET /api/directions/searches': searches({ searches: [{ ...SEARCH_ENTRY, running: 'this IDE' }], job: JOB }),
  'GET /api/directions/searches/s1': { ...SEARCH, running: 'this IDE', job: JOB } };
const stateText = () => byId('dirs-search-state').textContent;

test('the trial table: a row per trial with its state, kind, displays, seconds and the front', async () => {
  await openSearch(makeApp());
  assert.deepEqual([...byId('dirs-trials').children[0].children].map((c) => c.textContent), ['Trial', 'State', 'Kind', 'Refusals', 'KL divergence', 'Seconds', 'Front']);
  assert.deepEqual(rowsOf(byId('dirs-trials')), [['1', 'complete', 'startup', '5/10', '0.0100', '0.51', '●'], ['2', 'complete', 'startup', '1/10', '0.0500', '0.52', '●'],
    ['3', 'complete', 'startup', '3/10', '0.0200', '0.53', '●'], ['4', 'failed', 'tpe', '', '', '0.02', ''], ['5', 'complete', 'tpe', '4/10', '0.0300', '0.55', '']]);
  assert.deepEqual([...byId('dirs-trials').children].slice(1).map((r) => r.getAttribute('data-trial')), ['1', '2', '3', '4', '5']);
});

test('S3: a running job: the state line of 13.1 with Stop now and Stop after this trial; its progress repaints the header only', async () => {
  const app = makeApp();
  await openSearch(app, RUNNING);
  assert.equal(stateText(), 'Running trial 3 of 10 (2 finished): responses 5 of 64');
  assert.ok(button(byId('dirs-search-head'), 'Stop now'));
  assert.ok(button(byId('dirs-search-head'), 'Stop after this trial'));
  assert.equal(byId('dirs-search-actions'), null, 'no Resume while it runs');
  const first = view.firstChild, rerenders = app.calls.rerender;
  onDirectionsEvent({ type: 'direction_search', workspace: 'fixture', search: 's1', state: 'running', seconds: 4, trial: 3, trials_done: 2, n_trials: 10, phase: 'logits', done: 9, total: 64 }, app);
  assert.equal(stateText(), 'Running trial 3 of 10 (2 finished): first-token passes 9 of 64');
  onDirectionsEvent({ type: 'direction_search', workspace: 'fixture', search: 's1', state: 'running', seconds: 4.2, trial: 3, trials_done: 2, n_trials: 10, phase: null, done: 0, total: null }, app);
  assert.equal(stateText(), 'Running trial 3 of 10 (2 finished): starting');
  assert.equal(view.firstChild, first);
  assert.equal(app.calls.rerender, rerenders);
});

test('S3: preparing: the KL reference, then checks and replay', async () => {
  const app = makeApp();
  await openSearch(app, RUNNING);
  onDirectionsEvent({ type: 'direction_search', workspace: 'fixture', search: 's1', state: 'preparing', seconds: 0.01, phase: 'reference', done: 1, total: 6 }, app);
  assert.equal(stateText(), 'Preparing: the KL reference 1 of 6');
  onDirectionsEvent({ type: 'direction_search', workspace: 'fixture', search: 's1', state: 'preparing', seconds: 0.02, phase: null, done: 0, total: null }, app);
  assert.equal(stateText(), 'Preparing: checks and replay');
});

test('S4: a trial event reads the search again and repaints its trials, without redrawing the view or taking the focus', async () => {
  const app = makeApp();
  const s = await openSearch(app, RUNNING);
  const box = labelled(view, 'scorer 1 prompts part 1 texts');
  box.focus();
  assert.equal(document.activeElement, box);
  const reads = s.gets().filter((u) => u === '/api/directions/searches/s1').length;
  const sixth = { index: 6, state: 'complete', kind: 'tpe', presented: {}, values: [0.2, 0.04], displays: ['2/10', '0.0400'], seconds: 0.6, on_front: true };
  s.table['GET /api/directions/searches/s1'] = { ...SEARCH, running: 'this IDE', trials: [...SEARCH.trials, sixth], front: [2, 6, 3, 1], job: { ...JOB, state: 'trial', trial: 6, trials_done: 3 } };
  const first = view.firstChild, rerenders = app.calls.rerender;
  onDirectionsEvent({ type: 'direction_search', workspace: 'fixture', search: 's1', state: 'trial', seconds: 5, trial: 6, trial_state: 'complete', values: [0.2, 0.04],
    displays: ['2/10', '0.0400'], front: [2, 6, 3, 1], stream: 'dirs-search-x' }, app);
  await settle();
  assert.equal(s.gets().filter((u) => u === '/api/directions/searches/s1').length, reads + 1);
  assert.equal(rowsOf(byId('dirs-trials')).length, 6);
  assert.deepEqual(rowsOf(byId('dirs-trials'))[5], ['6', 'complete', 'tpe', '2/10', '0.0400', '0.6', '●']);
  assert.equal(stateText(), 'Running: 3 of 10 trials finished.');
  assert.equal(view.firstChild, first, 'the view is not redrawn');
  assert.equal(app.calls.rerender, rerenders);
  assert.equal(document.activeElement, box, 'the field keeps the focus');
  assert.ok(box.isConnected);
  assert.deepEqual(s.posts(), []);
});

test('S5: Stop after this trial sends after_trial and says so; Stop now sends now', async () => {
  const app = makeApp();
  const s = await openSearch(app, { ...RUNNING, 'POST /api/directions/searches/s1/stop': { ok: true, found: true } });
  button(byId('dirs-search-head'), 'Stop after this trial').click(); await settle();
  assert.deepEqual(s.last('POST', '/api/directions/searches/s1/stop').body, { when: 'after_trial' });
  assert.equal(button(byId('dirs-search-head'), 'stopping after this trial').disabled, true);
  button(byId('dirs-search-head'), 'Stop now').click(); await settle();
  assert.deepEqual(s.last('POST', '/api/directions/searches/s1/stop').body, { when: 'now' });
  assert.equal(button(byId('dirs-search-head'), 'Stopping…').disabled, true);
  assert.equal(s.posts().length, 2);
});

test('S5: a stop that fails says so and can be pressed again', async () => {
  const app = makeApp();
  const s = await openSearch(app, { ...RUNNING, 'POST /api/directions/searches/s1/stop': refusal('when must be now or after_trial') });
  button(byId('dirs-search-head'), 'Stop now').click(); await settle();
  assert.equal(byId('dirs-search-head').querySelector('[role=alert]').textContent, 'Stop failed: when must be now or after_trial. Try again.');
  s.table['POST /api/directions/searches/s1/stop'] = { ok: true, found: true };
  button(byId('dirs-search-head'), 'Stop now').click(); await settle();
  assert.equal(byId('dirs-search-head').querySelector('[role=alert]'), null);
});

test('S6: a finished event reads both lists again and redraws; Resume is gone when every trial ran; Run more sends n_additional_trials', async () => {
  const app = makeApp();
  const s = await openSearch(app, { ...RUNNING, 'POST /api/directions/searches/s1/run': { ...SEARCH, job: { ...JOB, state: 'preparing', trials_done: 10, n_trials: 15 } } });
  const done = { ...JOB, running: false, state: 'finished', trials_done: 10 };
  const entry = { ...SEARCH_ENTRY, trials: { complete: 10 }, running: null };
  s.table['GET /api/directions/searches'] = searches({ searches: [entry], job: done });
  s.table['GET /api/directions/searches/s1'] = { ...SEARCH, job: done };
  const lists = () => [s.gets().filter((u) => u === '/api/directions').length, s.gets().filter((u) => u === '/api/directions/searches').length];
  const [a, b] = lists(), rerenders = app.calls.rerender;
  onDirectionsEvent({ type: 'direction_search', workspace: 'fixture', search: 's1', state: 'finished', seconds: 9, reason: null, trials: { complete: 10 }, front: [2, 3, 1] }, app);
  await settle(20);
  assert.deepEqual(lists(), [a + 1, b + 1], 'both lists read again');
  assert.ok(app.calls.rerender > rerenders, 'the view is redrawn');
  assert.equal(stateText(), 'Finished: 10 complete.');
  assert.equal(buttons(byId('dirs-search'), 'Stop now').length, 0);
  assert.ok(!nodes(byId('dirs-search-actions')).some((n) => n.tag === 'button' && /^Resume/.test(n.textContent)), 'all 10 trials ran');
  assert.equal(labelled(view, 'more trials').value, '10');
  type(labelled(view, 'more trials'), '5');
  button(byId('dirs-search-actions'), 'Run more').click(); await settle(20);
  const sent = s.last('POST', '/api/directions/searches/s1/run').body;
  assert.deepEqual(Object.keys(sent), ['n_additional_trials', 'stream']);
  assert.equal(sent.n_additional_trials, 5);
  assert.match(sent.stream, /^dirs-search-[A-Za-z0-9_-]+$/);
});

test('S6: one trial left is said in the singular', async () => {
  const app = makeApp();
  await openSearch(app, { 'GET /api/directions/searches/s1': { ...SEARCH, search: { ...SEARCH.search, n_trials: 6 } } });
  assert.ok(button(byId('dirs-search-actions'), 'Resume (1 trial left)'), 'six trials asked, five finished');
  assert.equal(button(byId('dirs-search-actions'), 'Resume (1 trials left)'), null);
});

test('S6: a stopped search with trials left: Resume (n trials left) sends only a stream; stopped and failed runs are said', async () => {
  const app = makeApp();
  const stopped = { ...JOB, running: false, state: 'stopped', reason: 'stopped after a trial' };
  const s = await openSearch(app, { 'GET /api/directions/searches': searches({ job: stopped }), 'GET /api/directions/searches/s1': { ...SEARCH, job: stopped },
    'POST /api/directions/searches/s1/run': { ...SEARCH, job: { ...JOB, state: 'preparing' } } });
  assert.equal(stateText(), 'Stopped (stopped after a trial).');
  button(byId('dirs-search-actions'), 'Resume (5 trials left)').click(); await settle(20);
  assert.deepEqual(Object.keys(s.last('POST', '/api/directions/searches/s1/run').body), ['stream']);
  const failed = { ...JOB, running: false, state: 'failed', error: 'the model worker ended', event: { message: 'Load it again.' } };
  s.table['GET /api/directions/searches'] = searches({ job: failed });
  s.table['GET /api/directions/searches/s1'] = { ...SEARCH, job: failed };
  onDirectionsEvent({ type: 'direction_search', workspace: 'fixture', search: 's1', state: 'failed', seconds: 1, error: 'the model worker ended', event: { message: 'Load it again.' } }, app);
  assert.equal(app.keep.dirs.searchJob.running, false);
  await settle(20);
  assert.equal(stateText(), 'The run ended: the model worker ended Load it again.');
});

test('S7: only the library versions changed: the box, and once ticked accept_changed_libraries is sent', async () => {
  const app = makeApp();
  const compatible = { run: 'torch changed from 2.8.0 to 2.9.0 since the search was made', apply: null, checked_without_model: [] };
  const s = await openSearch(app, { 'GET /api/directions/searches': searches({ searches: [{ ...SEARCH_ENTRY, compatible }] }),
    'GET /api/directions/searches/s1': { ...SEARCH, compatible }, 'POST /api/directions/searches/s1/run': { ...SEARCH, compatible, job: { ...JOB, state: 'preparing' } } });
  assert.deepEqual([...byId('dirs-searches').children[1].children[0].querySelectorAll('.pill')].map((p) => [p.textContent, p.title]), [['cannot run', compatible.run]]);
  const resume = () => button(byId('dirs-search-actions'), 'Resume (5 trials left)');
  assert.equal(resume().disabled, true, 'not before the box is ticked');
  change(labelled(view, 'continue with the changed library versions'), true); await settle();
  assert.equal(resume().disabled, false);
  resume().click(); await settle(20);
  const sent = s.last('POST', '/api/directions/searches/s1/run').body;
  assert.deepEqual({ ...sent, stream: null }, { accept_changed_libraries: true, stream: null });
});

test('S7: the weights or the measurement changed: the message, no Resume, no Run more, no box', async () => {
  const app = makeApp();
  const compatible = { run: 'M', apply: 'N', checked_without_model: [] };
  await openSearch(app, { 'GET /api/directions/searches': searches({ searches: [{ ...SEARCH_ENTRY, compatible }] }), 'GET /api/directions/searches/s1': { ...SEARCH, compatible } });
  assert.deepEqual([...byId('dirs-searches').children[1].children[0].querySelectorAll('.pill')].map((p) => [p.textContent, p.title]), [['cannot run', 'M'], ['cannot commit', 'N']]);
  assert.equal(byId('dirs-search-actions').textContent, 'This search cannot run here now: M');
  assert.equal(labelled(view, 'continue with the changed library versions'), null);
  assert.equal(labelled(view, 'more trials'), null);
});

// ------------------------------------------------------------------------------------------------ the front
const cls = (n) => n.getAttribute('class');
const trialMarks = (root, tag) => [...root.querySelectorAll(tag)].filter((n) => /^dirs-trial\b/.test(cls(n) || ''));

test('S9: frontChart draws the worked example of 9.8: four circles, three on the front, at the formulas\' coordinates', () => {
  const chart = frontChart(SEARCH);
  assert.equal(chart.getAttribute('id'), 'dirs-front-chart');
  const circles = trialMarks(chart, 'circle');
  assert.equal(circles.length, 4, 'the complete trials; failed trial 4 is not drawn');
  assert.deepEqual(circles.filter((c) => cls(c) === 'dirs-trial front').map((c) => c.getAttribute('data-trial')).sort(), ['1', '2', '3']);
  const at = (i) => circles.find((c) => c.getAttribute('data-trial') === String(i));
  assert.deepEqual([at(2).getAttribute('cx'), at(2).getAttribute('cy'), at(2).getAttribute('r')], ['60.0', '12.0', '5']);
  assert.deepEqual([at(5).getAttribute('cx'), at(5).getAttribute('cy'), at(5).getAttribute('r')], ['333.0', '146.0', '4']);
  assert.equal(cls(at(5)), 'dirs-trial');
  // the front joined in its order: 2, 3, 1 (trial 3: cx 60 + 0.5 * 364, cy 12 + (1 - 0.25) * 268; trial 1: cx 424, cy 280)
  assert.equal(chart.querySelector('polyline').getAttribute('points'), '60.0,12.0 242.0,213.0 424.0,280.0');
  assert.equal(cls(chart.querySelector('polyline')), 'dirs-front-line');
  assert.equal(chart.querySelector('[data-axis="0"]').textContent, 'Refusals (minimize)');
  assert.equal(chart.querySelector('[data-axis="1"]').textContent, 'KL divergence (minimize)');
  // KL values 0.001 and 0.5: 500 times apart, so the y axis is on a log scale
  const wide = { ...SEARCH, trials: [{ ...SEARCH.trials[0], values: [0.5, 0.001] }, { ...SEARCH.trials[1], values: [0.1, 0.5] }], front: [2, 1] };
  const log = frontChart(wide);
  assert.match(log.querySelector('[data-axis="1"]').textContent, /, log scale$/);
  assert.doesNotMatch(log.querySelector('[data-axis="0"]').textContent, /log/);
  const pts = Object.fromEntries(trialMarks(log, 'circle').map((c) => [c.getAttribute('data-trial'), c.getAttribute('cy')]));
  assert.deepEqual(pts, { 1: '280.0', 2: '12.0' });
});

test('S9: axisScale: linear, log from two decades of positive values, the middle when every value is the same', () => {
  const lin = axisScale([0.1, 0.5, NaN, 0.3]);
  assert.deepEqual([lin.lo, lin.hi, lin.log, lin.t(0.3)], [0.1, 0.5, false, 0.49999999999999994]);
  const log = axisScale([0.001, 0.1]);
  assert.deepEqual([log.log, log.t(0.01)], [true, 0.5]);
  assert.equal(axisScale([-1, 1000]).log, false, 'not for a value at or below 0');
  assert.equal(axisScale([2, 2]).t(2), 0.5);
});

test('S10: three objectives: parallel axes, one line per complete trial; one objective: a strip', () => {
  const three = { ...SEARCH, search: { ...SEARCH.search, objectives: [...SEARCH.search.objectives, { index: 2, name: 'Length', direction: 'maximize' }] },
    trials: SEARCH.trials.map((t) => (t.values ? { ...t, values: [...t.values, t.index] } : t)) };
  const chart = frontChart(three);
  const lines = trialMarks(chart, 'polyline');
  assert.equal(lines.length, 4);
  assert.equal(chart.getAttribute('width'), '420', '120 k + 60');
  assert.equal(lines.find((l) => l.getAttribute('data-trial') === '2').getAttribute('points'), '40.0,220.0 160.0,12.0 280.0,168.0');
  assert.equal(lines.filter((l) => cls(l) === 'dirs-trial front').length, 3);
  const one = frontChart({ ...SEARCH, search: { ...SEARCH.search, objectives: [SEARCH.search.objectives[0]] } });
  assert.equal(one.getAttribute('height'), '80');
  assert.ok(trialMarks(one, 'circle').every((c) => c.getAttribute('cy') === '40.0'));
  assert.equal(trialMarks(one, 'circle').find((c) => c.getAttribute('data-trial') === '2').getAttribute('cx'), '60.0');
});

test("S11: the front as Heretic lists it; a click on the list, a point or a row picks the trial", async () => {
  const app = makeApp();
  await openSearch(app);
  const items = [...byId('dirs-front-list').children];
  assert.deepEqual(items.map((li) => li.textContent), ['[Trial   2] Refusals: 1/10, KL divergence: 0.0500',
    '[Trial   3] Refusals: 3/10, KL divergence: 0.0200', '[Trial   1] Refusals: 5/10, KL divergence: 0.0100']);
  items[1].dispatchEvent({ type: 'click' }); await settle();
  assert.deepEqual(app.keep.dirs.search.trial, { search: 's1', index: 3 });
  trialMarks(byId('dirs-front-chart'), 'circle').find((c) => c.getAttribute('data-trial') === '5').dispatchEvent({ type: 'click' }); await settle();
  assert.equal(app.keep.dirs.search.trial.index, 5);
  [...byId('dirs-trials').children].find((r) => r.getAttribute('data-trial') === '1').dispatchEvent({ type: 'click' }); await settle();
  assert.equal(app.keep.dirs.search.trial.index, 1);
  assert.match([...byId('dirs-trials').children][1].className, /\bon\b/);
});

test('S4: a trial event repaints the front too', async () => {
  const app = makeApp();
  const s = await openSearch(app, RUNNING);
  s.table['GET /api/directions/searches/s1'] = { ...SEARCH, running: 'this IDE', front: [2, 3], trials: SEARCH.trials.map((t) => ({ ...t, on_front: [2, 3].includes(t.index) })), job: JOB };
  const first = view.firstChild;
  onDirectionsEvent({ type: 'direction_search', workspace: 'fixture', search: 's1', state: 'trial', seconds: 5, trial: 6, trial_state: 'failed', values: null, displays: null, front: [2, 3] }, app);
  await settle();
  assert.equal(byId('dirs-front-list').children.length, 2);
  assert.equal(trialMarks(byId('dirs-front-chart'), 'circle').filter((c) => cls(c) === 'dirs-trial front').length, 2);
  assert.equal(view.firstChild, first);
});

// ------------------------------------------------------------------------------------------------ a trial, the baseline, committing
const TRIAL_ROUTES = { 'GET /api/directions/searches/s1/trials/3': TRIAL, 'GET /api/directions/searches/s1/trials/4': FAILED_TRIAL };
const pickRow = async (i) => { [...byId('dirs-trials').children].find((r) => r.getAttribute('data-trial') === String(i)).dispatchEvent({ type: 'click' }); await settle(); };
const trialButton = (text) => button(byId('dirs-trial-actions'), text);
// The baseline: the current weights with the search's request (made-up responses: 8 of the 10 refused).
const BASELINE = { state: 'current', version: 0, seconds: 0.4, candidate: null, rendering: { chat: false, system: null, prefix: null, chat_options: null },
  scores: [{ plugin: 'KeywordRate', name: 'Refusals', optimization: 'minimize', value: 0.8, display: '8/10', matches: 8, total: 10,
    responses: REFUSAL_PROMPTS.map((p, k) => ({ prompt: p, response: k < 8 ? 'I cannot help with that.' : `Step one for lock ${k + 1}.`, tokens: 6, match: k < 8 })) },
  { plugin: 'KLDivergence', name: 'KL divergence', optimization: 'minimize', value: 0.0, display: '0 (by definition)', by_definition: true, per_prompt: null }],
  done: 12, total: 12 };

test('S12: a picked trial: its record (a GET), the presented parameters, the scores, the script, the fit output and the environment', async () => {
  const app = makeApp();
  const s = await openSearch(app, TRIAL_ROUTES);
  await pickRow(3);
  assert.ok(s.gets().includes('/api/directions/searches/s1/trials/3'));
  const detail = byId('dirs-trial');
  assert.equal(detail.querySelector('h4').textContent, 'Trial 3 · complete · startup');
  assert.deepEqual(rowsOf(byId('dirs-trial-parameters')), Object.entries(TRIAL.presented));
  assert.equal(byId('dirs-trial-scores').textContent, 'Refusals: 3/10, KL divergence: 0.0200');
  assert.equal(byId('dirs-trial-script').textContent, TRIAL.script);
  assert.equal(detail.querySelector('details pre').textContent, TRIAL.candidate.output);
  assert.match(detail.textContent, /torch 2\.8\.0, 8 threads/);
  assert.match(detail.textContent, /seconds: script 0\.01, scoring 0\.51, total 0\.53/);
  // the responses of the refusal scorer: the search's prompts, this trial's rows; no baseline column yet
  assert.deepEqual([...byId('dirs-trial-responses-0').children[0].children].map((c) => c.textContent), ['prompt', 'this trial']);
  assert.deepEqual(rowsOf(byId('dirs-trial-responses-0'))[0], [REFUSAL_PROMPTS[0], 'I cannot help with that.refusal']);
  assert.equal(byId('dirs-trial-responses-1'), null, 'KL divergence has no responses');
  button(detail, 'copy to the weight script bar').click();
  assert.deepEqual(app.calls.setConsole, [TRIAL.script]);
  assert.match(detail.textContent, /this can take as long as the trial's own fit did/, 'Q4 beside Commit');
  assert.deepEqual(s.posts(), [], 'reading a trial only GETs');
});

test('S12: Score the baseline: the search\'s own request on the current weights; a Baseline row, a hollow mark, the baseline column; stale after a commit', async () => {
  const app = makeApp();
  const s = await openSearch(app, { ...TRIAL_ROUTES, 'POST /api/directions/scores': BASELINE });
  await pickRow(3);
  button(byId('dirs-search-baseline'), 'Score the baseline').click(); await settle(20);
  const sent = s.last('POST', '/api/directions/scores').body;
  assert.deepEqual(Object.keys(sent), ['state', 'scores', 'chat', 'system', 'prefix', 'chat_options', 'max_response_length', 'stream']);
  assert.deepEqual({ ...sent, stream: null }, { state: 'current', ...SEARCH.search.scoring.request, stream: null });
  assert.match(sent.stream, /^dirs-scores-[A-Za-z0-9_-]+$/);
  assert.deepEqual(rowsOf(byId('dirs-trials'))[0], ['Baseline', 'current weights', '', '8/10', '0 (by definition)', '', '']);
  assert.equal([...byId('dirs-front-chart').querySelectorAll('circle')].filter((c) => c.getAttribute('class') === 'dirs-baseline').length, 1);
  assert.deepEqual([...byId('dirs-trial-responses-0').children[0].children].map((c) => c.textContent), ['prompt', 'this trial', 'baseline']);
  const rows = [...byId('dirs-trial-responses-0').children].slice(1);
  assert.deepEqual([...rows[1].children].map((c) => c.textContent), [REFUSAL_PROMPTS[1], 'Step one for lock 2.', 'I cannot help with that.refusal']);
  assert.match(rows[1].className, /\bdirs-differs\b/);
  assert.doesNotMatch(rows[0].className, /dirs-differs/, 'both refused alike');
  app.state.status = { ...app.state.status, version: 1 };
  await app.rerender(); await settle();
  assert.ok(button(byId('dirs-search-baseline'), 'Score the baseline again (the weights changed)'));
  assert.equal(rowsOf(byId('dirs-trials'))[0][0], '1', 'a stale baseline is not drawn');
  assert.deepEqual([...byId('dirs-trial-responses-0').children[0].children].map((c) => c.textContent), ['prompt', 'this trial']);
});

test('S12: a failed trial: its error, the row\'s title, no commit; Use in Edit fills the Edit form with the trial\'s parameters and the settings', async () => {
  const app = makeApp();
  await openSearch(app, TRIAL_ROUTES);
  await pickRow(4);
  assert.equal(byId('dirs-trial').querySelector('[role=alert]').textContent, FAILED_TRIAL.error);
  assert.equal([...byId('dirs-trials').children].find((r) => r.getAttribute('data-trial') === '4').title, FAILED_TRIAL.error);
  assert.equal(trialButton('Commit trial 4'), null);
  assert.equal(trialButton('Check the commit'), null);
  await pickRow(3);
  trialButton('Use in Edit').click(); await settle();
  const st = app.keep.dirs;
  assert.equal(st.view, 'Edit');
  assert.equal(st.method, 'ablation');
  assert.equal(st.selected, 'refusal');
  const p = st.params.ablation;
  assert.deepEqual([p.direction_index, p.orthogonalize_direction, p.row_normalization, p.full_normalization_lora_rank], ['', true, 'full', '3']);
  assert.deepEqual(p.components['attn.o_proj'].fields, { max_weight: '1.3', max_weight_position: '0.8', min_weight: '0.65', min_weight_distance: '1' });
  assert.equal(labelled(view, 'mlp.down_proj.max_weight').value, '0.4');
});

test('S13: Check the commit, a refusal for a different edit and its box, then Commit with accept_different_edit; weightsChanged after it', async () => {
  const differs = { ok: false, error: 'the edit made now differs from the one trial 3 scored (pieces 89ab against 0123). Its scores do not describe this edit', search: 's1', trial: 3,
    digest: '89abcdef0123', scored_digest: '0123456789abcdef0123', edit_equals_scored: false };
  const committed = { ok: true, committed: true, patch: 'p7', n_changed: 3072, stored: 816, changed: [{ param: 'blocks.0.attn.W_O', changed: 3072, of: 3072, norm: 0.5, max_abs: 0.1, rank: 3, kinds: ['lowrank'], stored: 816 }],
    drift: { kl: 0.137, kl_max: 0.4, top1_agree: 0.66, n: 384 }, search: 's1', trial: 3, digest: '89abcdef0123', scored_digest: '0123456789abcdef0123', edit_equals_scored: false, status: { version: 1 } };
  const app = makeApp();
  const s = await openSearch(app, { ...TRIAL_ROUTES, 'POST /api/directions/searches/s1/trials/3/apply': (body) => (body.accept_different_edit ? committed : differs) });
  await pickRow(3);
  assert.equal(labelled(view, 'commit anyway'), null, 'no box before such a refusal');
  trialButton('Check the commit').click(); await settle(20);
  assert.deepEqual(s.last('POST', '/api/directions/searches/s1/trials/3/apply').body, { dry_run: true, enforce_scope: false, accept_different_edit: false, message: null });
  assert.equal(byId('dirs-trial-report').textContent, `Not committed: ${differs.error}`);
  assert.equal(labelled(view, 'commit anyway').checked, false, 'the box appears unticked');
  trialButton('Commit trial 3').click(); await settle(20);
  assert.deepEqual(s.last('POST', '/api/directions/searches/s1/trials/3/apply').body, { dry_run: false, enforce_scope: false, accept_different_edit: false, message: null },
    'not ticked: not accepted');
  assert.equal(app.calls.weightsChanged, 0);
  change(labelled(view, 'commit anyway'), true);
  // the console's own scope box decides enforce_scope
  const scope = document.createElement('input'); scope.id = 'enforce-scope'; scope.type = 'checkbox'; scope.checked = true; byId('console').append(scope);
  try {
    trialButton('Commit trial 3').click(); await settle(20);
  } finally { scope.remove(); }
  assert.deepEqual(s.last('POST', '/api/directions/searches/s1/trials/3/apply').body, { dry_run: false, enforce_scope: true, accept_different_edit: true, message: null });
  assert.equal(app.calls.weightsChanged, 1);
  assert.match(byId('dirs-trial-report').textContent, /^Committed patch p7/);
  assert.match(byId('dirs-trial-report').textContent, /The committed edit differs from the scored one \(accepted; recorded in the patch\)\./);
  assert.equal(byId('dirs-job').textContent, 'Committed trial 3 as patch p7.');
  assert.ok(!s.posts().some((c) => c.url === '/api/edit'), 'the trial-apply route only');
  assert.equal(app.calls.runScript.length, 0);
});

test('S13: picking another trial drops the refusal and its box; an answer that the tests could not run is said', async () => {
  const app = makeApp();
  const differs = { ok: false, error: 'differs', edit_equals_scored: false };
  const s = await openSearch(app, { ...TRIAL_ROUTES, 'GET /api/directions/searches/s1/trials/1': { ...TRIAL, index: 1 },
    'POST /api/directions/searches/s1/trials/3/apply': differs,
    'POST /api/directions/searches/s1/trials/1/apply': { __answer: true, status: 409, data: { error: 'the candidate tests need the model', status: 'not_run', event: { recommendations: ['Load it.'] } } } });
  await pickRow(3);
  trialButton('Check the commit').click(); await settle(20);
  assert.ok(labelled(view, 'commit anyway'));
  await pickRow(1);
  assert.equal(labelled(view, 'commit anyway'), null);
  trialButton('Commit trial 1').click(); await settle(20);
  assert.equal(s.last('POST', '/api/directions/searches/s1/trials/1/apply').body.accept_different_edit, false);
  assert.equal(byId('dirs-trial-report').textContent, 'Not committed: the candidate tests could not run: the candidate tests need the model Load it.');
});

test('S7, S13: Commit is held by a changed workspace, a running search, a checkpoint without edits and a running job', async () => {
  const held = async (routes, appOver = {}) => {
    const app = makeApp(appOver);
    await openSearch(app, { ...TRIAL_ROUTES, ...routes });
    await pickRow(3);
    return [trialButton('Commit trial 3'), trialButton('Check the commit')];
  };
  const compatible = { run: 'M', apply: 'N', checked_without_model: [] };
  let [commit, check] = await held({ 'GET /api/directions/searches': searches({ searches: [{ ...SEARCH_ENTRY, compatible }] }), 'GET /api/directions/searches/s1': { ...SEARCH, compatible } });
  assert.deepEqual([commit.disabled, commit.title, check.disabled], [true, 'N', true]);
  [commit] = await held(RUNNING);
  assert.equal(commit.disabled, true);
  assert.equal(commit.title, "Search 's1' is running in this IDE: committing an edit would end it.");
  [commit] = await held({}, { open: { capabilities: { logical_edit: { supported: false, reason: 'packed weights' } } } });
  assert.deepEqual([commit.disabled, commit.title], [true, 'edits are not supported for this checkpoint: packed weights']);
  [commit] = await held({ 'GET /api/directions': { ...LIST, requests: [{ type: 'direction_scores', workspace: 'fixture', state: 'scoring', done: 1, total: 2, stream: 'x-1' }] } });
  assert.equal(commit.disabled, true);
  assert.match(commit.title, /started elsewhere/);
  [commit] = await held({});
  assert.equal(commit.disabled, false, 'otherwise it can be committed');
});

test('S1: without Optuna a trial is still read and committed; Resume and Run more are held', async () => {
  const app = makeApp();
  const stopped = { ...JOB, running: false, state: 'stopped', reason: 'stopped by the person' };
  await openSearch(app, { ...TRIAL_ROUTES, 'GET /api/directions/searches': searches({ sampler: { optuna: null, numpy: '2.3.0', validated: '5.0.0' }, job: stopped }),
    'GET /api/directions/searches/s1': { ...SEARCH, job: stopped } });
  for (const t of ['Resume (5 trials left)', 'Run more']) {
    assert.equal(button(byId('dirs-search-actions'), t).disabled, true, t);
    assert.match(button(byId('dirs-search-actions'), t).title, /requirements-search\.txt/);
  }
  await pickRow(3);
  assert.equal(byId('dirs-trial').querySelector('h4').textContent, 'Trial 3 · complete · startup');
  assert.equal(trialButton('Commit trial 3').disabled, false);
});

test('O1: patchOrigin: the search, the trial and its displays; the warning for a different edit; nothing for another patch', () => {
  const origin = { kind: 'direction_search', search: 's1', search_id: SEARCH.search.id, trial: 4, method: 'ablation', values: [0.09375, 0.0427], displays: ['3/32', '0.0427'],
    scored_digest: 'aa', digest: 'aa', edit_equals_scored: true };
  const line = patchOrigin({ source: { kind: 'wscript', script: 'x', origin } });
  assert.equal(line.textContent, 'from search s1, trial 4: 3/32, 0.0427');
  assert.equal(line.querySelector('.pill'), null);
  const differs = patchOrigin({ source: { kind: 'wscript', script: 'x', origin: { ...origin, edit_equals_scored: false } } });
  assert.equal(differs.textContent, 'from search s1, trial 4: 3/32, 0.0427 · the committed edit differs from the scored one');
  assert.equal(differs.querySelector('span').className, 'pill warn');
  assert.equal(patchOrigin({ source: { kind: 'wscript', script: 'blocks[0].mlp.W_out *= 0.5' } }), null);
  assert.equal(patchOrigin({ source: { kind: 'wscript', origin: { kind: 'other' } } }), null);
});

test('S15: a reload while a search runs: the running header and Stop at once, from the job of the searches answer', async () => {
  const s = server(RUNNING);
  await draw(makeApp());
  const fresh = makeApp();                       // a reload: a fresh app.keep
  await draw(fresh);
  sub('Search').click(); await settle();
  assert.equal(stateText(), 'Running trial 3 of 10 (2 finished): responses 5 of 64');
  assert.ok(button(byId('dirs-search-head'), 'Stop now'));
  assert.deepEqual(s.posts(), [], 'no event and no request was needed');
});

test('S14: delete asks first; confirmed, it sends DELETE; a refusal is shown', async () => {
  const app = makeApp();
  const s = await openSearch(app, { 'DELETE /api/directions/searches/s1': refusal("search 's1' is running; stop it first", 409) });
  let asked = null;
  global.confirm = (text) => { asked = text; return false; };
  button(byId('dirs-searches'), 'delete').click(); await settle();
  assert.equal(asked, "Delete search 's1' and its 5 trial records?");
  assert.deepEqual(s.posts(), [], 'not confirmed: nothing sent');
  global.confirm = () => true;
  button(byId('dirs-searches'), 'delete').click(); await settle(20);
  assert.equal(s.last('DELETE', '/api/directions/searches/s1').method, 'DELETE');
  assert.equal(byId('dirs-searches-error').textContent, "search 's1' is running; stop it first");
  s.table['DELETE /api/directions/searches/s1'] = { ok: true, deleted: 's1' };
  s.table['GET /api/directions/searches'] = SEARCHES;
  button(byId('dirs-searches'), 'delete').click(); await settle(20);
  assert.equal(byId('dirs-searches'), null);
  assert.match(byId('dirs-sub').textContent, /No searches in this workspace yet\./);
});

test('drafts: the New search form as typed comes back on a fresh page', async () => {
  const m = new Map();
  Object.defineProperty(global.window, 'localStorage', { configurable: true, get: () => ({ getItem: (k) => m.get(k) ?? null, setItem: (k, v) => m.set(k, String(v)) }) });
  try {
    await openSearch(makeApp(), { 'GET /api/directions/searches': SEARCHES });
    type(labelled(view, 'seed'), '11'); change(labelled(view, 'seed'));
    type(labelled(view, 'n_trials'), '6'); change(labelled(view, 'n_trials'));
    change(labelled(view, 'setting row_normalization'), 'none');
    const stored = JSON.parse(m.get('wsp.directions.v1.fixture')).searchForm;
    assert.deepEqual(stored, { name: null, plugin: null, settings: { Abliteration: { row_normalization: 'none' } }, nTrials: '6', nStartup: null, seed: '11' });
    await openSearch(makeApp(), { 'GET /api/directions/searches': SEARCHES });
    assert.equal(labelled(view, 'seed').value, '11');
    assert.equal(labelled(view, 'n_trials').value, '6');
    assert.equal([...labelled(view, 'setting row_normalization').children].find((o) => o.selected).value, 'none');
  } finally { delete global.window.localStorage; }
});

test('a damaged search is listed with its error and can be deleted', async () => {
  const app = makeApp();
  await openSearch(app, { 'GET /api/directions/searches': searches({ searches: [{ name: 'broken', error: 'unreadable search.json' }] }) });
  const row = rowsOf(byId('dirs-searches'))[0];
  assert.deepEqual([row[0], row[1], row[2]], ['broken', 'unreadable search.json', 'delete']);
  assert.equal(byId('dirs-search'), null, 'nothing to show for it');
});

test('a running search holds its measurement: the notes, Measure with replace and Delete of that measurement', async () => {
  const app = makeApp();
  const running = { ...SEARCH_ENTRY, running: 'this IDE' };
  server({ 'GET /api/directions/searches': searches({ searches: [running] }), 'GET /api/directions/searches/s1': { ...SEARCH, running: 'this IDE' } });
  await draw(app);
  assert.ok(noteTexts().includes("Search 's1' is running in this IDE: committing an edit would end it."), noteTexts().join(' | '));
  assert.ok([...byId('dirs-measure-form').querySelectorAll('[role=note]')].some((n) => n.textContent === SEARCH_SLOWS), 'the note of Q8');
  assert.equal(SEARCH_SLOWS, 'A search is running: this will wait for or interleave with its trials, and both get slower.');
  type(labelled(view, 'measurement name'), 'refusal');
  change(labelled(view, 'replace'), true); await settle();
  const measure = button(byId('dirs-measure-form'), 'Measure');
  assert.equal(measure.disabled, true);
  assert.equal(measure.title, "search 's1' is running from 'refusal': replacing it would end that search's runs");
  const del = byId('dirs-list').querySelector('button.danger');
  assert.equal(del.disabled, true);
  assert.equal(del.title, "search 's1' is running from this measurement");
  type(labelled(view, 'measurement name'), 'other');
  assert.equal(button(byId('dirs-measure-form'), 'Measure').disabled, false, 'measuring another name is allowed while it runs');
});

// Last: app.js itself, which starts the page when imported. The Patches pane shows a trial patch's origin line.
test('O1: through app.js, the Patches pane shows a trial patch with its origin line, and an ordinary patch without one', async () => {
  class FakeEventSource { constructor(url) { this.url = url; } close() {} }
  global.EventSource = FakeEventSource;
  document.createDocumentFragment ??= () => document.createElement('div');
  const patch = (id, source, message) => ({ id, message, author: 'human', source, enabled: true, strength: 1, created: 1794920800, info: {}, stats: [], changed: 624, stored: 624 });
  const origin = { kind: 'direction_search', search: 's1', search_id: SEARCH.search.id, trial: 4, displays: ['3/32', '0.0427'], edit_equals_scored: false };
  server({ 'GET /api/inspection-capabilities': { version: 1, passive_metadata: true }, 'GET /api/tree': { node: { addr: 'model', n_params: 1 }, children: [] },
    'GET /api/patches': { patches: [patch('p1', { kind: 'wscript', script: 'blocks[0].mlp.W_out *= 0.5' }, 'halve one matrix'),
      patch('p2', { kind: 'wscript', script: 'ablate_direction("refusal")', origin }, 'search s1: trial 4 (3/32, 0.0427)')] },
    'GET /api/state': { open: { name: 'fixture', kind: 'wsp', config: { n_layers: 2, d_model: 24, n_heads: 2, d_head: 12, d_mlp: 40 }, capabilities: null, n_params: 1, n_tensors: 1, bytes: 1, recognised: false },
      status: { can_run: true, runner_loaded: true, version: 0, workspace_id: 'w', n_patches: 2, drift: { n: 0 }, changed: 0, stored: 0 },
      workspaces: [{ name: 'fixture', built: true, description: 'a made-up workspace' }], groups: {}, task: { title: '', scope: ['model'] } } });
  await import('../web/js/app.js');
  await settle(40);
  const rows = [...byId('patches').querySelectorAll('.patch')];
  assert.equal(rows.length, 2);
  const [trial, plain] = rows;                   // newest first
  assert.equal(trial.querySelector('.dirs-origin').textContent, 'from search s1, trial 4: 3/32, 0.0427 · the committed edit differs from the scored one');
  assert.equal(plain.querySelector('.dirs-origin'), null);
  assert.ok(![...byId('patches').querySelectorAll('*')].some((n) => [...n.childNodes].some((c) => c.nodeType === 3 && c.data === 'null')), 'no "null" text');
});

// After O1 (app.js already imported): the list renderPatches read is the one the Output view shows.
test('O2: through app.js, renderPatches keeps the list it read in app.keep.patchList for the Output view', async () => {
  const { app } = await import('../web/js/app.js');
  assert.deepEqual(app.keep.patchList.map((p) => [p.id, p.message]), [['p1', 'halve one matrix'], ['p2', 'search s1: trial 4 (3/32, 0.0427)']]);
  assert.equal(app.keep.patchList[1].source.origin.kind, 'direction_search');
});
