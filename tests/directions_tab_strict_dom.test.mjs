// The Directions tab's frame against the strict fake DOM (tests/strict_dom.mjs): its place in the shell, the state
// notes, this page's job slot and the requests running elsewhere, the drafts kept in the browser, and the rule that
// text from outside is put into the page as text. A fake fetch answers from tests/directions_fixtures.mjs; no server,
// no model, no network. Expectations come from the tab's build spec (5.3, 6.2, 8, 11, 13) and docs/DIRECTIONS.md.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { installStrictDom } from './strict_dom.mjs';
import { LIST, META, STATS, SEARCHES, SEARCH, SEARCH_ENTRY, TRIAL, server, hold, makeApp, draw, settle, labelled, button, buttons, type, change, byId, nodes } from './directions_fixtures.mjs';
import { runJob, busyReason, SUBVIEWS, DRAFT_KEY, fmtBytes, onDirectionsEvent, paramsFor, streamId } from '../web/js/directions.js';

const { view } = installStrictDom({ shell: true });
const noteTexts = () => [...byId('dirs-notes').children].map((n) => n.textContent);
const statusOf = (id) => byId(id).querySelector('[role=status]')?.textContent ?? '';
const list = (over) => ({ ...LIST, ...over });
const deleteButton = (root) => byId('dirs-list').querySelector('button.danger');
function memoryStorage() { const m = new Map(); return { m, sets: [], getItem: (k) => (m.has(k) ? m.get(k) : null), setItem(k, v) { this.sets.push([k, String(v)]); m.set(k, String(v)); }, removeItem: (k) => m.delete(k) }; }
const withStorage = (getter) => Object.defineProperty(global.window, 'localStorage', { get: getter, configurable: true });

test('fmtBytes: bytes, KiB, MiB and GiB as 13.1 writes them', () => {
  assert.equal(fmtBytes(0), '0 bytes');
  assert.equal(fmtBytes(1023), '1023 bytes');
  assert.equal(fmtBytes(4480), '4.4 KiB');
  assert.equal(fmtBytes(3 * 2 ** 20), '3.0 MiB');
  assert.equal(fmtBytes(2 ** 31), '2.00 GiB');
});

test('T2: opening the tab, switching every sub-view and selecting a measurement only GET metadata', async () => {
  const s = server({ 'GET /api/directions/refusal': { ...META, stats: STATS } });
  const app = makeApp();
  let root = await draw(app);
  assert.equal(root.querySelector('h2').textContent, 'Directions');
  const control = labelled(root, 'Directions view');
  assert.deepEqual([...control.children].map((b) => b.textContent), SUBVIEWS.map((v) => v.name));
  for (const name of SUBVIEWS.map((v) => v.name)) { button(labelled(view, 'Directions view'), name).click(); await settle(); }
  change(labelled(view, 'measurement'), 'refusal'); await settle();
  assert.equal(app.keep.dirs.selected, 'refusal');
  for (const name of SUBVIEWS.map((v) => v.name)) { button(labelled(view, 'Directions view'), name).click(); await settle(); }
  root = await draw(app);
  assert.deepEqual(s.posts(), [], 'no POST, no DELETE');
  const allowed = /^\/api\/directions(\/searches|\/refusal(\?orthogonalize_direction=(true|false))?)?$|^\/api\/probe\/chat$/;
  assert.ok(s.gets().every((u) => allowed.test(u)), s.gets().join(', '));
  for (const u of ['/api/directions', '/api/directions/searches', '/api/probe/chat']) assert.ok(s.gets().includes(u), u);
  assert.deepEqual(SUBVIEWS.map((v) => v.name), ['Measure', 'Layers', 'Edit', 'Compare', 'Search', 'Output']);
  assert.ok(s.gets().includes('/api/directions/refusal?orthogonalize_direction=true'), 'Layers read the selected measurement');
});

test('T3: a model that cannot be run: the note with its reason; the saved measurements stay selectable', async () => {
  server();
  const app = makeApp({ status: { can_run: false, run_error: 'X' } });
  const root = await draw(app);
  assert.deepEqual(noteTexts(), ['Measuring, scoring, prefix detection and searching run the model, and this model cannot be run here: X. Saved measurements, the per-layer view, plans and applying (untested) still work.']);
  assert.equal(button(byId('dirs-list'), 'select').disabled, false);
  assert.equal(labelled(root, 'measurement').disabled, false);
  const measure = button(byId('dirs-measure-form'), 'Measure');
  assert.equal(measure.disabled, true);
  assert.equal(measure.title, 'this model cannot be run here: X');
  assert.equal(button(byId('dirs-set-good'), 'Check').disabled, false, 'a Check needs no model');
  assert.equal(button(byId('dirs-prefix'), 'Detect').disabled, true, 'Detect runs the model');
  assert.equal(button(labelled(root, 'Directions view'), 'Layers').disabled, false, 'the per-layer view works without running the model');
});

test('T4: a model that is not loaded yet: the not-loaded note, and nothing disabled by it', async () => {
  server();
  const root = await draw(makeApp({ status: { runner_loaded: false } }));
  assert.deepEqual(noteTexts(), ['The model is not loaded: measuring, scoring, prefix detection and searching load it first.']);
  assert.equal(button(byId('dirs-list'), 'select').disabled, false);
  assert.equal(deleteButton(root).disabled, false);
  assert.equal(labelled(root, 'as a chat message').disabled, false);
});

test('T5: GGUF and no chat template: both notes; the chat box disabled and unticked with the reason', async () => {
  server({ 'GET /api/probe/chat': { available: false, reason: 'R' } });
  const app = makeApp({ open: { kind: 'gguf' } });
  const root = await draw(app);
  const notes = noteTexts();
  assert.ok(notes.some((t) => t.startsWith('GGUF: ') && t.includes('never dequantized')), notes.join(' | '));
  assert.ok(notes.includes('As a chat message: unavailable (R).'), notes.join(' | '));
  const box = labelled(root, 'as a chat message');
  assert.equal(box.disabled, true);
  assert.equal(box.checked, false);
  assert.match(box.parentElement.textContent, /unavailable: R/);
  assert.equal(labelled(root, 'system_prompt').disabled, true);
  assert.equal(labelled(root, 'response_prefix').disabled, true);
  assert.equal(labelled(root, 'chat options').disabled, true);
  assert.equal(button(byId('dirs-prefix'), 'Detect').disabled, true, 'Detect needs the chat template');
});

test('T6: edits not supported for the checkpoint: the note with its reason', async () => {
  server();
  await draw(makeApp({ open: { capabilities: { logical_edit: { supported: false, reason: 'packed weights' } } } }));
  assert.ok(noteTexts().includes('Edits are not supported for this checkpoint: packed weights.'), noteTexts().join(' | '));
});

test('T7: an architecture whose layers are not known (parameters null): the note', async () => {
  server({ 'GET /api/directions': list({ parameters: null }) });
  await draw(makeApp());
  assert.deepEqual(noteTexts(), ["This checkpoint's layers are not known, so directions cannot be measured or applied here."]);
  assert.equal(button(byId('dirs-measure-form'), 'Measure').disabled, true);
  assert.equal(button(byId('dirs-list'), 'select').disabled, false, 'the list stays usable');
});

test('T8: this page\'s job: its line, progress from its own events without redrawing the view, Stop, its outcome', async () => {
  const s = server();
  const app = makeApp();
  await draw(app);
  const st = app.keep.dirs;
  let sent = null, finish;
  const gate = new Promise((r) => { finish = r; });
  const done = runJob(app, st, { kind: 'measure', label: "Measuring 'refusal'", name: 'refusal' }, async (stream) => { sent = stream; await gate; return "Measured 'refusal' in 0.5 s."; });
  await settle();
  assert.match(sent, /^dirs-measure-[A-Za-z0-9_-]+$/);
  assert.ok(sent.length <= 64);
  assert.ok(app.calls.rerender >= 1, 'the start re-rendered the view');
  assert.equal(statusOf('dirs-job'), "Measuring 'refusal': starting; this loads the model if it is not loaded.");
  assert.notEqual(busyReason(st), '');
  assert.equal(deleteButton(view).disabled, true, 'a DELETE waits behind a running request: disabled');
  assert.equal(button(byId('dirs-measure-form'), 'Measure').disabled, true, 'Measure held');
  assert.equal(await runJob(app, st, { kind: 'prefix', label: 'x' }, async () => 'no'), null, 'a second job is refused');
  const first = view.firstChild, rerenders = app.calls.rerender;
  onDirectionsEvent({ type: 'directions', workspace: 'fixture', name: 'refusal', state: 'measuring', done: 4, total: 11, seconds: 1.2, stream: sent }, app);
  assert.equal(statusOf('dirs-job'), "Measuring 'refusal': 4 of 11 prompts (1.2 s)");
  assert.equal(view.firstChild, first, 'the view is not redrawn by a progress event');
  assert.equal(app.calls.rerender, rerenders);
  assert.deepEqual(Object.keys(st.foreign), [], 'its own events are not a request started elsewhere');
  button(byId('dirs-job'), 'Stop').click(); await settle();
  assert.deepEqual(s.last('POST', '/api/generation/stop').body, { stream: sent });
  assert.equal(button(byId('dirs-job'), 'Stopping…').disabled, true);
  finish(); await done; await settle();
  assert.equal(st.job, null);
  assert.equal(statusOf('dirs-job'), "Measured 'refusal' in 0.5 s.");
  assert.equal(deleteButton(view).disabled, false);
});

test('a stream id is made without crypto.randomUUID too: a page served over plain http on another address', async () => {
  server();
  const app = makeApp();
  await draw(app);
  const st = app.keep.dirs;
  const own = Object.getOwnPropertyDescriptor(globalThis, 'crypto');
  Object.defineProperty(globalThis, 'crypto', { value: {}, configurable: true });           // no randomUUID, as outside a secure context
  try {
    const seen = [];
    for (let i = 0; i < 2; i++) assert.equal(await runJob(app, st, { kind: 'measure', label: 'x' }, async (stream) => { seen.push(stream); return 'done'; }), 'done');
    seen.push(streamId('search'), streamId('search'));
    assert.match(seen[0], /^dirs-measure-[A-Za-z0-9_-]+$/);
    assert.match(seen[2], /^dirs-search-[A-Za-z0-9_-]+$/);
    assert.equal(new Set(seen).size, 4, 'each request has its own stream');
    assert.ok(seen.every((x) => x.length <= 64));
  } finally { Object.defineProperty(globalThis, 'crypto', own); }
  assert.match(streamId('scores'), /^dirs-scores-[0-9a-f]{8}-[0-9a-f-]{27}$/, 'with crypto.randomUUID: a UUID');
});

test('T9: a request started elsewhere (B2): its card, actions disabled, Stop it, and its end', async () => {
  const s = server({ 'GET /api/directions': list({ requests: [{ type: 'directions', workspace: 'fixture', name: 'x', state: 'measuring', done: 2, total: 8, stream: 'other-1' }] }) });
  const app = makeApp();
  await draw(app);
  const card = byId('dirs-foreign');
  assert.equal(card.children.length, 1);
  assert.equal(card.children[0].children[0].textContent, "A measurement 'x' is running (not started by this page): 2 of 8 prompts.");
  assert.equal(deleteButton(view).disabled, true);
  assert.match(deleteButton(view).title, /started elsewhere/);
  assert.equal(button(byId('dirs-measure-form'), 'Measure').disabled, true);
  button(card, 'Stop it').click(); await settle();
  assert.deepEqual(s.last('POST', '/api/generation/stop').body, { stream: 'other-1' });
  onDirectionsEvent({ type: 'directions', workspace: 'other', name: 'y', state: 'measuring', done: 1, total: 2, stream: 'z' }, app);
  assert.equal(Object.keys(app.keep.dirs.foreign).length, 1, 'another workspace\'s event is ignored');
  onDirectionsEvent({ type: 'directions', workspace: 'fixture', name: 'x', state: 'measuring', done: 5, total: 8, seconds: 2, stream: 'other-1' }, app);
  assert.equal(byId('dirs-foreign').children[0].children[0].textContent, "A measurement 'x' is running (not started by this page): 5 of 8 prompts.");
  onDirectionsEvent({ type: 'directions', workspace: 'fixture', name: 'x', state: 'stopped', done: 5, total: 8, stream: 'other-1' }, app);
  await settle();
  assert.equal(byId('dirs-foreign').children.length, 0, 'the card is gone');
  assert.equal(busyReason(app.keep.dirs), '');
  assert.equal(deleteButton(view).disabled, false);
  assert.equal(button(byId('dirs-measure-form'), 'Measure').disabled, false, 'Measure enabled again');
});

test('T9: a foreign request ending as measured reads the list again', async () => {
  const s = server({ 'GET /api/directions': list({ requests: [{ type: 'directions', workspace: 'fixture', name: 'x', state: 'measuring', done: 2, total: 8, stream: 'other-2' }] }) });
  const app = makeApp();
  await draw(app);
  const before = s.gets().filter((u) => u === '/api/directions').length;
  s.table['GET /api/directions'] = list({ directions: [META, { ...META, name: 'x' }] });
  onDirectionsEvent({ type: 'directions', workspace: 'fixture', name: 'x', state: 'measured', done: 8, total: 8, stream: 'other-2' }, app);
  await settle();
  assert.equal(s.gets().filter((u) => u === '/api/directions').length, before + 1);
  assert.deepEqual([...byId('dirs-list').children].slice(1).map((r) => r.children[0].textContent), ['refusal', 'x']);
  assert.equal(byId('dirs-foreign').children.length, 0);
});

test('T10: drafts: a blocked localStorage, a working one restored on a fresh page, and drafts too large to keep', async () => {
  const saved = Object.getOwnPropertyDescriptor(global.window, 'localStorage');
  try {
    // (a) the getter throws (site data blocked): the tab draws and keeps what is typed in the page
    withStorage(() => { throw new Error('SecurityError'); });
    server();
    let app = makeApp();
    let root = await draw(app);
    const sys = labelled(root, 'system_prompt');
    type(sys, 'Be brief.'); change(sys);
    assert.equal(app.keep.dirs.rendering.system, 'Be brief.');
    // (b) a working storage: a change writes the drafts; a fresh app.keep restores them
    const store = memoryStorage();
    withStorage(() => store);
    app = makeApp();
    root = await draw(app);
    assert.equal(labelled(root, 'system_prompt').value, 'You are a helpful assistant.', "Heretic's default system prompt");
    type(labelled(root, 'system_prompt'), 'Answer in French.');
    assert.equal(store.sets.length, 0, 'typing alone waits for the debounce');
    change(labelled(root, 'system_prompt'));
    assert.ok(store.m.has(DRAFT_KEY('fixture')), 'written under wsp.directions.v1.fixture');
    const drafts = JSON.parse(store.m.get('wsp.directions.v1.fixture'));
    assert.equal(drafts.version, 1);
    assert.equal(drafts.rendering.system, 'Answer in French.');
    const again = makeApp();
    root = await draw(again);
    assert.equal(again.keep.dirs.rendering.system, 'Answer in French.');
    assert.equal(labelled(root, 'system_prompt').value, 'Answer in French.');
    // a shape error ignores the stored drafts
    store.m.set(DRAFT_KEY('fixture'), JSON.stringify({ version: 1, rendering: { chat: 'yes' } }));
    const odd = makeApp();
    await draw(odd);
    assert.equal(odd.keep.dirs.rendering.system, 'You are a helpful assistant.');
    // (c) more than 256 KiB: nothing written, and the note says so
    store.sets.length = 0;
    const big = 'x'.repeat(300000);
    type(labelled(view, 'system_prompt'), big); change(labelled(view, 'system_prompt'));
    assert.ok(!store.sets.some(([, v]) => v.includes(big)), 'no setItem with the large drafts');
    assert.ok(noteTexts().includes('Drafts are larger than 256 KiB and are kept only until the page is reloaded.'), noteTexts().join(' | '));
  } finally {
    if (saved) Object.defineProperty(global.window, 'localStorage', saved); else delete global.window.localStorage;
  }
});

test('the list: a damaged entry, another model shape, and a rendering cut to 40 characters', async () => {
  server({ 'GET /api/directions': list({ directions: [
    { ...META, name: 'chatty', options: { chat: true, system: 'S'.repeat(50), prefix: '<think>\n', chat_options: null, winsorization_quantile: 1 }, module_io: { bytes: 4480, layers: 2 } },
    { ...META, name: 'wide', d_model: 32 },
    { name: 'broken', error: 'unreadable meta' }] }) });
  const root = await draw(makeApp());
  const rows = [...byId('dirs-list').children].slice(1);
  assert.deepEqual([...rows[0].children].slice(0, 6).map((c) => c.textContent),
    ['chatty', '3 good / 2 bad', 'base', `chat, system: ${'S'.repeat(40)}…, prefix: <think>\\u{000A}`, '4.4 KiB', '2026-11-17 13:03']);
  assert.match(rows[1].children[0].textContent, /other model shape/);
  assert.equal(button(rows[1], 'select').disabled, true);
  assert.equal(rows[2].children[1].textContent, 'unreadable meta');
  assert.equal(rows[2].children[1].getAttribute('colspan'), '5');
  const options = [...labelled(root, 'measurement').children];
  assert.deepEqual(options.map((o) => [o.textContent, o.disabled]),
    [['(none)', false], ['chatty', false], ['wide (other model shape)', true], ['broken (damaged)', true]]);
});

test('the rendering: a selected measurement measured otherwise is said, and "Use its rendering" takes it', async () => {
  server({ 'GET /api/directions': list({ directions: [{ ...META, options: { chat: true, system: 'Sys.', prefix: null, chat_options: { enable_thinking: false }, winsorization_quantile: 1 } }] }) });
  const app = makeApp();
  await draw(app);
  change(labelled(view, 'measurement'), 'refusal'); await settle();
  assert.match(byId('dirs-rendering').textContent, /The rendering differs from the one 'refusal' was measured with\./);
  button(byId('dirs-rendering'), 'Use its rendering').click(); await settle();
  assert.deepEqual(app.keep.dirs.rendering, { chat: true, system: 'Sys.', prefix: '', chatOptions: '{"enable_thinking":false}' });
  assert.doesNotMatch(byId('dirs-rendering').textContent, /differs/);
  type(labelled(view, 'chat options'), '[1]');
  assert.equal(byId('dirs-rendering').querySelector('[role=alert]').textContent, 'chat options: not a JSON object');
});

test('T10: the prompt sets are drafts too: kept in the page when storage is blocked, and never written above 256 KiB', async () => {
  const saved = Object.getOwnPropertyDescriptor(global.window, 'localStorage');
  try {
    withStorage(() => { throw new Error('SecurityError'); });
    server();
    const blocked = makeApp();
    await draw(blocked);
    type(labelled(view, 'good prompts part 1 texts'), 'Name three fruits.'); change(labelled(view, 'good prompts part 1 texts'));
    assert.equal(blocked.keep.dirs.sets.good.parts[0].text, 'Name three fruits.');
    const store = memoryStorage();
    withStorage(() => store);
    const app = makeApp();
    await draw(app);
    type(labelled(view, 'bad prompts part 1 texts'), 'Pick a lock.'); change(labelled(view, 'bad prompts part 1 texts'));
    assert.deepEqual(JSON.parse(store.m.get(DRAFT_KEY('fixture'))).sets.bad, { parts: [{ kind: 'texts', text: 'Pick a lock.' }] });
    const fresh = makeApp();
    await draw(fresh);
    assert.equal(labelled(view, 'bad prompts part 1 texts').value, 'Pick a lock.');
    store.sets.length = 0;
    const big = 'y'.repeat(300000);
    type(labelled(view, 'good prompts part 1 texts'), big); change(labelled(view, 'good prompts part 1 texts'));
    assert.ok(!store.sets.some(([, v]) => v.includes(big)), 'no setItem with it');
    assert.ok(noteTexts().includes('Drafts are larger than 256 KiB and are kept only until the page is reloaded.'));
    assert.equal(fresh.keep.dirs.sets.good.parts[0].text.length, 300000, 'kept in the page');
  } finally {
    if (saved) Object.defineProperty(global.window, 'localStorage', saved); else delete global.window.localStorage;
  }
});

test('B2: a scores, a prefix and a dataset-fetch request started elsewhere: the cards of 13.4, Stop only with a stream', async () => {
  const requests = [
    { type: 'direction_scores', workspace: 'fixture', state: 'scoring', seconds: 1.5, phase: 'responses', done: 3, total: 7, stream: 'sc-9' },
    { type: 'direction_prefix', workspace: 'fixture', state: 'detecting', seconds: 0.2, phase: 'common_prefix', done: 1, total: 4 },
    { type: 'direction_datasets', workspace: 'fixture', state: 'fetching', seconds: 0.4, phase: 'file', file: 'data/train-00000-of-00001.jsonl',
      done_bytes: 1024, total_bytes: 4096, stream: 'fetch-1' }];
  const s = server({ 'GET /api/directions': list({ requests }) });
  const app = makeApp();
  await draw(app);
  const cards = () => [...byId('dirs-foreign').children];
  assert.deepEqual(cards().map((c) => c.children[0].textContent), [
    'A scoring is running (not started by this page): responses 3 of 7.',
    'A response-prefix detection is running (not started by this page): 1 of at most 4 responses.',
    'A dataset download is running (not started by this page): data/train-00000-of-00001.jsonl, 1.0 KiB of 4.0 KiB.']);
  assert.deepEqual(cards().map((c) => buttons(c, 'Stop it').length), [1, 0, 1], 'no Stop for a request sent without a stream');
  assert.equal(button(byId('dirs-measure-form'), 'Measure').disabled, true);
  assert.equal(button(byId('dirs-set-good'), 'Check').disabled, true, 'a Check waits behind it too');
  button(cards()[2], 'Stop it').click(); await settle();
  assert.deepEqual(s.last('POST', '/api/generation/stop').body, { stream: 'fetch-1' });
  // the scoring's next events: the phase text of 13.1 (null: starting), then its end removes the card
  onDirectionsEvent({ type: 'direction_scores', workspace: 'fixture', state: 'scoring', seconds: 1.6, phase: 'reference', done: 1, total: 9, stream: 'sc-9' }, app);
  assert.equal(cards()[0].children[0].textContent, "A scoring is running (not started by this page): the base model's first-token passes 1 of 9.");
  onDirectionsEvent({ type: 'direction_scores', workspace: 'fixture', state: 'scoring', seconds: 0, phase: null, done: 0, total: null, stream: 'sc-9' }, app);
  assert.equal(cards()[0].children[0].textContent, 'A scoring is running (not started by this page): starting.');
  onDirectionsEvent({ type: 'direction_scores', workspace: 'fixture', state: 'scored', seconds: 2, done: 9, total: 9, stream: 'sc-9' }, app);
  onDirectionsEvent({ type: 'direction_datasets', workspace: 'fixture', state: 'stopped', seconds: 0.5, done_bytes: 1024, total_bytes: 4096, fetched: [], stream: 'fetch-1' }, app);
  await settle();
  assert.deepEqual(cards().map((c) => c.children[0].textContent), ['A response-prefix detection is running (not started by this page): 1 of at most 4 responses.']);
  assert.equal(button(byId('dirs-measure-form'), 'Measure').disabled, true, 'still one running');
  onDirectionsEvent({ type: 'direction_prefix', workspace: 'fixture', state: 'detected', seconds: 0.3, done: 4, total: 4 }, app);
  await settle();
  assert.equal(cards().length, 0);
  assert.equal(button(byId('dirs-measure-form'), 'Measure').disabled, false);
});

test('B2: two requests sent without a stream are two cards; an event that cannot tell them apart reads the list again', async () => {
  const one = { type: 'direction_scores', workspace: 'fixture', state: 'scoring', seconds: 1, phase: 'responses', done: 1, total: 4 };
  const two = { type: 'direction_scores', workspace: 'fixture', state: 'scoring', seconds: 2, phase: 'logits', done: 3, total: 9 };
  const s = server({ 'GET /api/directions': list({ requests: [one, two] }) });
  const app = makeApp();
  await draw(app);
  const cards = () => [...byId('dirs-foreign').children].map((c) => c.children[0].textContent);
  assert.deepEqual(cards(), ['A scoring is running (not started by this page): responses 1 of 4.',
    'A scoring is running (not started by this page): first-token passes 3 of 9.']);
  assert.deepEqual([...byId('dirs-foreign').children].map((c) => buttons(c, 'Stop it').length), [0, 0]);
  // one of them ends: its event names no stream, so the list (its entry removed before the event) says which
  const reads = s.gets().filter((u) => u === '/api/directions').length;
  s.table['GET /api/directions'] = list({ requests: [two] });
  onDirectionsEvent({ type: 'direction_scores', workspace: 'fixture', state: 'scored', seconds: 1.5, done: 4, total: 4 }, app);
  await settle();
  assert.equal(s.gets().filter((u) => u === '/api/directions').length, reads + 1);
  assert.deepEqual(cards(), ['A scoring is running (not started by this page): first-token passes 3 of 9.']);
  // the one left: its events move its card, its end removes it
  onDirectionsEvent({ ...two, done: 5 }, app);
  assert.deepEqual(cards(), ['A scoring is running (not started by this page): first-token passes 5 of 9.']);
  onDirectionsEvent({ ...two, state: 'scored', done: 9 }, app);
  await settle();
  assert.deepEqual(cards(), []);
  assert.equal(button(byId('dirs-measure-form'), 'Measure').disabled, false);
});

test('a late event of a job of this page that has already ended is not taken for a request started elsewhere', async () => {
  server();
  const app = makeApp();
  await draw(app);
  const st = app.keep.dirs;
  let sent = null;
  await runJob(app, st, { kind: 'scores', label: 'Scoring', stateLabel: 'current weights' }, async (stream) => { sent = stream; return 'Scored.'; });
  await settle();
  assert.equal(st.job, null);
  assert.equal(statusOf('dirs-job'), 'Scored.');
  const rerenders = app.calls.rerender;
  for (const state of ['scoring', 'scored']) {
    onDirectionsEvent({ type: 'direction_scores', workspace: 'fixture', state, seconds: 0.2, phase: 'responses', done: 2, total: 7, stream: sent }, app);
  }
  await settle();
  assert.deepEqual(Object.keys(st.foreign), []);
  assert.equal(byId('dirs-foreign').children.length, 0);
  assert.equal(busyReason(st), '');
  assert.equal(statusOf('dirs-job'), 'Scored.', 'the outcome stays');
  assert.equal(app.calls.rerender, rerenders, 'nothing redrawn');
  // a reload's list that still holds it (its terminal event not yet published) does not make a card either
  const s2 = server({ 'GET /api/directions': list({ requests: [{ type: 'direction_scores', workspace: 'fixture', state: 'scoring', seconds: 0.3, phase: 'logits', done: 6, total: 7, stream: sent }] }) });
  st.listKey = null;
  await app.rerender(); await settle();
  assert.ok(s2.gets().includes('/api/directions'));
  assert.equal(byId('dirs-foreign').children.length, 0);
});

test('a job that loaded the model: once the shell has the new status, the not-loaded note is gone', async () => {
  server();
  const app = makeApp({ status: { runner_loaded: false } });
  let answered;
  const status = new Promise((r) => { answered = r; });
  app.refreshStatus = async () => { app.calls.refreshStatus++; await status; app.state.status = { ...app.state.status, runner_loaded: true }; };
  await draw(app);
  const loaded = 'The model is not loaded: measuring, scoring, prefix detection and searching load it first.';
  assert.deepEqual(noteTexts(), [loaded]);
  const st = app.keep.dirs;
  const done = runJob(app, st, { kind: 'measure', label: "Measuring 'refusal'", name: 'refusal' }, async () => "Measured 'refusal' in 0.5 s.");
  await settle();
  assert.equal(app.calls.refreshStatus, 1, 'the status is read again after the job');
  assert.deepEqual(noteTexts(), [loaded], 'the status has not arrived yet');
  answered(); await done; await settle();
  assert.equal(app.state.status.runner_loaded, true);
  assert.deepEqual(noteTexts(), [], 'no "not loaded" note once the status says it is loaded');
  assert.equal(statusOf('dirs-job'), "Measured 'refusal' in 0.5 s.");
});

test('drafts: the measure form, the method, the Edit parameters and max_response_length come back on a fresh page', async () => {
  const saved = Object.getOwnPropertyDescriptor(global.window, 'localStorage');
  try {
    const store = memoryStorage();
    withStorage(() => store);
    server();
    const app = makeApp();
    await draw(app);
    const st = app.keep.dirs;
    type(labelled(view, 'measurement name'), 'mine'); change(labelled(view, 'measurement name'));
    change(labelled(view, 'variant'), 'current');
    type(labelled(view, 'winsorization_quantile'), '0.9'); change(labelled(view, 'winsorization_quantile'));
    change(labelled(view, 'module inputs and outputs'), false);
    st.method = 'ara';
    paramsFor(st, 'ablation').direction_index = '0';
    paramsFor(st, 'ablation').components['mlp.down_proj'].on = false;
    paramsFor(st, 'ara').lora_rank = '8';
    type(labelled(view, 'max_response_length'), '32'); change(labelled(view, 'max_response_length'));
    const stored = JSON.parse(store.m.get(DRAFT_KEY('fixture')));
    assert.deepEqual(stored.measure, { name: 'mine', variant: 'current', wq: '0.9', moduleIo: false });
    assert.equal(stored.method, 'ara');
    assert.equal(stored.maxResponseLength, '32');
    const fresh = makeApp();
    await draw(fresh);
    const again = fresh.keep.dirs;
    assert.notEqual(again, st);
    assert.deepEqual(again.measure, { name: 'mine', variant: 'current', wq: '0.9', moduleIo: false, replace: false });
    assert.equal(again.method, 'ara');
    assert.equal(again.maxResponseLength, '32');
    assert.equal(again.params.ablation.direction_index, '0');
    assert.equal(again.params.ablation.components['mlp.down_proj'].on, false);
    assert.equal(again.params.ablation.components['attn.o_proj'].fields.max_weight, '1.15');
    assert.equal(again.params.ara.lora_rank, '8');
    assert.equal(labelled(view, 'measurement name').value, 'mine');
    assert.equal(labelled(view, 'winsorization_quantile').value, '0.9');
    assert.equal(labelled(view, 'module inputs and outputs').checked, false);
    assert.equal(labelled(view, 'max_response_length').value, '32');
    assert.equal([...labelled(view, 'variant').children].find((o) => o.selected)?.value, 'current');
  } finally {
    if (saved) Object.defineProperty(global.window, 'localStorage', saved); else delete global.window.localStorage;
  }
});

test('X2: text from outside stays text: a prompt preview, a refusal and a measurement name are shown as typed', async () => {
  const EVIL = '<img src=x onerror=alert(1)>';
  const s = server({ 'GET /api/directions': list({ directions: [{ ...META, name: EVIL }] }),
    'POST /api/directions/datasets': (body) => (body.prompts[0] === 'refuse' ? { __answer: true, status: 400, data: { error: `DirectionError: ${EVIL}` } }
      : { cached: true, count: 1, prompts_sha256: 'ab', datasets: null, preview: [EVIL], fetched: [], fetched_bytes: 0, seconds: 0 }) });
  const app = makeApp();
  await draw(app);
  const exactly = (root) => nodes(root).filter((n) => n.textContent === EVIL && [...n.childNodes].every((c) => c.nodeType === 3));
  assert.ok(exactly(byId('dirs-list')).some((n) => n.tag === 'td'), 'the name in the list');
  type(labelled(view, 'good prompts part 1 texts'), EVIL);
  button(byId('dirs-set-good'), 'Check').click(); await settle();
  assert.ok(exactly(byId('dirs-set-good')).some((n) => n.tag === 'li'), 'the preview');
  type(labelled(view, 'bad prompts part 1 texts'), 'refuse');
  button(byId('dirs-set-bad'), 'Check').click(); await settle();
  assert.ok(exactly(byId('dirs-set-bad')).some((n) => n.getAttribute('role') === 'alert'), 'the refusal');
  assert.ok(!nodes(view).some((n) => n.tag === 'img'), 'no img element was made');
  // a line of the Edit view's script and a plan row; a scorer's name, a prompt and a model's response in Compare
  const script = `ablate_direction("refusal")\n# ${EVIL}`;
  s.table['POST /api/directions/script'] = { script, plan: { direction: 'refusal', row_normalization: 'none',
    rows: [{ layer: 0, component: EVIL, weight: 1, skipped: EVIL, direction: 'index 1', views: 1, stored: 48 }],
    totals: { layers: 1, components: 1, views: 1, stored: 48, skipped: 0 } } };
  s.table['POST /api/directions/scores'] = { state: 'current', scores: [
    { plugin: 'KeywordRate', name: EVIL, optimization: 'minimize', value: 1, display: EVIL, matches: 1, total: 1, responses: [{ prompt: EVIL, response: EVIL, tokens: 1, match: true }] },
    { plugin: 'KLDivergence', name: 'KL divergence', optimization: 'minimize', value: 0, display: '0', per_prompt: [0] }], done: 2, total: 2 };
  change(labelled(view, 'measurement'), EVIL); await settle();
  button(labelled(view, 'Directions view'), 'Edit').click(); await settle();
  button(byId('dirs-edit-actions'), 'Preview plan').click(); await settle();
  assert.equal(byId('dirs-script').textContent, script);
  assert.ok([...byId('dirs-script').childNodes].every((c) => c.nodeType === 3), 'the script is one text node');
  assert.ok(exactly(byId('dirs-plan-table')).filter((n) => n.tag === 'td').length === 2, 'the plan row');
  button(labelled(view, 'Directions view'), 'Compare').click(); await settle();
  type(labelled(view, 'scorer 1 prompts part 1 texts'), EVIL);
  type(labelled(view, 'scorer 2 prompts part 1 texts'), 'x');
  button(byId('dirs-compare-go'), 'Score').click(); await settle(20);
  assert.ok(exactly(byId('dirs-compare-table')).some((n) => n.tag === 'td'), 'the scorer name and its display');
  const responses = exactly(byId('dirs-responses-0'));
  assert.ok(responses.length >= 3, 'the prompt and both responses');
  assert.ok(!nodes(view).some((n) => n.tag === 'img'), 'no img element was made');
  // a search's name in the list and in its header (Search)
  s.table['GET /api/directions/searches'] = { ...SEARCHES, searches: [{ ...SEARCH_ENTRY, name: EVIL }] };
  s.table[`GET /api/directions/searches/${encodeURIComponent(EVIL)}`] = { ...SEARCH, search: { ...SEARCH.search, name: EVIL, notes: [EVIL] } };
  app.keep.dirs.listKey = null;
  button(labelled(view, 'Directions view'), 'Search').click(); await settle();
  assert.ok(exactly(byId('dirs-searches')).some((n) => n.tag === 'td'), 'the search name in the list');
  assert.ok(exactly(byId('dirs-search-head')).some((n) => n.getAttribute('role') === 'note'), "the search's note");
  assert.ok(byId('dirs-search-head').querySelector('h3').textContent.startsWith(`${EVIL} · ablation`), 'the name in the header');
  // a trial's response, its script and an error of its record
  s.table[`GET /api/directions/searches/${encodeURIComponent(EVIL)}/trials/3`] = { ...TRIAL, script: EVIL, presented: { direction_index: EVIL },
    scores: [{ ...TRIAL.scores[0], display: EVIL, rows: TRIAL.scores[0].rows.map((r, k) => (k ? r : { ...r, response: EVIL })) }, TRIAL.scores[1]],
    candidate: { ...TRIAL.candidate, commit_check: { ok: false, error: EVIL } } };
  [...byId('dirs-trials').children].find((r) => r.getAttribute('data-trial') === '3').dispatchEvent({ type: 'click' }); await settle();
  assert.ok(exactly(byId('dirs-trial-responses-0')).some((n) => n.tag === 'div'), "the trial's response");
  assert.equal(byId('dirs-trial-script').textContent, EVIL);
  assert.ok(exactly(byId('dirs-trial-parameters')).some((n) => n.tag === 'td'), 'a presented parameter');
  assert.ok(byId('dirs-trial').querySelector('[role=alert]').textContent.endsWith(EVIL), 'a commit check error');
  assert.ok(!nodes(view).some((n) => n.tag === 'img'), 'no img element was made');
});

test('X1: no markup is built from text in the tab\'s files', () => {
  const dir = new URL('../web/js/', import.meta.url);
  const files = readdirSync(dir).filter((f) => /^directions.*\.js$/.test(f));
  assert.ok(files.includes('directions.js') && files.includes('directions_measure.js'), files.join(', '));
  for (const f of files) {
    const text = readFileSync(new URL(f, dir), 'utf8');
    for (const bad of [/innerHTML/, /outerHTML/, /insertAdjacentHTML/, /document\.write/, /\bhtml\s*:/]) assert.doesNotMatch(text, bad, `${f}: ${bad}`);
    assert.doesNotMatch(text, /\btip\(/, `${f}: the HTML tooltip helper`);
  }
});

// Last: app.js itself, which starts the page when imported (its events stay connected for the rest of the file).
test('T1: the shell has a Directions tab after Probe, selection-free, and hands it the direction events', async () => {
  class FakeEventSource { constructor(url) { this.url = url; FakeEventSource.last = this; } close() {} }
  global.EventSource = FakeEventSource;
  document.createDocumentFragment ??= () => document.createElement('div');   // the explorer tree builds into one
  const state = hold();
  const s = server({ 'GET /api/inspection-capabilities': { version: 1, passive_metadata: true }, 'GET /api/state': state,
    'GET /api/tree': { node: { addr: 'model', n_params: 1 }, children: [] }, 'GET /api/patches': { patches: [] },
    'GET /api/directions/searches/s1': { search: { name: 's1' }, trials: [], front: [], running: null, job: null } });
  const { app } = await import('../web/js/app.js');
  const { SELECTION_FREE } = await import('../web/js/shell.js');
  const names = [...byId('tabs').children].map((b) => b.textContent);
  assert.equal(names[names.indexOf('Probe') + 1], 'Directions', names.join(', '));
  assert.ok(SELECTION_FREE.has('Directions'));
  app.tab = 'Directions';
  await settle();
  state.release({ open: { name: 'fixture', kind: 'wsp', config: { n_layers: 2, d_model: 24, n_heads: 2, d_head: 12, d_mlp: 40 }, capabilities: null, n_params: 1, n_tensors: 1, bytes: 1, recognised: false },
    status: { can_run: true, runner_loaded: true, version: 0, workspace_id: 'w', n_patches: 0, drift: { n: 0 }, changed: 0, stored: 0 },
    workspaces: [{ name: 'fixture', built: true, description: 'a made-up workspace' }], groups: {}, task: { title: '', scope: ['model'] } });
  await settle(40);
  assert.equal(view.querySelector('h2')?.textContent, 'Directions');
  assert.equal(view.dataset.view, 'Directions');
  app.rerender(); await settle(20);
  assert.equal(view.querySelector('h2').textContent, 'Directions');
  assert.ok(FakeEventSource.last, 'the page listens to /api/events');
  FakeEventSource.last.onmessage({ data: JSON.stringify({ type: 'direction_search', workspace: 'fixture', search: 's1', state: 'trial', trial: 1 }) });
  await settle();
  assert.ok(s.gets().includes('/api/directions/searches/s1'), s.gets().join(', '));
  assert.deepEqual(s.posts(), [], 'none of this sent a POST');
  assert.ok(nodes(view).some((n) => n.id === 'dirs-notes'));
});
