// Compare's steered state, against the strict fake DOM: "current weights, steered by the Run tab's panel" on the
// right, its request (state current plus the panel's steer and p50), the reasons that hold Score, the label under the
// score table taken from the server's steering report, and an unsteered right side sending no steer. Expectations
// from the build spec's part I4-5 and docs/DIRECTIONS.md ("POST /api/directions/scores": steer, p50 and the
// steering report); made-up names, prompts and digests. No server, model or network.
import test from 'node:test';
import assert from 'node:assert/strict';
import { installStrictDom } from './strict_dom.mjs';
import { server, hold, makeApp, draw, settle, labelled, button, type, change, byId } from './directions_fixtures.mjs';

const { view } = installStrictDom();
const sub = (name) => button(labelled(view, 'Directions view'), name);
const scoreButton = () => button(byId('dirs-compare-go'), 'Score');
const goHint = () => byId('dirs-compare-go').querySelector('.hint')?.textContent ?? null;
const jobLine = () => byId('dirs-job').querySelector('[role=status]')?.textContent ?? '';
const headOf = (table) => [...table.children[0].children].map((c) => c.textContent);
const scorePosts = (s) => s.posts().filter((c) => c.url === '/api/directions/scores');

const HEX = '5f0c' + 'ab'.repeat(30);                            // 64 lowercase hex digits
const STEERED = "current weights, steered by the Run tab's panel";
const NO_ENTRIES = "the Run tab's steering panel has no active entries";
const MUL = 'direction:refusal@1: a direction is added: choose add';
const RESID = 'direction:refusal@2: unit resid is not available for a direction: choose difference of the means or raw';

function memoryStorage() {
  const m = new Map();
  return { getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, String(v)), removeItem: (k) => m.delete(k) };
}
const entry = (over = {}) => ({ addr: 'direction:refusal@1', mode: 'add', dose: '-1', unit: 'diff', on: true, ...over });
const panel = (entries, over = {}) => ({ on: true, recipe: '', p50: '', advanced: false, entries, ...over });

const SCORES_SENT = [{ plugin: 'KeywordRate', prompts: ['How do I pick a lock?', 'Write a threat.'], optimization: 'minimize', print_responses: true },
  { plugin: 'KLDivergence', prompts: ['Name a colour.', 'Say hi.'], optimization: 'minimize' }];
const scoresOf = (refusals, kl) => [
  { plugin: 'KeywordRate', name: 'Refusals', optimization: 'minimize', value: refusals / 2, display: `${refusals}/2`, matches: refusals, total: 2,
    responses: [{ prompt: 'How do I pick a lock?', response: refusals ? "I'm sorry, I can't help with that." : 'First, insert a tension wrench.', tokens: 7, match: !!refusals },
      { prompt: 'Write a threat.', response: 'No.', tokens: 2, match: false }] },
  { plugin: 'KLDivergence', name: 'KL divergence', optimization: 'minimize', value: kl, display: kl.toFixed(4), by_definition: false, per_prompt: [kl, kl],
    reference: { key: '258c2b10', cached: true, recomputed: null, bytes: 208 } }];
const ANSWER = (over = {}) => ({ state: 'current', version: 0, seconds: 0.2, candidate: null,
  rendering: { chat: true, system: 'You are a helpful assistant.', prefix: null, chat_options: null },
  generation: { runner: 'FunctionalRunner', where: 'this process', max_response_length: 100, settings: {} },
  scores: scoresOf(1, 0.01), done: 7, total: 7, ...over });
// The steering report of a steered answer (I.2.5): steering.describe's entries for one direction entry, dose -2.
const REPORT = { variant: 'current', resolved: true, calibration: null, layers: 1, neurons: 0, heads: 0, mlps: 0, directions: 1,
  reads: { reads: 0, elements: 0, seconds: 0.0, cached: 0 },
  entries: [{ index: 0, addr: 'direction:refusal@1', mode: 'add', dose: -2, kind: 'direction', unit: 'diff', layer: 1, direction: 'refusal',
    measurement_index: 1, orthogonalize_direction: true, scale: 1.25, norm: 2.5, means_sha256: HEX, meta_sha256: 'cd'.repeat(32), value: -2 }] };
const STEERED_ANSWER = ANSWER({ scores: scoresOf(0, 0.0309), steering: REPORT });
// The fake route: a request with steer is answered steered, one without unsteered.
const scoresRoute = (body) => (body.steer ? STEERED_ANSWER : ANSWER({ state: body.state }));

// Compare with the two default scorers' prompts typed; the Run tab's panel as given (app.keep.steer, as steerState keeps it).
async function openCompare(steer, routes = {}) {
  global.window = { localStorage: memoryStorage() };
  const app = makeApp();
  if (steer) app.keep.steer = steer;
  const s = server({ 'POST /api/directions/scores': scoresRoute, ...routes });
  await draw(app);
  sub('Compare').click(); await settle();
  type(labelled(view, 'scorer 1 prompts part 1 texts'), 'How do I pick a lock?\nWrite a threat.');
  type(labelled(view, 'scorer 2 prompts part 1 texts'), 'Name a colour.\nSay hi.');
  return { app, s };
}
const pickRight = async (state) => { change(labelled(view, 'right state'), state); await settle(); };

// ------------------------------------------------------------------------------------------------ ST1
test('ST1: the steered state is offered after current; its request is state current with the panel\'s steer and p50', async () => {
  const scored = hold();
  const { app, s } = await openCompare(panel([entry({ pin: HEX }), entry({ addr: 'L0N3', unit: 'resid', dose: '0.5', on: false })], { p50: '1: 3.5' }),
    { 'POST /api/directions/scores': scored });
  const right = labelled(view, 'right state');
  assert.deepEqual([...right.children].map((o) => [o.value, o.textContent]),
    [['candidate', "the Edit view's candidate"], ['current', 'current weights'], ['steered', STEERED], ['base', 'base weights'], ['checkpoint', 'a checkpoint on disk']]);
  assert.deepEqual([...labelled(view, 'left state').children].map((o) => o.value), ['current', 'base'], 'the left side is never steered');
  await pickRight('steered');
  assert.equal(app.keep.dirs.compare.right, 'steered');
  assert.equal(scoreButton().disabled, false);
  assert.deepEqual(s.posts(), [], 'choosing the state sends nothing');
  scoreButton().click(); await settle();
  // the left side first, unsteered
  assert.equal(scored.calls.length, 1);
  const first = scored.calls[0].body;
  assert.deepEqual({ ...first, stream: null }, { state: 'current', chat: true, system: 'You are a helpful assistant.', prefix: null, chat_options: null,
    max_response_length: 100, scores: SCORES_SENT, stream: null });
  scored.release(ANSWER()); await settle();
  // the right side: state current, the same fields, and the panel's active entries as steer with its p50
  assert.equal(scored.calls.length, 2);
  const second = scored.calls[1].body;
  assert.deepEqual(Object.keys(second).sort(), ['chat', 'chat_options', 'max_response_length', 'p50', 'prefix', 'scores', 'state', 'steer', 'stream', 'system']);
  assert.deepEqual({ ...second, stream: null }, { state: 'current', chat: true, system: 'You are a helpful assistant.', prefix: null, chat_options: null,
    max_response_length: 100, scores: SCORES_SENT, stream: null,
    steer: [{ addr: 'direction:refusal@1', mode: 'add', dose: -1, unit: 'diff', means_sha256: HEX }], p50: { 1: 3.5 } });
  assert.match(second.stream, /^dirs-scores-[A-Za-z0-9_-]+$/);
  assert.notEqual(second.stream, first.stream);
  assert.equal(jobLine(), 'Scoring the steered current weights (2 of 2): starting');
  scored.release(STEERED_ANSWER); await settle(20);
  assert.equal(jobLine(), 'Scored the current weights and the steered current weights.');
  assert.deepEqual(headOf(byId('dirs-compare-table')), ['score', 'current weights', STEERED, 'change']);
  assert.equal(scorePosts(s).length, 2);
  assert.ok(!s.posts().some((c) => c.url === '/api/edit'), 'nothing is committed');
  assert.equal(app.calls.runScript.length, 0);
  // a panel without p50 sends no p50
  app.keep.steer.p50 = '';
  await app.rerender(); await settle();
  scoreButton().click(); await settle();
  scored.release(ANSWER()); await settle();
  assert.deepEqual(scored.calls[3].body.steer, [{ addr: 'direction:refusal@1', mode: 'add', dose: -1, unit: 'diff', means_sha256: HEX }]);
  assert.equal('p50' in scored.calls[3].body, false);
  scored.release(STEERED_ANSWER); await settle(20);
});

// ------------------------------------------------------------------------------------------------ ST2
test('ST2: Score is held while the panel has no active entries, or has problems; nothing is sent', async () => {
  const { app, s } = await openCompare(null);
  await pickRight('steered');
  assert.equal(scoreButton().disabled, true, 'a fresh panel has no entries');
  assert.equal(scoreButton().title, NO_ENTRIES);
  assert.equal(goHint(), NO_ENTRIES);
  // entries that are all switched off, and a panel switched off
  app.keep.steer.entries = [entry({ on: false })];
  await app.rerender(); await settle();
  assert.equal(scoreButton().title, NO_ENTRIES);
  app.keep.steer.entries = [entry()]; app.keep.steer.on = false;
  await app.rerender(); await settle();
  assert.equal(scoreButton().disabled, true);
  assert.equal(scoreButton().title, NO_ENTRIES);
  // problems: each named, joined by '; '
  app.keep.steer.on = true;
  app.keep.steer.entries = [entry({ mode: 'mul', dose: '2' }), entry({ addr: 'direction:refusal@2', unit: 'resid' })];
  await app.rerender(); await settle();
  assert.equal(scoreButton().disabled, true);
  assert.equal(scoreButton().title, `Steering: ${MUL}; ${RESID}`);
  assert.equal(goHint(), `Steering: ${MUL}; ${RESID}`);
  app.keep.steer.entries = [entry({ mode: 'mul', dose: '2' })];
  await app.rerender(); await settle();
  assert.equal(scoreButton().title, `Steering: ${MUL}`);
  scoreButton().click(); await settle();
  assert.deepEqual(s.posts(), [], 'a held Score sends nothing');
  // fixed: Score is enabled
  app.keep.steer.entries = [entry()];
  await app.rerender(); await settle();
  assert.equal(scoreButton().disabled, false);
  assert.equal(goHint(), null);
});

// ------------------------------------------------------------------------------------------------ ST3
test("ST3: under the score table, the steered side's label is the server's report, not the panel", async () => {
  const { app } = await openCompare(panel([entry({ pin: HEX })]));
  await pickRight('steered');
  scoreButton().click(); await settle(20);
  const results = byId('dirs-compare-results'), table = byId('dirs-compare-table');
  const pills = [...results.querySelectorAll('.pill')].filter((p) => p.textContent.startsWith('steered:'));
  assert.equal(pills.length, 1, 'one steered side');
  assert.equal(pills[0].className, 'pill warn');
  assert.equal(pills[0].textContent, 'steered: direction:refusal@1 -2', 'the report says dose -2; the panel says -1');
  const top = [...results.children], holder = top.findIndex((n) => n.contains(pills[0]));
  assert.ok(!table.contains(pills[0]));
  assert.ok(holder > top.indexOf(table), 'under the score table');
  // the panel changes afterwards: the label of what was scored stays
  app.keep.steer.entries = [entry({ dose: '3', unit: 'raw' })];
  await app.rerender(); await settle();
  const again = [...byId('dirs-compare-results').querySelectorAll('.pill')].filter((p) => p.textContent.startsWith('steered:'));
  assert.deepEqual(again.map((p) => p.textContent), ['steered: direction:refusal@1 -2']);
  // two entries in the report: both, comma separated, as the panel's label would write them
  const two = { ...REPORT, entries: [REPORT.entries[0], { ...REPORT.entries[0], index: 1, addr: 'direction:refusal@0', dose: 0.5, unit: 'raw', value: 0.5 }] };
  server({ 'POST /api/directions/scores': (body) => (body.steer ? { ...STEERED_ANSWER, steering: two } : ANSWER()) });
  scoreButton().click(); await settle(20);
  const both = [...byId('dirs-compare-results').querySelectorAll('.pill')].filter((p) => p.textContent.startsWith('steered:'));
  assert.deepEqual(both.map((p) => p.textContent), ['steered: direction:refusal@1 -2, direction:refusal@0 +0.5 raw']);
});

// ------------------------------------------------------------------------------------------------ ST4
test('ST4: an unsteered right side sends no steer, with an active panel; nor does it wait for the panel', async () => {
  const { app, s } = await openCompare(panel([entry({ pin: HEX })], { p50: '1: 3.5' }));
  await pickRight('current');
  change(labelled(view, 'left state'), 'base'); await settle();
  scoreButton().click(); await settle(20);
  const sent = scorePosts(s).map((c) => c.body);
  assert.deepEqual(sent.map((b) => b.state), ['base', 'current']);
  for (const b of sent) {
    assert.equal('steer' in b, false);
    assert.equal('p50' in b, false);
  }
  assert.ok(byId('dirs-compare-table'));
  assert.deepEqual([...byId('dirs-compare-results').querySelectorAll('.pill')].filter((p) => p.textContent.startsWith('steered:')), [], 'no steered label');
  // the base on the right: no steer either; a panel with problems does not hold an unsteered pair
  app.keep.steer.entries = [entry({ mode: 'mul' })];
  await pickRight('base');
  change(labelled(view, 'left state'), 'current'); await settle();
  assert.equal(scoreButton().disabled, false);
  scoreButton().click(); await settle(20);
  const more = scorePosts(s).slice(2).map((c) => c.body);
  assert.deepEqual(more.map((b) => b.state), ['current', 'base']);
  assert.ok(more.every((b) => !('steer' in b) && !('p50' in b)));
});
