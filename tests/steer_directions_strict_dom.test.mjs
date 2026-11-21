// Steering along a measured direction in the page, against the strict fake DOM: the target direction:<name>@<k> in
// the Run tab's steering panel (what it sends, what it keeps across a reload, recipe lines), the Layers view's steer
// button, and the Probe tab's neuron list refusing a direction. Expectations from docs/STEERING.md and
// docs/DIRECTIONS.md (the Directions tab, Layers); made-up names and digests. No server, model or network.
import test from 'node:test';
import assert from 'node:assert/strict';
import { installStrictDom } from './strict_dom.mjs';
import { META, STATS, server, makeApp, draw, settle, labelled, button, nodes, type, change, byId } from './directions_fixtures.mjs';
import { parseTarget, parseRecipe, steerRequest, saveSteer, loadSteer, steerPanel, entryReach } from '../web/js/steer.js';
import { probeView } from '../web/js/probe.js';

const { view } = installStrictDom();
const HEX = '5f0c' + 'ab'.repeat(30);                            // 64 lowercase hex digits
const HELP = 'write a neuron L34N3838, a head L11H3, an MLP blocks.12.mlp or a group group:name, a direction direction:name@12';
const RESID = 'unit resid is not available for a direction: choose difference of the means or raw';

function memoryStorage() {
  const m = new Map();
  return { m, getItem: (k) => (m.has(k) ? m.get(k) : null), setItem: (k, v) => m.set(k, String(v)), removeItem: (k) => m.delete(k) };
}
// A fresh browser storage for each test (steerState reads it when the panel state is not in app.keep).
const freshStorage = () => { const s = memoryStorage(); global.window = { localStorage: s }; return s; };
const entry = (over = {}) => ({ addr: 'direction:refusal@1', mode: 'add', dose: '-1', unit: 'diff', on: true, ...over });
const state = (...entries) => ({ on: true, recipe: '', p50: '', entries });
const options = (select) => [...select.children].map((o) => ({ value: o.value, text: o.textContent, disabled: !!o.disabled, selected: !!o.selected }));

// ------------------------------------------------------------------------------------------------ SD1
test('SD1: a direction target parses', () => {
  assert.deepEqual(parseTarget('direction:refusal@12'), { addr: 'direction:refusal@12', kind: 'direction' });
  assert.deepEqual(parseTarget('  direction:my-set_2.v1@0 '), { addr: 'direction:my-set_2.v1@0', kind: 'direction' });
  assert.deepEqual(parseTarget('Direction:refusal@3'), { addr: 'direction:refusal@3', kind: 'direction' }, 'the prefix is written in lower case');
  for (const bad of ['direction:a b@1', 'direction:x@01', 'direction:x', 'direction:@1', 'direction:-x@1', 'direction:x@1.5']) {
    assert.equal(parseTarget(bad).error, `unknown target "${bad}": ${HELP}`, bad);
  }
  assert.equal(parseTarget('').error, `no target: ${HELP}`);
  assert.deepEqual(entryReach(entry({ addr: 'direction:refusal@12' })), { neurons: 0, text: 'a direction entering layer 12', problem: '' });
});

// ------------------------------------------------------------------------------------------------ SD2
test('SD2: what the panel sends for a direction', () => {
  const r = steerRequest(state(entry({ pin: HEX })));
  assert.deepEqual(r.problems, []);
  assert.deepEqual(r.fields, { steer: [{ addr: 'direction:refusal@1', mode: 'add', dose: -1, unit: 'diff', means_sha256: HEX }] });
  assert.equal(r.label, 'direction:refusal@1 -1');
  // ortho false adds orthogonalize_direction: false; true (the server's default) and no pin send neither
  assert.deepEqual(steerRequest(state(entry({ pin: HEX, ortho: false }))).fields.steer,
    [{ addr: 'direction:refusal@1', mode: 'add', dose: -1, unit: 'diff', means_sha256: HEX, orthogonalize_direction: false }]);
  assert.deepEqual(steerRequest(state(entry({ ortho: true }))).fields.steer, [{ addr: 'direction:refusal@1', mode: 'add', dose: -1, unit: 'diff' }]);
  const raw = steerRequest(state(entry({ unit: 'raw', dose: '0.25' })));
  assert.deepEqual(raw.fields.steer, [{ addr: 'direction:refusal@1', mode: 'add', dose: 0.25, unit: 'raw' }]);
  assert.equal(raw.label, 'direction:refusal@1 +0.25 raw');
  // a neuron entry never sends the direction fields (the server refuses unknown fields)
  assert.deepEqual(steerRequest(state(entry({ addr: 'L1N2', unit: 'resid', pin: HEX, ortho: false }))).fields.steer,
    [{ addr: 'L1N2', mode: 'add', dose: -1, unit: 'resid' }]);
  // mul and resid are problems: nothing is sent
  const mul = steerRequest(state(entry({ mode: 'mul' })));
  assert.equal(mul.fields, null);
  assert.deepEqual(mul.problems, ['direction:refusal@1: a direction is added: choose add']);
  const resid = steerRequest(state(entry({ unit: 'resid' })));
  assert.equal(resid.fields, null);
  assert.deepEqual(resid.problems, [`direction:refusal@1: ${RESID}`]);

  // the panel: the unit select offers diff and raw; multiply is disabled; the reach is said
  freshStorage();
  const st = state(entry({ pin: HEX }), entry({ addr: '', unit: 'resid', dose: '0.5' }));
  const app = { keep: { steer: st }, state: { open: { name: 'fixture' } } };
  const box = steerPanel(app);
  assert.deepEqual(options(labelled(box, 'steer entry 1 unit')), [{ value: 'diff', text: 'difference of the means', disabled: false, selected: true },
    { value: 'raw', text: 'raw (unit direction)', disabled: false, selected: false }]);
  assert.deepEqual(options(labelled(box, 'steer entry 1 mode')).map((o) => [o.value, o.disabled]), [['add', false], ['mul', true]]);
  assert.equal(labelled(box, 'steer entry 1 reach').textContent, 'pushes a direction entering layer 1');
  assert.equal(labelled(box, 'steering header').textContent, 'not steered (fix 1 entry)', 'the empty second entry');
  // retyping the index keeps the pin; another measurement drops it
  const retype = (i, text) => { const t = labelled(box, `steer entry ${i} target`); type(t, text); change(t); };
  retype(1, 'direction:refusal@0');
  assert.deepEqual(st.entries[0], entry({ addr: 'direction:refusal@0', pin: HEX }));
  retype(1, 'direction:other@0');
  assert.deepEqual(st.entries[0], entry({ addr: 'direction:other@0' }));
  // a new entry typed as a direction is added in diff; typed back as a neuron, in resid
  retype(2, 'direction:refusal@1');
  assert.equal(st.entries[1].mode, 'add');
  assert.equal(st.entries[1].unit, 'diff');
  assert.equal(labelled(box, 'steering header').textContent, 'the edited model is steered: direction:other@0 -1, direction:refusal@1 +0.5');
  retype(2, 'L3N4');
  assert.equal(st.entries[1].unit, 'resid');
  retype(2, 'L3H4');
  retype(2, 'direction:refusal@1');
  assert.equal(st.entries[1].mode, 'add', 'a head was multiplied; a direction is added');
});

// ------------------------------------------------------------------------------------------------ SD3
test('SD3: the panel keeps pin, ortho and diff across a reload', () => {
  const s = freshStorage();
  const st = state(entry({ pin: HEX, ortho: false }), entry({ addr: 'direction:refusal@0', unit: 'raw', ortho: true, dose: '2' }),
    entry({ addr: 'L1N2', unit: 'diff', dose: '0.5' }), entry({ addr: 'L1N3', unit: 'raw', dose: '0.5', on: false }),
    entry({ addr: 'direction:refusal@2', unit: 'resid' }));
  assert.equal(saveSteer('fixture', st), true);
  const stored = JSON.parse(s.getItem('wsp.steer.v1.fixture'));
  assert.deepEqual(stored.entries[0], entry({ pin: HEX, ortho: false }));
  assert.deepEqual(stored.entries[2], entry({ addr: 'L1N2', unit: 'diff', dose: '0.5' }), 'saved as it is; no pin or ortho for an entry without them');
  const back = loadSteer('fixture');
  assert.deepEqual(back.entries, [entry({ pin: HEX, ortho: false }), entry({ addr: 'direction:refusal@0', unit: 'raw', ortho: true, dose: '2' }),
    entry({ addr: 'L1N2', unit: 'resid', dose: '0.5' }), entry({ addr: 'L1N3', unit: 'raw', dose: '0.5', on: false }),
    entry({ addr: 'direction:refusal@2', unit: 'diff' })]);
  // what the reloaded panel sends is what the saved one sent (for the direction entries)
  assert.deepEqual(steerRequest(state(back.entries[0])).fields, steerRequest(state(st.entries[0])).fields);
  // a pin that is not text is not kept; nor an ortho that is not true or false
  s.setItem('wsp.steer.v1.odd', JSON.stringify({ on: true, entries: [{ addr: 'direction:refusal@1', mode: 'add', dose: '-1', unit: 'diff', pin: 7, ortho: 'no' }] }));
  assert.deepEqual(loadSteer('odd').entries, [entry()]);
});

// ------------------------------------------------------------------------------------------------ SD4
const layers = async (stats, over = {}) => {
  freshStorage();
  const s = server({ 'GET /api/directions/refusal': { ...META, stats, means_sha256: HEX, ...over } });
  const app = makeApp();
  await draw(app);
  change(labelled(view, 'measurement'), 'refusal'); await settle();
  button(labelled(view, 'Directions view'), 'Layers').click(); await settle();
  return { s, app };
};
const steerCells = () => [...byId('dirs-layers-table').children].map((r) => [...r.children].at(-1));

test('SD4: the Layers view adds the entry', async () => {
  const { s, app } = await layers(STATS);
  const cells = steerCells();
  assert.equal(cells[0].textContent, 'steer', 'the last column');
  const buttons = nodes(byId('dirs-layers-table')).filter((n) => n.tag === 'button');
  assert.deepEqual(buttons.map((b) => [b.getAttribute('aria-label'), b.textContent, !!b.disabled]),
    [['steer along index 0', 'steer -1', false], ['steer along index 1', 'steer -1', false]], 'no button for index n');
  assert.equal(buttons[1].title, "adds direction:refusal@1 with dose -1 to the Run tab's steering panel (-1 removes the behaviour, +1 adds it); nothing runs");
  assert.equal(cells[3].textContent, '–');
  assert.equal(cells[3].title, 'index 2 is after the final norm; no layer reads it');
  const before = s.calls.length;
  labelled(view, 'steer along index 1').click(); await settle();
  assert.deepEqual(app.keep.steer.entries, [entry({ pin: HEX, ortho: true })]);
  assert.equal(byId('dirs-layers-note').textContent, "direction:refusal@1 -1 added to the Run tab's steering panel; Run, Chat and the Probe tab's Next token use it.");
  assert.deepEqual(s.posts(), [], 'nothing is sent');
  assert.deepEqual(s.calls.slice(before).filter((c) => !c.url.startsWith('/api/directions')), [], 'no request outside the tab\'s own reads');
  assert.deepEqual(steerRequest(app.keep.steer).fields, { steer: [{ addr: 'direction:refusal@1', mode: 'add', dose: -1, unit: 'diff', means_sha256: HEX }] });
  assert.deepEqual(JSON.parse(global.window.localStorage.getItem('wsp.steer.v1.fixture')).entries, [entry({ pin: HEX, ortho: true })], 'kept for the Run tab');
  // the view's orthogonalize_direction goes with the entry
  app.keep.dirs.params.ablation.orthogonalize_direction = false;
  await app.rerender(); await settle();
  labelled(view, 'steer along index 0').click(); await settle();
  assert.deepEqual(app.keep.steer.entries[1], entry({ addr: 'direction:refusal@0', pin: HEX, ortho: false }));
  assert.deepEqual(steerRequest(app.keep.steer).fields.steer[1],
    { addr: 'direction:refusal@0', mode: 'add', dose: -1, unit: 'diff', means_sha256: HEX, orthogonalize_direction: false });
  assert.deepEqual(s.posts(), []);
});

test('SD4: an unusable index before the final norm has a disabled button; an answer without means_sha256 adds no pin', async () => {
  const stats = [STATS[0], { ...STATS[1], usable: false }, { ...STATS[1], index: 2 }, { ...STATS[2], index: 3 }];
  const { s, app } = await layers(stats, { means_sha256: undefined });
  const buttons = nodes(byId('dirs-layers-table')).filter((n) => n.tag === 'button');
  assert.deepEqual(buttons.map((b) => [b.getAttribute('aria-label'), !!b.disabled, b.title]),
    [['steer along index 0', false, "adds direction:refusal@0 with dose -1 to the Run tab's steering panel (-1 removes the behaviour, +1 adds it); nothing runs"],
      ['steer along index 1', true, 'index 1 is not usable'],
      ['steer along index 2', false, "adds direction:refusal@2 with dose -1 to the Run tab's steering panel (-1 removes the behaviour, +1 adds it); nothing runs"]]);
  assert.equal(steerCells()[4].title, 'index 3 is after the final norm; no layer reads it');
  labelled(view, 'steer along index 1').click(); await settle();
  assert.equal(app.keep.steer, undefined, 'a disabled button adds nothing');
  labelled(view, 'steer along index 2').click(); await settle();
  assert.deepEqual(app.keep.steer.entries, [entry({ addr: 'direction:refusal@2', ortho: true })]);
  assert.deepEqual(s.posts(), []);
});

// ------------------------------------------------------------------------------------------------ SD5
test('SD5: recipe lines', () => {
  const read = (text) => parseRecipe(text);
  assert.deepEqual(read('direction:refusal@2 -0.5'), { entries: [{ addr: 'direction:refusal@2', mode: 'add', dose: -0.5, unit: 'diff' }], problems: [] });
  assert.deepEqual(read('direction:refusal@2 -0.5 raw').entries, [{ addr: 'direction:refusal@2', mode: 'add', dose: -0.5, unit: 'raw' }]);
  assert.deepEqual(read('direction:refusal@2 -0.5 diff').entries, [{ addr: 'direction:refusal@2', mode: 'add', dose: -0.5, unit: 'diff' }]);
  assert.deepEqual(read('direction:refusal@2 -0.5 resid'), { entries: [], problems: [{ item: 'direction:refusal@2 -0.5 resid', reason: RESID }] });
  assert.deepEqual(read('direction:refusal@2 x2').problems, [{ item: 'direction:refusal@2 x2', reason: 'a direction is added: write a signed dose, e.g. -0.5' }]);
  assert.deepEqual(read('L5N9 +1 diff').problems,
    [{ item: 'L5N9 +1 diff', reason: 'diff (the difference of the means) is a unit of a direction only; a neuron is pushed in resid or raw' }]);
  assert.deepEqual(read('L5N9 +1').entries, [{ addr: 'L5N9', mode: 'add', dose: 1, unit: 'resid' }], 'a neuron stays in resid');
  const mixed = read('direction:refusal@1 +1, L5N9 -0.5\ndirection:refusal@01 -1');
  assert.deepEqual(mixed.entries, [{ addr: 'direction:refusal@1', mode: 'add', dose: 1, unit: 'diff' }, { addr: 'L5N9', mode: 'add', dose: -0.5, unit: 'resid' }]);
  assert.equal(mixed.problems.length, 1);
  assert.match(mixed.problems[0].reason, /^unknown target: /);
});

// ------------------------------------------------------------------------------------------------ SD6
test('SD6: the Probe tab refuses a direction in its neuron list, before anything is sent', async () => {
  freshStorage();
  const s = server();
  const app = { sel: 'model', tab: 'Probe', variant: 'current', keep: {}, state: { open: { name: 'fixture', kind: 'wsp' }, status: { can_run: true, version: 0 } },
    rerender() {}, select() {} };
  app.keep.probe = { view: 'reads', reads: { text: '', neurons: '' }, fires: { text: '', k: 20, sign: 'both' }, next: { prompt: '', source: 'panel', typed: '', top: 10 },
    contrast: { a: '', b: '', k: 20 } };
  const root = document.createElement('div');
  view.replaceChildren(root);
  await probeView(root, app); await settle();
  type(labelled(root, 'reads text'), 'abc');
  type(labelled(root, 'reads neurons'), 'L0N1, direction:refusal@1, L3H1');
  button(root, 'Read').click(); await settle();
  assert.match(root.textContent, /direction:refusal@1 is not a neuron\. L3H1 is not a neuron\./);
  assert.deepEqual(s.posts(), []);
});

// ------------------------------------------------------------------------------------------------ SD8
// The panel's Check table: a direction entry by the length of its vector and what one unit is (norm = |dose| × scale,
// value = the dose); a neuron entry beside it as before. Expected cells typed by hand (fmt of 0.25 is 0.25).
const HEAD = ['entry', 'constant added', 'output column length', 'layer residual norm (p50)', 'p50 source'];
const checkRows = (box) => {
  const table = nodes(box).find((n) => n.tag === 'table' && n.textContent.includes('constant added'));
  return [...table.children].map((r) => [...r.children].map((c) => c.textContent));
};
async function checkWith(entries, answerEntries) {
  freshStorage();
  const s = server({ 'POST /api/steer/resolve': { ok: true, entries: answerEntries, calibration: { needed: false, cached: false, ran: false }, reads: { reads: 0 } } });
  const app = { keep: { steer: state(...entries) }, state: { open: { name: 'fixture' } } };
  const box = steerPanel(app);
  view.replaceChildren(box);
  button(box, 'Check').click(); await settle();
  return { s, box };
}
const NEURON = { index: 1, addr: 'L1N3', mode: 'add', dose: 0.75, kind: 'neurons', unit: 'resid', layer: 1, neurons: 1, p50: { 1: 4 }, norm: 0.5, value: 6, source: 'measured' };
const DIRECTION = { index: 0, addr: 'direction:refusal@1', mode: 'add', dose: -0.5, kind: 'direction', unit: 'diff', layer: 1, direction: 'refusal',
  measurement_index: 1, orthogonalize_direction: true, scale: 0.5, norm: 0.25, means_sha256: HEX, meta_sha256: 'cd'.repeat(32), value: -0.5 };

test('SD8: Check describes a direction entry by its vector', async () => {
  const { s, box } = await checkWith([entry({ dose: '-0.5', pin: HEX }), entry({ addr: 'L1N3', unit: 'resid', dose: '0.75' })], [DIRECTION, NEURON]);
  assert.equal(s.last('POST', '/api/steer/resolve').body.steer.length, 2);
  const rows = checkRows(box);
  assert.deepEqual(rows[0], HEAD, 'the header is unchanged');
  assert.deepEqual(rows[1], ['direction:refusal@1 -0.5', 'vector of length 0.25', '–', '–', 'unit diff: one unit is 0.5']);
  assert.deepEqual(rows[2], ['L1N3 +0.75', '6', '0.5', '4', 'measured'], 'a neuron entry as before');
  assert.equal(rows.length, 3);
});

test('SD8: Check describes a raw direction entry: the dose is the length', async () => {
  const raw = { ...DIRECTION, addr: 'direction:refusal@0', dose: 2, unit: 'raw', layer: 0, measurement_index: 0, scale: 1, norm: 2, value: 2 };
  const { box } = await checkWith([entry({ addr: 'direction:refusal@0', unit: 'raw', dose: '2' })], [raw]);
  const rows = checkRows(box);
  assert.deepEqual(rows[0], HEAD);
  assert.deepEqual(rows[1], ['direction:refusal@0 +2 raw', 'vector of length 2', '–', '–', 'raw (the dose is the length)']);
});
