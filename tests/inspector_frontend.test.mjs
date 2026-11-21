// Model-free inspector selections UI: actions come from the server's plan, refusals and failures are shown where
// the person clicked, patch actions wait visibly behind a running generation, breadcrumbs navigate, selections
// are saved in one request, and the before/after view shows base → current. Fake DOM and network only.
import test from 'node:test';
import assert from 'node:assert/strict';

class ClassList {
  constructor(el) { this.el = el; }
  add(c) { const s = new Set((this.el.className || '').split(' ').filter(Boolean)); s.add(c); this.el.className = [...s].join(' '); }
  remove(c) { this.el.className = (this.el.className || '').split(' ').filter((x) => x && x !== c).join(' '); }
  contains(c) { return (this.el.className || '').split(' ').includes(c); }
}
class Element {
  constructor(tag) { this.tag = tag; this.nodeType = 1; this.children = []; this.style = {}; this.events = {}; this.disabled = false; this.value = ''; this._text = ''; this.className = ''; this.classList = new ClassList(this); }
  get textContent() { return this._text; }
  set textContent(v) { this._text = String(v); this.children = []; }        // as in a browser: replaces the children
  append(...kids) { this.children.push(...kids); }
  replaceChildren(...kids) { this.children = kids; }
  addEventListener(name, fn) { this.events[name] = fn; }
  setAttribute(k, v) { this[k] = v; }
  click() { return this.events.click?.({ target: this, stopPropagation() {} }); }
  querySelectorAll(sel) { const tags = sel.split(',').map((s) => s.trim()); return walk(this).filter((n) => tags.includes(n.tag)); }
  getContext() { return { createImageData: (w, h) => ({ data: new Uint8ClampedArray(4 * w * h) }), putImageData() {}, drawImage() {}, strokeRect() {} }; }
  getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 100 }; }
  get offsetWidth() { return 10; } get offsetHeight() { return 10; }
}
global.document = { createElement: (t) => new Element(t), createElementNS: (ns, t) => new Element(t), createTextNode: (text) => ({ nodeType: 3, textContent: text }),
  body: new Element('body'), documentElement: { dataset: {} }, querySelector: () => null };
global.window = { devicePixelRatio: 1, innerWidth: 1000, innerHeight: 800 };
global.atob = (b) => Buffer.from(b, 'base64').toString('binary');

const { actionsRow, breadcrumb, groupSummary, saveSelection, compareView, patchAction, trackGeneration, generationRunning, busyText,
  settleGeneration, STALE_MS, defaultName, groupsEvent, compensatesNote, trackEditProgress, startGeneration, eventsLost,
  beginEdit, endEdit } = await import('../web/js/selection.js');
const { api, generationStarted, generationSettled } = await import('../web/js/util.js');
const { rowsTable } = await import('../web/js/atlas.js');

const walk = (root) => [root, ...(root.children || []).flatMap((c) => (c && c.nodeType === 1 ? walk(c) : []))];
const text = (node) => (node.textContent || '') + (node.children || []).map((c) => (c.nodeType === 1 ? text(c) : c.textContent || '')).join('');
const button = (root, label) => walk(root).find((n) => n.tag === 'button' && text(n) === label);
const status = (root) => walk(root).find((n) => n.role === 'status');
const tick = () => new Promise((r) => setImmediate(r));
const wait = (ms) => new Promise((r) => setTimeout(r, ms));
const response = (data, status = 200) => ({ ok: status < 400, status, statusText: 'x', headers: { get: () => 'application/json' }, json: async () => data });
const b64 = (arr) => Buffer.from(new Float32Array(arr).buffer).toString('base64');

function app(extra = {}) {
  return { keep: {}, state: { groups: {}, status: { workspace_id: 'w', version: 0 } }, selected: [], scripts: [], console: [],
    select(a) { this.selected.push(a); }, setConsole(t) { this.console.push(t); },
    async runScript(script, opts) { this.scripts.push([script, opts]); return { ok: true, committed: true, n_changed: 16, patch: 'P1' }; }, ...extra };
}
const enabled = { scale: { enabled: true, reason: 'multiplies the selected weights (16 weights)' }, ablate: { enabled: true, reason: 'sets the selected weights (16 weights) to zero' },
  console: { enabled: true, reason: 'puts the scaling script in the console' }, revert: { enabled: false, reason: 'nothing here has been edited' }, weights: 16, other: [], n_other: 0 };
const sliceNode = { addr: 'model.layers.34.mlp.down_proj.weight[:, 3838]', kind: 'tensor', actions: enabled,
  crumbs: [{ addr: 'model', label: 'model' }, { addr: 'blocks.34', label: 'block 34' }, { addr: 'blocks.34.mlp', label: 'mlp' },
    { addr: 'model.layers.34.mlp.down_proj.weight', label: 'mlp.down_proj.weight' }, { addr: 'model.layers.34.mlp.down_proj.weight[:, 3838]', label: '[:, 3838]' }],
  jumps: [{ addr: 'blocks.34.mlp.neuron.3838', label: 'L34N3838', kind: 'unit' }] };

test('scale on a slice asks the server for the script and applies it as one edit', async () => {
  const a = app(); const sent = [];
  global.fetch = async (url, o) => { sent.push([url, JSON.parse(o.body)]); return response({ script: 'P("x")[:, 3838] *= 0.5', neurons: 0, heads: 0, weights: 16, other: [] }); };
  const row = actionsRow(a, sliceNode);
  assert.equal(button(row, 'scale ×').disabled, false);
  assert.equal(button(row, 'revert to base').disabled, true);
  assert.equal(button(row, 'revert to base').title, 'nothing here has been edited');
  await button(row, 'scale ×').click(); await tick();
  assert.deepEqual(sent, [['/api/selection/script', { addr: sliceNode.addr, action: 'scale', factor: 0.5 }]]);
  assert.deepEqual(a.scripts, [['P("x")[:, 3838] *= 0.5', { message: `scale ${sliceNode.addr} by 0.5` }]]);
  assert.match(text(status(row)), /Done: 16 weights changed, as patch P1/);
  assert.equal(button(row, 'scale ×').disabled, false);
  assert.equal(a.keep.lastAction.addr, sliceNode.addr);                 // survives the redraw after the edit
  const again = actionsRow(a, sliceNode);
  assert.match(text(status(again)), /Done: 16 weights changed/);
  assert.match(text(status(actionsRow(a, sliceNode))), /Done/);         // a second redraw (the patch event) keeps it
  assert.equal(text(status(actionsRow(a, { ...sliceNode, addr: 'L1N2' }))), '');
});

test('an action that cannot be done is disabled and says why in plain words', () => {
  const why = 'This is 267,386,880 weights (about 1.1 GB as FP32 edits), too large for a weight patch (the limit is 60,000,000 weights). Select fewer neurons or a smaller slice.';
  const off = { enabled: false, reason: why };
  const row = actionsRow(app(), { ...sliceNode, kind: 'component', actions: { scale: off, ablate: off, console: off, revert: { enabled: false, reason: 'nothing here has been edited' }, n_other: 0 } });
  for (const l of ['scale ×', 'ablate', '→ console', 'revert to base']) assert.equal(button(row, l).disabled, true, l);
  assert.match(text(row), /Cannot edit: .*too large for a weight patch/);
});

test('a refusal or a script that changes nothing is shown where the person clicked', async () => {
  global.fetch = async () => response({ error: 'Nothing in this selection has been edited, so there is nothing to revert.', refused: true }, 400);
  const a = app();
  const row = actionsRow(a, { ...sliceNode, actions: { ...enabled, revert: { enabled: true, reason: 'x' } } });
  await button(row, 'revert to base').click(); await tick();
  assert.equal(a.scripts.length, 0);
  assert.match(text(status(row)), /nothing to revert/);
  assert.match(status(row).className, /err/);
  global.fetch = async () => response({ script: 's', weights: 1, other: [] });
  a.runScript = async () => ({ ok: true, committed: false, note: 'the script changed no weights' });
  await button(row, 'zero').click(); await tick();
  assert.equal(text(status(row)), 'the script changed no weights');
  a.runScript = async () => ({ ok: false, error: 'line 1: boom' });
  await button(row, 'zero').click(); await tick();
  assert.equal(text(status(row)), 'line 1: boom');
});

test('→ console puts the server-built script in the console', async () => {
  global.fetch = async (url, o) => response({ script: `scale_neurons(34, [3838], ${JSON.parse(o.body).factor})`, weights: 1, other: [] });
  const a = app();
  const row = actionsRow(a, sliceNode);
  await button(row, '→ console').click(); await tick();
  assert.deepEqual(a.console, ['scale_neurons(34, [3838], 0.5)']);
});

test('the breadcrumb navigates each level and jumps from a slice to its neuron', () => {
  const a = app();
  const nav = breadcrumb(a, sliceNode);
  assert.equal(text(nav).replace(/\s+/g, ' '), 'model›block 34›mlp›mlp.down_proj.weight›[:, 3838]→ L34N3838');
  button(nav, 'block 34').click(); button(nav, 'mlp').click(); button(nav, '→ L34N3838').click();
  assert.deepEqual(a.selected, ['blocks.34', 'blocks.34.mlp', 'blocks.34.mlp.neuron.3838']);
  assert.equal(button(nav, '[:, 3838]'), undefined);                      // where you are is not a link
});

test('a selection page shows its size and a capped member list', () => {
  const g = groupSummary(app(), { selection: { neurons: 12000, heads: 2, tensors: 0, layers: [3, 4, 5, 9], weights: 1e6, by_layer: { 3: 6000, 9: 6000 },
    n_members: 450, members: Array.from({ length: 200 }, (_, i) => `L3N${i}`), other: [], n_other: 0 } });
  assert.match(text(g), /12\.0k neurons · 2 heads/);
  assert.match(text(g), /in 4 layers \(3–5, 9\)/);
  assert.match(text(g), /and 250 more/);
  assert.equal(walk(g).filter((n) => n.tag === 'button').length, 200);
});

test('saving a selection is one request; an existing name is added to', async () => {
  const sent = [];
  global.fetch = async (url, o) => { sent.push(JSON.parse(o.body)); return response({ groups: { red: { members: ['blocks.3.mlp.neurons.0-99'] } }, summary: { neurons: 100 } }); };
  let refreshed = 0;
  const a = app({ async refreshStatus() { refreshed++; } }); const st = new Element('span');
  await saveSelection(a, { selector: { word: ['red'] } }, { ask: () => 'red', status: st });
  assert.equal(refreshed, 1);                                              // reads are scoped to the new identity
  assert.deepEqual(sent[0], { name: 'red', selector: { word: ['red'] }, mode: 'replace' });
  assert.match(st.textContent, /Saved as selection "red": 100 neurons/);
  await saveSelection(a, { neurons: [[3, 5]] }, { ask: () => 'red', status: st });
  assert.equal(sent[1].mode, 'add');
  assert.match(st.textContent, /Added to selection "red"/);
  assert.equal(await saveSelection(a, { neurons: [] }, { ask: () => '' }), null);
  assert.equal(sent.length, 2);
});

test('atlas rows can be checked and saved as one selection without opening the row', async () => {
  const sent = [];
  global.fetch = async (url, o) => { sent.push([url, JSON.parse(o.body)]); return response({ groups: {}, summary: { neurons: 2 } }); };
  globalThis.prompt = () => 'picked';
  let renders = 0;
  const a = app({ rerender() { renders++; } });
  const st = {};
  const t = rowsTable([{ layer: 1, neuron: 2, mech: 'm' }, { layer: 3, neuron: 4, mech: 'm' }, { layer: 5, neuron: 6, mech: 'm' }], st, a);
  const boxes = walk(t).filter((n) => n.tag === 'input' && n.type === 'checkbox');
  assert.equal(boxes.length, 3);
  for (const i of [0, 1]) { boxes[i].events.click({ stopPropagation() {} }); boxes[i].events.change({ target: { checked: true } }); }
  const save = button(t, 'Save checked (2)');
  assert.ok(save && !save.disabled);
  await save.click(); await tick();
  assert.deepEqual(sent, [['/api/groups', { name: 'picked', neurons: [[1, 2], [3, 4]], mode: 'replace' }]]);
  assert.equal(renders, 0);
});

test('before/after shows base, current and difference, with base → current on hover', () => {
  const m = { grid: [2, 2], factor: [1, 1], region: [0, 2, 0, 2], row_offset: 0, col_offset: 0, row_role: 'resid', col_role: 'neuron', pool: 'mean',
    base: b64([1, 2, 3, 4]), current: b64([1, 4, 3, 8]), delta: b64([0, 2, 0, 4]), scale: 8, delta_scale: 4, dtype: 'Q4_K',
    stats: { base: { n: 4, norm: 5.477, min: 1, max: 4 }, current: { n: 4, norm: 9.95, min: 1, max: 8 }, delta: { n: 4, norm: 4.47, changed: 2, max_abs: 4 } } };
  let single = 0;
  const v = compareView(app(), m, { onSingle: () => single++ });
  const canvases = walk(v).filter((n) => n.tag === 'canvas');
  assert.equal(canvases.length, 3);
  assert.match(text(v), /base.*current.*difference/s);
  assert.match(text(v), /5\.477 → 9\.95/);
  assert.match(text(v), /1 … 4 → 1 … 8/);
  assert.match(text(v), /2 of 4 weights/);
  assert.match(text(v), /Stored as Q4_K: "base" is the reconstructed values/);
  canvases[1].events.mousemove({ clientX: 60, clientY: 10 });          // row 0, col 1
  const tipEl = document.body.children.at(-1);
  assert.match(tipEl.innerHTML, /<b>2 → 4<\/b>/);
  assert.match(tipEl.innerHTML, /Δ 2/);
  button(v, 'single view').click();
  assert.equal(single, 1);
});

test('generation events mark a running generation until it ends', () => {
  const a = app({ editWaitsForGeneration: true });
  assert.equal(generationRunning(a), false);
  trackGeneration(a, { type: 'generation', stream: 's1', variant: 'current', state: 'started' });
  assert.equal(generationRunning(a), true);
  assert.equal(busyText(a), 'waiting for the running generation…');
  trackGeneration(a, { type: 'generation', stream: 's1', variant: 'current', state: 'complete' });
  assert.equal(generationRunning(a), true);                              // the base model may follow at once
  assert.equal(generationRunning(a, Date.now() + 5000), false);
  trackGeneration(a, { type: 'other' });
  assert.equal(busyText({ generating: new Map() }), 'working…');
});

test('a patch action waiting behind a generation says so, disables its row, and refreshes after', async () => {
  const a = app({ editWaitsForGeneration: true });
  trackGeneration(a, { type: 'generation', stream: 's1', variant: 'current', state: 'started' });
  let release; let refreshed = 0;
  global.fetch = () => new Promise((r) => { release = () => r(response({ ok: true, missing: false, patches: [] })); });
  const row = new Element('div'), st = new Element('div');
  const del = new Element('button'), box = new Element('input');
  row.append(del, box, st);
  const act = patchAction(a, row, st, async () => { refreshed++; });
  const done = act({ id: 'p1', action: 'delete' });
  await wait(350);
  assert.equal(st.textContent, 'waiting for the running generation…');
  assert.equal(del.disabled, true); assert.equal(box.disabled, true);
  assert.ok(row.classList.contains('busy'));
  release(); await done;
  assert.equal(del.disabled, false);
  assert.equal(refreshed, 1);
  assert.equal(st.textContent, '');
});

test('a failed patch action shows its error in the panel and refreshes; an already deleted patch is fine', async () => {
  const a = app(); let refreshed = 0;
  global.fetch = async () => response({ error: 'ValueError: strength must be a number' }, 400);
  const row = new Element('div'), st = new Element('div');
  await patchAction(a, row, st, async () => { refreshed++; })({ id: 'p1', action: 'set', strength: 'x' });
  assert.equal(refreshed, 1);
  assert.equal(st.textContent, 'ValueError: strength must be a number');
  assert.deepEqual(a.keep.patchNote, { text: 'ValueError: strength must be a number', bad: true });
  a.keep.patchNote = null;
  global.fetch = async () => response({ ok: true, missing: true, patches: [] });
  await patchAction(a, row, st, async () => { refreshed++; })({ id: '0001-88468', action: 'delete' });
  assert.equal(refreshed, 2);
  assert.match(a.keep.patchNote.text, /already gone/);
  assert.equal(a.keep.patchNote.bad, false);
});

// ---------------------------------------------------------------- review fixes
test('the waiting text needs a shared lock, ends when this tab\'s own request settles, and times out only after lost events', async () => {
  const a = app();
  const t0 = Date.now();
  trackGeneration(a, { type: 'generation', stream: 's1', variant: 'current', state: 'started' }, t0);
  assert.equal(busyText(a, t0), 'working…');                             // no runtime worker: edits do not wait
  a.editWaitsForGeneration = true;
  assert.equal(busyText(a, t0), 'waiting for the running generation…');
  // a long prompt read sends nothing: a stream with no final event still runs, however long it is silent
  assert.equal(busyText(a, t0 + 5 * STALE_MS), 'waiting for the running generation…');
  eventsLost(a, t0 + 10);                                                 // the connection dropped: its end may be lost
  assert.equal(busyText(a, t0 + STALE_MS - 1), 'waiting for the running generation…');
  assert.equal(busyText(a, t0 + STALE_MS + 1), 'working…');              // so it times out: not forever
  trackGeneration(a, { type: 'generation', stream: 's1', variant: 'current', state: 'token' }, t0 + 20);
  assert.equal(busyText(a, t0 + 5 * STALE_MS), 'waiting for the running generation…');   // heard from again after the drop
  settleGeneration(a, { stream: 's1' }, t0);                              // this tab's /api/run came back
  assert.equal(generationRunning(a, t0 + 5000), false);
  // the hooks util.api calls when a run/chat request starts and settles, however it ends
  const seen = [];
  const started = (path, body) => seen.push(['start', path, body.stream]);
  const hook = (path, body) => seen.push(['end', path, body.stream]);
  generationStarted.add(started);
  generationSettled.add(hook);
  global.fetch = async () => response({ error: 'boom' }, 500);
  await assert.rejects(api('/api/run', { prompt: 'x', stream: 's2' }));
  global.fetch = async () => response({ ok: true });
  await api('/api/chat', { stream: 's3' });
  await api('/api/state');
  generationStarted.delete(started);
  generationSettled.delete(hook);
  assert.deepEqual(seen, [['start', '/api/run', 's2'], ['end', '/api/run', 's2'], ['start', '/api/chat', 's3'], ['end', '/api/chat', 's3']]);
});

test('this tab\'s own unsettled run waits however silent, even after lost events; its settling ends it', () => {
  const a = app({ editWaitsForGeneration: true });
  const t0 = Date.now();
  startGeneration(a, { stream: 'mine' }, t0);                              // before any event of it arrives
  assert.equal(busyText(a, t0), 'waiting for the running generation…');
  eventsLost(a, t0 + 1);
  assert.equal(busyText(a, t0 + 10 * STALE_MS), 'waiting for the running generation…');
  trackGeneration(a, { type: 'generation', stream: 'mine', variant: 'current', state: 'started' }, t0 + 2);
  assert.equal(busyText(a, t0 + 10 * STALE_MS), 'waiting for the running generation…');   // still this tab's
  settleGeneration(a, { stream: 'mine' }, t0 + 3);
  assert.equal(busyText(a, t0 + 3 + 5000), 'working…');
  startGeneration(a, {}, t0);                                              // no stream id: nothing to track
  assert.equal(a.generating.size, 1);
});

test('only this tab\'s own edit progress is shown, and it ends the waiting text', () => {
  const a = app({ editWaitsForGeneration: true });
  const now = Date.now();
  trackGeneration(a, { type: 'generation', stream: 'other', variant: 'current', state: 'started' }, now);
  const id = beginEdit(a);
  assert.match(id, /^edit-/);
  assert.notEqual(beginEdit(a), id);
  trackEditProgress(a, { type: 'edit_progress', line: 9, lines: 10 }, now);                      // no id: another client
  trackEditProgress(a, { type: 'edit_progress', request: 'edit-x', line: 8, lines: 10 }, now);   // someone else's
  assert.equal(busyText(a, now + 100), 'waiting for the running generation…');
  trackEditProgress(a, { type: 'edit_progress', request: id, line: 2, lines: 10 }, now);
  assert.equal(busyText(a, now + 100), 'working… (line 2 of 10)');          // it holds the lock: not waiting
  endEdit(a, id);
  assert.equal(a.editProgress, null);
  trackEditProgress(a, { type: 'edit_progress', request: id, line: 3, lines: 10 }, now);         // after it settled
  assert.equal(a.editProgress, null);
});

test('actionsRow and the patch list learn whether edits wait for a generation', () => {
  const a = app();
  actionsRow(a, { ...sliceNode, edit_waits_for_generation: true });
  assert.equal(a.editWaitsForGeneration, true);
  actionsRow(a, { ...sliceNode, edit_waits_for_generation: false });
  assert.equal(a.editWaitsForGeneration, false);
});

test('a long edit shows how far it has got', () => {
  const a = app();
  const now = Date.now();
  const request = beginEdit(a);
  trackEditProgress(a, { type: 'edit_progress', request, line: 3, lines: 50, window: 7, windows: 23 }, now);
  assert.equal(busyText(a, now + 100), 'working… (line 3 of 50, read 7 of 23)');
  assert.equal(busyText(a, now + 5000), 'working…');
});

test('the save dialog offers the suggested name, never an existing selection', async () => {
  assert.equal(defaultName('picked', []), 'picked');
  assert.equal(defaultName('picked', ['picked', 'picked 2']), 'picked 3');
  assert.equal(defaultName('', ['a']), 'selection');
  const offered = [];
  const groups = { a: { members: ['L1N3'] }, b: { members: [] } };
  global.fetch = async (url, o) => response({ groups, summary: { neurons: 1 } });
  const a = app();
  a.state.groups = groups;
  await saveSelection(a, { members: ['group:a'] }, { ask: (m, d) => { offered.push(d); return d; } });
  await saveSelection(a, { members: ['L1N2'] }, { suggest: 'a', ask: (m, d) => { offered.push(d); return d; } });
  assert.deepEqual(offered, ['selection', 'a 2']);                       // was: the first existing selection, "a"
});

test('a groups event redraws the list and the inspector, not the view (Find is not run again)', async () => {
  const urls = [];
  global.fetch = async (url) => { urls.push(url); return response({ groups: { picked: { members: [] } }, open: {} }); };
  const a = app({ rerender() { throw new Error('view redrawn'); } });
  let lists = 0, inspectors = 0;
  await groupsEvent(a, { renderGroups: () => lists++, renderInspector: () => inspectors++ });
  assert.deepEqual(urls, ['/api/state']);
  assert.equal(lists, 1); assert.equal(inspectors, 1);
  assert.deepEqual(Object.keys(a.state.groups), ['picked']);
});

test('ticked rows and the save result survive a redraw of the table', async () => {
  global.fetch = async () => response({ groups: {}, summary: { neurons: 2 } });
  globalThis.prompt = (m, d) => d;
  const a = app(); const st = {};
  const rows = [{ layer: 1, neuron: 2, mech: 'm' }, { layer: 3, neuron: 4, mech: 'm' }];
  const t = rowsTable(rows, st, a);
  const boxes = walk(t).filter((n) => n.tag === 'input' && n.type === 'checkbox');
  boxes[1].events.change({ target: { checked: true } });
  await button(t, 'Save checked (1)').click(); await tick();
  const again = rowsTable(rows, st, a);                                   // the patch / groups event redraw
  const b2 = walk(again).filter((n) => n.tag === 'input' && n.type === 'checkbox');
  assert.deepEqual(b2.map((b) => !!b.checked), [false, true]);
  assert.ok(button(again, 'Save checked (1)') && !button(again, 'Save checked (1)').disabled);
  assert.match(text(again), /Saved as selection "picked": 2 neurons/);
  const fewer = rowsTable([rows[0]], st, a);                              // a row no longer listed is not saved
  assert.ok(button(fewer, 'Save checked (0)').disabled);
});

test('a revert patch says what it cancels, and warns when that has changed', () => {
  const rev = { id: 'P2', source: { kind: 'wscript', compensates: [{ id: 'P1', strength: 1, message: 'scale L1N3' }] } };
  const ok = compensatesNote(rev, [{ id: 'P1', enabled: true, strength: 1 }, rev]);
  assert.match(text(ok), /cancels the edits of P1 "scale L1N3" at ×1/);
  assert.doesNotMatch(ok.className, /err/);
  const off = compensatesNote(rev, [{ id: 'P1', enabled: false, strength: 1 }, rev]);
  assert.match(text(off), /P1 \(disabled since\).*no longer there/);
  assert.match(off.className, /err/);
  assert.match(text(compensatesNote(rev, [rev])), /P1 \(deleted since\)/);
  assert.match(text(compensatesNote(rev, [{ id: 'P1', enabled: true, strength: 0.5 }])), /now at ×0\.5/);
  assert.equal(compensatesNote({ id: 'P1', source: { kind: 'wscript' } }, []), null);
});

test('a patch change that a revert depends on is made, and the panel warns', async () => {
  const a = app();
  global.fetch = async () => response({ ok: true, missing: false, patches: [], warning: 'Patch P1 was disabled, but revert patch P2 …' });
  const row = new Element('div'), st = new Element('div');
  await patchAction(a, row, st, async () => {})({ id: 'P1', action: 'set', enabled: false });
  assert.deepEqual(a.keep.patchNote, { text: 'Patch P1 was disabled, but revert patch P2 …', bad: true });
});

test('a plain tensor is zeroed, not ablated, in the button, the reasons and the message', async () => {
  const a = app();
  global.fetch = async () => response({ script: 's', weights: 16, other: [] });
  const row = actionsRow(a, sliceNode);
  assert.ok(button(row, 'zero') && !button(row, 'ablate'));
  assert.match(text(row), /"zero" sets the selected weights/);
  await button(row, 'zero').click(); await tick();
  assert.equal(a.scripts[0][1].message, `zero ${sliceNode.addr}`);
  const unit = actionsRow(app(), { ...sliceNode, kind: 'unit', addr: 'L1N3' });
  assert.ok(button(unit, 'ablate'));
  assert.match(text(unit), /"ablate" sets/);
});

test('a selection page labels its size as every part', () => {
  const g = groupSummary(app(), { selection: { neurons: 2, heads: 0, tensors: 0, layers: [1], weights: 96, by_layer: { 1: 2 }, n_members: 1, members: ['x'], other: [], n_other: 0 } });
  assert.match(text(g), /96 weights in all parts/);
});

// ---------------------------------------------------------------- a large edit asks first (wsp.edit_limit.cost)
const costOf = (weights, confirm) => ({ weights, disk_bytes: weights * 4, memory_bytes: weights * 12, confirm_above: 60_000_000, confirm,
  gguf: true, cancellable: false, estimate_seconds: null, message: `Apply an edit of ${weights} weights? A running edit cannot be stopped.` });

test('an edit above the threshold asks first, and declining applies nothing', async () => {
  const asked = [];
  const a = app({ confirm: (m) => { asked.push(m); return false; } });
  global.fetch = async () => response({ script: 'scale_neurons(3, range(0, 9000), 0.5)', neurons: 9000, heads: 0, weights: 383_273_184, other: [], cost: costOf(383_273_184, true) });
  const row = actionsRow(a, sliceNode);
  await button(row, 'scale ×').click(); await tick();
  assert.equal(asked.length, 1);
  assert.match(asked[0], /Apply an edit of 383273184 weights\? A running edit cannot be stopped\./);
  assert.deepEqual(a.scripts, []);                                          // nothing was sent to /api/edit
  assert.equal(text(status(row)), 'Not applied: nothing was changed.');
  a.confirm = (m) => { asked.push(m); return true; };
  await button(row, 'scale ×').click(); await tick();
  assert.equal(asked.length, 2);
  assert.deepEqual(a.scripts, [['scale_neurons(3, range(0, 9000), 0.5)', { message: `scale ${sliceNode.addr} by 0.5`, weights: 383_273_184 }]]);
  assert.match(text(status(row)), /Done: 16 weights changed/);
});

test('an edit at or below the threshold applies without asking', async () => {
  const a = app({ confirm: () => { throw new Error('asked'); } });
  global.fetch = async () => response({ script: 'P("x")[:, 3838] *= 0.5', neurons: 0, heads: 0, weights: 16, other: [], cost: costOf(16, false) });
  const row = actionsRow(a, sliceNode);
  await button(row, 'scale ×').click(); await tick();
  assert.equal(a.scripts.length, 1);
  assert.match(text(status(row)), /Done: 16 weights changed/);
});

test('without a way to ask, a large edit is not applied', async () => {
  const a = app();
  const saved = globalThis.confirm; delete globalThis.confirm;
  try {
    global.fetch = async () => response({ script: 's', neurons: 0, heads: 0, weights: 70_000_000, other: [], cost: costOf(70_000_000, true) });
    const row = actionsRow(a, sliceNode);
    await button(row, 'scale ×').click(); await tick();
    assert.equal(a.scripts.length, 0);
    assert.equal(text(status(row)), 'Not applied: nothing was changed.');
  } finally { if (saved) globalThis.confirm = saved; }
});

test('a running edit shows how much is written and the time left', () => {
  const a = {};
  const id = beginEdit(a, { weights: 1000, now: 0 });
  trackEditProgress(a, { type: 'edit_progress', request: id, line: 1, lines: 2, window: 3, windows: 9, stored: 50 }, 1000);
  assert.equal(busyText(a, 1000), 'working… (line 1 of 2, read 3 of 9)');    // under a tenth written: no estimate yet
  trackEditProgress(a, { type: 'edit_progress', request: id, line: 2, lines: 2, window: 5, windows: 9, stored: 500 }, 10_000);
  assert.equal(busyText(a, 10_000), 'working… (line 2 of 2, read 5 of 9, 50% written, about 10 s left)');
  trackEditProgress(a, { type: 'edit_progress', request: id, line: 2, lines: 2, stored: 250 }, 120_000);
  assert.match(busyText(a, 120_000), /25% written, about 6 min left/);
  endEdit(a, id);
  assert.equal(a.editStarts.size, 0);
  const plain = beginEdit(a);                                                // a console edit: no total, no estimate
  trackEditProgress(a, { type: 'edit_progress', request: plain, line: 1, lines: 3, stored: 10 }, Date.now());
  assert.equal(busyText(a), 'working… (line 1 of 3)');
});
