// The Atlas tab's observed stage against a DOM that behaves like a browser's where the looser fake of
// atlas_wiring_frontend.test.mjs does not: `children` is an HTMLCollection (element nodes only, indexable and
// iterable, no array methods) and text is held in text nodes (the fake of programmer_strict_dom.test.mjs).
// Fake network only; no server, no model.
import test from 'node:test';
import assert from 'node:assert/strict';
import { atlasView, onAtlasEvent } from '../web/js/atlas.js';

class HTMLCollection {
  constructor(nodes) { nodes.forEach((n, i) => { this[i] = n; }); this.length = nodes.length; }
  item(i) { return this[i] ?? null; }
  *[Symbol.iterator]() { for (let i = 0; i < this.length; i++) yield this[i]; }
}
class Text { constructor(text) { this.nodeType = 3; this.data = String(text); this.parentElement = null; } get textContent() { return this.data; } set textContent(v) { this.data = String(v); } }
class Element {
  constructor(tag) { this.tag = tag; this.nodeType = 1; this.childNodes = []; this.style = {}; this.events = {}; this.disabled = false; this.value = ''; this.className = ''; this.parentElement = null; this.scrollTop = 0; this.scrollHeight = 0; }
  get children() { return new HTMLCollection(this.childNodes.filter((n) => n.nodeType === 1)); }
  get firstChild() { return this.childNodes[0] ?? null; }
  get textContent() { return this.childNodes.map((n) => n.textContent).join(''); }
  set textContent(v) { this.childNodes = []; if (String(v) !== '') this.append(new Text(v)); }
  append(...nodes) { for (const n of nodes) { const node = typeof n === 'string' ? new Text(n) : n; node.parentElement = this; this.childNodes.push(node); } }
  replaceChildren(...nodes) { this.childNodes = []; this.append(...nodes); }
  removeChild(n) { this.childNodes = this.childNodes.filter((x) => x !== n); n.parentElement = null; return n; }
  addEventListener(name, fn) { this.events[name] = fn; }
  setAttribute(k, v) { this[k] = v; }
  click() { return this.events.click?.({ target: this }); }
}
const walk = (root) => [root, ...root.childNodes.filter((n) => n.nodeType === 1).flatMap(walk)];
global.document = { createElement: (t) => new Element(t), createElementNS: (ns, t) => new Element(t), createTextNode: (t) => new Text(t), getElementById: () => null };
const tick = () => new Promise((r) => setImmediate(r));
const settle = async () => { for (let k = 0; k < 6; k++) await tick(); };
const json = (data) => ({ ok: true, status: 200, headers: { get: () => 'application/json' }, json: async () => data });

const NOTE = 'This stage loads and runs the model on about 5,000 tokens of probe text, one window at a time.';
const built = (o = {}) => ({
  built: true, can_build: true, can_observe: true, observed_unavailable: null, observed_note: NOTE, model: 'fake', variant: 'base',
  stages: { weights: { neurons: 10, layers: [0, 1] } }, stale_layers: [], wiring_progress: null, wiring_built: false, build_running: null, job: null,
  layers: { stages: [{ from: 0, to: 1, desc: 'all layers' }], layers: [] }, validation: null, shift: null, residual: null, ...o });
const observed = built({ stages: { weights: { neurons: 10, layers: [0, 1] }, observed: { tokens: 4992 } },
  validation: { tokens: 4992, source: 'the bundled probe text', loss: 2.5, peak_signed_share: 0.4,
    checks: [{ claim: 'Firing sign follows gate–value alignment', verdict: 'holds', text: 'The sign matches for 90% of the 10 neurons.' }] } });

function serve(status) {
  const seen = { builds: [], status };
  global.fetch = async (path, opt) => {
    if (opt?.body) { seen.builds.push(JSON.parse(opt.body)); return json({ running: true }); }
    if (String(path).startsWith('/api/atlas/estimate')) return json({ search_s: 60, read_s: 1, read_known: true, total_s: 61, memory_bytes: 1e9, waits: ['Run'], continuing: null });
    return json(seen.status);
  };
  return seen;
}
function makeApp() {
  const app = { tab: 'Atlas', variant: 'current', keep: {}, state: { status: { workspace_id: 'w', version: 0 } }, renders: 0, root: null,
    async rerender() { this.renders++; this.root = new Element('root'); await atlasView(this.root, this); } };
  return app;
}
const byId = (root, id) => walk(root).find((n) => n.id === id);
const buttons = (root) => walk(root).filter((n) => n.tag === 'button');

test('a built atlas without the observed stage offers to add it alone, with the note, and sends only that stage', async () => {
  const seen = serve(built()), app = makeApp(), root = new Element('root');
  await atlasView(root, app); await settle();
  const card = byId(root, 'atlas-add-observed');
  assert.ok(card, 'the card is drawn');
  assert.equal(card.children.length, 3);                                    // sentence, note, button: elements only
  assert.equal(byId(card, 'atlas-observed-note').textContent, NOTE);
  assert.equal(root.textContent.split(NOTE).length, 2);                     // the note once
  const add = buttons(card).find((b) => b.textContent === 'Add the observed stage');
  await add.click(); await settle();
  assert.deepEqual(seen.builds, [{ stages: ['observed'], fetch: true, force: false, variant: 'base', workspace_id: 'w', version: 0 }]);
  assert.equal(app.renders, 1);
});

test('an edited-weights atlas sends its own variant; after the build the tab shows the observed results without a reload', async () => {
  const seen = serve(built({ variant: 'current' })), app = makeApp(), root = new Element('root');
  await atlasView(root, app); await settle();
  await buttons(byId(root, 'atlas-add-observed'))[0].click(); await settle();
  assert.equal(seen.builds[0].variant, 'current'); assert.equal(seen.builds[0].force, false);
  seen.status = observed;                                                   // the build finished on the server
  onAtlasEvent({ done: true, workspace_id: 'w', version: 0 }, app); await settle();
  const text = app.root.textContent;
  assert.match(text, /How well the map held up/);
  assert.match(text, /Forward pass over 4,?992 tokens of the bundled probe text/);
  assert.match(text, /Firing sign follows gate–value alignment/);
  assert.equal(byId(app.root, 'atlas-add-observed'), undefined);             // nothing left to add
  assert.doesNotMatch(text, /Not checked: the observed stage has not run/);
});

test('no card when the stage cannot be built, and a rebuild is asked for when the edited weights changed', async () => {
  for (const status of [built({ observed_unavailable: 'This model cannot be run here, so the map will be weights-only and firing signs will be predictions.' }),
                        built({ can_observe: false }), observed]) {
    serve(status);
    const root = new Element('root');
    await atlasView(root, makeApp()); await settle();
    assert.equal(byId(root, 'atlas-add-observed'), undefined);
  }
  const seen = serve(built({ variant: 'current', stale_layers: [0, 1] })), root = new Element('root');
  await atlasView(root, makeApp()); await settle();
  const card = byId(root, 'atlas-add-observed');
  assert.match(card.textContent, /have changed since it was built, so adding it needs a rebuild/);
  assert.equal(buttons(card).length, 0); assert.equal(seen.builds.length, 0);
});

test('the build panel shows the worker note under the Observed box in a strict DOM', async () => {
  const seen = serve({ built: false, can_build: true, can_observe: true, observed_unavailable: null, observed_note: NOTE, wiring_auto: false, neurons: 2, model: 'fake' });
  const root = new Element('root');
  await atlasView(root, makeApp()); await settle();
  const note = byId(root, 'atlas-observed-note');
  assert.equal(note.textContent, NOTE); assert.equal(note.childNodes[0].nodeType, 3);
  const boxes = walk(root).filter((n) => n.tag === 'input');
  boxes[2].events.change({ target: { checked: true } });
  await buttons(root).find((b) => /Build atlas/.test(b.textContent)).click(); await settle();
  assert.deepEqual(seen.builds[0].stages, ['weights', 'observed']);
});
