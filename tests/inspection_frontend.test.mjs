// node --test tests/inspection_frontend.test.mjs; all networking/DOM is fake.
import test from 'node:test';
import assert from 'node:assert/strict';
import { InspectionClient, inspection, calculation } from '../web/js/inspection.js';
import { weightsView } from '../web/js/weights.js';
import { systemsView } from '../web/js/systems.js';

const tick = () => new Promise((resolve) => setImmediate(resolve));
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };
const makeApp = () => ({ sel: 'blocks.0.mlp.neuron.0', tab: 'Weights', variant: 'current', keep: {}, state: { status: { workspace_id: 'a', version: 0 }, open: { recognised: true, config: { n_layers: 1 } } } });

class Element {
  constructor(tag) { this.tag = tag; this.nodeType = 1; this.children = []; this.style = {}; this.events = {}; this.disabled = false; this.hidden = false; this.textContent = ''; }
  append(...kids) { this.children.push(...kids); }
  addEventListener(event, fn) { this.events[event] = fn; }
  setAttribute(k, v) { this[k] = v; }
  click() { this.events.click?.({ target: this }); }
}
global.document = { createElement: (t) => new Element(t), createTextNode: (text) => ({ nodeType: 3, textContent: text }) };
function descendants(root, tag) { return [root, ...(root.children || []).flatMap((child) => descendants(child, tag))].filter((e) => e.tag === tag); }


test('requests coalesce and completed results reuse by workspace/version/options; failures retry', async () => {
  const app = makeApp(), pending = deferred(); let calls = 0;
  const client = new InspectionClient(() => { calls++; return pending.promise; });
  const first = client.read(app, '/node', { addr: 'a' });
  assert.equal(first, client.read(app, '/node', { addr: 'a' }));
  await tick(); assert.equal(calls, 1);
  pending.resolve({ ok: true }); await first;
  await client.read(app, '/node', { addr: 'a' }); assert.equal(calls, 1);
  app.state.status.version++; await client.read(app, '/node', { addr: 'a' }); assert.equal(calls, 2);
  app.state.status.workspace_id = 'b'; app.state.status.version = 0;
  await client.read(app, '/node', { addr: 'a' }); assert.equal(calls, 3);
  await client.read(app, '/node', { addr: 'a', variant: 'base' }); assert.equal(calls, 4);
  client.send = () => { calls++; throw new Error('failed'); };
  await assert.rejects(client.read(app, '/node', { addr: 'b' }), /failed/);
  client.send = () => { calls++; return {}; };
  await client.read(app, '/node', { addr: 'b' }); assert.equal(calls, 6);
});

test('old failure cannot evict a new workspace request, POST retains its method and scope', async () => {
  const app = makeApp(), old = deferred(), current = deferred();
  const calls = [];
  const client = new InspectionClient((...args) => { calls.push(args); return calls.length === 1 ? old.promise : current.promise; });
  const a = client.read(app, '/ablation', { suite: 's' }, true); const rejection = assert.rejects(a, /old/);
  await tick(); app.state.status.workspace_id = 'b';
  const b = client.read(app, '/ablation', { suite: 's' }, true);
  await tick(); old.reject(new Error('old')); await rejection;
  assert.equal(client.read(app, '/ablation', { suite: 's' }, true), b);
  assert.deepEqual(calls[1], ['/ablation', { suite: 's', workspace_id: 'b', version: 0 }]);
  current.resolve({}); await b;
});

test('calculation waits for a click, disables repeats, joins rerenders, and ignores stale completion', async () => {
  const app = makeApp(), pending = deferred(); let calls = 0, renders = 0;
  inspection.entries.clear(); inspection.scope = null;
  inspection.send = () => { calls++; return pending.promise; };
  const root = new Element('root');
  calculation(root, app, { path: '/node', label: 'Calculate', render: () => renders++ });
  await tick(); assert.equal(calls, 0);
  const button = descendants(root, 'button')[0]; button.click(); button.click();
  assert.equal(button.disabled, true); await tick(); assert.equal(calls, 1);
  const rerender = new Element('root');
  calculation(rerender, app, { path: '/node', label: 'Calculate', render: () => renders++ });
  assert.equal(descendants(rerender, 'button')[0].disabled, true);
  app.sel = 'blocks.0.mlp.neuron.1'; pending.resolve({}); await tick();
  assert.equal(renders, 0);
  assert.equal(button.disabled, false);
});

test('failure restores explicit retry and persistent inspector control works after tab switch', async () => {
  const app = makeApp(); let calls = 0, rendered = 0;
  inspection.entries.clear(); inspection.scope = null;
  inspection.send = () => { calls++; if (calls === 1) throw new Error('failed'); return {}; };
  const root = new Element('root');
  calculation(root, app, { path: '/stats', label: 'Statistics', tabBound: false, render: () => rendered++ });
  const button = descendants(root, 'button')[0];
  app.tab = 'Systems'; button.click(); await tick();
  assert.equal(button.disabled, false); assert.match(button.textContent, /Retry/); assert.equal(calls, 1);
  await tick(); assert.equal(calls, 1);
  button.click(); await tick(); assert.equal(calls, 2); assert.equal(rendered, 1);
  assert.equal(button.disabled, true); assert.equal(button.style.display, 'none');
  button.click(); await tick(); assert.equal(calls, 2); assert.equal(rendered, 1);
});

test('passive neuron view sends only metadata, and stale navigation starts no follow-up work', async () => {
  const app = makeApp(); const calls = [];
  inspection.entries.clear(); inspection.scope = null;
  inspection.send = (path) => { calls.push(path); return path.startsWith('/api/node') ? {
    addr: app.sel, kind: 'unit', level: 5, n_params: 16, stats: {}, analysis: 'neuron', slices: []
  } : { children: [] }; };
  const root = new Element('root');
  await weightsView(root, app);
  assert.equal(calls.length, 2); assert.ok(calls.every((p) => !p.includes('analysis=1')));
  assert.equal(descendants(root, 'button').length, 1);
  inspection.entries.clear();
  const pending = deferred(); inspection.send = () => pending.promise;
  const stale = weightsView(new Element('root'), app, () => false);
  pending.resolve({}); await stale;
});

test('Systems does not compute connections or remembered ablation on opening', async () => {
  const app = makeApp(); app.tab = 'Systems'; app.keep.sys = { overlay: 'ablation' };
  inspection.entries.clear(); inspection.scope = null;
  let calls = 0; inspection.send = () => { calls++; throw new Error('implicit computation'); };
  const root = new Element('root'); await systemsView(root, app); await tick();
  assert.equal(calls, 0); assert.equal(descendants(root, 'button').length, 2);
  assert.equal(descendants(root, 'button')[1].disabled, true);
});

test('old server fails capability handshake before state, inspection, or event subscription', async () => {
  const nodes = new Map();
  document.querySelector = (selector) => {
    if (!nodes.has(selector)) nodes.set(selector, new Element('div'));
    return nodes.get(selector);
  };
  global.window = {};
  let subscriptions = 0;
  global.EventSource = class { constructor() { subscriptions++; } };
  const calls = [];
  global.fetch = async (path) => {
    calls.push(path);
    return { ok: false, statusText: 'Not Found', headers: { get: () => 'application/json' }, json: async () => ({ detail: 'Not Found' }) };
  };
  await import('../web/js/app.js');
  await tick();
  assert.deepEqual(calls, ['/api/inspection-capabilities']);
  assert.equal(subscriptions, 0);
  assert.equal(window.app.state, null);
});

test('bounded Systems controls share pending work and explicit cancel retains request scope', async () => {
  const app=makeApp();app.tab='Systems';app.state.open.n_params=630000000;app.state.open.config.n_layers=8;
  app.rerender=()=>{};
  global.window=new EventTarget();
  inspection.entries.clear();inspection.scope=null;
  const pending=deferred();let calls=0;
  inspection.send=()=>{calls++;return pending.promise;};
  const sent=[];global.fetch=async(path,options)=>{sent.push([path,JSON.parse(options.body)]);return {ok:true,headers:{get:()=> 'application/json'},json:async()=>({cancelled:true})};};
  const first=new Element('root');await systemsView(first,app);
  assert.equal(calls,0);assert.equal(descendants(first,'input').length,2);
  const buttons=descendants(first,'button');assert.equal(buttons[1].disabled,true);
  buttons[0].click();await tick();assert.equal(calls,1);
  const second=new Element('root');await systemsView(second,app);await tick();assert.equal(calls,1);
  buttons[1].click();await tick();assert.equal(sent[0][0],'/api/wiring/cancel');
  assert.deepEqual(sent[0][1],{top:4000,layers:'0',workspace_id:'a',version:0});
  pending.reject(new Error('Wiring analysis cancelled'));await tick();
  assert.match(buttons[0].textContent,/Retry/);assert.equal(calls,1);
});
