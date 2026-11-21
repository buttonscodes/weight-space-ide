// The shell (app.js, shell.js): a tree selection, the tab bar and its Programmer marker, and mounting a view (afterMount
// before the scroll position is restored). The helpers in shell.js are checked on their own, then app.js itself is
// imported into tests/strict_dom.mjs and driven, so that its wiring to them is checked too. No server or model.
import test from 'node:test';
import assert from 'node:assert/strict';
import { installStrictDom, flush, byText } from './strict_dom.mjs';
import { selectNode, tabBar, mountView, viewKey } from '../web/js/shell.js';

const { view } = installStrictDom({ shell: true });
global.setInterval = () => 1;                          // the Programmer ticker is not needed here
global.clearInterval = () => {};
const T0 = 1_800_000_000;
const json = (data, ok = true) => ({ ok, status: ok ? 200 : 404, statusText: 'x', headers: { get: () => 'application/json' }, json: async () => data });
const el = (tag = 'div', kids = 0) => { const n = document.createElement(tag); for (let k = 0; k < kids; k++) n.append(document.createElement('p')); return n; };

// ---------------------------------------------------------------------------------------------------- shell.js
test('selectNode: the inspector always follows; the view is redrawn only when it draws the selection or is not shown', () => {
  const calls = [];
  const ui = (shown) => ({ markSelected: (a) => calls.push('mark ' + a), renderTabs: () => calls.push('tabs'), renderView: () => calls.push('view'),
    renderInspector: () => calls.push('inspector'), shownView: () => shown });
  const app = { sel: 'model', tab: 'Programmer' };
  selectNode(app, 'blocks.0', undefined, ui('Programmer'));
  assert.deepEqual(calls.splice(0), ['mark blocks.0', 'tabs', 'inspector']);
  assert.equal(app.sel, 'blocks.0');
  app.tab = 'Weights';
  selectNode(app, 'embed', undefined, ui('Weights'));
  assert.deepEqual(calls.splice(0), ['mark embed', 'tabs', 'view', 'inspector']);
  selectNode(app, 'unembed', 'Trace', ui('Tests'));        // the Tests view's trace button: the Trace view is drawn
  assert.deepEqual(calls.splice(0), ['mark unembed', 'tabs', 'view', 'inspector']);
  assert.equal(app.tab, 'Trace');
});

test('tabBar: the marker is text in the Programmer button; update() redraws the bar only when it changes, never the view', () => {
  const host = el();
  const app = { tab: 'Weights' };
  let marker = null, views = 0, tops = 0;
  const bar = tabBar(app, ['Weights', 'Programmer'], { host: () => host, marker: () => marker, renderView: () => views++, scrollToTop: () => tops++ });
  bar.render();
  const texts = () => [...host.children].map((b) => b.textContent);
  assert.deepEqual(texts(), ['Weights', 'Programmer']);
  const first = host.firstChild;
  bar.update();
  assert.equal(host.firstChild, first, 'no change, no redraw');
  marker = { state: 'working', text: 'working', title: 'A programmer turn is running.' };
  bar.update();
  assert.deepEqual(texts(), ['Weights', 'Programmer · working']);
  const p = byText(host, 'button', 'Programmer · working');
  assert.equal(p.title, 'A programmer turn is running.');
  assert.match(p.className, /\bmarked working\b/);
  const shown = host.firstChild;
  marker = { ...marker, text: 'working' };
  bar.update();
  assert.equal(host.firstChild, shown, 'the same state: not redrawn');
  assert.equal(views, 0, 'updating the marker never draws the view');
  p.click();
  assert.equal(app.tab, 'Programmer');
  assert.equal(views, 1);
  byText(host, 'button', 'Programmer · working').click();  // the shown Programmer tab again: to its top, no redraw
  assert.deepEqual([views, tops], [1, 1]);
});

test('mountView: afterMount runs once the box is in the page, then the scroll position of the same view is restored', () => {
  const page = view;
  const order = [];
  const first = el('div', 80);
  mountView(page, first, 'Programmer', 'model');
  assert.equal(page.dataset.view, 'Programmer');
  page.scrollTop = 1e9;
  const y = page.scrollTop;
  assert.ok(y > 0);
  // The next box is short until its afterMount adds the rest (as the Programmer tab moves its log in).
  const next = el('div', 2);
  next.afterMount = () => { order.push(['afterMount', next.isConnected, page.scrollTop]); for (let k = 0; k < 78; k++) next.append(document.createElement('p')); };
  mountView(page, next, 'Programmer', 'blocks.0');         // another selection: the Programmer view keeps its place
  assert.deepEqual(order, [['afterMount', true, 0]]);
  assert.equal(page.scrollTop, y, 'restored after afterMount, at its full height');
  mountView(page, el('div', 80), 'Weights', 'blocks.0');
  assert.equal(page.scrollTop, 0, 'another view starts at its top');
  assert.equal(viewKey('Weights', 'embed') === viewKey('Weights', 'blocks.0'), false);
  const thrower = el('div', 1); thrower.afterMount = () => { throw new Error('boom'); };
  const err = console.error; console.error = () => {};
  try { mountView(page, thrower, 'Run', 'model'); } finally { console.error = err; }
  assert.equal(page.firstChild, thrower, 'a failing afterMount leaves the view mounted');
});

// ---------------------------------------------------------------------------------------------------- app.js itself
function fakeServer() {
  const events = [{ kind: 'user', i: 0, t: T0, text: 'BRIEF' }, { kind: 'init', i: 1, t: T0 + 1, session: 's-1', model: 'm' }];
  for (let k = 0; k < 40; k++) events.push({ kind: 'tool_use', i: events.length, t: T0 + 2 + k, id: 'c' + k, name: 'mcp__wsp__inspect', input: {} },
    { kind: 'tool_result', i: events.length + 1, t: T0 + 2.5 + k, id: 'c' + k, text: 'ok' });
  const s = { capability: { version: 0 }, posts: [], gets: [], events,
    run: { id: 'run-1', harness: 'claude', native: true, status: 'running', started: T0, ended: null, pending: [], mode: 'tools', model: '', billing: 'subscription', workspace: '/ws/a' } };
  global.Date.now = () => (T0 + 60) * 1000;
  global.fetch = async (url, o = {}) => {
    const u = String(url);
    if (o.method === 'POST') { s.posts.push(u); return json({}); }
    s.gets.push(u);
    if (u === '/api/inspection-capabilities') return json(s.capability);
    if (u.startsWith('/api/harness/auth')) return json({ status: 'ready', method: 'claude.ai', billing: 'subscription', csrf_token: 't' });
    if (u === '/api/harness') return json({ runs: [s.run], log: [], now: T0 + 60, csrf_token: 't', workspace: '/ws/a' });
    if (u.startsWith('/api/harness/conversations')) return json({ conversations: [], now: T0 + 60, csrf_token: 't' });
    if (u.startsWith('/api/harness?')) return json({ ...s.run, events: s.events, now: T0 + 60 });
    if (u.startsWith('/api/node')) return json({ error: 'no node here' }, false);
    return json({});
  };
  return s;
}

test('app.js: a tree click leaves the Programmer view as it is, a re-render keeps its place, and the marker follows the run', async () => {
  const s = fakeServer();
  const { app } = await import('../web/js/app.js');
  await flush(10);                                             // its load() stops at the capability check (version 0)
  assert.ok(s.gets.includes('/api/inspection-capabilities'));
  app.state = { open: { name: 'a' }, status: { workspace_id: 'w', version: 1 }, task: { title: 'Paris to Rome', scope: ['model'] }, suites: [],
    harnesses: [{ name: 'claude', label: 'Claude Code', installed: true, native: true }] };
  app.tab = 'Programmer';
  app.rerender();
  await flush(30);
  const log = view.querySelector('#agent-log');
  assert.ok(log && log.isConnected, 'the log is in the page (renderView called afterMount)');
  const box = view.querySelector('#agent-log-box');
  assert.equal(box.scrollTop, box.scrollHeight - box.clientHeight, 'the log at its end');
  view.scrollTop = 1e9;
  const y = view.scrollTop, shown = view.firstChild;
  assert.ok(y > 0);
  // A tree click: the inspector follows, the view stays.
  app.select('blocks.0');
  await flush(10);
  assert.equal(view.firstChild, shown, 'not rebuilt');
  assert.equal(view.scrollTop, y, 'not moved');
  assert.ok(s.gets.some((u) => u.startsWith('/api/node?addr=blocks.0')), 'the inspector read the new selection');
  assert.match(document.querySelector('#inspector').textContent, /no node here/);
  // A re-render after the selection changed (a run event, the theme button): the same place, the log in the page.
  app.rerender();
  await flush(30);
  assert.notEqual(view.firstChild, shown, 'a new view');
  assert.equal(view.scrollTop, y, 'kept across a re-render after the selection changed');
  assert.ok(view.querySelector('#agent-log').isConnected);
  // The tab bar's marker: from the run's state, and redrawn when the run changes, without drawing the view.
  const tabs = () => [...document.querySelector('#tabs').children].map((b) => b.textContent);
  app.runChanged();
  assert.ok(tabs().includes('Programmer · working'), tabs().join(','));
  const now = view.firstChild;
  app.keep.prog.run = { ...app.keep.prog.run, status: 'waiting', pending: [{ request_id: '1', title: 'Bash' }] };
  app.runChanged();
  assert.ok(tabs().includes('Programmer · waiting for you'), tabs().join(','));
  assert.equal(view.firstChild, now, 'the marker never redraws the view');
  // Clicking the shown Programmer tab: to the top, where the open run is.
  byText(document.querySelector('#tabs'), 'button', 'Programmer · waiting for you').click();
  assert.equal(view.scrollTop, 0);
  assert.equal(view.firstChild, now);
  assert.deepEqual(s.posts, [], 'none of this sent a POST');
});
