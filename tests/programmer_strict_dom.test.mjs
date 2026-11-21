// The Programmer tab's log against a DOM that behaves like a browser's where the looser fake DOM of
// programmer_frontend.test.mjs does not: `children` is an HTMLCollection (element nodes only, indexable and
// iterable, no array methods) and text is held in text nodes. The first real run failed to draw with
// "node.children[0].children.at is not a function" while every fake-DOM test passed. No server, provider or model.
import test from 'node:test';
import assert from 'node:assert/strict';
import { programmerView, onAgentEvent, tickProgrammer, stopTicker } from '../web/js/work.js';
import { installStrictDom, mount, all } from './strict_dom.mjs';

// The DOM itself is tests/strict_dom.mjs (shared with programmer_return_frontend.test.mjs); a view is mounted the way
// the shell mounts it, since a browser writes a live region and scrolls only inside the document.
const { view } = installStrictDom();
const walk = all;
const Element = class { constructor(tag) { return document.createElement(tag); } };
global.setInterval = () => 1;
global.clearInterval = () => {};
const tick = () => new Promise((r) => setImmediate(r));
const settle = async () => { for (let k = 0; k < 6; k++) await tick(); };
const json = (data) => ({ ok: true, status: 200, headers: { get: () => 'application/json' }, json: async () => data });

const T0 = 1_800_000_000;
const run = { id: 'run-1', harness: 'claude', native: true, status: 'running', started: T0, ended: null, pending: [], mode: 'tools', model: '', billing: 'subscription' };
const events = [
  { kind: 'user', i: 0, t: T0, text: 'task' },
  { kind: 'init', i: 1, t: T0 + 1, session: 's', model: 'm', n_tools: 17, mcp: [{ name: 'wsp', status: 'connected' }] },
  { kind: 'thinking_delta', i: 2, t: T0 + 2, id: 'm:0', text: 'thinking' },
  { kind: 'tool_use', i: 3, t: T0 + 3, id: 'call-1', name: 'mcp__wsp__overview', input: {} },
  { kind: 'tool_result', i: 4, t: T0 + 9, id: 'call-1', text: 'ok' },
  { kind: 'tool_use', i: 5, t: T0 + 10, id: 'call-2', name: 'mcp__wsp__inspect', input: { what: 'card', address: 'L38N3838' } },
];
function serve(now) {
  global.fetch = async (url) => {
    const u = String(url);
    if (u.startsWith('/api/harness/auth')) return json({ status: 'ready', method: 'claude.ai', billing: 'subscription', csrf_token: 't' });
    if (u === '/api/harness') return json({ runs: [{ workspace: '/ws/a', ...run }], log: [], now: now(), csrf_token: 't', workspace: '/ws/a' });
    if (u.startsWith('/api/harness/conversations')) return json({ conversations: [], now: now(), csrf_token: 't' });
    if (u.startsWith('/api/harness?')) return json({ ...run, events, now: now() });
    return json({});
  };
}
const makeApp = () => ({ tab: 'Programmer', keep: {}, sel: 'model', variant: 'current', rerender() {}, refreshStatus: async () => {},
  state: { task: { title: 'T', instructions: 'I', scope: ['model'], suite: null, preserve: null }, suites: [],
    harnesses: [{ name: 'claude', label: 'Claude Code', installed: true, native: true }, { name: 'custom', label: 'custom command', installed: true }] } });
const timers = (root) => walk(root).filter((n) => n.tag === 'span' && /^ev-time/.test(n.className)).map((n) => n.textContent);
const logOf = (root) => walk(root).find((n) => n.id === 'agent-log');

test('a run with tool calls draws its log and timers in a DOM whose children are an HTMLCollection', async () => {
  const realNow = Date.now;
  try {
    let server = T0 + 25;
    global.Date.now = () => server * 1000;
    serve(() => server);
    const app = makeApp(), root = new Element('root');
    await programmerView(root, app); mount(view, root); await settle();      // threw "…children.at is not a function" before the fix
    const log = logOf(root);
    assert.ok(log, 'the log is drawn');
    assert.equal(log.children.length, 6, 'every saved event has its card');
    assert.deepEqual(timers(root), ['took 6 s', 'running · 15 s']);             // a finished call's duration, an open call's timer
    assert.match(root.textContent, /Running inspect/);

    // A tool call arriving live is appended with its own timer; nothing throws inside the event handler.
    onAgentEvent({ type: 'agent', run: 'run-1', t: server, event: { kind: 'tool_result', i: 6, t: server, id: 'call-2', text: 'ok' } }, app);
    onAgentEvent({ type: 'agent', run: 'run-1', t: server + 1, event: { kind: 'tool_use', i: 7, t: server + 1, id: 'call-3', name: 'mcp__wsp__atlas', input: { action: 'unit' } } }, app);
    server += 5;
    await tickProgrammer(app, Date.now());
    assert.equal(logOf(root).children.length, 8);
    assert.deepEqual(timers(root), ['took 6 s', 'took 15 s', 'running · 4 s']);

    // Drawing the tab again (coming back to it) shows the same log.
    const again = new Element('root');
    await programmerView(again, app); mount(view, again); await settle();
    assert.equal(logOf(again).children.length, 8);
    assert.deepEqual(timers(again), ['took 6 s', 'took 15 s', 'running · 4 s']);
  } finally { global.Date.now = realNow; stopTicker(); }
});
