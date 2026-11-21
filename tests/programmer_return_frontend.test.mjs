// Coming back to a running programmer conversation: the run first on the tab, the log kept (and its scroll position)
// across re-renders, tab switches and events that arrive meanwhile, the tab bar's marker, and no request but reads on
// navigation. Written against tests/strict_dom.mjs, which fails where a browser would. No server, provider or model.
import test from 'node:test';
import assert from 'node:assert/strict';
import { installStrictDom, mount, all, byText, flush } from './strict_dom.mjs';
import { programmerView, onAgentEvent, tickProgrammer, stopTicker, tickerCount, programmerMarker, seedProgrammer } from '../web/js/work.js';
import { SELECTION_FREE, viewKey, renderOnSelect, tabText } from '../web/js/shell.js';

const { view } = installStrictDom();
global.setInterval = () => 1;                          // the ticker is driven by hand (tickProgrammer)
global.clearInterval = () => {};
const T0 = 1_800_000_000;
const json = (data, ok = true, status = ok ? 200 : 409) => ({ ok, status, statusText: 'x', headers: { get: () => 'application/json' }, json: async () => data });
const realNow = Date.now;

function server({ run, events = [], conversations = [] }) {
  const s = { run, events, conversations, posts: [], gets: [], now: T0 + 60, workspace: '/ws/a' };
  global.Date.now = () => s.now * 1000;
  global.fetch = async (url, o = {}) => {
    const u = String(url);
    if (o.method === 'POST') {
      const body = JSON.parse(o.body || '{}');
      s.posts.push({ url: u, body });
      if (u === '/api/task') return json({ task: body });
      if (u === '/api/harness/resume') return json({ ...s.run, events: s.events, now: s.now });
      return json({ ...(s.run || {}), now: s.now });
    }
    s.gets.push(u);
    if (u.startsWith('/api/harness/conversation?')) return json(s.transcript);
    if (u.startsWith('/api/harness/auth')) return json({ status: 'ready', method: 'claude.ai', billing: 'subscription', csrf_token: 't' });
    if (u === '/api/harness') return json({ runs: s.runs || (s.run ? [s.run] : []), log: [], now: s.now, csrf_token: 't', workspace: s.workspace });
    if (u.startsWith('/api/harness/conversations')) return json({ conversations: s.conversations, now: s.now, csrf_token: 't' });
    if (u.startsWith('/api/harness?')) { const since = +(new URLSearchParams(u.split('?')[1]).get('since') || 0); return json({ ...s.run, events: s.events.slice(since), now: s.now }); }
    return json({});
  };
  return s;
}
function makeApp() {
  return { tab: 'Programmer', keep: {}, sel: 'model', variant: 'current', rerenders: 0, markers: 0,
    rerender() { this.rerenders++; }, runChanged() { this.markers++; }, refreshStatus: async () => {},
    state: { task: { title: 'Paris to Rome', instructions: 'Make it say Rome', scope: ['model'], suite: null, preserve: null }, suites: [],
      harnesses: [{ name: 'claude', label: 'Claude Code', installed: true, native: true }, { name: 'custom', label: 'custom command', installed: true }] } };
}
// What app.js renderView does: build detached, mount, then afterMount.
async function render(app) { const box = document.createElement('div'); await programmerView(box, app); mount(view, box); await flush(); return box; }
const run = (status = 'running', extra = {}) => ({ id: 'run-1', harness: 'claude', native: true, status, started: T0, ended: null, pending: [], mode: 'tools', model: '', billing: 'subscription', conversation: 'conv-20261117-010000-0000abcd', workspace: '/ws/a', ...extra });
const turn = (n) => {
  const out = [{ kind: 'user', i: 0, t: T0, text: 'BRIEF' }, { kind: 'init', i: 1, t: T0 + 1, session: 's-1', model: 'm' }];
  for (let k = 0; k < n; k++) out.push({ kind: 'tool_use', i: out.length, t: T0 + 2 + k, id: 'c' + k, name: 'mcp__wsp__inspect', input: { address: `L1N${k}` } },
    { kind: 'tool_result', i: out.length + 1, t: T0 + 2.5 + k, id: 'c' + k, text: 'ok' });
  return out;
};
const live = { id: 'conv-20261117-010000-0000abcd', started: T0, provider: 'claude', mode: 'tools', status: 'running', n_events: 3, task: 'Paris to Rome', native: true, live: true, resume: { ok: false, code: 'open', reason: 'This is the conversation running now.' } };
const order = (root) => [...root.children].map((n) => n.id || n.tag + (n.className ? '.' + n.className.split(' ')[0] : ''));
const logBox = () => view.querySelector('#agent-log-box');
const done = () => { stopTicker(); global.Date.now = realNow; };

test('an open conversation is shown first: status, prompts, log, follow-up and its buttons above the folded setup', async () => {
  const s = server({ run: run('waiting', { pending: [{ request_id: '7', title: 'Bash', input: { command: 'ls' } }] }), events: turn(3), conversations: [live] });
  const app = makeApp();
  const root = await render(app);
  assert.deepEqual(order(root), ['div.row', 'agent-run', 'details.agent-setup', 'agent-conversations']);
  const section = root.querySelector('#agent-run');
  assert.deepEqual([...section.children].map((n) => n.id || n.className), ['row', 'agent-status-slot', 'card agent-prompt', 'agent-log-box', 'card']);
  assert.equal(section.querySelector('.agent-status-slot').firstChild.className, 'agent-status prompt');
  assert.match(section.querySelector('.agent-prompt h3').textContent, /^Waiting for your response: Bash$/);
  assert.ok(byText(section, 'button', 'Interrupt turn') && byText(section, 'button', 'Close conversation') && byText(section, 'button', 'Send message'));
  const details = root.querySelector('details.agent-setup');
  assert.equal(details.open, false, 'folded');
  assert.equal(details.querySelector('summary').textContent, 'Task, harness and sign-in of this conversation (fixed while it is open): Paris to Rome');
  assert.ok(!byText(root, 'button', 'Start programmer'));
  assert.ok(details.querySelectorAll('input, select, textarea').every((c) => c.disabled), 'its controls stay disabled');
  assert.equal(s.posts.length, 0);
  // The person opens the fold: it stays open across a re-render.
  details.open = true; details.dispatchEvent({ type: 'toggle' });
  assert.equal((await render(app)).querySelector('details.agent-setup').open, true);
  done();
});

test('with no open conversation the page keeps its order: setup, conversations, then the last run', async () => {
  server({ run: run('closed', { ended: T0 + 30 }), events: turn(1) });
  const root = await render(makeApp());
  assert.deepEqual(order(root), ['div.row', 'div', 'agent-run']);              // header, setup, the last run
  assert.ok(root.children[1].querySelector('#agent-auth'), 'the setup holds sign-in, task and harness');
  assert.ok(!root.querySelector('details'));
  const ids = all(root).map((n) => n.id).filter(Boolean);
  assert.ok(ids.indexOf('agent-auth') < ids.indexOf('agent-run'), 'setup before the run');
  assert.ok(byText(root, 'button', 'Start programmer'));
  assert.ok(!byText(root.querySelector('#agent-run'), 'button', 'Close conversation'));
  done();
});

test('the live row of the conversation list brings the run into view, sending nothing', async () => {
  const s = server({ run: run(), events: turn(2), conversations: [live] });
  const root = await render(makeApp());
  const show = byText(root.querySelector('#agent-conversations'), 'button', 'Show at the top');
  assert.ok(show, 'a control, not the words "shown below"');
  assert.doesNotMatch(root.textContent, /shown below/);
  show.click();
  assert.equal(root.querySelector('#agent-run').scrolledIntoView, 1);
  assert.equal(document.activeElement, root.querySelector('#agent-run h3'));
  assert.equal(s.posts.length, 0);
  done();
});

test('the log node is kept across renders and tabs; events that arrive on another tab are in it on return', async () => {
  const s = server({ run: run(), events: turn(2) });
  const app = makeApp();
  await render(app);
  const log = view.querySelector('#agent-log');
  assert.equal(log.children.length, 6);
  await render(app);
  assert.equal(view.querySelector('#agent-log'), log, 'not rebuilt by a re-render');
  app.tab = 'Weights';
  view.replaceChildren(document.createElement('div'));       // another view is shown
  const before = app.rerenders, heard = app.markers;
  for (let k = 0; k < 3; k++) onAgentEvent({ type: 'agent', run: 'run-1', t: T0 + 50 + k, event: { kind: 'text_delta', i: 6 + k, t: T0 + 50 + k, id: 'm:1', text: 'piece ' } }, app);
  onAgentEvent({ type: 'agent', run: 'run-1', t: T0 + 55, event: { kind: 'tool_use', i: 9, t: T0 + 55, id: 'c9', name: 'mcp__wsp__atlas', input: { action: 'unit' } } }, app);
  assert.equal(app.rerenders, before, 'events on another tab never redraw the view');
  assert.equal(app.markers - heard, 4, 'but the tab bar hears of each');
  app.tab = 'Programmer';
  s.events = app.keep.prog.events;
  const root = await render(app);
  assert.equal(root.querySelector('#agent-log'), log, 'the same node, with what arrived meanwhile');
  assert.equal(log.children.length, 8);
  assert.match(log.textContent, /piece piece piece/);
  assert.match(root.querySelector('.agent-status-text').textContent, /^Running atlas unit/);
  // A new run is a new log.
  onAgentEvent({ type: 'agent', run: 'run-2', t: T0 + 70, event: { kind: 'user', i: 0, t: T0 + 70, text: 'next' } }, app);
  s.run = run('running', { id: 'run-2' }); s.events = app.keep.prog.events;
  assert.notEqual((await render(app)).querySelector('#agent-log'), log);
  done();
});

test('the log stays at its end while a turn runs; once scrolled up it keeps its place through events, re-renders and tabs', async () => {
  const s = server({ run: run(), events: turn(30) });
  const app = makeApp();
  await render(app);
  let box = logBox();
  assert.ok(box.scrollHeight > box.clientHeight, 'the log scrolls');
  assert.equal(box.scrollTop, box.scrollHeight - box.clientHeight, 'at its end after the render');
  onAgentEvent({ type: 'agent', run: 'run-1', t: T0 + 40, event: { kind: 'text', i: 62, t: T0 + 40, text: 'more' } }, app);
  await flush();
  assert.equal(box.scrollTop, box.scrollHeight - box.clientHeight, 'follows a new event');
  // Scrolled up by the person.
  box.scrollTop = 100; await flush();
  onAgentEvent({ type: 'agent', run: 'run-1', t: T0 + 41, event: { kind: 'text', i: 63, t: T0 + 41, text: 'again' } }, app);
  await flush();
  assert.equal(box.scrollTop, 100, 'a new event does not move it');
  // A prompt appears: the run event re-renders the whole tab.
  s.run = run('waiting', { pending: [{ request_id: '1', title: 'Bash' }] });
  onAgentEvent({ type: 'run', run: s.run }, app);
  s.events = app.keep.prog.events;
  await render(app);
  assert.notEqual(logBox(), box, 'a new box…');
  const old = box;
  box = logBox();
  assert.equal(box.scrollTop, 100, '…at the same place');
  old.dispatchEvent({ type: 'scroll' });                       // a late event (a wheel scroll in flight) for the old box
  assert.deepEqual([app.keep.prog.logEl.follow, app.keep.prog.logEl.scroll], [false, 100], 'the old box does not speak for the log');
  // Away and back.
  app.tab = 'Weights'; view.replaceChildren(document.createElement('div'));
  app.tab = 'Programmer';
  await render(app);
  assert.equal(logBox().scrollTop, 100);
  // Back to the end: following again.
  logBox().scrollTop = 1e9; await flush();
  await render(app);
  assert.equal(logBox().scrollTop, logBox().scrollHeight - logBox().clientHeight);
  done();
});

// A conversation's log starts at its end and follows its output, whatever the previous conversation's log was doing.
async function scrolledUpThenClosed(s, app) {
  await render(app);
  logBox().scrollTop = 100; await flush();
  assert.equal(app.keep.prog.logEl.follow, false, 'run A scrolled up');
  s.run = run('closed', { ended: T0 + 50 });
  onAgentEvent({ type: 'run', run: s.run }, app);
  s.events = app.keep.prog.events;
  await render(app);
  assert.equal(logBox().scrollTop, 100, 'the ended run keeps its place');
}
const atEnd = (box) => box.scrollTop === box.scrollHeight - box.clientHeight;
const stream = (app, id, from, n) => { for (let k = 0; k < n; k++) onAgentEvent({ type: 'agent', run: id, t: T0 + 90 + k, event: { kind: 'text', i: from + k, t: T0 + 90 + k, text: 'line ' + k } }, app); };

test('a new conversation started after closing one opens at its end and follows its output', async () => {
  const s = server({ run: run(), events: turn(30) });
  const app = makeApp();
  await scrolledUpThenClosed(s, app);
  s.run = run('starting', { id: 'run-2' }); s.events = [];
  byText(view, 'button', 'Start programmer').click();
  await flush();
  assert.ok(s.posts.some((p) => p.url === '/api/harness/start'));
  await render(app);
  stream(app, 'run-2', 0, 50);
  await flush();
  const log = view.querySelector('#agent-log');
  assert.equal(log.runId, 'run-2');
  assert.equal(log.children.length, 50);
  assert.ok(atEnd(logBox()), `at its end (${logBox().scrollTop} of ${logBox().scrollHeight - logBox().clientHeight})`);
  assert.equal(log.follow, true, 'following');
  await render(app);                                           // and through a re-render
  stream(app, 'run-2', 50, 5); await flush();
  assert.ok(atEnd(logBox()));
  done();
});

test('a resumed conversation opens at its end and follows its output', async () => {
  const s = server({ run: run(), events: turn(30) });
  const id = 'conv-20261117-010000-0000abcd';
  s.conversations = [{ id, started: T0, provider: 'claude', mode: 'tools', status: 'closed', n_events: 62, task: 'Paris to Rome', native: true, live: false, resume: { ok: true } }];
  s.transcript = { meta: { id, started: T0, provider: 'claude', mode: 'tools', status: 'closed', native: true, task: 'Paris to Rome' }, events: turn(30), resume: { ok: true }, now: s.now };
  const app = makeApp();
  await scrolledUpThenClosed(s, app);
  byText(view.querySelector('#agent-conversations'), 'button', 'Open').click();
  await flush();
  await render(app);
  s.run = run('starting', { id: 'run-3', resumed: true }); s.events = turn(30);
  byText(view.querySelector('#agent-conversation'), 'button', 'Resume').click();
  await flush();
  assert.ok(s.posts.some((p) => p.url === '/api/harness/resume'));
  await render(app);
  assert.equal(view.querySelector('#agent-log').runId, 'run-3');
  assert.ok(atEnd(logBox()), 'its earlier events, at their end');
  stream(app, 'run-3', 62, 10); await flush();
  assert.ok(atEnd(logBox()), 'and following what the resumed session says');
  done();
});

test('after a workspace switch the tab shows no run of the other workspace, only this one\'s conversations', async () => {
  const s = server({ run: run('closed', { ended: T0 + 30, workspace: '/ws/other' }), events: turn(2) });
  s.conversations = [{ id: 'conv-20261117-020000-0000abcd', started: T0 + 5, provider: 'codex', mode: 'tools', status: 'closed', n_events: 9, task: 'Here', native: true, live: false, resume: { ok: true } }];
  const app = makeApp();
  await seedProgrammer(app);
  assert.equal(app.keep.prog.run, null, 'no marker from the other workspace\'s run');
  const root = await render(app);
  assert.equal(root.querySelector('#agent-run'), null, 'no run shown');
  assert.ok(!s.gets.some((u) => u.startsWith('/api/harness?id=')), 'its events are not fetched');
  assert.match(root.querySelector('#agent-conversations').textContent, /Here/);
  assert.match(root.textContent, /No run yet/);
  // The open workspace's own run, when the server has one, is shown: the last of them.
  s.run = run('closed', { ended: T0 + 40 });
  s.runs = [run('closed', { id: 'run-0', ended: T0 + 10 }), run('closed', { id: 'run-x', workspace: '/ws/other' }), s.run,
    run('closed', { id: 'run-y', workspace: '/ws/other' })];
  const fresh = makeApp();
  await seedProgrammer(fresh);
  assert.equal(fresh.keep.prog.run.id, 'run-1');
  const shown = makeApp();
  assert.equal((await render(shown)).querySelector('#agent-run .mono')?.textContent, 'run-1');
  done();
});

test('a log rebuilt for the same run (its billing changed at init) keeps where the person left it', async () => {
  const s = server({ run: run(), events: turn(30) });
  const app = makeApp();
  await render(app);
  const before = view.querySelector('#agent-log');
  logBox().scrollTop = 100; await flush();
  s.run = run('running', { billing: 'per_call' });             // the provider said it uses an API key: costs are shown
  onAgentEvent({ type: 'run', run: s.run }, app);
  s.events = app.keep.prog.events;
  await render(app);
  assert.notEqual(view.querySelector('#agent-log'), before, 'rebuilt with its costs');
  assert.equal(logBox().scrollTop, 100, 'at the same place');
  assert.equal(app.keep.prog.logEl.follow, false);
  done();
});

test('a re-render keeps the page where it was: the shared log moves only once the new view is mounted', async () => {
  const s = server({ run: run(), events: turn(30), conversations: [live] });
  const app = makeApp();
  await render(app);
  view.scrollTop = 1e9;                                          // the page scrolled to its bottom
  const y = view.scrollTop;
  assert.ok(y > 0);
  // A render that is built but never mounted (superseded) leaves the log in the page that is shown.
  const shown = view.querySelector('#agent-log');
  await programmerView(document.createElement('div'), app);
  assert.equal(view.querySelector('#agent-log'), shown);
  assert.equal(view.scrollTop, y, 'building a view does not shorten the page that is shown');
  s.run = run('waiting', { pending: [{ request_id: '1', title: 'Bash' }] });
  onAgentEvent({ type: 'run', run: s.run }, app);
  await render(app);
  assert.equal(view.scrollTop, y);
  done();
});

test('coming back to the tab restarts the ticking status line; leaving stops it', async () => {
  server({ run: run(), events: turn(1) });
  const app = makeApp();
  await render(app);
  assert.equal(tickerCount(), 1);
  app.tab = 'Weights';
  await tickProgrammer(app);
  assert.equal(tickerCount(), 0);
  app.tab = 'Programmer';
  await render(app);
  assert.equal(tickerCount(), 1);
  done();
});

test('the tab marker says what the conversation does, in text, and disappears when none is open', () => {
  const m = (status, extra) => programmerMarker({ run: run(status, extra) });
  assert.equal(m('starting').text, 'working');
  assert.equal(m('running').text, 'working');
  assert.equal(m('interrupting').text, 'working');
  assert.deepEqual([m('waiting').state, m('waiting').text], ['prompt', 'waiting for you']);
  assert.equal(m('running', { pending: [{ request_id: '1', title: 'Bash' }] }).text, 'waiting for you');
  assert.equal(m('idle').text, 'turn finished');
  for (const status of ['closed', 'failed', 'interrupted', 'stopped', 'done', 'closing']) assert.equal(m(status), null, status);
  assert.equal(programmerMarker({ run: { id: 'r', harness: 'custom', status: 'running' } }).text, 'working');
  assert.equal(programmerMarker(undefined), null);
  assert.equal(programmerMarker({ run: null }), null);
  for (const status of ['running', 'waiting', 'idle']) assert.ok(m(status).title.length > 20, 'a title says it in a sentence');
  assert.equal(tabText('Programmer', m('waiting')), 'Programmer · waiting for you');
  assert.equal(tabText('Programmer', null), 'Programmer');
});

test('at page load the run is read for the marker with one GET of the runs, and the tab still loads its events', async () => {
  const s = server({ run: run('waiting', { pending: [{ request_id: '1', title: 'Bash' }] }), events: turn(2) });
  const app = makeApp();
  app.tab = 'Weights';
  await seedProgrammer(app);
  await seedProgrammer(app);                                  // e.g. a second app.load before the tab is opened
  assert.deepEqual(s.gets, ['/api/harness']);
  assert.equal(programmerMarker(app.keep.prog).text, 'waiting for you');
  app.tab = 'Programmer';
  const root = await render(app);
  assert.ok(s.gets.includes('/api/harness?id=run-1'), 'the tab fetched the events');
  assert.equal(root.querySelector('#agent-log').children.length, 6);
  assert.equal(s.posts.length, 0);
  // Seeding never replaces a run the page already has.
  const again = s.gets.length;
  await seedProgrammer(app);
  assert.equal(s.gets.length, again);
  done();
});

test('a reload in the middle of a run shows it first again, with its whole log at its end', async () => {
  server({ run: run(), events: turn(25) });
  await render(makeApp());
  const fresh = makeApp();                                     // a new page: nothing carried over
  view.replaceChildren(document.createElement('div'));
  const root = await render(fresh);
  assert.equal(order(root)[1], 'agent-run');
  assert.equal(root.querySelector('#agent-log').children.length, 52);
  assert.equal(logBox().scrollTop, logBox().scrollHeight - logBox().clientHeight);
  done();
});

test('navigating sends no POST: renders, events, ticks, leaving, seeding and coming back only read', async () => {
  const s = server({ run: run(), events: turn(3), conversations: [live] });
  const app = makeApp();
  await render(app);
  for (const tab of ['Weights', 'Atlas', 'Systems', 'Run', 'Probe', 'Trace', 'Tests', 'Chunks', 'Export', 'Settings']) {
    app.tab = tab; view.replaceChildren(document.createElement('div'));
    onAgentEvent({ type: 'agent', run: 'run-1', t: T0 + 9, event: { kind: 'raw', i: s.events.length, t: T0 + 9, text: tab } }, app);
    s.events = app.keep.prog.events;
    await tickProgrammer(app);
    await seedProgrammer(app);
  }
  app.tab = 'Programmer';
  await render(app);
  s.now += 30; await tickProgrammer(app);
  assert.deepEqual(s.posts, []);
  assert.ok(s.gets.every((u) => !/stop|close|interrupt|send|respond|start|resume/.test(u)));
  done();
});

test('an action error on the open conversation is shown in the run, where the button was', async () => {
  const s = server({ run: run(), events: turn(1) });
  const app = makeApp();
  let root = await render(app);
  global.fetch = async (url, o = {}) => (o.method === 'POST' ? json({ error: 'Agent did not acknowledge the command' }, false) : json({ runs: [], conversations: [], now: T0, csrf_token: 't' }));
  byText(root.querySelector('#agent-run'), 'button', 'Interrupt turn').click();
  await flush();
  server({ run: s.run, events: s.events });
  root = await render(app);
  assert.equal(root.querySelector('#agent-run [role=alert]').textContent, 'Agent did not acknowledge the command');
  done();
});

test('the shell: a tree selection redraws only views that draw the selection, or a view not yet shown', () => {
  for (const tab of ['Weights', 'Systems', 'Atlas']) assert.ok(!SELECTION_FREE.has(tab), tab + ' draws the selection (or is left as it was)');
  for (const tab of ['Programmer', 'Run', 'Probe', 'Trace', 'Tests', 'Chunks', 'Export', 'Settings']) assert.ok(SELECTION_FREE.has(tab), tab);
  assert.equal(renderOnSelect('Programmer', 'Programmer'), false);
  assert.equal(renderOnSelect('Weights', 'Weights'), true);
  assert.equal(renderOnSelect('Trace', 'Tests'), true, 'the Tests view sets app.tab = Trace and then selects: Trace is drawn');
  assert.equal(viewKey('Programmer', 'blocks.0'), viewKey('Programmer', 'model'), 'the scroll position does not depend on the selection');
  assert.equal(viewKey('Weights', 'blocks.0'), 'Weightsblocks.0');
});
