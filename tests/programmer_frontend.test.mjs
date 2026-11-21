// Model-free and provider-free Programmer tab: the status line, tool-call timers, cost display and earlier
// conversations. Every server answer is faked; no bridge, provider or model is involved.
import test from 'node:test';
import assert from 'node:assert/strict';
import { programmerView, onAgentEvent, programmerStatus, toolCalls, toolLabel, tickProgrammer, refreshProgrammerStatus, SILENCE_S, tickerCount, stopTicker, setEndedRefreshMs } from '../web/js/work.js';
import { installStrictDom, mount, all } from './strict_dom.mjs';

const tick = () => new Promise((r) => setImmediate(r));
const settle = async () => { for (let k = 0; k < 6; k++) await tick(); };
// The browser-like fake DOM of tests/strict_dom.mjs (until 2026-11-17 this file had a looser one, under which the
// tab's "children.at is not a function" failure passed). A view is mounted the way the shell mounts it.
const { view: VIEW } = installStrictDom();
const intervals = [];
global.setInterval = (fn, ms) => { intervals.push(fn); return intervals.length; };
global.clearInterval = () => {};
const walk = all;
const text = (node) => node.textContent ?? '';
const button = (root, label) => walk(root).find((n) => n.tag === 'button' && text(n) === label);
const byRole = (root, role) => walk(root).find((n) => n.role === role);
const TOKEN = 'tok-1';
const json = (data, ok = true, status = ok ? 200 : 409) => ({ ok, status, statusText: 'Conflict', headers: { get: () => 'application/json' }, json: async () => data });

// A fake clock: the browser's clock runs `skew` ms ahead of the server's.
function clock(serverStart, skew) {
  let server = serverStart;
  global.Date.now = () => server * 1000 + skew;
  return { set(s) { server = s; }, add(s) { server += s; }, get server() { return server; } };
}
const realNow = Date.now;

function makeApp(over = {}) {
  return { tab: 'Programmer', keep: {}, sel: 'model', variant: 'current', rerenders: 0, rerender() { this.rerenders++; },
    refreshStatus: async () => {}, state: { task: { title: 'Rename', instructions: 'Make Paris say Rome', scope: ['model'], suite: null, preserve: null }, suites: [],
      harnesses: [{ name: 'claude', label: 'Claude Code', installed: true, native: true }, { name: 'codex', label: 'Codex', installed: true, native: true },
        { name: 'custom', label: 'custom command', installed: true }] }, ...over };
}

// The server: GET /api/harness and /api/harness?id=..., conversations, auth; POSTs recorded.
function server({ run = null, events = [], now, conversations = [], auth = { status: 'ready', method: 'claude.ai', billing: 'subscription' }, post = {} } = {}) {
  const s = { run, events, posts: [], gets: [], now, conversations, conversation: {} };
  global.fetch = async (url, o = {}) => {
    const u = String(url);
    if (o.method === 'POST') {
      const body = JSON.parse(o.body || '{}');
      s.posts.push({ url: u, body, token: o.headers?.['X-WSP-Auth'] });
      if (u.startsWith('/api/harness/auth')) return json({ ...auth, csrf_token: TOKEN });
      const answer = post[u];
      if (typeof answer === 'function') return answer(body);
      if (answer && answer.headers) return answer;           // a prepared response (json(data, ok))
      return json(answer ?? { ...(s.run || {}), now: s.now() });
    }
    s.gets.push(u);
    if (u.startsWith('/api/harness/auth')) return json({ ...auth, csrf_token: TOKEN });
    // The open workspace, and each run's own (the tab shows only this workspace's runs).
    if (u === '/api/harness') return json({ runs: s.run ? [{ workspace: '/ws/a', ...s.run }] : [], log: [], now: s.now(), csrf_token: TOKEN, workspace: '/ws/a' });
    if (u.startsWith('/api/harness/conversations')) return json({ conversations: s.conversations, now: s.now(), csrf_token: TOKEN, locking: s.locking || null });
    if (u.startsWith('/api/harness/conversation?')) { const id = new URLSearchParams(u.split('?')[1]).get('id'); return json(s.conversation[id] || { error: 'Unknown conversation' }, !!s.conversation[id]); }
    if (u.startsWith('/api/harness?')) {
      const p = new URLSearchParams(u.split('?')[1]);
      const since = +(p.get('since') || 0);
      return json({ ...s.run, events: s.events.slice(since), now: s.now() });
    }
    if (u.startsWith('/api/brief')) return json('brief');
    return json({});
  };
  return s;
}
async function view(app) { const root = document.createElement('div'); await programmerView(root, app); mount(VIEW, root); return root; }
const statusOf = (root) => { const live = byRole(root, 'status'); return { live: text(live), time: text(live.parentElement.children[1]) }; };
const toolTimes = (root) => walk(root).filter((n) => n.tag === 'span' && /^ev-time/.test(n.className)).map(text);

const T0 = 1_800_000_000;
const use = (i, t, id, name, input) => ({ kind: 'tool_use', i, t, id, name, input });
const res = (i, t, id, extra = {}) => ({ kind: 'tool_result', i, t, id, text: 'ok', ...extra });
const nativeRun = (status = 'running', extra = {}) => ({ id: 'run-1', harness: 'claude', native: true, status, started: T0, ended: null, pending: [], mode: 'tools', model: '', ...extra });

test('a running tool is named with its main argument and timed from the server clock, not from ticks', async () => {
  const c = clock(T0 + 100, 7_000);                        // browser clock 7 s ahead of the server
  const events = [{ kind: 'user', i: 0, t: T0, text: 'brief' }, { kind: 'text', i: 1, t: T0 + 40, text: 'Looking' },
    use(2, T0 + 49, 'a', 'mcp__wsp__inspect', { address: 'L41N12481', what: 'card' })];
  const s = server({ run: nativeRun(), events, now: () => c.server });
  const app = makeApp();
  const root = await view(app);
  assert.deepEqual(statusOf(root), { live: 'Running inspect card L41N12481', time: '51 s · turn 1 min 40 s' });
  assert.equal(byRole(root, 'status').getAttribute('aria-live'), 'polite');
  assert.deepEqual(toolTimes(root), ['running · 51 s']);
  // The tab was in the background for two minutes: no tick ran, the next one is still right.
  c.add(120);
  await tickProgrammer(app);
  assert.equal(statusOf(root).time.startsWith('2 min 51 s · turn 3 min 40 s'), true, statusOf(root).time);
  assert.deepEqual(toolTimes(root), ['running · 2 min 51 s']);
  assert.ok(s.gets.some((u) => u.startsWith('/api/harness?id=run-1&since=3')), 'missed events are fetched from the first unseen index');
  Date.now = realNow;
});

test('calls issued together: one runs, the rest wait behind it, and each shows its own time and duration', async () => {
  const c = clock(T0 + 75, 0);
  const events = [{ kind: 'user', i: 0, t: T0, text: 'go' }, use(1, T0, 'a', 'inspect', { address: 'L1N1', what: 'card' }),
    use(2, T0, 'b', 'inspect', { address: 'L1N2', what: 'card' }), use(3, T0, 'c', 'inspect', { address: 'L1N3', what: 'card' }),
    use(4, T0, 'd', 'inspect', { address: 'L1N4', what: 'card' }), res(5, T0 + 50, 'a')];
  server({ run: nativeRun(), events, now: () => c.server });
  const app = makeApp();
  const root = await view(app);
  assert.deepEqual(statusOf(root), { live: 'Running inspect card L1N2 · 2 more calls waiting behind it', time: '25 s · turn 1 min 15 s' });
  assert.deepEqual(toolTimes(root), ['took 50 s', 'running · 25 s', 'waiting behind another call · 1 min 15 s', 'waiting behind another call · 1 min 15 s']);
  // The second result arrives over the event stream.
  c.set(T0 + 100);
  onAgentEvent({ type: 'agent', run: 'run-1', t: T0 + 100, event: res(6, T0 + 100, 'b') }, app);
  assert.deepEqual(statusOf(root).live, 'Running inspect card L1N3 · 1 more call waiting behind it');
  assert.deepEqual(toolTimes(root).slice(0, 3), ['took 50 s', 'took 50 s · waited 50 s first', 'running · 0 s']);
  Date.now = realNow;
});

test('a reload in the middle of a run shows the same status as before it', async () => {
  const c = clock(T0 + 30, -3_000);
  const events = [{ kind: 'user', i: 0, t: T0, text: 'go' }, use(1, T0 + 10, 'a', 'atlas', { action: 'search', selector: 'word:Paris' })];
  server({ run: nativeRun(), events, now: () => c.server });
  const before = statusOf(await view(makeApp()));
  const after = statusOf(await view(makeApp()));           // a fresh page: new app.keep, nothing carried over
  assert.deepEqual(before, after);
  assert.deepEqual(after, { live: 'Running atlas search word:Paris', time: '20 s · turn 30 s' });
  Date.now = realNow;
});

test('waiting for the model, then a long silence said plainly without calling the run stuck', async () => {
  const c = clock(T0 + 12, 0);
  const events = [{ kind: 'user', i: 0, t: T0, text: 'go' }, use(1, T0 + 2, 'a', 'overview', {}), res(2, T0 + 5, 'a')];
  server({ run: nativeRun(), events, now: () => c.server });
  const app = makeApp();
  const root = await view(app);
  assert.deepEqual(statusOf(root), { live: 'Waiting for the model', time: '7 s since the last output · turn 12 s' });
  c.set(T0 + 5 + SILENCE_S + 15);
  refreshProgrammerStatus(app.keep.prog);
  const st = statusOf(root);
  assert.equal(st.live, 'Waiting for the model · no output for over 1 min 0 s');
  assert.match(st.time, /no output for 1 min 15 s$/);
  assert.doesNotMatch(st.live + st.time, /stuck|dead|hung/i);
  Date.now = realNow;
});

test('the live sentence changes only with the state; the seconds tick beside it', async () => {
  const c = clock(T0 + 10, 0);
  server({ run: nativeRun(), events: [{ kind: 'user', i: 0, t: T0, text: 'go' }, use(1, T0 + 1, 'a', 'run', { prompt: 'The capital of France is' })], now: () => c.server });
  const app = makeApp();
  const root = await view(app);
  const live = byRole(root, 'status');
  const writes = [];
  let value = live.textContent;
  Object.defineProperty(live, 'textContent', { get: () => value, set: (v) => { writes.push(v); value = v; } });
  for (let k = 0; k < 5; k++) { c.add(1); refreshProgrammerStatus(app.keep.prog); }
  assert.deepEqual(writes, [], 'no announcement while only the seconds change');
  assert.equal(statusOf(root).time, '14 s · turn 15 s');
  Date.now = realNow;
});

test('finished, failed, stopped, idle and interrupted runs each say so; open calls of an ended turn are not completed', () => {
  const ev = [{ kind: 'user', i: 0, t: T0, text: 'go' }, use(1, T0 + 1, 'a', 'edit', { script: 'x', message: 'scale' }), { kind: 'result', i: 2, t: T0 + 20, ok: true }];
  assert.deepEqual(programmerStatus(nativeRun('idle'), ev, T0 + 99), { state: 'idle', live: 'Turn finished · waiting for your message', time: 'turn took 20 s' });
  assert.equal(programmerStatus(nativeRun('idle'), [...ev.slice(0, 2), { kind: 'result', i: 2, t: T0 + 3, ok: false }], T0 + 9).live, 'Turn stopped or failed · waiting for your message');
  assert.equal(programmerStatus(nativeRun('failed', { ended: T0 + 30 }), [...ev, { kind: 'error', i: 3, t: T0 + 30, text: 'Agent bridge exited (1). Start a new conversation.' }], T0 + 99).live,
    'Failed: Agent bridge exited (1). Start a new conversation.');
  assert.equal(programmerStatus({ id: 'r', status: 'stopped', started: T0, ended: T0 + 5 }, [], T0 + 9).live, 'Stopped');
  assert.deepEqual(programmerStatus({ id: 'r', status: 'done', started: T0, ended: T0 + 65 }, [], T0 + 99), { state: 'done', live: 'Finished', time: 'ran 1 min 5 s' });
  assert.equal(programmerStatus(nativeRun('closed'), ev, T0 + 99).live, 'Conversation closed');
  assert.equal(programmerStatus(nativeRun('waiting', { pending: [{ request_id: '1', title: 'Write' }] }), [...ev.slice(0, 2), { kind: 'prompt', i: 2, t: T0 + 4, title: 'Write' }], T0 + 10).live, 'Waiting for your response: Write');
  assert.deepEqual(toolCalls(ev, true).map((c) => c.state), ['not_completed']);
  assert.deepEqual(toolCalls(ev.slice(0, 2), false).map((c) => c.state), ['not_completed']);
  assert.equal(toolLabel({ name: 'mcp__wsp__edit', input: { script: 'x', message: 'scale L3 neurons' } }), 'edit scale L3 neurons');
  assert.equal(toolLabel({ name: 'commandExecution', input: { command: 'python fix.py' } }), 'commandExecution python fix.py');
});

// ------------------------------------------------------------------------------------------------ cost
const resultEv = { kind: 'result', i: 2, t: T0 + 20, ok: true, turns: 4, duration_ms: 20000, input_tokens: 12000, output_tokens: 900, cost_usd: 0.42 };
const finished = [{ kind: 'user', i: 0, t: T0, text: 'go' }, { kind: 'text', i: 1, t: T0 + 10, text: 'done' }, resultEv];

test('a subscription run shows no dollar cost and no budget field; turns, time and tokens stay', async () => {
  const c = clock(T0 + 30, 0);
  const s = server({ run: nativeRun('idle', { billing: 'subscription' }), events: finished, now: () => c.server });
  const app = makeApp();
  app.keep.prog = { budget: 3 };                           // a value typed before: never sent for a subscription
  const root = await view(app);
  const body = text(root);
  assert.match(body, /4 turns · 20 s · 12.0k tokens in · 900 out/);
  assert.doesNotMatch(body, /\$|USD|budget/i);
  Date.now = realNow;
});

test('a sign-in billed per call keeps the Claude budget field and the cost figure', async () => {
  const c = clock(T0 + 30, 0);
  const s = server({ run: nativeRun('closed', { billing: 'per_call' }), events: finished, now: () => c.server,
    auth: { status: 'ready', method: 'api_key', billing: 'per_call' }, post: { '/api/task': json({ task: {} }) } });
  const app = makeApp();
  let root = await view(app);
  assert.match(text(root), /est\. \$0\.42 \(billed per call\)/);
  const budget = walk(root).find((n) => n.tag === 'label' && /Claude budget \(USD, billed per call\)/.test(text(n)));
  assert.ok(budget, 'budget field shown');
  app.keep.prog.budget = 2.5;
  root = await view(app);
  await button(root, 'Start programmer').click(); await settle();
  assert.equal(s.posts.find((p) => p.url === '/api/harness/start').body.budget, 2.5);
  Date.now = realNow;
});

test('a subscription start sends no budget even if one was typed earlier; a custom harness keeps its own cost estimate', async () => {
  const c = clock(T0 + 30, 0);
  const s = server({ now: () => c.server, post: { '/api/task': json({ task: {} }) } });
  const app = makeApp();
  app.keep.prog = { budget: 4 };
  const root = await view(app);
  await button(root, 'Start programmer').click(); await settle();
  assert.equal(s.posts.find((p) => p.url === '/api/harness/start').body.budget, 0);
  const c2 = clock(T0 + 30, 0);
  server({ run: { id: 'run-2', harness: 'custom', status: 'done', started: T0, ended: T0 + 20 }, events: finished, now: () => c2.server });
  const legacy = makeApp();
  legacy.keep.prog = { harness: 'custom' };
  assert.match(text(await view(legacy)), /est\. \$0\.42 \(harness estimate\)/);
  Date.now = realNow;
});

// ------------------------------------------------------------------------------------------------ conversations
const savedEvents = [{ kind: 'user', i: 0, t: T0, text: 'BRIEF' }, { kind: 'init', i: 1, t: T0 + 1, session: 'session-1', model: 'm' },
  use(2, T0 + 5, 'a', 'mcp__wsp__overview', {}), res(3, T0 + 9, 'a'), { kind: 'result', i: 4, t: T0 + 12, ok: true, turns: 2 },
  { kind: 'user', i: 5, t: T0 + 30, text: 'look at L41' }, use(6, T0 + 31, 'b', 'mcp__wsp__inspect', { address: 'L41N12481', what: 'card' })];
const ok = { ok: true, code: null, reason: null };
const conv = (over = {}) => ({ id: 'conv-20261116-101500-0000abcd', started: T0, ended: T0 + 40, provider: 'claude', mode: 'tools', model: '', status: 'interrupted',
  n_events: savedEvents.length, task: 'Paris to Rome', session: 'session-1', native: true, live: false, elsewhere: false, billing: 'subscription', resume: ok, ...over });
function savedServer(c, extra = {}) {
  const cl = clock(T0 + 600, 0);
  const s = server({ now: () => cl.server, conversations: [c], post: { '/api/task': json({ task: {} }) }, ...extra });
  s.conversation[c.id] = { meta: { ...c, interrupted: 'The IDE server stopped while a turn was running.' }, events: savedEvents, resume: c.resume, now: T0 + 600 };
  return s;
}

test('after a restart the tab lists earlier conversations and reads one without contacting a provider', async () => {
  const s = savedServer(conv());
  const app = makeApp();
  let root = await view(app);
  const table = walk(root).find((n) => n.id === 'agent-conversations');
  assert.match(text(table), /Claude Code · tools/); assert.match(text(table), /Paris to Rome/);
  assert.deepEqual(walk(table).filter((n) => n.tag === 'td').map(text).slice(1, 5), ['Claude Code · tools', 'Paris to Rome', 'interrupted', '7']);
  await button(root, 'Open').click(); await settle();
  root = await view(app);
  const box = walk(root).find((n) => n.id === 'agent-conversation');
  assert.match(text(box), /Interrupted: The IDE server stopped while a turn was running\./);
  assert.match(text(box), /look at L41/);
  assert.deepEqual(walk(box).filter((n) => /^ev-time/.test(n.className)).map(text), ['took 4 s', 'not completed']);
  assert.equal(button(box, 'Resume').disabled, false); assert.equal(button(box, 'New conversation').disabled, false);
  assert.deepEqual(s.posts.filter((p) => !p.url.startsWith('/api/harness/auth')), [], 'opening and reading posted nothing');
  assert.ok(s.gets.includes('/api/harness/conversation?id=conv-20261116-101500-0000abcd'));
  Date.now = realNow;
});

test('Resume sends the conversation and the optional message, then shows the resumed run with its history', async () => {
  const resumed = { id: 'run-9', harness: 'claude', native: true, status: 'running', started: T0 + 600, pending: [], conversation: 'conv-20261116-101500-0000abcd', resumed: true, billing: 'subscription',
    events: [...savedEvents, { kind: 'resumed', i: 7, t: T0 + 600, text: 'Resumed with the provider\'s own session. The workspace may have changed since this conversation last ran (weights, patches, tests); the model was told so and given the current patch list.' }], now: T0 + 600 };
  const s = savedServer(conv(), { post: { '/api/harness/resume': json(resumed) } });
  const app = makeApp();
  let root = await view(app);
  await button(root, 'Open').click(); await settle();
  root = await view(app);
  const box = walk(root).find((n) => n.id === 'agent-conversation');
  const message = walk(box).find((n) => n.tag === 'textarea');
  message.dispatchEvent({ type: 'input', target: { value: 'carry on with L42' } });
  await button(box, 'Resume').click(); await settle();
  assert.deepEqual(s.posts.find((p) => p.url === '/api/harness/resume').body, { conversation: 'conv-20261116-101500-0000abcd', text: 'carry on with L42' });
  const st = app.keep.prog;
  assert.equal(st.run.id, 'run-9'); assert.equal(st.viewing, null); assert.equal(st.events.length, 8);
  s.run = { ...resumed }; delete s.run.events; s.events = resumed.events;
  root = await view(app);
  assert.match(text(root), /RESUMED|resumed/); assert.match(text(root), /workspace may have changed/);
  Date.now = realNow;
});

for (const [code, reason] of [
  ['gone', 'Claude Code no longer has session session-1 for this workspace\'s agent directory, so it cannot be resumed. Start a new conversation.'],
  ['task', 'The task changed since this conversation started (instructions). Resuming would continue work on a different task: start a new conversation, or set the task back.'],
  ['no_session', 'No Claude Code session id was recorded (the provider never confirmed the conversation started), so there is nothing to resume. Start a new conversation.'],
  ['legacy', 'This was a custom command run. Those run once and keep no session the IDE could continue. Start a new conversation.'],
  ['workspace', 'This conversation ran in /old/ws; the workspace is now at /new/ws, where Claude Code keeps no session for it. Start a new conversation.'],
  ['busy', 'Close the current programmer conversation before resuming another.']]) {
  test(`a conversation that cannot be resumed (${code}) says why and still offers New conversation`, async () => {
    const s = savedServer(conv({ resume: { ok: false, code, reason } }));
    const app = makeApp();
    let root = await view(app);
    await button(root, 'Open').click(); await settle();
    root = await view(app);
    const box = walk(root).find((n) => n.id === 'agent-conversation');
    assert.equal(text(walk(box).find((n) => n.id === 'resume-refusal')), 'Resume is not available: ' + reason);
    assert.equal(button(box, 'Resume').disabled, true);
    assert.equal(button(box, 'New conversation').disabled, false);
    await button(box, 'New conversation').click(); await settle();
    assert.ok(s.posts.some((p) => p.url === '/api/harness/start'), 'New conversation starts one');
    assert.ok(!s.posts.some((p) => p.url === '/api/harness/resume'));
    Date.now = realNow;
  });
}

test('a refusal the server gives at the click is shown where Resume was clicked', async () => {
  const s = savedServer(conv(), { post: { '/api/harness/resume': json({ error: 'Claude Code no longer has session session-1. Start a new conversation.', refusal: 'gone' }, false) } });
  const app = makeApp();
  let root = await view(app);
  await button(root, 'Open').click(); await settle();
  root = await view(app);
  await button(walk(root).find((n) => n.id === 'agent-conversation'), 'Resume').click(); await settle();
  root = await view(app);
  const box = walk(root).find((n) => n.id === 'agent-conversation');
  assert.match(text(box), /no longer has session session-1/);
  assert.equal(app.keep.prog.run, null);
  Date.now = realNow;
});

test('Resume waits for sign-in like Start does', async () => {
  savedServer(conv(), { auth: { status: 'signed_out' } });
  const app = makeApp();
  let root = await view(app);
  await button(root, 'Open').click(); await settle();
  root = await view(app);
  const box = walk(root).find((n) => n.id === 'agent-conversation');
  assert.equal(button(box, 'Resume').disabled, true);
  assert.match(text(box), /Resume is not available: Sign in to Claude Code above before resuming\./);
  Date.now = realNow;
});

test('a page left open across a server restart stops showing the run as working', async () => {
  const c = clock(T0 + 20, 0);
  const s = server({ run: nativeRun(), events: [{ kind: 'user', i: 0, t: T0, text: 'go' }, use(1, T0 + 2, 'a', 'overview', {})], now: () => c.server });
  const app = makeApp();
  const root = await view(app);
  assert.equal(statusOf(root).live, 'Running overview');
  // The server restarts: it no longer knows the run.
  global.fetch = async (url) => String(url).startsWith('/api/harness?id=') ? json({ error: 'Unknown run' }, false, 404) : json({});
  c.add(30);
  await tickProgrammer(app);
  assert.equal(app.keep.prog.run.status, 'interrupted');
  assert.ok(app.rerenders > 0);
  assert.match(programmerStatus(app.keep.prog.run, app.keep.prog.events, c.server).live, /^Interrupted: the IDE server restarted/);
  assert.deepEqual(toolCalls(app.keep.prog.events, false).map((x) => x.state), ['not_completed']);
  Date.now = realNow;
});

// ------------------------------------------------------------------------------------------------ review fixes
test('every Programmer POST carries the token the server handed out, for native and custom harnesses alike', async () => {
  const c = clock(T0 + 30, 0);
  const s = server({ now: () => c.server, post: { '/api/task': json({ task: {} }) } });
  const app = makeApp();
  let root = await view(app);
  await button(root, 'Start programmer').click(); await settle();
  const custom = makeApp();
  custom.keep.prog = { harness: 'custom' };                // no sign-in request: the token comes from GET /api/harness
  root = await view(custom);
  await button(root, 'Start programmer').click(); await settle();
  await button(root, 'Prepare only').click(); await settle();
  const harnessPosts = s.posts.filter((p) => p.url.startsWith('/api/harness/'));
  assert.ok(harnessPosts.length >= 3);
  assert.deepEqual([...new Set(harnessPosts.map((p) => p.token))], [TOKEN]);
  Date.now = realNow;
});

test('one ticker at most: renders reuse it; an ended run, leaving the tab or switching workspace stops it', async () => {
  stopTicker();
  const c = clock(T0 + 30, 0);
  const s = server({ run: nativeRun(), events: [{ kind: 'user', i: 0, t: T0, text: 'go' }, use(1, T0 + 2, 'a', 'overview', {})], now: () => c.server });
  const app = makeApp();
  const before = intervals.length;
  for (let k = 0; k < 3; k++) await view(app);
  assert.equal(tickerCount(), 1); assert.equal(intervals.length - before, 1, 'one interval for three renders');
  app.tab = 'Tests';
  await tickProgrammer(app);
  assert.equal(tickerCount(), 0, 'leaving the tab stops it');
  app.tab = 'Programmer'; await view(app);
  assert.equal(tickerCount(), 1);
  app.keep = {};                                           // a workspace switch resets app.keep
  await tickProgrammer(app);
  assert.equal(tickerCount(), 0, 'a workspace switch stops it');
  // Switched while the tab stays open, and the new workspace's first event arrives before any render: the old
  // workspace's ticker does not carry on for the new state.
  await view(app);
  assert.equal(tickerCount(), 1);
  app.keep = {};
  onAgentEvent({ type: 'agent', run: 'run-7', event: { kind: 'user', i: 0, t: T0 + 30, text: 'other' } }, app);
  await tickProgrammer(app);
  assert.equal(tickerCount(), 0, 'the ticker belonged to the previous workspace');
  // An ended run with a call that was still open never starts one.
  s.run = nativeRun('closed', { ended: T0 + 20 });
  const ended = makeApp();
  await view(ended);
  assert.equal(tickerCount(), 0);
  // The run ends while the tab shows it: the next tick stops the ticker.
  s.run = nativeRun();
  const live = makeApp();
  await view(live);
  assert.equal(tickerCount(), 1);
  onAgentEvent({ type: 'run', run: nativeRun('closed', { ended: T0 + 40 }) }, live);
  await tickProgrammer(live);
  assert.equal(tickerCount(), 0);
  Date.now = realNow;
});

test('the status live region is one node kept across renders, written only once it is in the document', async () => {
  stopTicker();
  const c = clock(T0 + 10, 0);
  server({ run: nativeRun(), events: [{ kind: 'user', i: 0, t: T0, text: 'go' }, use(1, T0 + 1, 'a', 'overview', {})], now: () => c.server });
  const app = makeApp();
  const first = byRole(await view(app), 'status');
  const second = byRole(await view(app), 'status');
  assert.equal(first, second, 'the same live region node, re-attached');
  assert.equal(walk(await view(app)).filter((n) => n.role === 'status').length, 1);
  // Detached (a render builds the view before mounting it): the new sentence waits until the node is attached.
  let connected = false;
  Object.defineProperty(first, 'isConnected', { get: () => connected });
  onAgentEvent({ type: 'agent', run: 'run-1', event: res(2, T0 + 9, 'a') }, app);
  assert.equal(text(first), 'Running overview');
  connected = true;
  await new Promise((r) => setTimeout(r, 5));
  assert.equal(text(first), 'Waiting for the model');
  stopTicker();
  Date.now = realNow;
});

test('a resumed marker ends the calls before it: an interrupted call is not shown as running during resume startup', () => {
  const events = [{ kind: 'user', i: 0, t: T0, text: 'go' }, use(1, T0 + 1, 'a', 'mcp__wsp__inspect', { address: 'L1N1' }),
    { kind: 'resumed', i: 2, t: T0 + 500, text: 'Resuming' }];
  assert.deepEqual(toolCalls(events, true).map((x) => x.state), ['not_completed']);
  const st = programmerStatus(nativeRun('starting', { started: T0 + 500 }), events, T0 + 505);
  assert.equal(st.state, 'starting');
  for (const kind of ['resume_failed', 'init', 'error']) {
    assert.deepEqual(toolCalls([...events.slice(0, 2), { kind, i: 2, t: T0 + 3 }], true).map((x) => x.state), ['not_completed'], kind);
  }
});

test('only IDE tool calls queue; a provider\'s own tools run from the moment they are issued', () => {
  const events = [{ kind: 'user', i: 0, t: T0, text: 'go' }, use(1, T0 + 1, 'b', 'Bash', { command: 'ls' }),
    use(2, T0 + 1, 'x', 'mcp__wsp__inspect', { address: 'L1N1', what: 'card' }), use(3, T0 + 1, 'y', 'mcp__wsp__overview', {}),
    use(4, T0 + 2, 'c', 'commandExecution', { command: 'pwd' }), use(5, T0 + 2, 'z', 'inspect', { address: 'L1N2' })];
  const calls = toolCalls(events, true);
  assert.deepEqual(calls.map((x) => [x.id, x.state]), [['b', 'running'], ['x', 'running'], ['y', 'queued'], ['c', 'running'], ['z', 'queued']]);
  const st = programmerStatus(nativeRun(), events, T0 + 10);
  assert.equal(st.live, 'Running inspect card L1N1 · 2 more calls running · 2 more calls waiting behind it');
  // A Bash result does not move the IDE queue: the next IDE call starts when the IDE answered.
  const after = toolCalls([...events, res(6, T0 + 4, 'b'), res(7, T0 + 6, 'x')], true);
  assert.deepEqual(after.find((x) => x.id === 'b'), { ...after.find((x) => x.id === 'b'), start: T0 + 1, end: T0 + 4, state: 'done' });
  assert.equal(after.find((x) => x.id === 'y').start, T0 + 6);
});

test('an unreadable saved conversation is one row; the others are listed and open as usual', async () => {
  const s = savedServer(conv());
  s.conversations = [{ id: 'conv-20261116-090000-00000bad', unreadable: 'meta.json is not valid JSON' }, conv()];
  const app = makeApp();
  const root = await view(app);
  const table = walk(root).find((n) => n.id === 'agent-conversations');
  assert.match(text(table), /unreadable conversation: meta\.json is not valid JSON/);
  assert.equal(walk(table).filter((n) => n.tag === 'button' && text(n) === 'Open').length, 1);
  assert.equal(button(root, 'Start programmer').disabled, false);
  Date.now = realNow;
});

test('a conversation recorded as gone offers Try again, which asks the server for one more attempt', async () => {
  const s = savedServer(conv({ resume: { ok: false, code: 'gone', reason: 'Claude Code has no session session-1. Start a new conversation, or try again.', retry: true } }),
    { post: { '/api/harness/resume': json({ error: 'still gone', refusal: 'gone' }, false) } });
  const app = makeApp();
  let root = await view(app);
  await button(root, 'Open').click(); await settle();
  root = await view(app);
  const box = walk(root).find((n) => n.id === 'agent-conversation');
  assert.equal(button(box, 'Resume').disabled, true);
  await button(box, 'Try again').click(); await settle();
  assert.deepEqual(s.posts.find((p) => p.url === '/api/harness/resume').body, { conversation: 'conv-20261116-101500-0000abcd', text: '', retry: true });
  Date.now = realNow;
});

test('a conversation another IDE process holds can be marked interrupted, only after a confirmation', async () => {
  const held = conv({ elsewhere: true, owner_pid: 4242, status: 'running',
    resume: { ok: false, code: 'elsewhere', reason: 'Another IDE process on this machine (pid 4242) holds this conversation.', release: true } });
  const s = savedServer(held, { post: { '/api/harness/conversation/release': json({ conversation: { ...held, elsewhere: false, status: 'interrupted' } }) } });
  const app = makeApp();
  let root = await view(app);
  await button(root, 'Open').click(); await settle();
  root = await view(app);
  let box = walk(root).find((n) => n.id === 'agent-conversation');
  const asked = [];
  global.confirm = (message) => { asked.push(message); return false; };
  await button(box, 'It is not running; mark interrupted').click(); await settle();
  assert.equal(asked.length, 1); assert.match(asked[0], /pid 4242/);
  assert.equal(s.posts.filter((p) => p.url === '/api/harness/conversation/release').length, 0, 'declined: nothing sent');
  global.confirm = () => true;
  await button(box, 'It is not running; mark interrupted').click(); await settle();
  assert.deepEqual(s.posts.find((p) => p.url === '/api/harness/conversation/release').body, { conversation: held.id, confirm: true });
  assert.equal(s.posts.find((p) => p.url === '/api/harness/conversation/release').token, TOKEN);
  delete global.confirm;
  Date.now = realNow;
});

test('a per-call Claude resume says its budget starts again from zero', async () => {
  savedServer(conv({ billing: 'per_call', budget: 2.5 }), { auth: { status: 'ready', method: 'api_key', billing: 'per_call' } });
  const app = makeApp();
  let root = await view(app);
  await button(root, 'Open').click(); await settle();
  root = await view(app);
  const note = walk(root).find((n) => n.id === 'resume-budget');
  assert.match(text(note), /\$2\.5 budget again, counted from zero/);
  Date.now = realNow;
});

test('the conversation list is fetched once more shortly after a run ends, even if the final run event was missed', async () => {
  stopTicker(); setEndedRefreshMs(5);
  const c = clock(T0 + 30, 0);
  const s = server({ run: nativeRun(), events: [{ kind: 'user', i: 0, t: T0, text: 'go' }], now: () => c.server });
  const app = makeApp();
  await view(app);
  const before = app.rerenders;
  onAgentEvent({ type: 'run', run: nativeRun('failed', { ended: T0 + 31 }) }, app);
  const afterEvent = app.rerenders;
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(app.rerenders, afterEvent + 1, 'one more render (and list fetch) after the end');
  // The final event missed: the next sync finds the run ended and the same one refresh follows.
  const missed = makeApp();
  s.run = nativeRun();
  await view(missed);
  s.run = nativeRun('closed', { ended: T0 + 40 });
  c.add(20);
  await tickProgrammer(missed);
  const n = missed.rerenders;
  await new Promise((r) => setTimeout(r, 30));
  assert.equal(missed.rerenders, n + 1);
  assert.ok(before >= 0);
  setEndedRefreshMs(1500); stopTicker();
  Date.now = realNow;
});

test('a workspace on a filesystem without file locks says so above its conversations', async () => {
  const s = savedServer(conv());
  s.locking = 'The filesystem of this workspace does not support file locks, so saved conversations cannot be protected against two IDE processes using the same one.';
  const root = await view(makeApp());
  assert.match(text(walk(root).find((n) => n.id === 'conversations-locking')), /does not support file locks/);
  Date.now = realNow;
});
