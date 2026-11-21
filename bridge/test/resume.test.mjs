// Resuming a recorded session: the option each provider is given, and what happens when it cannot be continued.
// Fakes only: an injected SDK object, a fake RPC class and a fake `codex app-server` script. No provider is contacted.
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { ClaudeAdapter, CodexAdapter, resumePromptGuard, MISMATCH_SENT } from '../adapters.mjs';
import { messageQueue } from '../protocol.mjs';

const tick = () => new Promise(resolve => setImmediate(resolve));
const config = { cwd: '/tmp/ws/agent', executable: 'test-agent', mode: 'tools', server: null, budget: 0 };
function recorder() {
  const events = [], states = [], errors = [];
  return { events, states, errors, args: [e => events.push(e), s => states.push(s), e => errors.push(e)] };
}
function fakeSdk({ known = true, messages = [], infoError = null } = {}) {
  const output = messageQueue();
  const sdk = { infoAsked: [], messagesAsked: [], queries: [], interrupted: 0,
    async getSessionInfo(id, options) { sdk.infoAsked.push({ id, options }); if (infoError) throw infoError; return known ? { sessionId: id, summary: 'earlier' } : undefined; },
    async getSessionMessages(id, options) { sdk.messagesAsked.push({ id, options }); return messages; },
    query(input) { sdk.queries.push(input.options); return { [Symbol.asyncIterator]: () => output[Symbol.asyncIterator](), close: () => output.end(), interrupt: async () => { sdk.interrupted++; } }; } };
  return { sdk, output };
}

test('Claude resume checks the session exists for the agent directory, then passes it as the SDK resume option', async () => {
  const r = recorder(); const { sdk, output } = fakeSdk();
  const a = new ClaudeAdapter({ ...config, resume: 'sess-1' }, ...r.args, sdk);
  await a.start();
  assert.deepEqual(sdk.infoAsked, [{ id: 'sess-1', options: { dir: '/tmp/ws/agent' } }]);
  assert.deepEqual(sdk.messagesAsked, []);                 // a found session needs no second look
  assert.equal(sdk.queries[0].resume, 'sess-1');
  assert.equal(sdk.queries[0].forkSession, undefined);
  output.push({ type: 'system', subtype: 'init', session_id: 'sess-1', tools: [] }); await tick();
  assert.equal(r.events.find(e => e.kind === 'init').session, 'sess-1');
  assert.deepEqual(r.errors, []);
  a.close(); await a.pump;
});

test('Claude resume is gone only when getSessionInfo has nothing and getSessionMessages is empty; refused before any query', async () => {
  const r = recorder(); const { sdk } = fakeSdk({ known: false, messages: [] });
  const a = new ClaudeAdapter({ ...config, resume: 'sess-gone' }, ...r.args, sdk);
  await assert.rejects(a.start(), (error) => error.resume === true && error.gone === true && /has no session sess-gone/.test(error.message));
  assert.deepEqual(sdk.messagesAsked, [{ id: 'sess-gone', options: { dir: '/tmp/ws/agent', limit: 1 } }]);
  assert.deepEqual(sdk.queries, []);
});

test('an ambiguous getSessionInfo (undefined, but the session has messages) goes ahead with the resume', async () => {
  const r = recorder(); const { sdk, output } = fakeSdk({ known: false, messages: [{ type: 'user', uuid: 'u1' }] });
  const a = new ClaudeAdapter({ ...config, resume: 'sess-side' }, ...r.args, sdk);
  await a.start();
  assert.equal(sdk.queries[0].resume, 'sess-side');
  output.push({ type: 'system', subtype: 'init', session_id: 'sess-side', tools: [] }); await tick();
  assert.deepEqual(r.errors, []);
  a.close(); await a.pump;
});

test('a session check that throws is a failed attempt, never gone', async () => {
  const r = recorder(); const { sdk } = fakeSdk({ infoError: new Error('EACCES reading the project store') });
  const a = new ClaudeAdapter({ ...config, resume: 'sess-1' }, ...r.args, sdk);
  await assert.rejects(a.start(), (error) => error.resume === true && error.gone === false && /could not check session sess-1: EACCES/.test(error.message));
  assert.deepEqual(sdk.queries, []);
});

test('Claude starting a different session than the one resumed is a resume failure, not a new conversation', async () => {
  const r = recorder(); const { sdk, output } = fakeSdk();
  const a = new ClaudeAdapter({ ...config, resume: 'sess-1' }, ...r.args, sdk);
  await a.start();
  output.push({ type: 'system', subtype: 'init', session_id: 'sess-fresh', tools: [] }); await tick(); await tick();
  assert.equal(r.events.filter(e => e.kind === 'init').length, 0);
  assert.equal(r.errors.length, 1); assert.equal(r.errors[0].resume, true); assert.equal(r.errors[0].gone, false);
  assert.match(r.errors[0].message, /sess-fresh/);
  assert.equal(sdk.interrupted, 1);                        // the turn the other session started is stopped at once
  a.close();
});

test('a fresh Claude conversation asks for no session and passes no resume option', async () => {
  const r = recorder(); const { sdk } = fakeSdk();
  const a = new ClaudeAdapter(config, ...r.args, sdk); await a.start();
  assert.deepEqual(sdk.infoAsked, []); assert.equal('resume' in sdk.queries[0], false);
  a.close(); await a.pump;
});

class FakeRpc {
  constructor(command, args, options, notification, request) { this.calls = []; this.notify = notification; this.closed = false; this.answer = FakeRpc.answer; }
  async request(method, params) {
    this.calls.push({ method, params });
    if (method === 'thread/resume') return this.answer(params);
    if (method === 'thread/start') return { thread: { id: 'thread-new' } };
    if (method === 'turn/start') return { turn: { id: 'turn-1' } };
    return {};
  }
  write(message) { this.calls.push(message); }
  close() { this.closed = true; }
}

test('Codex resume uses thread/resume with the recorded thread and the same settings, and never thread/start', async () => {
  const r = recorder(); FakeRpc.answer = params => ({ thread: { id: params.threadId } });
  const a = new CodexAdapter({ ...config, resume: 'thread-7', model: 'm' }, ...r.args, FakeRpc); await a.start();
  const resume = a.rpc.calls.find(c => c.method === 'thread/resume');
  assert.deepEqual(resume.params, { threadId: 'thread-7', cwd: '/tmp/ws/agent', approvalPolicy: 'untrusted', sandbox: 'read-only', model: 'm' });
  assert.equal(a.rpc.calls.some(c => c.method === 'thread/start'), false);
  assert.equal(r.events.find(e => e.kind === 'init').session, 'thread-7');
  await a.send('continue');
  assert.equal(a.rpc.calls.find(c => c.method === 'turn/start').params.threadId, 'thread-7');
  a.close();
});

test('Codex resume errors and a different thread id are resume failures with nothing sent', async () => {
  for (const [answer, words] of [[() => { throw new Error('no rollout found'); }, /could not resume thread thread-7: no rollout found/],
    [() => ({ thread: { id: 'thread-new' } }), /answered with thread thread-new instead of resuming thread-7/]]) {
    const r = recorder(); FakeRpc.answer = answer;
    const a = new CodexAdapter({ ...config, resume: 'thread-7' }, ...r.args, FakeRpc);
    await assert.rejects(a.start(), (error) => error.resume === true && error.gone === false && words.test(error.message));
    assert.equal(a.rpc.calls.some(c => ['thread/start', 'turn/start'].includes(c.method)), false);
    assert.equal(r.events.length, 0);
  }
});

test('the bridge process reports a refused resume as resume_failed and closes', async () => {
  const fake = fileURLToPath(new URL('./fake-codex.mjs', import.meta.url));
  const bridge = spawn(process.execPath, [fileURLToPath(new URL('../index.mjs', import.meta.url))], { env: { ...process.env, FAKE_RESUME: 'gone' }, stdio: ['pipe', 'pipe', 'inherit'] });
  const frames = [];
  const lines = createInterface({ input: bridge.stdout });
  lines.on('line', line => frames.push(JSON.parse(line)));
  bridge.stdin.write(JSON.stringify({ method: 'start', params: { provider: 'codex', executable: fake, cwd: process.cwd(), mode: 'tools', server: null, budget: 0, resume: 'thread-9', brief: 'resume notice' } }) + '\n');
  await new Promise(resolve => bridge.on('close', resolve));
  assert.deepEqual(frames.map(f => f.event?.kind || f.status), ['resume_failed', 'failed', 'closed']);
  assert.match(frames[0].event.text, /could not resume thread thread-9: no rollout found for thread id thread-9/);
  assert.equal(frames[0].event.gone, false);               // what Codex answers for a missing thread is not known: retryable
});

// The resume guard hook: its logic only. Whether the CLI calls UserPromptSubmit hooks before sending the prompt
// is not exercised here (no real CLI).
test('the resume guard blocks a prompt about to go to another session and lets the resumed one through', async () => {
  const seen = [];
  const guard = resumePromptGuard('sess-1', sid => seen.push(sid));
  assert.deepEqual(await guard({ hook_event_name: 'UserPromptSubmit', session_id: 'sess-1', prompt: 'p' }), { continue: true });
  const blocked = await guard({ hook_event_name: 'UserPromptSubmit', session_id: 'sess-new', prompt: 'p' });
  assert.equal(blocked.continue, false); assert.equal(blocked.decision, 'block');
  assert.equal(blocked.hookSpecificOutput.hookEventName, 'UserPromptSubmit');
  assert.deepEqual(seen, ['sess-new']);
  // No usable session id, or another event: let it through; the init check decides.
  for (const input of [{ hook_event_name: 'UserPromptSubmit' }, { hook_event_name: 'UserPromptSubmit', session_id: '' },
    { hook_event_name: 'UserPromptSubmit', session_id: 7 }, { hook_event_name: 'SessionStart', session_id: 'sess-new' }, undefined, null]) {
    assert.deepEqual(await guard(input), { continue: true });
  }
  assert.deepEqual(seen, ['sess-new']);
});

test('a Claude resume registers the guard; a mismatch it reports ends the attempt as not gone; a fresh start has no hook', async () => {
  const r = recorder(); const { sdk } = fakeSdk();
  const a = new ClaudeAdapter({ ...config, resume: 'sess-1' }, ...r.args, sdk);
  await a.start();
  const hook = sdk.queries[0].hooks.UserPromptSubmit[0].hooks[0];
  assert.deepEqual(await hook({ hook_event_name: 'UserPromptSubmit', session_id: 'sess-1' }), { continue: true });
  assert.equal(r.errors.length, 0);
  const out = await hook({ hook_event_name: 'UserPromptSubmit', session_id: 'sess-other' });
  assert.equal(out.decision, 'block');
  assert.equal(r.errors.length, 1); assert.equal(r.errors[0].resume, true); assert.equal(r.errors[0].gone, false);
  assert.match(r.errors[0].message, /blocked before it was sent/);
  a.close(); await a.pump;
  const r2 = recorder(); const fresh = fakeSdk();
  const b = new ClaudeAdapter(config, ...r2.args, fresh.sdk); await b.start();
  assert.equal('hooks' in fresh.sdk.queries[0], false);
  b.close(); await b.pump;
});

test('a mismatch the init check catches says the resume message may have reached the new session', async () => {
  const r = recorder(); const { sdk, output } = fakeSdk();
  const a = new ClaudeAdapter({ ...config, resume: 'sess-1' }, ...r.args, sdk);
  await a.start();
  output.push({ type: 'system', subtype: 'init', session_id: 'sess-fresh', tools: [] }); await tick(); await tick();
  assert.ok(r.errors[0].message.endsWith(MISMATCH_SENT));
  assert.match(MISMATCH_SENT, /task title, the current patch list and your optional message/);
  a.close();
});
