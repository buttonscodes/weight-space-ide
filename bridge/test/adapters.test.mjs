import test from 'node:test';
import assert from 'node:assert/strict';
import { ClaudeAdapter, CodexAdapter, Prompts } from '../adapters.mjs';
import { messageQueue, RpcProcess } from '../protocol.mjs';
import { once } from 'node:events';

const tick = () => new Promise(resolve => setImmediate(resolve));
const config = { cwd: '/tmp', executable: 'test-agent', mode: 'files', server: null, budget: 0 };
function recorder() {
  const events = [], states = [], errors = [];
  return { events, states, errors, args: [e => events.push(e), s => states.push(s), e => errors.push(e)] };
}

class FakeRpc {
  constructor(command, args, options, notification, request, exit) {
    this.command = command; this.args = args; this.notify = notification; this.inbound = request; this.exit = exit; this.calls = []; this.closed = false;
  }
  async request(method, params) {
    this.calls.push({ method, params });
    if (method === 'thread/start') return { thread: { id: 'thread-1' } };
    if (method === 'turn/start') { if (this.rejectTurn) throw new Error('turn/start timed out'); return { turn: { id: 'turn-1' } }; }
    return {};
  }
  write(message) { this.calls.push(message); }
  close() { this.closed = true; }
}

test('Codex keeps one thread across turns, rejects concurrent sends, streams and interrupts', async () => {
  const r = recorder(); const a = new CodexAdapter(config, ...r.args, FakeRpc);
  await a.start(); await a.send('first');
  await assert.rejects(a.send('duplicate'), /Wait/);
  a.rpc.notify('item/agentMessage/delta', { threadId: 'thread-1', itemId: 'm1', delta: 'Hi' });
  a.rpc.notify('item/completed', { threadId: 'thread-1', item: { id: 'm1', type: 'agentMessage', text: 'Hi' } });
  assert.equal(r.events.filter(e => e.kind === 'text').length, 0);
  a.rpc.notify('turn/completed', { threadId: 'other', turn: { status: 'completed' } });
  assert.equal(a.busy, true);
  a.rpc.notify('turn/completed', { threadId: 'thread-1', turn: { status: 'completed' } });
  await a.send('second'); await a.interrupt();
  assert.equal(a.rpc.calls.filter(c => c.method === 'thread/start').length, 1);
  assert.deepEqual(a.rpc.calls.filter(c => c.method === 'turn/start').map(c => c.params.threadId), ['thread-1', 'thread-1']);
  assert.equal(a.rpc.calls.at(-1).method, 'turn/interrupt');
  a.rpc.notify('turn/completed', { threadId: 'thread-1', turn: { status: 'interrupted' } });
  assert.equal(r.events.at(-1).ok, false);
  assert.equal(r.states.at(-1), 'idle');
  a.close(); assert.equal(a.rpc.closed, true);
});

test('Codex approval and questions resolve once; tools mode declines permission expansion', async () => {
  const r = recorder(); const a = new CodexAdapter(config, ...r.args, FakeRpc); await a.start();
  const response = a.rpc.inbound('item/commandExecution/requestApproval', { threadId: a.threadId, command: 'echo hi' });
  const id = r.events.at(-1).request_id;
  a.prompts.respond(id, { allow: true });
  assert.deepEqual(await response, { decision: 'accept' });
  assert.throws(() => a.prompts.respond(id, { allow: true }), /no longer/);
  const question = a.rpc.inbound('item/tool/requestUserInput', { questions: [{ id: 'q', question: 'Which?' }] });
  a.prompts.respond(r.events.at(-1).request_id, { allow: true, answers: { q: 'A' } });
  assert.deepEqual(await question, { answers: { q: { answers: ['A'] } } });
  a.config = { ...config, mode: 'tools' };
  assert.deepEqual(await a.rpc.inbound('item/commandExecution/requestApproval', {}), { decision: 'decline' });
  await assert.rejects(a.rpc.inbound('item/permissions/requestApproval', {}), /Unsupported/);
  await assert.rejects(a.rpc.inbound('item/fileChange/requestApproval', { threadId: 'other' }), /another/);
  a.close();
});

test('ambiguous Codex send failure closes the conversation rather than making it retryable', async () => {
  const r = recorder(); const a = new CodexAdapter(config, ...r.args, FakeRpc); await a.start(); a.rpc.rejectTurn = true;
  await assert.rejects(a.send('do once'), /timed out/);
  assert.equal(a.closed, true); assert.equal(r.errors.length, 1);
  await assert.rejects(a.send('retry'), /Wait/);
});

test('Claude streams follow-up turns over one query and routes permissions through SDK', async () => {
  const r = recorder(); let calls = 0, options, prompt; const output = messageQueue();
  const sdk = { query(input) { calls++; options = input.options; prompt = input.prompt[Symbol.asyncIterator]();
    return { [Symbol.asyncIterator]: () => output[Symbol.asyncIterator](), interrupt: async () => { output.push({ type: 'result', is_error: true, result: 'Interrupted' }); }, close: () => output.end() }; } };
  const a = new ClaudeAdapter(config, ...r.args, sdk); await a.start();
  assert.equal(options.systemPrompt.preset, 'claude_code'); assert.equal(options.pathToClaudeCodeExecutable, 'test-agent');
  await a.send('first'); assert.equal((await prompt.next()).value.message.content, 'first');
  await assert.rejects(a.send('duplicate'), /Wait/);
  output.push({ type: 'system', subtype: 'init', session_id: 'claude-1', tools: [] });
  output.push({ type: 'stream_event', event: { type: 'message_start', message: { id: 'm1' } } });
  output.push({ type: 'stream_event', event: { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'hello' } } });
  output.push({ type: 'assistant', message: { id: 'm1', content: [{ type: 'text', text: 'hello' }] } });
  output.push({ type: 'result', is_error: false }); await tick();
  assert.equal(r.events.filter(e => e.kind === 'text').length, 0);
  await a.send('second'); assert.equal((await prompt.next()).value.message.content, 'second'); assert.equal(calls, 1);
  const permission = options.canUseTool('Bash', { command: 'echo x' }, {});
  a.prompts.respond(r.events.at(-1).request_id, { allow: false }); assert.equal((await permission).behavior, 'deny');
  await a.interrupt(); await tick(); assert.equal(r.states.at(-1), 'idle');
  a.close(); await a.pump; assert.deepEqual(r.errors, []);
});

test('Claude tools mode denies non-WSP tools; close rejects outstanding approval', async () => {
  const r = recorder(); let options; const stream = messageQueue();
  const sdk = { query(input) { options = input.options; return { [Symbol.asyncIterator]: () => stream[Symbol.asyncIterator](), close: () => stream.end() }; } };
  const a = new ClaudeAdapter({ ...config, mode: 'tools' }, ...r.args, sdk); await a.start();
  assert.deepEqual(options.tools, []);
  assert.equal((await options.canUseTool('Bash', {}, {})).behavior, 'deny');
  assert.equal((await options.canUseTool('mcp__wsp__overview', {}, {})).behavior, 'allow');
  a.close(); await a.pump;
  const prompts = new Prompts(() => {}); const pending = prompts.ask({ title: 'test' });
  assert.throws(() => prompts.respond('1', { allow: 'yes' }), /Malformed/);
  prompts.cancel(); assert.deepEqual(await pending, { allow: false });
});

test('real stdio transport services server requests while a client request is pending', async () => {
  const fixture = new URL('./rpc-fixture.mjs', import.meta.url);
  let prompted = false;
  const rpc = new RpcProcess(process.execPath, [fixture.pathname], {}, () => {}, async (method) => {
    assert.equal(method, 'ask'); prompted = true; return { approved: false };
  }, () => {});
  assert.deepEqual(await rpc.request('test'), { received: true }); assert.equal(prompted, true);
  rpc.close(); await once(rpc.child, 'close');
});

test('Codex retains interrupt requested before turn/start acknowledgement', async () => {
  const r = recorder(); const a = new CodexAdapter(config, ...r.args, FakeRpc); await a.start();
  let acknowledge;
  const original = a.rpc.request.bind(a.rpc);
  a.rpc.request = (method, params) => method === 'turn/start'
    ? new Promise(resolve => { acknowledge = resolve; }) : original(method, params);
  const sending = a.send('slow acknowledgement');
  await a.interrupt();
  assert.equal(a.interruptRequested, true);
  acknowledge({ turn: { id: 'slow-turn' } }); await sending;
  assert.deepEqual(a.rpc.calls.at(-1), { method: 'turn/interrupt', params: { threadId: 'thread-1', turnId: 'slow-turn' } });
  a.close();
});

test('delayed previous turn acknowledgement cannot overwrite the current interrupt target', async () => {
  const r = recorder(); const a = new CodexAdapter(config, ...r.args, FakeRpc); await a.start();
  const acknowledgements = [];
  const original = a.rpc.request.bind(a.rpc);
  a.rpc.request = (method, params) => method === 'turn/start'
    ? new Promise(resolve => { acknowledgements.push(resolve); }) : original(method, params);
  const first = a.send('one');
  a.notification('turn/started', { threadId: a.threadId, turn: { id: 'one' } });
  a.notification('turn/completed', { threadId: a.threadId, turn: { id: 'one', status: 'completed' } });
  const second = a.send('two');
  a.notification('turn/started', { threadId: a.threadId, turn: { id: 'two' } });
  acknowledgements[0]({ turn: { id: 'one' } }); await first;
  assert.equal(a.turnId, 'two'); await a.interrupt();
  assert.equal(a.rpc.calls.at(-1).params.turnId, 'two');
  acknowledgements[1]({ turn: { id: 'two' } }); await second; a.close();
});

test('Codex WSP MCP tool elicitation requires explicit approval; unknown forms stay declined', async () => {
  const r = recorder(); const a = new CodexAdapter({ ...config, mode: 'tools' }, ...r.args, FakeRpc); await a.start(); await a.send('overview');
  const form = { serverName: 'wsp', threadId: a.threadId, turnId: a.turnId, mode: 'form',
    message: 'Allow overview?', requestedSchema: { type: 'object', properties: {} }, _meta: { codex_approval_kind: 'mcp_tool_call' } };
  const response = a.request('mcpServer/elicitation/request', form);
  assert.equal(r.events.at(-1).title, 'Allow WSP tool');
  a.prompts.respond(r.events.at(-1).request_id, { allow: true });
  assert.deepEqual(await response, { action: 'accept', content: null, _meta: null });
  for (const patch of [{ serverName: 'other' }, { mode: 'url' }, { turnId: 'old' }, { requestedSchema: { type: 'object', properties: { secret: { type: 'string' } } } }]) {
    assert.equal((await a.request('mcpServer/elicitation/request', { ...form, ...patch })).action, 'cancel');
  }
  const pending = a.request('mcpServer/elicitation/request', form); a.close();
  assert.equal((await pending).action, 'cancel');
});

test('Claude init says which API key source the session uses, so the IDE can tell a subscription from per-call billing', async () => {
  const r = recorder(); const output = messageQueue();
  const sdk = { query() { return { [Symbol.asyncIterator]: () => output[Symbol.asyncIterator](), close: () => output.end() }; } };
  const a = new ClaudeAdapter(config, ...r.args, sdk); await a.start();
  output.push({ type: 'system', subtype: 'init', session_id: 'claude-1', tools: [], apiKeySource: 'none' });
  output.push({ type: 'system', subtype: 'init', session_id: 'claude-1', tools: [], apiKeySource: 'ANTHROPIC_API_KEY' }); await tick();
  assert.deepEqual(r.events.filter(e => e.kind === 'init').map(e => e.api_key_source), ['none', 'ANTHROPIC_API_KEY']);
  a.close(); await a.pump;
});
