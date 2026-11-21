import { RpcProcess, messageQueue } from './protocol.mjs';

const textOf = value => typeof value === 'string' ? value : JSON.stringify(value ?? '');
// Raised when a recorded session cannot be continued; index.mjs reports it as resume_failed. `gone` is set only
// for an answer that the session does not exist; any other failure is one failed attempt that can be retried.
export function resumeError(message, { gone = false } = {}) { const error = new Error(message); error.resume = true; error.gone = gone; return error; }

// What may have reached a different session before it was stopped (the server says the same, native_harness.py).
export const MISMATCH_SENT = 'The resume message (the task title, the current patch list and your optional message) may have been sent to that new session before it was stopped. The resumed session itself was not changed.';

// A UserPromptSubmit hook for a Claude resume (SDK Options.hooks; its input carries BaseHookInput.session_id).
// It blocks a prompt about to go to any session other than `expected`, so the resume message is never sent to a
// fresh one, and reports the mismatch. When the field is absent or not a string it lets the prompt through and
// the init check decides. Whether the CLI calls it before sending is not verified here (no real CLI was run).
export function resumePromptGuard(expected, onMismatch) {
  return async (input) => {
    const sid = input && input.hook_event_name === 'UserPromptSubmit' ? input.session_id : undefined;
    if (typeof sid !== 'string' || !sid || sid === expected) return { continue: true };
    onMismatch(sid);
    return { continue: false, decision: 'block', stopReason: `Session ${sid} is not the resumed session ${expected}.`,
      reason: `Blocked: the resume of ${expected} would have gone to session ${sid}.`,
      hookSpecificOutput: { hookEventName: 'UserPromptSubmit', suppressOriginalPrompt: true } };
  };
}

// Every prompt is answered once. Interrupt/close rejects outstanding requests.
export class Prompts {
  constructor(emit) { this.emit = emit; this.pending = new Map(); this.nextId = 1; }
  ask(prompt, signal) {
    const requestId = String(this.nextId++);
    return new Promise(resolve => {
      const cancel = () => this.respond(requestId, { allow: false }, false);
      this.pending.set(requestId, { resolve, signal, cancel });
      if (signal?.aborted) return cancel();
      signal?.addEventListener('abort', cancel, { once: true });
      this.emit({ kind: 'prompt', request_id: requestId, ...prompt });
    });
  }
  respond(id, answer, strict = true) {
    if (!answer || typeof answer !== 'object' || typeof answer.allow !== 'boolean' ||
        (answer.answers != null && (typeof answer.answers !== 'object' || Array.isArray(answer.answers)))) {
      throw new Error('Malformed approval response');
    }
    const entry = this.pending.get(String(id));
    if (!entry) { if (strict) throw new Error('This request is no longer pending'); return; }
    this.pending.delete(String(id)); entry.signal?.removeEventListener('abort', entry.cancel);
    entry.resolve(answer); this.emit({ kind: 'prompt_resolved', request_id: String(id) });
  }
  cancel() { for (const id of [...this.pending.keys()]) this.respond(id, { allow: false }); }
}

export class ClaudeAdapter {
  constructor(config, emit, changed, failed, sdk) {
    this.config = config; this.emit = emit; this.changed = changed; this.failed = failed; this.sdk = sdk;
    this.prompts = new Prompts(emit); this.inbox = messageQueue(); this.closed = false; this.busy = false;
    this.streamed = new Set(); this.messageId = ''; this.turnNumber = 0;
  }
  async start() {
    const sdk = this.sdk || await import('@anthropic-ai/claude-agent-sdk');
    const c = this.config;
    if (c.resume) {
      // Checked before anything is sent. The SDK documents getSessionInfo as returning undefined "if the session file
      // is not found, is a sidechain session, or has no extractable summary", so undefined alone is ambiguous.
      // getSessionMessages is documented to return an "empty array if session not found": only both answers together
      // are taken to mean the session is gone. Anything else goes ahead, and the init check below is the backstop.
      let noInfo = false, messages;
      try {
        if (typeof sdk.getSessionInfo === 'function') noInfo = (await sdk.getSessionInfo(c.resume, { dir: c.cwd })) == null;
        if (noInfo && typeof sdk.getSessionMessages === 'function') messages = await sdk.getSessionMessages(c.resume, { dir: c.cwd, limit: 1 });
      } catch (error) {
        throw resumeError(`Claude Code could not check session ${c.resume}: ${error.message}`);
      }
      if (noInfo && Array.isArray(messages) && messages.length === 0) {
        throw resumeError(`Claude Code has no session ${c.resume} with any messages for this workspace's agent directory, so it cannot be resumed.`, { gone: true });
      }
    }
    const env = { ...process.env };
    // A parent coding session is not the child session. Keep normal login homes and
    // provider environment intact; never copy or rewrite credentials.
    for (const name of ['CLAUDECODE', 'CLAUDE_CODE_SESSION_ID', 'CODEX_THREAD_ID']) delete env[name];
    delete env.CLAUDE_CODE_ENTRYPOINT;
    this.query = sdk.query({ prompt: this.inbox, options: {
      cwd: c.cwd, env, pathToClaudeCodeExecutable: c.executable,
      systemPrompt: { type: 'preset', preset: 'claude_code' },
      settingSources: ['user', 'project', 'local'], includePartialMessages: true, supportedDialogKinds: [],
      ...(c.resume ? { resume: c.resume } : {}),
      mcpServers: c.server ? { wsp: c.server } : {}, strictMcpConfig: true,
      permissionMode: 'default', ...(c.mode === 'tools' ? { tools: [] } : {}),
      // First line on a resume: block the resume message if it is about to go to another session. The init
      // check in read() stays the second line (and the only one if the CLI never calls the hook).
      ...(c.resume ? { hooks: { UserPromptSubmit: [{ hooks: [resumePromptGuard(c.resume, sid => {
        this.failed(resumeError(`Claude Code was about to continue session ${sid} instead of resuming ${c.resume}; the resume message was blocked before it was sent, and the conversation was closed. The resumed session itself was not changed.`));
      })] }] } } : {}),
      ...(c.model ? { model: c.model } : {}), ...(c.budget ? { maxBudgetUsd: c.budget } : {}),
      canUseTool: async (name, input, context) => {
        if (c.mode === 'tools' && !name.startsWith('mcp__wsp__')) return { behavior: 'deny', message: 'Only WSP tools are enabled in this conversation.' };
        if (name.startsWith('mcp__wsp__')) return { behavior: 'allow', updatedInput: input };
        const questions = name === 'AskUserQuestion' && Array.isArray(input.questions) ? input.questions.map((q, i) => ({
          id: String(i), question: q.question, options: q.options, multiSelect: q.multiSelect
        })) : undefined;
        const answer = await this.prompts.ask({ title: name, input, questions }, context.signal);
        if (!answer.allow) return { behavior: 'deny', message: 'Declined by user.' };
        const answers = {};
        if (questions) for (const q of questions) answers[q.question] = answer.answers?.[q.id] || '';
        return { behavior: 'allow', updatedInput: questions ? { ...input, answers } : input };
      }
    } });
    this.pump = this.read().catch(error => { if (!this.closed) this.failed(error); });
    this.changed('idle');
  }
  async read() {
    for await (const event of this.query) {
      if (this.closed) return;
      if (event.type === 'system' && event.subtype === 'init') {
        if (this.config.resume && event.session_id !== this.config.resume) {
          // Claude Code reports the session only when a turn starts (system/init), so the first message has already
          // gone to that other session: stop it at once. The resumed session itself was not touched.
          this.query.interrupt?.().catch(() => {});
          throw resumeError(`Claude Code started a different session (${event.session_id}) instead of resuming ${this.config.resume}; it was stopped and closed. ${MISMATCH_SENT}`);
        }
        // apiKeySource 'none' means no API key is in use (a claude.ai login); anything else is billed per call.
        this.emit({ kind: 'init', session: event.session_id, model: event.model, mcp: event.mcp_servers, n_tools: event.tools?.length, api_key_source: event.apiKeySource });
      } else if (event.type === 'stream_event') {
        const e = event.event;
        if (e.type === 'message_start') this.messageId = e.message?.id || `turn-${this.turnNumber}`;
        if (e.type === 'content_block_delta' && ['text_delta', 'thinking_delta'].includes(e.delta?.type)) {
          const id = `${this.messageId}:${e.index}`;
          this.streamed.add(id);
          this.emit({ kind: e.delta.type === 'text_delta' ? 'text_delta' : 'thinking_delta', id, text: e.delta.text || e.delta.thinking || '' });
        }
      } else if (event.type === 'assistant') {
        for (const [index, block] of (event.message?.content || []).entries()) {
          const id = `${event.message.id || this.messageId}:${index}`;
          if (['text', 'thinking'].includes(block.type) && !this.streamed.has(id)) this.emit({ kind: block.type, text: block.text || block.thinking });
          if (block.type === 'tool_use') this.emit({ kind: 'tool_use', id: block.id, name: block.name, input: block.input });
        }
      } else if (event.type === 'user') {
        for (const b of Array.isArray(event.message?.content) ? event.message.content : []) if (b.type === 'tool_result') {
          this.emit({ kind: 'tool_result', id: b.tool_use_id, text: textOf(b.content), error: !!b.is_error });
        }
      } else if (event.type === 'result') {
        this.busy = false;
        this.prompts.cancel();
        this.emit({ kind: 'result', ok: !event.is_error, text: event.result || (event.errors || []).join('\n'), cost_usd: event.total_cost_usd,
          turns: event.num_turns, input_tokens: event.usage?.input_tokens, output_tokens: event.usage?.output_tokens });
        this.changed('idle');
      }
    }
    if (!this.closed) throw new Error('Claude conversation ended; start a new conversation to continue.');
  }
  async send(text) {
    if (this.closed || this.busy) throw new Error('Wait for the current turn to finish');
    this.busy = true; this.turnNumber++; this.streamed.clear(); this.changed('running');
    this.emit({ kind: 'user', text });
    this.inbox.push({ type: 'user', message: { role: 'user', content: text }, parent_tool_use_id: null, session_id: '' });
  }
  async interrupt() { this.prompts.cancel(); if (this.busy) await this.query.interrupt(); }
  close() { this.closed = true; this.prompts.cancel(); this.inbox.end(); this.query?.close(); }
}

export class CodexAdapter {
  constructor(config, emit, changed, failed, Rpc = RpcProcess) {
    this.config = config; this.emit = emit; this.changed = changed; this.failed = failed; this.Rpc = Rpc;
    this.prompts = new Prompts(emit); this.busy = false; this.closed = false; this.streamed = new Set();
  }
  async start() {
    const c = this.config;
    if (c.budget) throw new Error('Codex app-server has no USD budget cap; clear the budget before starting.');
    const s = c.server;
    // Replace the MCP table for this process; other configured MCP servers are not part of this task.
    const table = s ? `{wsp={command=${JSON.stringify(s.command)},args=${JSON.stringify(s.args)},env={PYTHONPATH=${JSON.stringify(s.env.PYTHONPATH)}},tool_timeout_sec=600}}` : '{}';
    const args = ['-c', `mcp_servers=${table}`, 'app-server'];
    const env = { ...process.env }; delete env.CODEX_THREAD_ID;
    this.rpc = new this.Rpc(c.executable, args, { cwd: c.cwd, env },
      (method, params) => this.notification(method, params), (method, params) => this.request(method, params),
      error => { if (!this.closed) this.failed(error); });
    await this.rpc.request('initialize', { clientInfo: { name: 'weight_space_ide', title: 'Weight-space IDE', version: '1.0.0' }, capabilities: { experimentalApi: true, mcpServerOpenaiFormElicitation: false } });
    this.rpc.write({ method: 'initialized' });
    const settings = { cwd: c.cwd, approvalPolicy: 'untrusted', sandbox: c.mode === 'tools' ? 'read-only' : 'workspace-write',
      ...(c.model ? { model: c.model } : {}) };
    let result;
    if (c.resume) {
      // thread/resume by id (the app-server's own method) with the same settings as
      // thread/start; an error or another thread id means the session is not continued, and nothing is sent.
      try { result = await this.rpc.request('thread/resume', { threadId: c.resume, ...settings }); }
      catch (error) { throw resumeError(`Codex could not resume thread ${c.resume}: ${error.message}`); }
      if (result?.thread?.id !== c.resume) throw resumeError(`Codex answered with thread ${result?.thread?.id || '(none)'} instead of resuming ${c.resume}; nothing was sent.`);
    } else result = await this.rpc.request('thread/start', settings);
    if (!result?.thread?.id) throw new Error('Codex did not return a thread ID');
    this.threadId = result.thread.id;
    this.emit({ kind: 'init', session: this.threadId, model: result.model || c.model });
    this.changed('idle');
  }
  async send(text) {
    if (this.closed || this.busy) throw new Error('Wait for the current turn to finish');
    const generation = this.sendGeneration = (this.sendGeneration || 0) + 1;
    this.turnId = null;
    this.busy = true; this.streamed.clear(); this.changed('running'); this.emit({ kind: 'user', text });
    try {
      const result = await this.rpc.request('turn/start', { threadId: this.threadId, input: [{ type: 'text', text, text_elements: [] }] });
      // Completion may precede the acknowledgement; do not resurrect a completed turn.
      if (this.busy && this.sendGeneration === generation) {
        this.turnId = result?.turn?.id || this.turnId;
        if (this.interruptRequested) await this.deliverInterrupt();
      }
    } catch (error) {
      // A completed older send no longer owns a later turn's transport state.
      if (this.sendGeneration !== generation) return;
      this.failed(error); this.close(); throw error;
    }
  }
  notification(method, p) {
    if (this.closed || (p.threadId && this.threadId && p.threadId !== this.threadId)) return;
    if (method === 'turn/started') {
      this.turnId = p.turn?.id;
      if (this.interruptRequested) this.deliverInterrupt().catch(error => this.failed(error));
    }
    else if (method === 'item/agentMessage/delta' || method === 'item/reasoning/summaryTextDelta') {
      const id = `${p.itemId}:${p.summaryIndex || 0}`; this.streamed.add(p.itemId);
      this.emit({ kind: method.includes('reasoning') ? 'thinking_delta' : 'text_delta', id, text: p.delta || '' });
    } else if (method === 'item/started' || method === 'item/completed') {
      const item = p.item || {}, done = method === 'item/completed';
      if (done && item.type === 'agentMessage' && !this.streamed.has(item.id)) this.emit({ kind: 'text', text: item.text || '' });
      else if (['mcpToolCall', 'commandExecution', 'fileChange'].includes(item.type)) {
        this.emit(done ? { kind: 'tool_result', id: item.id, text: textOf(item.result ?? item.aggregatedOutput ?? item.error ?? item.changes), error: item.status === 'failed' }
          : { kind: 'tool_use', id: item.id, name: item.tool || item.type, input: item.arguments || { command: item.command, changes: item.changes } });
      }
    } else if (method === 'turn/completed') {
      this.busy = false; this.turnId = null; this.interruptRequested = false; this.prompts.cancel();
      this.emit({ kind: 'result', ok: p.turn?.status === 'completed', text: p.turn?.error?.message || (p.turn?.status === 'interrupted' ? 'Interrupted' : '') });
      this.changed('idle');
    } else if (method === 'error') this.emit({ kind: 'raw', text: p.error?.message || textOf(p) });
  }
  async request(method, p) {
    if (p.threadId && p.threadId !== this.threadId) throw new Error('Request is for another conversation');
    if (['item/commandExecution/requestApproval', 'item/fileChange/requestApproval'].includes(method)) {
      if (this.config.mode === 'tools') return { decision: 'decline' };
      const answer = await this.prompts.ask({ title: method.includes('commandExecution') ? 'Run command' : 'Change files', input: p });
      return { decision: answer.allow === true ? 'accept' : 'decline' };
    }
    if (method === 'item/tool/requestUserInput' && Array.isArray(p.questions)) {
      const answer = await this.prompts.ask({ title: 'Agent question', questions: p.questions });
      return { answers: Object.fromEntries(p.questions.map(q => [q.id, { answers: answer.allow ? [answer.answers?.[q.id] || ''] : [] }])) };
    }
    if (method === 'mcpServer/elicitation/request') {
      const schema = p.requestedSchema;
      const emptyForm = schema == null || (schema.type === 'object' && schema.properties &&
        typeof schema.properties === 'object' && !Array.isArray(schema.properties) && !Object.keys(schema.properties).length);
      // Codex wraps MCP tool permission in a zero-field form. Only this known
      // WSP approval is renderable; URL/auth/general forms remain declined.
      if (p.serverName !== 'wsp' || p.threadId !== this.threadId || !p.turnId || p.turnId !== this.turnId ||
          p._meta?.codex_approval_kind !== 'mcp_tool_call' || (p.mode != null && p.mode !== 'form') || !emptyForm) {
        this.emit({ kind: 'raw', text: 'Declined an unsupported MCP approval or form. Only active WSP tool confirmations are supported.' });
        return { action: 'cancel', content: null };
      }
      const answer = await this.prompts.ask({ title: 'Allow WSP tool', input: { message: p.message, server: p.serverName } });
      return answer.allow === true ? { action: 'accept', content: null, _meta: null }
        : { action: 'cancel', content: null, _meta: { codex_approval_cancellation_reason: 'user' } };
    }
    // Elicitation URLs, permission expansion, dynamic tools and unknown protocol
    // additions are unsupported, never implicitly approved.
    throw new Error(`Unsupported agent request: ${method}`);
  }
  async deliverInterrupt() {
    if (this.busy && this.turnId && this.interruptRequested) {
      this.interruptRequested = false;
      await this.rpc.request('turn/interrupt', { threadId: this.threadId, turnId: this.turnId });
    }
  }
  async interrupt() { this.prompts.cancel(); this.interruptRequested = this.busy; await this.deliverInterrupt(); }
  close() { this.closed = true; this.prompts.cancel(); this.rpc?.close(); }
}
