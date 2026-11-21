import { createInterface } from 'node:readline';
import { ClaudeAdapter, CodexAdapter } from './adapters.mjs';

let adapter, closing = false;
const output = message => process.stdout.write(JSON.stringify(message) + '\n');
const emit = event => output({ event });
const state = status => output({ status });
// A resume the provider cannot honour is reported as such (kind resume_failed), never as a fresh conversation.
function fail(error) {
  if (closing) return;
  emit(error.resume ? { kind: 'resume_failed', text: error.message, gone: error.gone === true } : { kind: 'error', text: error.message });
  state('failed'); close(1);
}
function close(code = 0) {
  if (closing) return;
  closing = true; adapter?.close(); state('closed');
  // Python owns the entire process group and supplies a final bounded kill.
  setTimeout(() => process.exit(code), 100).unref();
  lines.close(); process.stdin.destroy();
}
const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
lines.on('line', line => {
  let message;
  try { message = JSON.parse(line); } catch { return fail(new Error('Invalid bridge command')); }
  // Start/send operations serialize themselves. Reading stays live for responses.
  Promise.resolve().then(async () => {
    const { method, params = {} } = message;
    if (method === 'start') {
      if (adapter) throw new Error('Conversation already started');
      if (!['claude', 'codex'].includes(params.provider)) throw new Error('Unknown provider');
      adapter = params.provider === 'claude' ? new ClaudeAdapter(params, emit, state, fail) : new CodexAdapter(params, emit, state, fail);
      await adapter.start(); await adapter.send(params.brief);
    } else {
      if (!adapter || closing) throw new Error('Conversation is not open');
      if (method === 'send') { if (typeof params.text !== 'string' || !params.text.trim()) throw new Error('Enter a message'); await adapter.send(params.text); }
      else if (method === 'respond') adapter.prompts.respond(params.request_id, params);
      else if (method === 'interrupt') await adapter.interrupt();
      else if (method === 'close') close();
      else throw new Error('Unknown bridge command');
    }
    if (message.id != null) output({ id: message.id, result: {} });
  }).catch(error => { if (message.id != null) output({ id: message.id, error: error.message }); else fail(error); });
});
lines.on('close', () => { if (!closing) close(); });
process.on('SIGTERM', () => close());
process.on('SIGINT', () => close());
process.on('uncaughtException', fail);
process.on('unhandledRejection', fail);
