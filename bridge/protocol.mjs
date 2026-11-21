import { createInterface } from 'node:readline';
import { spawn } from 'node:child_process';

// JSON-RPC on stdio. Requests and server prompts are dispatched independently:
// waiting for a turn or an approval never blocks reading its reply.
export class RpcProcess {
  constructor(command, args, options, onNotification, onRequest, onExit) {
    this.pending = new Map();
    this.nextId = 1;
    this.child = spawn(command, args, { ...options, stdio: ['pipe', 'pipe', 'pipe'] });
    this.child.stderr.on('data', chunk => process.stderr.write(chunk));
    this.lines = createInterface({ input: this.child.stdout });
    this.lines.on('line', line => {
      try {
        const message = JSON.parse(line);
        if (!message || typeof message !== 'object') throw new Error('invalid protocol frame');
        if (message.method && message.id != null) {
          Promise.resolve(onRequest(message.method, message.params || {})).then(
            result => this.write({ id: message.id, result }),
            error => this.write({ id: message.id, error: { code: -32601, message: error.message } })
          ).catch(() => {});
        } else if (message.method) onNotification(message.method, message.params || {});
        else if (message.id != null) {
          const pending = this.pending.get(message.id);
          if (pending) {
            this.pending.delete(message.id); clearTimeout(pending.timer);
            message.error ? pending.reject(new Error(message.error.message || 'RPC failed')) : pending.resolve(message.result);
          }
        }
      } catch (error) { this.fail(error); this.child.kill('SIGTERM'); }
    });
    this.child.stdin.on('error', error => this.fail(error));
    this.child.on('error', error => { this.fail(error); onExit(error); });
    this.child.on('close', (code, signal) => {
      const error = new Error(`Codex app-server exited (${signal || code})`);
      this.fail(error); onExit(error);
    });
  }
  write(message) {
    if (this.closed || !this.child.stdin.writable) throw new Error('Agent connection closed');
    this.child.stdin.write(JSON.stringify(message) + '\n');
  }
  request(method, params = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error(`${method} timed out`)); }, 30000);
      this.pending.set(id, { resolve, reject, timer });
      try { this.write({ id, method, params }); }
      catch (error) { clearTimeout(timer); this.pending.delete(id); reject(error); }
    });
  }
  fail(error) {
    this.closed = true;
    for (const p of this.pending.values()) { clearTimeout(p.timer); p.reject(error); }
    this.pending.clear();
  }
  close() { this.fail(new Error('Agent connection closed')); this.child.stdin.end(); this.child.kill('SIGTERM'); }
}

export function messageQueue() {
  let wake, ended = false;
  const queue = [];
  return {
    push(message) { if (ended) throw new Error('Conversation closed'); queue.push(message); wake?.(); },
    end() { ended = true; wake?.(); },
    async *[Symbol.asyncIterator]() {
      while (!ended) {
        if (queue.length) yield queue.shift();
        else await new Promise(resolve => { wake = resolve; });
      }
    }
  };
}
