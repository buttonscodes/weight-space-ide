// Authentication belongs to the installed harness. This bridge only exposes
// sanitized readiness and a bounded, explicitly requested sign-in flow.
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { RpcProcess } from './protocol.mjs';

export function safeLoginUrl(provider, value) {
  try {
    const url = new URL(value);
    if (url.protocol !== 'https:' || url.username || url.password || url.port || url.hash) return null;
    if (provider === 'claude') {
      if (!['claude.com', 'claude.ai'].includes(url.hostname) || !['/cai/oauth/authorize', '/oauth/authorize'].includes(url.pathname)) return null;
      const redirect = new URL(url.searchParams.get('redirect_uri') || '');
      if (url.searchParams.get('code') !== 'true' || redirect.href !== 'https://platform.claude.com/oauth/code/callback') return null;
    } else if (provider === 'codex') {
      if (!['auth.openai.com', 'chatgpt.com'].includes(url.hostname) || !['/oauth/authorize', '/authorize'].includes(url.pathname)) return null;
    } else return null;
    return url.href;
  } catch { return null; }
}

export function claudeLoginUrl(output) {
  const clean = output.replace(/\x1b\][\s\S]*?(?:\x07|\x1b\\)/g, '').replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '');
  for (const match of clean.matchAll(/(https:\/\/\S+)\s/g)) {
    const url = safeLoginUrl('claude', match[1]);
    if (url) return url;
  }
  return null;
}

export function claudeReadiness(value) {
  if (!value || typeof value.loggedIn !== 'boolean') throw new Error('Invalid authentication status');
  const known = new Set(['claude.ai', 'oauth_token', 'api_key', 'auth_token', 'bedrock', 'vertex', 'foundry', 'none']);
  return { status: value.loggedIn ? 'ready' : 'signed_out', method: known.has(value.authMethod) ? value.authMethod : 'configured' };
}
export function codexReadiness(value) {
  if (!value || typeof value.requiresOpenaiAuth !== 'boolean') throw new Error('Invalid authentication status');
  const ready = value.requiresOpenaiAuth === false || !!value.account;
  const known = new Set(['apiKey', 'chatgpt', 'amazonBedrock', 'gcp']);
  return { status: ready ? 'ready' : 'signed_out', method: value.requiresOpenaiAuth === false ? 'configured provider' : known.has(value.account?.type) ? value.account.type : 'none' };
}

export class AuthFlow {
  constructor(config, emit, { spawnImpl = spawn, Rpc = RpcProcess } = {}) {
    this.config = config; this.emit = emit; this.spawn = spawnImpl; this.Rpc = Rpc;
    this.cancelled = false; this.needsCode = false; this.codeSent = false; this.loginId = null;
    this.env = { ...process.env };
    for (const key of ['CLAUDECODE', 'CLAUDE_CODE_SESSION_ID', 'CLAUDE_CODE_ENTRYPOINT', 'CODEX_THREAD_ID']) delete this.env[key];
  }
  async claudeStatus() {
    return new Promise((resolve, reject) => {
      let output = '', settled = false;
      const child = this.spawn(this.config.executable, ['auth', 'status', '--json'], {
        cwd: this.config.cwd, env: this.env, stdio: ['ignore', 'pipe', 'ignore']
      });
      this.child = child;
      const timer = setTimeout(() => { child.kill('SIGTERM'); reject(new Error('Status timed out')); }, 15000);
      child.stdout.on('data', chunk => { output = (output + chunk).slice(-16000); });
      child.on('error', () => { clearTimeout(timer); settled = true; reject(new Error('Status failed')); });
      child.on('close', () => {
        clearTimeout(timer); if (settled) return;
        try { resolve(claudeReadiness(JSON.parse(output))); }
        catch { reject(new Error('Could not read Claude authentication status')); }
      });
    });
  }
  async codexConnection() {
    this.rpc = new this.Rpc(this.config.executable, ['app-server'], { cwd: this.config.cwd, env: this.env },
      (method, p) => {
        if (method === 'account/login/completed') {
          if (p.loginId === this.loginId) this.completeLogin?.(p.success === true);
          else this.earlyCompletion = p;
        }
      }, async () => { throw new Error('Unsupported authentication request'); },
      () => { if (!this.cancelled) this.completeLogin?.(false); });
    await this.rpc.request('initialize', { clientInfo: { name: 'weight_space_ide_auth', version: '1.0.0' }, capabilities: {} });
    this.rpc.write({ method: 'initialized' });
  }
  async status() {
    if (this.config.provider === 'claude') return this.claudeStatus();
    if (!this.rpc) await this.codexConnection();
    return codexReadiness(await this.rpc.request('account/read', { refreshToken: false }));
  }
  async run() {
    try {
      const before = await this.status();
      if (this.cancelled) return;
      if (this.config.operation === 'status' || before.status === 'ready') { this.emit(before); return; }
      this.emit({ status: 'signing_in' });
      if (this.config.provider === 'claude') await this.claudeLogin();
      else await this.codexLogin();
      if (this.cancelled) return;
      this.emit({ status: 'verifying' });
      const after = await this.status();
      if (after.status !== 'ready') throw new Error('Sign-in was not verified');
      this.emit(after);
    } catch {
      if (!this.cancelled) this.emit({ status: 'error', message: this.codeSent ? 'The sign-in code was not accepted. Start sign-in again.' : 'Unable to verify sign-in. Check the installed CLI and try again.' });
    } finally { this.rpc?.close(); this.child?.kill('SIGTERM'); }
  }
  async claudeLogin() {
    await new Promise((resolve, reject) => {
      let scanned = '', published = false;
      this.child = this.spawn(this.config.executable, ['auth', 'login', '--claudeai'], {
        cwd: this.config.cwd, env: { ...this.env, BROWSER: fileURLToPath(new URL('./no-browser.sh', import.meta.url)) },
        stdio: ['pipe', 'pipe', 'pipe']
      });
      const scan = chunk => {
        if (published || this.cancelled) return;
        scanned = (scanned + chunk.toString()).slice(-16000);
        const url = claudeLoginUrl(scanned);
        if (url) { published = true; scanned = ''; this.needsCode = true; this.emit({ status: 'signing_in', url, needs_code: true }); }
      };
      this.child.stdout.on('data', scan); this.child.stderr.on('data', scan);
      this.child.stdin.on('error', () => reject(new Error('Sign-in input closed')));
      this.child.on('error', () => reject(new Error('Sign-in failed')));
      this.child.on('close', code => code === 0 ? resolve() : reject(new Error('Sign-in failed')));
    });
  }
  async codexLogin() {
    const completed = new Promise(resolve => { this.completeLogin = resolve; });
    const login = await this.rpc.request('account/login/start', { type: 'chatgpt' });
    this.loginId = login?.loginId;
    if (this.cancelled) { if (this.loginId) await this.rpc.request('account/login/cancel', { loginId: this.loginId }); return; }
    const url = safeLoginUrl('codex', login?.authUrl);
    if (!this.loginId || !url) throw new Error('Unsafe or missing sign-in URL');
    this.emit({ status: 'signing_in', url, needs_code: false });
    if (this.earlyCompletion?.loginId === this.loginId) this.completeLogin(this.earlyCompletion.success === true);
    if (!await completed) throw new Error('Sign-in failed');
    this.loginId = null;
  }
  code(value) {
    if (this.config.provider !== 'claude' || !this.needsCode || this.codeSent || this.cancelled ||
        typeof value !== 'string' || !/^[A-Za-z0-9._~+\/=\-]+(?:#[A-Za-z0-9._~+\/=\-]+)?$/.test(value) || value.length > 4096) {
      throw new Error('Invalid or expired sign-in code');
    }
    this.codeSent = true; this.needsCode = false;
    this.child.stdin.write(value + '\n');
    this.emit({ status: 'verifying', needs_code: false });
  }
  async cancel() {
    this.cancelled = true;
    if (this.rpc && this.loginId) {
      await Promise.race([this.rpc.request('account/login/cancel', { loginId: this.loginId }).catch(() => {}), new Promise(r => setTimeout(r, 1000))]);
    }
    this.completeLogin?.(false); this.rpc?.close(); this.child?.kill('SIGTERM');
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const lines = createInterface({ input: process.stdin });
  let flow, closing = false;
  const output = frame => process.stdout.write(JSON.stringify(frame) + '\n');
  const finish = async () => {
    if (closing) return; closing = true;
    await flow?.cancel(); lines.close(); process.stdin.destroy();
    setTimeout(() => process.exit(0), 50).unref();
  };
  lines.on('line', line => {
    let m;
    try { m = JSON.parse(line); } catch { output({ status: 'error', message: 'Invalid authentication command' }); void finish(); return; }
    if (m.method === 'start' && !flow) {
      if (!['claude', 'codex'].includes(m.params?.provider) || !['status', 'login'].includes(m.params?.operation)) return void finish();
      flow = new AuthFlow(m.params, output);
      void flow.run().finally(finish);
    } else if (m.method === 'code') {
      try { flow.code(m.code); output({ id: m.id, ok: true }); }
      catch { output({ id: m.id, error: 'Invalid or expired sign-in code' }); }
    } else if (m.method === 'cancel') void finish();
  });
  lines.on('close', () => { void finish(); });
  process.on('SIGTERM', () => { void finish(); });
  process.on('SIGINT', () => { void finish(); });
  process.on('uncaughtException', () => { output({ status: 'error', message: 'Authentication process failed' }); void finish(); });
  process.on('unhandledRejection', () => { output({ status: 'error', message: 'Authentication process failed' }); void finish(); });
}
