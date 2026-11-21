// A minimal Chrome DevTools Protocol driver for the opt-in browser check: it launches its own headless Chrome with a
// fresh profile in a temporary directory, on a free local port, and drives one page. Node 22+ (global WebSocket).
import { spawn } from 'node:child_process';
import { existsSync, writeFileSync } from 'node:fs';
import { delimiter, join } from 'node:path';
import net from 'node:net';

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// WSP_CHROME, else the usual install locations, else chrome/chromium on PATH. null when none is found.
export function findChrome() {
  if (process.env.WSP_CHROME) return existsSync(process.env.WSP_CHROME) ? process.env.WSP_CHROME : null;
  const fixed = ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/Applications/Chromium.app/Contents/MacOS/Chromium',
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'];
  for (const p of fixed) if (existsSync(p)) return p;
  for (const dir of (process.env.PATH || '').split(delimiter)) {
    for (const name of ['google-chrome', 'google-chrome-stable', 'chromium', 'chromium-browser']) {
      if (dir && existsSync(join(dir, name))) return join(dir, name);
    }
  }
  return null;
}

// A port in 41000-49000 that is free on 127.0.0.1 now (never an IDE's usual 8765 or 43879).
export async function freePort(taken = new Set()) {
  for (let k = 0; k < 200; k++) {
    const port = 41000 + Math.floor(Math.random() * 8000);
    if (taken.has(port)) continue;
    const ok = await new Promise((resolve) => {
      const s = net.createServer();
      s.once('error', () => resolve(false));
      s.listen(port, '127.0.0.1', () => s.close(() => resolve(true)));
    });
    if (ok) { taken.add(port); return port; }
  }
  throw new Error('no free port in 41000-49000');
}

export async function launch({ chrome, port, profile, width = 1400, height = 1000 }) {
  const proc = spawn(chrome, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
    '--no-first-run', '--no-default-browser-check', '--disable-extensions', '--disable-background-networking',
    '--disable-sync', '--disable-component-update', '--metrics-recording-only', '--password-store=basic',
    '--use-mock-keychain', `--window-size=${width},${height}`, 'about:blank'], { stdio: 'ignore' });
  let target;
  for (let k = 0; k < 150 && !target; k++) {
    await sleep(100);
    try { target = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === 'page'); } catch {}
  }
  if (!target) { proc.kill('SIGKILL'); throw new Error('Chrome did not start'); }
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  let id = 0; const waiting = new Map(); const handlers = [];
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.id && waiting.has(msg.id)) { const { res, rej } = waiting.get(msg.id); waiting.delete(msg.id); msg.error ? rej(new Error(JSON.stringify(msg.error))) : res(msg.result); }
    else if (msg.method) for (const h of handlers) h(msg);
  };
  const send = (method, params = {}) => new Promise((res, rej) => { const k = ++id; waiting.set(k, { res, rej }); ws.send(JSON.stringify({ id: k, method, params })); });
  const page = {
    console: [], posts: [], proc,
    send, on: (fn) => handlers.push(fn),
    async eval(expr) {
      const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true });
      if (r.exceptionDetails) throw new Error('eval: ' + (r.exceptionDetails.exception?.description || r.exceptionDetails.text));
      return r.result.value;
    },
    async goto(url) { await send('Page.navigate', { url }); await page.waitLoad(); },
    async reload() { await send('Page.reload', { ignoreCache: true }); await page.waitLoad(); },
    async waitLoad() { for (let k = 0; k < 100; k++) { await sleep(100); try { if (await page.eval('document.readyState') === 'complete') return; } catch {} } },
    async waitFor(expr, ms = 15000, label = expr) {
      const end = Date.now() + ms;
      while (Date.now() < end) { try { if (await page.eval(expr)) return true; } catch {} await sleep(150); }
      throw new Error('timed out waiting for: ' + label);
    },
    async viewport(w, h) { await send('Emulation.setDeviceMetricsOverride', { width: w, height: h, deviceScaleFactor: 1, mobile: false }); },
    async shot(path) { if (!path) return null; const r = await send('Page.captureScreenshot', { format: 'png' }); writeFileSync(path, Buffer.from(r.data, 'base64')); return path; },
    async click(expr) { return page.eval(`(() => { const el = ${expr}; if (!el) throw new Error('no element: ' + ${JSON.stringify(expr)}); el.click(); return true; })()`); },
    async close() {
      try { await Promise.race([send('Browser.close'), sleep(2000)]); } catch {}
      for (let k = 0; k < 30 && proc.exitCode === null && proc.signalCode === null; k++) await sleep(100);
      if (proc.exitCode === null && proc.signalCode === null) { try { proc.kill('SIGKILL'); } catch {} }
    },
  };
  await send('Page.enable'); await send('Runtime.enable'); await send('Log.enable'); await send('Network.enable');
  page.on((m) => {
    if (m.method === 'Runtime.exceptionThrown') page.console.push({ type: 'exception', text: m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text });
    else if (m.method === 'Runtime.consoleAPICalled' && ['error', 'warning', 'assert'].includes(m.params.type))
      page.console.push({ type: m.params.type, text: m.params.args.map((a) => a.value ?? a.description ?? '').join(' ') });
    else if (m.method === 'Log.entryAdded' && ['error', 'warning'].includes(m.params.entry.level)) page.console.push({ type: 'log-' + m.params.entry.level, text: m.params.entry.text + ' ' + (m.params.entry.url || '') });
    else if (m.method === 'Network.requestWillBeSent' && m.params.request.method === 'POST') page.posts.push({ url: m.params.request.url, at: Date.now() });
  });
  await page.viewport(width, height);
  return page;
}
