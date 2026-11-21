// The Run tab's chat "Send:" toggle in a real headless Chrome, against tests/browser/fake_ide.py (the IDE's own routes
// and pages on a temporary root with two tiny random models). It only draws the view and presses the toggle: no chat
// message is sent and no model runs. Opt-in, like programmer.browser.test.mjs beside it; run from weight-space-ide/ as
//
//     WSP_BROWSER_TESTS=1 node --test tests/browser/chat_toggle.browser.test.mjs
//
// Skipped unless WSP_BROWSER_TESTS=1 and a Chrome is found (WSP_CHROME, else the usual locations). The Python is
// WSP_PYTHON, else the active virtual environment's, else python3. Free ports on 127.0.0.1 in 41000-49000; root, model
// cache and Chrome profile live in a temporary directory that is removed at the end, with the server and the browser.
// WSP_BROWSER_SHOTS=<dir> saves screenshots there.
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtempSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { findChrome, freePort, launch, sleep } from './cdp.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const APP_DIR = join(HERE, '..', '..');
const CHROME = process.env.WSP_BROWSER_TESTS === '1' ? findChrome() : null;
const skip = process.env.WSP_BROWSER_TESTS !== '1' ? 'opt-in: set WSP_BROWSER_TESTS=1 to draw the chat toggle in a real browser'
  : !CHROME ? 'no Chrome found: set WSP_CHROME to its executable' : false;
const PYTHON = process.env.WSP_PYTHON || (process.env.VIRTUAL_ENV ? join(process.env.VIRTUAL_ENV, 'bin', 'python') : 'python3');
const SHOTS = process.env.WSP_BROWSER_SHOTS || null;
const shotPath = (name) => (SHOTS ? join(SHOTS, name) : null);
const TMP = skip ? null : mkdtempSync(join(tmpdir(), 'wsp-browser-chat-'));
const ports = new Set();
let server = null, page = null;

async function startServer(root) {
  const port = await freePort(ports);
  mkdirSync(root, { recursive: true });
  const proc = spawn(PYTHON, [join(HERE, 'fake_ide.py'), '--root', root, '--port', String(port)], { cwd: APP_DIR, stdio: ['ignore', 'pipe', 'pipe'], detached: true });
  const s = { proc, port, base: `http://127.0.0.1:${port}`, log: '' };
  proc.stdout.on('data', (d) => { s.log += d; });
  proc.stderr.on('data', (d) => { s.log += d; });
  for (let k = 0; k < 300; k++) {
    if (proc.exitCode !== null) throw new Error('the fake IDE exited:\n' + s.log.slice(-2000));
    try { if ((await fetch(s.base + '/fake/state')).ok) return s; } catch {}
    await sleep(100);
  }
  throw new Error('the fake IDE did not start:\n' + s.log.slice(-2000));
}
async function stopServer(s) {
  if (!s) return;
  const gone = () => s.proc.exitCode !== null || s.proc.signalCode !== null;
  try { process.kill(-s.proc.pid, 'SIGTERM'); } catch {}
  for (let k = 0; k < 50 && !gone(); k++) await sleep(100);
  if (!gone()) { try { process.kill(-s.proc.pid, 'SIGKILL'); } catch {} }
}
after(async () => {
  if (page) await page.close();
  await stopServer(server);
  if (TMP) rmSync(TMP, { recursive: true, force: true });
});
process.on('exit', () => { if (server) { try { process.kill(-server.proc.pid, 'SIGKILL'); } catch {} } });

const LABEL = 'Send one message to both models, or a separate message to each';
const errors = () => page.console.filter((c) => c.type === 'exception' || c.type === 'error' || c.type === 'log-error' && !/favicon/.test(c.text));
const half = (text) => `[...document.querySelector('[aria-label="${LABEL}"]').querySelectorAll('button')].find((b) => b.textContent === ${JSON.stringify(text)})`;
// Geometry of the chat view: the toggle, its halves, the settings row, the steering panel and the message boxes.
const layout = () => page.eval(`(() => {
  const r = (el) => { if (!el) return null; const b = el.getBoundingClientRect(); return { left: Math.round(b.left), right: Math.round(b.right), top: Math.round(b.top), bottom: Math.round(b.bottom), width: Math.round(b.width), height: Math.round(b.height) }; };
  const view = document.querySelector('#view'), toggle = document.querySelector('[aria-label="${LABEL}"]');
  const clear = [...view.querySelectorAll('button')].find((b) => b.textContent === 'clear');
  const steer = [...view.querySelectorAll('.card')].find((c) => /steer/i.test(c.textContent));
  const box = (name) => view.querySelector('input[aria-label="' + name + '"]');
  const halves = toggle ? [...toggle.querySelectorAll('button')] : [];
  const visible = (el) => { if (!el) return false; const b = el.getBoundingClientRect(), s = getComputedStyle(el); return b.width > 0 && b.height > 0 && s.visibility !== 'hidden' && s.display !== 'none' && Number(s.opacity) > 0.4; };
  return { innerWidth: window.innerWidth, innerHeight: window.innerHeight, scrollWidth: document.documentElement.scrollWidth,
    view: r(view), viewScrollWidth: view.scrollWidth, viewClientWidth: view.clientWidth,
    toggle: r(toggle), lead: toggle ? toggle.parentElement.firstChild.textContent : null,
    halves: halves.map((b) => ({ text: b.textContent, pressed: b.getAttribute('aria-pressed'), disabled: b.disabled, visible: visible(b), fontSize: getComputedStyle(b).fontSize, ...r(b) })),
    inSettingsRow: !!(toggle && clear && clear.parentElement.contains(toggle)), settings: r(clear && clear.parentElement), steer: r(steer),
    one: r(box('Message to both models')), base: r(box('Message to the base model')), edited: r(box('Message to the edited model')),
    headings: [...view.querySelectorAll('.split > div > .small.muted')].map((d) => d.textContent).filter((t) => /^To the/.test(t)) };
})()`);

function checks(s, width, check) {
  check(s.innerWidth === width, `${width}: viewport is ${width} wide`, s.innerWidth);
  check(s.scrollWidth <= s.innerWidth, `${width}: no horizontal page overflow`, { scrollWidth: s.scrollWidth, innerWidth: s.innerWidth });
  check(s.viewScrollWidth <= s.viewClientWidth, `${width}: no horizontal overflow inside the view`, { scrollWidth: s.viewScrollWidth, clientWidth: s.viewClientWidth });
  check(!!s.toggle && s.lead === 'Send:', `${width}: the toggle is drawn with its "Send:" lead-in`, s.lead);
  check(!s.inSettingsRow, `${width}: the toggle is not in the settings row`);
  check(s.toggle && s.settings && s.toggle.top >= s.settings.bottom, `${width}: below the settings row`, { toggle: s.toggle, settings: s.settings });
  check(s.toggle && s.steer && s.toggle.top >= s.steer.bottom, `${width}: below the steering panel`, { toggle: s.toggle, steer: s.steer });
  check(s.halves.length === 2 && s.halves.every((b) => b.visible && b.left >= s.view.left && b.right <= s.view.right && b.width > 60),
    `${width}: both halves fully visible inside the view, without scrolling sideways`, s.halves);
  check(s.halves.every((b) => parseFloat(b.fontSize) >= 13), `${width}: the toggle is at body text size`, s.halves.map((b) => b.fontSize));
}

test('the chat Send toggle is visible above the box and switches between one box and two side by side, at 1400 and 900 px', { skip, timeout: 120000 }, async () => {
  const fails = [];
  const check = (ok, msg, detail) => { if (!ok) fails.push(msg + (detail !== undefined ? '  ' + JSON.stringify(detail) : '')); };
  server = await startServer(join(TMP, 'root'));
  page = await launch({ chrome: CHROME, port: await freePort(ports), profile: join(TMP, 'chrome-profile'), width: 1400, height: 1000 });
  await page.goto(server.base + '/');
  await page.waitFor(`document.querySelectorAll('#tree .node').length > 0`);
  await page.click(`[...document.querySelectorAll('#tabs button')].find((b) => b.textContent.startsWith('Run'))`);
  await page.waitFor(`document.querySelector('#view').dataset.view === 'Run' && [...document.querySelectorAll('#view .seg button')].some((b) => b.textContent === 'chat')`, 15000, 'Run tab');
  check(await page.eval(`!document.querySelector('[aria-label="${LABEL}"]')`), 'completion mode: no Send toggle');
  await page.click(`[...document.querySelectorAll('#view .seg button')].find((b) => b.textContent === 'chat')`);
  await page.waitFor(`!!document.querySelector('[aria-label="${LABEL}"]')`, 10000, 'chat mode toggle');
  const postsBefore = page.posts.length;
  for (const [width, height] of [[1400, 1000], [900, 900]]) {
    await page.viewport(width, height); await sleep(500);
    let s = await layout();
    checks(s, width, check);
    check(s.halves.map((b) => [b.text, b.pressed]).join() === 'One message to both models,true,A separate message to each model,false', `${width}: one message to both is pressed`, s.halves);
    check(!!s.one && !s.base && !s.edited && s.toggle.bottom <= s.one.top, `${width}: one box, just below the toggle`, { toggle: s.toggle, one: s.one });
    // With no conversation yet the toggle is on screen when chat opens (at 1400 px; a narrower pane wraps more above it).
    if (width === 1400) check(s.toggle.bottom <= Math.min(s.view.bottom, s.innerHeight), `${width}: the toggle is on screen without scrolling the view`, { toggle: s.toggle, view: s.view });
    await page.shot(shotPath(`chat-toggle-${width}-one.png`));
    await page.click(half('A separate message to each model'));
    await page.waitFor(`!!document.querySelector('input[aria-label="Message to the base model"]')`, 5000, `${width}: separate boxes`);
    await sleep(200);
    s = await layout();
    checks(s, width, check);
    check(s.halves.map((b) => [b.text, b.pressed]).join() === 'One message to both models,false,A separate message to each model,true', `${width}: separate is pressed`, s.halves);
    check(!s.one && s.base && s.edited, `${width}: two boxes`, { one: s.one, base: s.base, edited: s.edited });
    check(s.base && s.edited && Math.abs(s.base.top - s.edited.top) <= 2 && s.base.right <= s.edited.left, `${width}: the two boxes side by side, base left, edited right`, { base: s.base, edited: s.edited });
    check(s.base && s.edited && s.base.left >= s.view.left && s.edited.right <= s.view.right, `${width}: both boxes inside the view`, { base: s.base, edited: s.edited, view: s.view });
    check(s.toggle && s.base && s.toggle.bottom <= s.base.top, `${width}: the toggle stays above the boxes`);
    check(s.headings.join() === 'To the base model,To the edited model', `${width}: each column says which model it talks to`, s.headings);
    await page.shot(shotPath(`chat-toggle-${width}-separate.png`));
    await page.click(half('One message to both models'));
    await page.waitFor(`!!document.querySelector('input[aria-label="Message to both models"]')`, 5000, `${width}: back to one box`);
    await sleep(200);
    s = await layout();
    check(!!s.one && !s.base && !s.edited, `${width}: one box again`, { one: s.one, base: s.base });
    check(s.halves[0]?.pressed === 'true', `${width}: one message to both pressed again`, s.halves);
  }
  const posts = page.posts.slice(postsBefore).map((p) => p.url.replace(server.base, ''));
  check(!posts.some((u) => /\/api\/(chat|run)\b/.test(u)), 'no chat or run request was sent', posts);
  check(errors().length === 0, 'no exception or console error', errors());
  assert.deepEqual(fails, [], `chat toggle: ${fails.length} check(s) failed`);
});
