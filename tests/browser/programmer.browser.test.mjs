// The Programmer tab in a real headless Chrome, against tests/browser/fake_ide.py (the IDE's own routes and pages, a
// fake programmer run, two tiny random models). Opt-in, because it draws in a browser, which the ordinary node suite
// does not: run it from weight-space-ide/ with the Python environment active, as
//
//     WSP_BROWSER_TESTS=1 node --test tests/browser/programmer.browser.test.mjs
//
// It is skipped unless WSP_BROWSER_TESTS=1 and a Chrome is found (WSP_CHROME, else the usual locations). The Python is
// WSP_PYTHON, else the active virtual environment's, else python3. Ports are chosen free on 127.0.0.1 in 41000-49000;
// root, model cache and Chrome profile live in a temporary directory that is removed at the end, with the server and
// the browser it started. No provider, CLI, sign-in, network call or real model is involved. WSP_BROWSER_SHOTS=<dir>
// saves screenshots there.
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
const skip = process.env.WSP_BROWSER_TESTS !== '1' ? 'opt-in: set WSP_BROWSER_TESTS=1 to draw the Programmer tab in a real browser'
  : !CHROME ? 'no Chrome found: set WSP_CHROME to its executable' : false;
const PYTHON = process.env.WSP_PYTHON || (process.env.VIRTUAL_ENV ? join(process.env.VIRTUAL_ENV, 'bin', 'python') : 'python3');
const SHOTS = process.env.WSP_BROWSER_SHOTS || null;
const shotPath = (name) => (SHOTS ? join(SHOTS, name) : null);

const TMP = skip ? null : mkdtempSync(join(tmpdir(), 'wsp-browser-'));
const ports = new Set();
const servers = new Set();
let page = null;

// ------------------------------------------------------------------------------------------- server and browser
async function startServer(root, { autostart = false } = {}) {
  const port = await freePort(ports);
  mkdirSync(root, { recursive: true });
  const proc = spawn(PYTHON, [join(HERE, 'fake_ide.py'), '--root', root, '--port', String(port), ...(autostart ? ['--autostart'] : [])],
    { cwd: APP_DIR, stdio: ['ignore', 'pipe', 'pipe'], detached: true });
  const server = { proc, port, base: `http://127.0.0.1:${port}`, log: '' };
  proc.stdout.on('data', (d) => { server.log += d; });
  proc.stderr.on('data', (d) => { server.log += d; });
  servers.add(server);
  for (let k = 0; k < 300; k++) {
    if (proc.exitCode !== null) throw new Error('the fake IDE exited:\n' + server.log.slice(-2000));
    try { if ((await fetch(server.base + '/fake/state')).ok) return server; } catch {}
    await sleep(100);
  }
  throw new Error('the fake IDE did not start:\n' + server.log.slice(-2000));
}
async function stopServer(server, signal = 'SIGTERM') {
  if (!servers.has(server)) return;
  const gone = () => server.proc.exitCode !== null || server.proc.signalCode !== null;
  try { process.kill(-server.proc.pid, signal); } catch {}
  for (let k = 0; k < 50 && !gone(); k++) await sleep(100);
  if (!gone()) { try { process.kill(-server.proc.pid, 'SIGKILL'); } catch {} for (let k = 0; k < 30 && !gone(); k++) await sleep(100); }
  servers.delete(server);
}
let launches = 0;
async function openBrowser() {
  if (page) { await page.close(); page = null; }            // one browser at a time, each with its own profile
  page = await launch({ chrome: CHROME, port: await freePort(ports), profile: join(TMP, `chrome-profile-${++launches}`), width: 1400, height: 1000 });
  return page;
}
async function closeBrowser() { if (page) { await page.close(); page = null; } }
after(async () => {
  if (page) await page.close();
  for (const s of [...servers]) await stopServer(s, 'SIGKILL');
  if (TMP) rmSync(TMP, { recursive: true, force: true });
});
process.on('exit', () => { for (const s of servers) { try { process.kill(-s.proc.pid, 'SIGKILL'); } catch {} } });

// ------------------------------------------------------------------------------------------- checks and readers
function checker() {
  const fails = [];
  const check = (ok, msg, detail) => { if (!ok) fails.push(msg + (detail !== undefined ? '  ' + JSON.stringify(detail) : '')); };
  return { check, done: (what) => assert.deepEqual(fails, [], `${what}: ${fails.length} check(s) failed`) };
}
const fake = (server, what) => fetch(server.base + '/fake/' + what, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
const serverRuns = async (server) => (await (await fetch(server.base + '/fake/state')).json()).runs;
const tabText = () => page.eval(`[...document.querySelectorAll('#tabs button')].find((b) => b.textContent.startsWith('Programmer')).textContent`);
const tab = (name) => page.click(`[...document.querySelectorAll('#tabs button')].find((b) => b.textContent.startsWith(${JSON.stringify(name)}))`);
const button = (scope, text) => page.click(`[...document.querySelectorAll(${JSON.stringify(scope + ' button')})].find((b) => b.textContent === ${JSON.stringify(text)})`);
const markerIs = (text) => `[...document.querySelectorAll('#tabs button')].some((b) => b.textContent === ${JSON.stringify(text)})`;
const st = () => page.eval(`(() => {
  const v = document.querySelector('#view'), run = document.querySelector('#agent-run'), box = document.querySelector('#agent-log-box');
  const setup = document.querySelector('details.agent-setup'), log = document.querySelector('#agent-log');
  return { view: v.dataset.view, viewScroll: v.scrollTop, runTop: run ? Math.round(run.getBoundingClientRect().top) : null,
    setupTop: setup ? Math.round(setup.getBoundingClientRect().top) : null, setupOpen: setup ? setup.open : null,
    cards: log?.children.length ?? null, runId: log?.runId ?? null, logText: log?.textContent ?? '',
    logScroll: box ? Math.round(box.scrollTop) : null, logGap: box ? Math.round(box.scrollHeight - box.scrollTop - box.clientHeight) : null,
    status: document.querySelector('.agent-status-text[role=status]')?.textContent ?? null,
    time: document.querySelector('.agent-status-time')?.textContent ?? null,
    prompt: document.querySelector('.agent-prompt h3')?.textContent ?? null,
    inspector: document.querySelector('#inspector h2')?.textContent ?? null, innerWidth: window.innerWidth, innerHeight: window.innerHeight,
    scrollWidth: document.documentElement.scrollWidth, workspace: window.app?.state?.open?.name ?? null,
    runStatus: window.app?.keep?.prog?.run?.status ?? null, runShown: window.app?.keep?.prog?.run?.id ?? null };
})()`);
const tabsCut = () => page.eval(`(() => { const bar = document.querySelector('#tabs').getBoundingClientRect();
  return [...document.querySelectorAll('#tabs button')].filter((b) => { const r = b.getBoundingClientRect(); return r.right > bar.right + 1 || r.left < bar.left - 1 || r.bottom > bar.bottom + 1; }).map((b) => b.textContent); })()`);
// Trace asks the tiny model for a token it does not have ('T' is not in its vocabulary): that 4xx is the model's.
const errors = () => page.console.filter((c) => c.type === 'exception' || c.type === 'error' || (c.type === 'log-error' && !/favicon|api\/trace/.test(c.text)));

// A log that opens at its end and follows: sampled a few times while the run writes.
async function follows(check, label, samples = 3) {
  let last = null;
  for (let k = 0; k < samples; k++) {
    await sleep(2000);
    const s = await st();
    check(s.logGap !== null && s.logGap <= 40, `${label}: at its end after ${2 * (k + 1)} s`, { gap: s.logGap, scroll: s.logScroll });
    last = s;
  }
  return last;
}

let A = null;                                                  // the first fake IDE (a run started at once)

// ------------------------------------------------------------------------------------------- the scenarios
test('coming back to a running conversation: first on the tab, tree clicks and tabs leave it as it is, the marker on every tab', { skip, timeout: 240000 }, async () => {
  const { check, done } = checker();
  A = await startServer(join(TMP, 'root-a'), { autostart: true });
  await openBrowser();
  await page.goto(A.base + '/');
  await page.waitFor(`document.querySelectorAll('#tree .node').length > 0`);
  check((await page.eval('window.innerWidth')) === 1400, 'viewport is 1400 wide');
  await page.waitFor(markerIs('Programmer · working'), 10000, 'marker working at page load');
  check(await page.eval(`[...document.querySelectorAll('#tabs button')].find((b) => b.textContent.startsWith('Programmer')).title.length > 10`), 'marker has a title');
  await tab('Programmer');
  await page.waitFor(`document.querySelector('#agent-run') && document.querySelector('#agent-log')?.children.length > 3`);
  await sleep(2500);
  let s = await st();
  check(s.runTop !== null && s.runTop < 200 && s.setupTop > s.runTop, 'run section first, setup folded below it', s);
  check(s.setupOpen === false, 'setup cards folded under one line');
  check(s.logGap <= 40, 'log at its end while a turn runs', s.logGap);
  check(!!s.status && /Running|Waiting for the model/.test(s.status), 'status line says what runs', s.status);
  await page.shot(shotPath('01-run-in-progress.png'));
  const t1 = s.time; await sleep(2200); s = await st();
  check(s.time !== t1, 'status line ticks', [t1, s.time]);
  check(s.logGap <= 40, 'log follows new events', s.logGap);
  // Three tree clicks: no rebuild, no scroll change, inspector updated.
  await page.eval(`window.__view = document.querySelector('#view').firstChild; document.querySelector('#view').scrollTop = 60;`);
  await sleep(300);
  const before = await st();
  const postsBefore = page.posts.length;
  for (const addr of ['embed', 'blocks.0', 'unembed']) {
    await page.click(`document.querySelector('#tree .node[data-addr="${addr}"]')`);
    await page.waitFor(`document.querySelector('#inspector h2')?.textContent === ${JSON.stringify(addr)}`, 8000, 'inspector ' + addr);
    await sleep(700);
    s = await st();
    const same = await page.eval(`document.querySelector('#view').firstChild === window.__view`);
    check(same && s.viewScroll === before.viewScroll && s.inspector === addr && s.logGap <= 40, `tree click ${addr}: view not rebuilt, scroll kept, inspector updated`, { same, viewScroll: s.viewScroll, inspector: s.inspector, logGap: s.logGap });
  }
  await page.shot(shotPath('02-after-three-tree-clicks.png'));
  // A re-render of the same tab after the selection changed (the theme button re-renders the view) keeps its place.
  await page.click(`[...document.querySelectorAll('#topbar button')].find((b) => b.textContent === '◐')`); await sleep(1500);
  s = await st();
  check(s.viewScroll === before.viewScroll && s.runTop !== null, 'a same-tab re-render after a selection change keeps the view scroll', s.viewScroll);
  await page.click(`[...document.querySelectorAll('#topbar button')].find((b) => b.textContent === '◐')`); await sleep(1200);
  // Every other tab, with the marker; events keep arriving meanwhile.
  const cardsBefore = (await st()).cards;
  for (const t of ['Weights', 'Atlas', 'Systems', 'Run', 'Probe', 'Trace', 'Tests', 'Chunks', 'Export', 'Settings']) {
    await tab(t); await sleep(1300);
    check((await tabText()).startsWith('Programmer · '), `marker shown on the ${t} tab`, await tabText());
    if (t === 'Weights') await page.shot(shotPath('03-weights-tab-marker.png'));
  }
  await sleep(3000);
  await tab('Programmer');
  await page.waitFor(`document.querySelector('#agent-log')`);
  await sleep(600);
  s = await st();
  check(s.cards > cardsBefore && s.view === 'Programmer' && s.runTop < 200, 'back on Programmer: the run at the top with what arrived meanwhile', { cardsBefore, cards: s.cards });
  check(s.logGap <= 40, 'back on Programmer: log at its end', s.logGap);
  const t2 = s.time; await sleep(2200);
  check((await st()).time !== t2, 'back on Programmer: status line ticks again');
  await page.shot(shotPath('04-back-on-programmer.png'));
  const navPosts = page.posts.slice(postsBefore).map((p) => p.url.replace(A.base, ''));
  check(!navPosts.some((u) => u.startsWith('/api/harness')), 'navigation (tree, every tab, back) sent no Programmer POST', navPosts);
  // Scrolled up: the position is kept through events, a re-render and a tab round trip.
  await page.eval(`document.querySelector('#agent-log-box').scrollTop = 120`);
  await sleep(2500);
  s = await st();
  check(Math.abs(s.logScroll - 120) <= 2, 'scrolled up: new events do not move the log', s.logScroll);
  await tab('Weights'); await sleep(1500); await tab('Programmer'); await page.waitFor(`document.querySelector('#agent-log-box')`); await sleep(500);
  s = await st();
  check(Math.abs(s.logScroll - 120) <= 2, 'scrolled up: kept across leaving and coming back', s.logScroll);
  // A permission prompt: a run event re-renders the tab; the log keeps its place; the marker changes.
  await fake(A, 'prompt');
  await page.waitFor(`document.querySelector('.agent-prompt')`, 20000, 'prompt card');
  await sleep(800);
  s = await st();
  check(/Bash/.test(s.prompt) && /Waiting for your response/.test(s.status), 'prompt shown in the run section', { prompt: s.prompt, status: s.status });
  check(Math.abs(s.logScroll - 120) <= 2, 'log scroll survives the re-render a prompt causes', s.logScroll);
  check(await tabText() === 'Programmer · waiting for you', 'marker: waiting for you', await tabText());
  const promptTop = await page.eval(`Math.round(document.querySelector('.agent-prompt').getBoundingClientRect().top)`);
  check(promptTop < s.innerHeight, 'prompt visible without scrolling the page', promptTop);
  check((await tabsCut()).length === 0, 'every tab fully visible with the longest marker (1400 px)', await tabsCut());
  await page.shot(shotPath('05-permission-prompt.png'));
  await tab('Weights'); await sleep(800);
  await page.shot(shotPath('06-weights-tab-waiting-marker.png'));
  await tab('Programmer'); await page.waitFor(`document.querySelector('.agent-prompt')`); await sleep(400);
  await page.eval(`(() => { const b = document.querySelector('#agent-log-box'); b.scrollTop = b.scrollHeight; })()`); await sleep(300);
  await button('.agent-prompt', 'Allow once');
  await page.waitFor(`!document.querySelector('.agent-prompt')`, 10000, 'prompt answered');
  await sleep(800);
  check(await tabText() === 'Programmer · working', 'marker back to working after the answer', await tabText());
  // The live row's control brings the run into view.
  await page.eval(`document.querySelector('#view').scrollTop = 100000`); await sleep(300);
  const liveBtn = await page.eval(`!![...document.querySelectorAll('#agent-conversations button')].find((b) => b.textContent === 'Show at the top')`);
  check(liveBtn, 'live row has a "Show at the top" control');
  if (liveBtn) {
    await button('#agent-conversations', 'Show at the top'); await sleep(500);
    s = await st();
    check(s.runTop >= 0 && s.runTop < 200, 'the control brings the run into view', s.runTop);
  }
  // Reload mid-run: the marker from the first tab, then the run restored.
  await page.reload(); await page.waitFor(`document.querySelectorAll('#tree .node').length > 0`);
  await page.waitFor(markerIs('Programmer · working'), 10000, 'marker after reload');
  await tab('Programmer');
  await page.waitFor(`document.querySelector('#agent-log')?.children.length > 3`); await sleep(1500);
  s = await st();
  check(s.runTop < 200 && s.logGap <= 40 && /Running|Waiting for the model/.test(s.status), 'reload mid-run: the run restored at the top, log at its end', { runTop: s.runTop, logGap: s.logGap, status: s.status });
  await page.shot(shotPath('07-after-reload.png'));
  // Narrow window.
  await page.viewport(900, 900); await sleep(800);
  s = await st();
  check(s.innerWidth === 900 && s.scrollWidth <= s.innerWidth, 'narrow 900 px: no horizontal page overflow', { innerWidth: s.innerWidth, scrollWidth: s.scrollWidth });
  check(s.runTop < 300, 'narrow: run still first', s.runTop);
  check((await tabsCut()).length === 0, 'narrow: every tab and the marker fully visible', await tabsCut());
  await page.shot(shotPath('08-narrow-900.png'));
  await page.viewport(1400, 1000); await sleep(400);
  // End the turn (marker "turn finished"), scrolled up as the person left it.
  await fake(A, 'end');
  await page.waitFor(markerIs('Programmer · turn finished'), 45000, 'marker idle');
  await sleep(500);
  await page.shot(shotPath('09-idle.png'));
  check(errors().length === 0, 'no exception or console error', errors());
  done('coming back to a running conversation');
});

test('a new conversation, and a resumed one, open at their end and follow their output after a scrolled-up one', { skip, timeout: 180000 }, async () => {
  assert.ok(A && page, 'runs after the first scenario');
  const { check, done } = checker();
  // The finished first conversation, scrolled up, then closed.
  await page.eval(`document.querySelector('#agent-log-box').scrollTop = 30`); await sleep(600);
  check((await st()).logScroll === 30, 'first conversation scrolled up', (await st()).logScroll);
  await button('#agent-run', 'Close conversation');
  await page.waitFor(markerIs('Programmer'), 10000, 'marker gone');
  await sleep(2000);
  let s = await st();
  check(s.setupTop === null, 'closed: today\'s order again (no folded setup)', s.setupTop);
  await page.shot(shotPath('10-closed.png'));
  const first = s.runId;
  // Start programmer: a new conversation's log.
  await button('', 'Start programmer');
  await page.waitFor(`document.querySelector('#agent-log')?.runId && document.querySelector('#agent-log').runId !== ${JSON.stringify(first)}`, 15000, 'new conversation log');
  await page.waitFor(`document.querySelector('#agent-log').children.length > 6`, 30000, 'new conversation output');
  s = await follows(check, 'a new conversation after a scrolled-up one');
  check(s.cards > 6, 'its output arrives', s.cards);
  await page.shot(shotPath('11-new-conversation-follows.png'));
  // Scrolled up again, ended and closed; then resumed from the list.
  const second = s.runId;
  await page.eval(`document.querySelector('#agent-log-box').scrollTop = 30`); await sleep(600);
  await fake(A, 'end');
  await page.waitFor(markerIs('Programmer · turn finished'), 45000, 'second idle');
  await button('#agent-run', 'Close conversation');
  await page.waitFor(markerIs('Programmer'), 10000, 'second closed');
  await sleep(1500);
  await page.click(`[...document.querySelectorAll('#agent-conversations button')].find((b) => b.textContent === 'Open')`);
  await page.waitFor(`document.querySelector('#agent-conversation')`);
  await sleep(500);
  await button('#agent-conversation', 'Resume');
  await page.waitFor(`document.querySelector('#agent-log')?.runId && ![${JSON.stringify(first)}, ${JSON.stringify(second)}].includes(document.querySelector('#agent-log').runId)`, 15000, 'resumed conversation log');
  await page.waitFor(markerIs('Programmer · working'), 15000, 'resumed marker');
  s = await follows(check, 'a resumed conversation after a scrolled-up one');
  check(/Resuming the provider's own session/.test(s.logText), 'it is the resumed conversation');
  await page.shot(shotPath('12-resumed-follows.png'));
  await fake(A, 'end');
  await page.waitFor(markerIs('Programmer · turn finished'), 45000, 'resumed idle');
  await button('#agent-run', 'Close conversation');
  await page.waitFor(markerIs('Programmer'), 10000, 'resumed closed');
  check(errors().length === 0, 'no exception or console error', errors());
  done('new and resumed conversations');
});

test('after a workspace switch the tab shows no run of the other workspace, and the run again on switching back', { skip, timeout: 120000 }, async () => {
  assert.ok(A && page, 'runs after the first scenario');
  const { check, done } = checker();
  await sleep(1500);
  const last = (await serverRuns(A)).at(-1);
  check(last.status === 'closed', 'the last run of the first workspace is closed', last.status);
  const pick = async (name) => {
    await page.eval(`(() => { const s = document.querySelector('#topbar select'); s.value = ${JSON.stringify(name)}; s.dispatchEvent(new Event('change')); })()`);
    await page.waitFor(`window.app?.state?.open?.name === ${JSON.stringify(name)} && document.querySelector('#view').dataset.view === 'Programmer'`, 30000, 'opened ' + name);
    await sleep(1500);
  };
  await pick('rnd2');
  let s = await st();
  check(s.workspace === 'rnd2' && s.view === 'Programmer', 'the other workspace, on the Programmer tab', s);
  check(s.runTop === null && s.runShown === null, 'no run of the first workspace shown', { runTop: s.runTop, run: s.runShown });
  check(await page.eval(`!document.querySelector('#agent-conversations')`), 'none of its conversations listed (it has none)');
  check(await page.eval(`/No run yet/.test(document.querySelector('#view').textContent)`), 'the empty state');
  check(await tabText() === 'Programmer', 'no marker', await tabText());
  await page.shot(shotPath('13-other-workspace.png'));
  await pick('rnd');
  s = await st();
  check(s.runShown === last.id && s.runTop !== null, 'back in the first workspace: its last run shown', { shown: s.runShown, last: last.id });
  check(await page.eval(`document.querySelectorAll('#agent-conversations tr').length - 1`) === 2, 'its two conversations listed (the second was resumed)');
  check(errors().length === 0, 'no exception or console error', errors());
  done('workspace switch');
});

test('the first fake IDE and its browser are stopped', { skip }, async () => {
  await closeBrowser();
  if (A) await stopServer(A);
});

test('a server killed during a first turn and restarted: the conversation is listed resumable and Resume continues its session', { skip, timeout: 120000 }, async () => {
  const { check, done } = checker();
  const root = join(TMP, 'root-b');
  let server = await startServer(root, { autostart: true });
  let run = null;
  for (let k = 0; k < 300 && !(run && run.events >= 20); k++) { await sleep(100); run = (await serverRuns(server)).at(-1); }
  check(run && run.events >= 20 && run.status === 'running' && !!run.session, 'mid first turn, the session given', run);
  await stopServer(server, 'SIGKILL');                         // the server dies at once, mid-turn
  server = await startServer(root);
  await openBrowser();
  await page.goto(server.base + '/');
  await page.waitFor(`document.querySelectorAll('#tree .node').length > 0`);
  await sleep(1000);
  check(await tabText() === 'Programmer', 'no run in this server: no marker', await tabText());
  await tab('Programmer');
  await page.waitFor(`document.querySelector('#agent-conversations')`);
  const rows = await page.eval(`[...document.querySelectorAll('#agent-conversations tr')].slice(1).map((r) => [...r.children].map((c) => c.textContent))`);
  check(rows.length === 1 && rows[0][3] === 'interrupted' && rows[0][5] === 'Open', 'the cut-off conversation is listed as interrupted, with Open', rows);
  await button('#agent-conversations', 'Open');
  await page.waitFor(`document.querySelector('#agent-conversation')`);
  await sleep(500);
  const resume = await page.eval(`(() => { const b = [...document.querySelectorAll('#agent-conversation button')].find((x) => x.textContent === 'Resume'); return { disabled: b.disabled, refusal: document.querySelector('#resume-refusal')?.textContent || null }; })()`);
  check(resume.disabled === false && resume.refusal === null, 'Resume is available (not "no session was recorded")', resume);
  await page.shot(shotPath('14-after-restart-resumable.png'));
  await button('#agent-conversation', 'Resume');
  await page.waitFor(`document.querySelector('#agent-run h3')?.textContent === 'Open conversation'`, 15000, 'resumed run shown');
  await page.waitFor(markerIs('Programmer · working'), 15000, 'marker');
  await sleep(4000);
  const log = await page.eval(`document.querySelector('#agent-log').textContent`);
  check(log.includes("Resuming the provider's own session " + run.session), 'resumed with the session saved at init', run.session);
  check(/\[Weight-space IDE\] This conversation is being resumed/.test(log), 'the resume message was sent as the first message');
  const s = await st();
  check(s.logGap <= 40, 'the resumed log at its end', s.logGap);
  await page.shot(shotPath('15-resumed.png'));
  await button('#agent-run', 'Close conversation');
  await page.waitFor(markerIs('Programmer'), 10000, 'closed');
  check(errors().length === 0, 'no exception or console error', errors());
  done('restart and resume');
  await closeBrowser();
  await stopServer(server);
});
