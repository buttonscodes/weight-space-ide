// Tests (the specification), Chunks (weights as text within a token budget), Programmer (task + harness + live log).
import { h, $, api, q, fmt, fmtInt, tabs, field, esc } from './util.js';

// ------------------------------------------------------------------------------------------------ tests
export async function testsView(el, app) {
  const st = app.keep.tests || (app.keep.tests = { suite: null, json: false });
  const suites = app.state.suites || [];
  const again = () => app.rerender();
  if (!st.suite || !suites.find((s) => s.name === st.suite)) st.suite = (app.state.task && app.state.task.suite) || (suites[0] && suites[0].name);
  el.append(h('div', { class: 'row' }, h('h2', {}, 'Tests'), h('span', { class: 'muted' }, 'the behaviour a weight program is written against; run on the base and the edited model')));
  const newSuite = async () => {
    const name = prompt('New suite name (letters, digits, - _):');
    if (!name) return;
    app.state.suites = (await api('/api/suite', { name, suite: { description: '', cases: [] } })).suites; st.suite = name; again();
  };
  el.append(h('div', { class: 'row', style: { margin: '8px 0' } },
    h('select', { onchange: (e) => { st.suite = e.target.value; again(); } }, suites.map((s) => h('option', { value: s.name, selected: s.name === st.suite }, `${s.name} (${s.n})`))),
    h('button', { onclick: newSuite }, 'new suite'),
    st.suite && app.state.task.suite !== st.suite ? h('button', { onclick: async () => { app.state.task = (await api('/api/task', { ...app.state.task, suite: st.suite })).task; await app.refreshStatus(); again(); } }, 'use as the task\'s suite')
      : st.suite ? h('span', { class: 'pill good' }, 'task suite') : null,
    h('button', { class: 'ghost', onclick: () => { st.json = !st.json; again(); } }, st.json ? 'table' : 'edit as JSON')));
  if (!app.state.status.can_run) el.append(h('p', { class: 'pill warn', style: { display: 'inline-block' } }, 'this model cannot be run here yet, so suites can be written but not evaluated'));
  if (!st.suite) { el.append(h('p', { class: 'muted' }, 'No suites yet. A suite is the specification a weight program is written against; add one with "new suite".')); return; }
  const suite = await api('/api/suite?' + q({ name: st.suite }));
  const save = async (s) => { st.result = null; app.state.suites = (await api('/api/suite', { name: st.suite, suite: s })).suites; await app.refreshStatus(); again(); };
  if (st.json) {
    const ta = h('textarea', { rows: 22, value: JSON.stringify(suite, null, 1) });
    el.append(ta, h('div', { class: 'row', style: { marginTop: '6px' } }, h('button', { class: 'primary', onclick: () => { try { save(JSON.parse(ta.value)); } catch (e) { alert(e.message); } } }, 'Save')));
    return;
  }
  if (!app.state.status.can_run) { el.append(h('pre', { class: 'doc' }, JSON.stringify(suite, null, 1))); return; }
  let cur, base;
  if (app.state.open.kind === 'gguf') {
    const currentKey = () => `${app.state.status.workspace_id ?? app.state.open.name}:${app.state.status.version}:${st.suite}`;
    const key = currentKey();
    const valid = () => app.keep.tests === st && currentKey() === key;
    const refresh = () => { if (valid() && (!app.tab || app.tab === 'Tests')) again(); };
    const notRun = (record) => ({name:st.suite,passed:0,total:suite.cases.length,not_run:suite.cases.length,
      cases:suite.cases.map((c)=>({...c,name:c.name || c.prompt,pass:null,status:'not_run',event:record,value:null}))});
    const progress = h('span',{class:'muted'},st.busy ? 'Running base and current sequentially…' : 'Tests run only on explicit request.');
    const run = h('button',{disabled:!!st.busy,onclick:async()=>{
      if (st.busy || !valid()) return;
      st.busy=true;run.disabled=true;progress.textContent='Running base and current sequentially…';
      let phase='base', result={key};
      try {
        result.base = await api('/api/tests?' + q({suite:st.suite,variant:'base'}));
        if (!valid()) return;
        phase='cur';
        result.cur = await api('/api/tests?' + q({suite:st.suite}));
        if (!valid()) return;
        st.result=result;await app.refreshStatus();
      } catch(e) {
        if (!valid()) return;
        if (e.payload?.status === 'not_run' || e.event) {
          result.event=e.event;result[phase]=e.payload?.report || notRun(e.event);
          result.base ||= notRun(e.event);result.cur ||= notRun(e.event);
          st.result=result;
        } else { st.result={key,error:e.message}; }
      } finally { st.busy=false;run.disabled=false;refresh(); }
    }},st.result?.key === key && (st.result.event || st.result.error) ? 'Retry tests' : 'Run tests');
    el.append(h('div',{class:'row'},run,progress));
    if (st.result?.key !== key) {el.append(h('pre',{class:'doc'},JSON.stringify(suite,null,1)));return;}
    if (st.result.error) {el.append(h('p',{role:'status'},st.result.error));return;}
    ({cur,base}=st.result);
    if (st.result.event) {
      const record=st.result.event;
      el.append(h('div',{class:'card',role:'status'},h('b',{},`NOT RUN · ${record.code}`),
        h('p',{},record.message),
        record.measured != null ? h('pre',{class:'small'},JSON.stringify({measured:record.measured,limit:record.limit},null,1)) : null,
        h('p',{},(record.recommendations || []).join(' '))));
    }
  } else {
    [cur,base] = await Promise.all([api('/api/tests?' + q({ suite: st.suite })), api('/api/tests?' + q({ suite: st.suite, variant: 'base' }))]);
  }
  const summary = (label,r) => `${label} ${r.passed}/${r.total} pass${r.not_run ? ` · ${r.not_run} NOT RUN` : ''}`;
  el.append(h('p', {}, h('b', {}, summary('edited',cur)), h('span', { class: 'muted' }, `   ${summary('base',base)}   ${suite.description || ''}`)));
  // add a case
  const f = { type: 'next_token', prompt: '', expect: '' };
  const kind = h('select', { onchange: (e) => { f.type = e.target.value; } }, ['next_token', 'not_token', 'completion', 'preserve'].map((k) => h('option', {}, k)));
  el.append(h('div', { class: 'row card', style: { margin: '8px 0', alignItems: 'flex-end' } }, field('type', kind),
    field('prompt', h('input', { type: 'text', style: { width: '320px', fontFamily: 'var(--mono)' }, oninput: (e) => { f.prompt = e.target.value; } })),
    field('expect / forbid / contains', h('input', { type: 'text', style: { width: '160px', fontFamily: 'var(--mono)' }, oninput: (e) => { f.expect = e.target.value; } })),
    h('button', { onclick: () => {
      if (!f.prompt) return;
      const c = { type: f.type, prompt: f.prompt };
      if (f.type === 'next_token') c.expect = f.expect; else if (f.type === 'not_token') c.forbid = f.expect; else if (f.type === 'completion') { c.contains = f.expect; c.max_tokens = 20; } else c.max_kl = 0.05;
      save({ ...suite, cases: [...suite.cases, c] });
    } }, 'add case')));
  const tbl = h('table', { class: 'grid' }, h('tr', {}, h('th', {}, ''), h('th', {}, 'case'), h('th', {}, 'wants'), h('th', {}, 'base'), h('th', {}, 'edited'), h('th', { class: 'num', title: 'distance from flipping: logit margin, or KL for preserve' }, 'margin'), h('th', {}, '')));
  const wasRun = (c) => c && typeof c.pass === 'boolean' && c.status !== 'not_run';
  const mark = (c) => !wasRun(c) ? h('span', {class:'muted'}, 'NOT RUN')
    : h('span', { style: { color: c.pass ? 'var(--good)' : 'var(--crit)' } }, c.pass ? '✓ ' : '✗ ', h('span', { class: 'tok' }, c.error ? 'error' : (c.got ?? '')));
  const rank = (c) => !wasRun(c) ? 2 : Number(c.pass);
  const order = cur.cases.map((c, i) => i).sort((a, b) => (rank(cur.cases[a]) - rank(cur.cases[b])) || (a - b));
  for (const i of order.slice(0, 400)) {
    const c = cur.cases[i], b = base.cases[i], spec = suite.cases[i];
    const wants = spec.type === 'next_token' ? [].concat(spec.expect).join(' | ') : spec.type === 'not_token' ? 'not ' + [].concat(spec.forbid).join(' | ')
      : spec.type === 'completion' ? Object.entries(spec).filter(([k]) => ['contains', 'not_contains', 'equals', 'matches'].includes(k)).map(([k, v]) => `${k} ${v}`).join(', ') : `KL ≤ ${spec.max_kl ?? 0.05}`;
    tbl.append(h('tr', { title: c.error || c.event?.message || '' }, h('td', {}, !wasRun(c) || !wasRun(b) || c.pass === b.pass ? '' : h('span', { class: 'pill ' + (c.pass ? 'good' : 'bad') }, c.pass ? 'fixed' : 'broken')),
      h('td', { class: 'mono' }, c.name), h('td', { class: 'mono' }, wants), h('td', {}, mark(b)), h('td', {}, mark(c)), h('td', { class: 'num' }, fmt(c.value)),
      h('td', {}, h('button', { class: 'ghost', title: 'trace this prompt', onclick: () => { app.keep.trace = { prompt: spec.prompt, target: spec.type === 'next_token' ? [].concat(spec.expect)[0] : '' }; app.tab = 'Trace'; app.select(app.sel); } }, '⌕'),
        h('button', { class: 'ghost danger', title: 'remove case', onclick: () => save({ ...suite, cases: suite.cases.filter((_, j) => j !== i) }) }, '×'))));
  }
  el.append(tbl);
}

// ------------------------------------------------------------------------------------------------ chunks
const CONTEXT = 1_000_000;
export async function chunksView(el, app) {
  const st = app.keep.chunks || (app.keep.chunks = { budget: 40000, format: 'auto', plan: null, open: null, text: '' });
  if (app.keep.chunkSel) { st.sel = app.keep.chunkSel; app.keep.chunkSel = null; st.plan = null; }
  if (st.sel == null) st.sel = (app.state.task.scope || ['model']).join(', ');
  const again = () => app.rerender();
  el.append(h('div', { class: 'row' }, h('h2', {}, 'Chunks'), h('span', { class: 'muted' }, 'weights as text for the programmer model, cut along the hierarchy to fit a token budget')));
  const plan = async () => {
    try { st.plan = await api('/api/plan', { selection: st.sel.split(',').map((s) => s.trim()).filter(Boolean), budget: st.budget, format: st.format }); st.open = null; st.err = null; }
    catch (e) { st.err = e.message; st.plan = null; }
    again();
  };
  el.append(h('div', { class: 'row', style: { margin: '8px 0', alignItems: 'flex-end' } },
    field('selection (addresses)', h('input', { type: 'text', value: st.sel, style: { width: '330px', fontFamily: 'var(--mono)' }, onchange: (e) => { st.sel = e.target.value; } })),
    h('button', { class: 'ghost', style: { alignSelf: 'flex-end' }, onclick: () => { st.sel = app.sel; plan(); } }, 'use selection'),
    field('tokens per chunk', h('input', { type: 'number', value: st.budget, step: 5000, min: 2000, style: { width: '100px' }, onchange: (e) => { st.budget = +e.target.value; } })),
    field('format', h('select', { onchange: (e) => { st.format = e.target.value; } }, [['auto', 'auto — most faithful that fits'], ['g3', 'g3 — 3 significant digits (5.3 tok/weight)'], ['q', 'q — integers × scale (2.5 tok/weight)'],
      ['sparse', 'sparse — top 8 per row'], ['svd', 'svd — rank-4 factors + vocabulary'], ['stats', 'stats — summaries only']].map(([v, l]) => h('option', { value: v, selected: st.format === v }, l)))),
    h('button', { class: 'primary', onclick: plan }, 'Plan')));
  if (st.err) el.append(h('p', { style: { color: 'var(--crit)' } }, st.err));
  if (!st.plan) { if (!st.err) plan(); return; }
  const p = st.plan;
  const pct = Math.min(100, 100 * p.est_tokens / CONTEXT);
  el.append(h('div', { class: 'card' },
    h('div', { class: 'row' }, h('b', {}, `${fmtInt(p.n_params)} weights → ${p.chunks.length} chunk${p.chunks.length === 1 ? '' : 's'}, ~${fmtInt(p.est_tokens)} tokens in format ${p.format}`),
      h('span', { class: 'pill ' + (p.fits_one_context ? 'good' : 'warn') }, p.fits_one_context ? 'fits in one 1M context with room to work' : 'too large for one context: read chunk by chunk, or use a lossier format')),
    h('div', { class: 'budget', style: { margin: '8px 0 3px' }, title: `${fmtInt(p.est_tokens)} of 1,000,000 tokens` }, h('i', { class: p.est_tokens > 700000 ? 'over' : '', style: { width: pct + '%' } }), h('span', { class: 'mark', style: { left: '70%' } })),
    h('div', { class: 'small muted', style: { position: 'relative', height: '16px' } }, h('span', { style: { position: 'absolute', left: 0 } }, '0'),
      h('span', { style: { position: 'absolute', left: '70%', transform: 'translateX(-50%)' } }, 'working limit 700k'), h('span', { style: { position: 'absolute', right: 0 } }, '1M context')),
    h('div', { class: 'small muted', style: { marginTop: '6px' } }, 'whole selection by format: ', Object.entries(p.totals_by_format).map(([k, v]) => `${k} ${fmtInt(v)}`).join(' · '), ' tokens (estimates from measured per-number costs)')));
  const tbl = h('table', { class: 'grid', style: { marginTop: '10px' } }, h('tr', {}, h('th', {}, '#'), h('th', {}, 'chunk'), h('th', { class: 'num' }, 'weights'), h('th', { class: 'num' }, 'tokens'), h('th', { style: { width: '30%' } }, 'of chunk budget')));
  for (const c of p.chunks.slice(0, 300)) {
    tbl.append(h('tr', { class: 'clickable', style: st.open === c.id ? { background: 'var(--surface-2)' } : {}, onclick: async () => { st.open = c.id; st.text = await api('/api/chunk?' + q({ id: c.id, variant: app.variant })); again(); } },
      h('td', {}, c.id), h('td', { class: 'mono' }, c.title), h('td', { class: 'num' }, fmtInt(c.n_params)), h('td', { class: 'num' }, fmtInt(c.est_tokens)),
      h('td', {}, h('div', { class: 'budget', style: { height: '8px' } }, h('i', { style: { width: Math.min(100, 100 * c.est_tokens / p.budget) + '%' } })))));
  }
  el.append(h('div', { class: 'split', style: { alignItems: 'start' } }, h('div', { class: 'scroll', style: { maxHeight: '520px' } }, tbl),
    st.open == null ? h('p', { class: 'muted', style: { marginTop: '14px' } }, 'Click a chunk to see exactly what the programmer model would read: a header saying where the weights sit and what they do, then the weights.')
      : h('div', {}, h('div', { class: 'row', style: { margin: '10px 0 4px' } }, h('b', {}, `chunk ${st.open}`), h('span', { class: 'muted small' }, `${fmtInt(st.text.length)} characters`),
        h('button', { class: 'ghost', onclick: () => navigator.clipboard.writeText(st.text) }, 'copy')), h('pre', { class: 'doc', style: { marginTop: 0 } }, st.text))));
}

// ------------------------------------------------------------------------------------------------ programmer
// The status line and the tool-call timers are computed from event times (server clock, seconds) and the
// server's clock as estimated from a response's `now`, never from counted ticks, so they are right after a
// reload, a reconnect or a stretch in a background tab.
export const SILENCE_S = 60;          // "no output for N s" once nothing at all has arrived for this long
const ACTIVE = ['starting', 'running', 'waiting', 'interrupting'];
// The IDE's own MCP tools (wsp/tools.py). The IDE answers them one at a time, so calls issued together queue
// behind each other; a provider's own tools (Bash, Read, Codex commands and file changes) do not queue there.
export const IDE_TOOLS = new Set(['overview', 'list_nodes', 'plan_chunks', 'read_chunk', 'read_weights', 'inspect', 'vocab', 'run',
  'trace', 'atlas', 'edit', 'rome', 'run_tests', 'patches', 'checkout_files', 'commit_files', 'finish']);
export const isIdeTool = (name) => { const n = String(name || ''); return n.startsWith('mcp__wsp__') || IDE_TOOLS.has(n); };
// Events that begin or end a turn or a run: open calls before one of them were never answered.
const BOUNDARY = ['result', 'user', 'init', 'resumed', 'resume_failed', 'error'];
const MAIN_ARGS = ['address', 'addr', 'selector', 'selection', 'prompt', 'chunk_id', 'id', 'text', 'query', 'suite', 'command', 'message', 'summary'];

export function fmtDuration(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${s % 60} s`;
  return `${Math.floor(s / 3600)} h ${Math.floor(s / 60) % 60} min`;
}

// "inspect card L41N12481": the tool's short name and its main argument.
export function toolLabel(ev) {
  const name = String(ev.name || 'tool').replace(/^mcp__wsp__/, '');
  const inp = ev.input && typeof ev.input === 'object' ? ev.input : {};
  const short = (v) => { const t = (Array.isArray(v) ? v.join(', ') : typeof v === 'object' ? JSON.stringify(v) : String(v)).replace(/\s+/g, ' ').trim(); return t.length > 60 ? t.slice(0, 59) + '…' : t; };
  const parts = [name];
  for (const k of ['what', 'action']) if (typeof inp[k] === 'string' && inp[k]) { parts.push(inp[k]); break; }
  const main = MAIN_ARGS.find((k) => inp[k] != null && inp[k] !== '');
  if (main) parts.push(short(inp[main]));
  return parts.join(' ');
}

// The tool calls of a run, in the order issued. IDE tool calls issued together are answered one after another, so
// such a call is taken to start when it was issued or when the previous IDE result arrived, whichever is later; the
// first open one is running and the rest wait behind it. Any other tool runs from the moment it was issued. A turn
// or run boundary (BOUNDARY), or a run that is no longer active, leaves the open calls before it "not completed":
// they are never shown as running.
export function toolCalls(events, active) {
  const calls = [], open = [];
  let lastResult = -Infinity;
  for (const ev of events) {
    if (ev.kind === 'tool_use') { const c = { id: ev.id, ev, ide: isIdeTool(ev.name), issued: ev.t, start: null, end: null, state: 'open' }; calls.push(c); open.push(c); }
    else if (ev.kind === 'tool_result') {
      const at = open.findIndex((c) => c.id === ev.id);
      if (at < 0) continue;
      const c = open.splice(at, 1)[0];
      c.start = c.ide ? Math.max(c.issued, lastResult) : c.issued; c.end = ev.t; c.state = 'done'; c.error = !!ev.error;
      if (c.ide) lastResult = ev.t;
    } else if (BOUNDARY.includes(ev.kind)) { for (const c of open.splice(0)) c.state = 'not_completed'; }
  }
  let ideOpen = 0;
  for (const c of open) {
    if (!active) { c.state = 'not_completed'; continue; }
    if (!c.ide) { c.state = 'running'; c.start = c.issued; continue; }
    c.state = ideOpen === 0 ? 'running' : 'queued'; c.start = ideOpen === 0 ? Math.max(c.issued, lastResult) : null;
    ideOpen++;
  }
  return calls;
}

export function toolTimeText(c, now) {
  if (c.state === 'running') return `running · ${fmtDuration(now - c.start)}`;
  if (c.state === 'queued') return `waiting behind another call · ${fmtDuration(now - c.issued)}`;
  if (c.state === 'not_completed') return 'not completed';
  if (c.state !== 'done') return '';
  const waited = c.start - c.issued;
  return `took ${fmtDuration(c.end - c.start)}` + (waited >= 1 ? ` · waited ${fmtDuration(waited)} first` : '') + (c.error ? ' · error' : '');
}

// {state, live, time}: `live` is the sentence a screen reader announces (it changes only when the state does);
// `time` is the ticking part beside it. `now` is the server's clock in seconds.
export function programmerStatus(run, events, now) {
  if (!run) return null;
  const status = run.status, active = ACTIVE.includes(status);
  const last = events.length ? events[events.length - 1].t : run.started;
  const quiet = now - (last ?? now);
  let turnStart = run.started;
  for (const ev of events) if (ev.kind === 'user' && ev.t != null) turnStart = ev.t;
  const turn = `turn ${fmtDuration(now - (turnStart ?? now))}`;
  const silent = active && status !== 'waiting' && !run.pending?.length && quiet >= SILENCE_S;
  const tail = silent ? ` · no output for ${fmtDuration(quiet)}` : '';
  const said = silent ? ` · no output for over ${fmtDuration(SILENCE_S)}` : '';
  const lastOf = (kind) => { for (let k = events.length - 1; k >= 0; k--) if (events[k].kind === kind) return events[k]; return null; };
  if (status === 'starting') return { state: 'starting', live: (run.native ? 'Starting the conversation' : 'Starting the harness') + said, time: fmtDuration(now - (run.started ?? now)) + tail };
  if (status === 'waiting' || (active && run.pending?.length)) {
    const p = lastOf('prompt');
    const title = run.pending?.[0]?.title || p?.title;
    return { state: 'prompt', live: 'Waiting for your response' + (title ? ': ' + title : ''), time: p?.t != null ? fmtDuration(now - p.t) : '' };
  }
  if (status === 'interrupting') return { state: 'interrupting', live: 'Interrupting the turn' + said, time: turn + tail };
  if (active) {
    const calls = toolCalls(events, true);
    const running = calls.filter((c) => c.state === 'running'), queued = calls.filter((c) => c.state === 'queued').length;
    if (running.length) {
      const first = running.find((c) => c.ide) || running[0], others = running.length - 1;
      return { state: 'tool', live: `Running ${toolLabel(first.ev)}` + (others ? ` · ${others} more call${others === 1 ? '' : 's'} running` : '')
        + (queued ? ` · ${queued} more call${queued === 1 ? '' : 's'} waiting behind it` : '') + said,
        time: `${fmtDuration(now - first.start)} · ${turn}` + tail };
    }
    return { state: 'model', live: 'Waiting for the model' + said, time: `${fmtDuration(quiet)} since the last output · ${turn}` + tail };
  }
  const result = lastOf('result'), lastUser = lastOf('user');
  if (status === 'idle') {
    const ok = result && (!lastUser || result.i > lastUser.i) ? result.ok : null;
    const took = ok != null && lastUser && result.t >= lastUser.t ? `turn took ${fmtDuration(result.t - lastUser.t)}` : '';
    return { state: 'idle', live: (ok === false ? 'Turn stopped or failed' : ok ? 'Turn finished' : 'Ready') + ' · waiting for your message', time: took };
  }
  const ran = run.ended && run.started ? `ran ${fmtDuration(run.ended - run.started)}` : '';
  if (status === 'failed') { const e = lastOf('error'); return { state: 'failed', live: 'Failed' + (e?.text ? ': ' + String(e.text).split('\n')[0].slice(0, 200) : ''), time: ran }; }
  if (status === 'interrupted') return { state: 'interrupted', live: 'Interrupted' + (run.interrupted ? ': ' + run.interrupted : ''), time: ran };
  if (status === 'stopped') return { state: 'stopped', live: 'Stopped', time: ran };
  if (status === 'done') return { state: 'done', live: 'Finished', time: ran };
  if (status === 'closing') return { state: 'closing', live: 'Closing the conversation', time: '' };
  return { state: status, live: status === 'closed' ? 'Conversation closed' : String(status), time: ran };
}

// The server's clock (seconds) as this browser sees it; the offset comes from a response's `now`.
export function serverNow(st, clientMs = Date.now()) { return (clientMs + (st.clockOffset || 0)) / 1000; }
export function noteServerTime(st, now, sentMs, receivedMs = Date.now()) {
  if (typeof now === 'number' && Number.isFinite(now)) st.clockOffset = now * 1000 - (sentMs + receivedMs) / 2;
}

// The status sentence is a polite live region. One node per Programmer tab state (st) is kept across renders and
// re-attached, and its text is written only while it is in the document (a render builds the view detached and
// mounts it afterwards), so a change is a change of an attached live region. Not tested with a real screen reader.
function announce(st, text) {
  const node = st.statusLive;
  if (node.isConnected === false) {                  // not mounted yet: write once the view is in the document
    st.liveWanted = text;
    if (!st.liveRetry) {
      let tries = 0;
      const retry = () => {
        if (st.statusLive !== node) { st.liveRetry = null; return; }
        if (node.isConnected === false && ++tries < 20) { st.liveRetry = setTimeout(retry, 50); return; }
        st.liveRetry = null;
        if (node.isConnected !== false && node.textContent !== st.liveWanted) node.textContent = st.liveWanted;
      };
      st.liveRetry = setTimeout(retry, 0);
    }
    return;
  }
  if (node.textContent !== text) node.textContent = text;
}

// Redraws the status line and every tool call's timer from st.events; called on each event and once a second.
export function refreshProgrammerStatus(st, clientMs = Date.now()) {
  const now = serverNow(st, clientMs);
  const status = programmerStatus(st.run, st.events, now);
  if (st.statusLive && status) {
    announce(st, status.live);
    st.statusTime.textContent = status.time;
    st.statusBox.className = 'agent-status ' + status.state;
  }
  if (st.toolNodes) {
    for (const c of toolCalls(st.events, ACTIVE.includes(st.run?.status))) {
      const node = st.toolNodes.get(c.id);
      if (node) { node.textContent = toolTimeText(c, now); node.className = 'ev-time ' + c.state; }
    }
  }
  return status;
}

// A dollar figure only for a run that pays per call: a legacy/custom harness (its own estimate) or a native sign-in
// billed per API call. A subscription run's result never shows one (the server drops it too).
function evCard(ev, costNote = null) {
  const k = ev.kind;
  if (k === 'user') return h('div', { class: 'ev user' }, h('div', { class: 'ev-kind' }, 'you'), ev.text);
  if (k === 'prompt' || k === 'prompt_resolved') return h('div', { class: 'ev small muted' }, k === 'prompt' ? 'Waiting for your response: ' + ev.title : 'Response sent');
  if (k === 'resumed') return h('div', { class: 'ev result' }, h('div', { class: 'ev-kind' }, 'resumed'), ev.text);
  if (k === 'resume_failed') return h('div', { class: 'ev result bad' }, h('div', { class: 'ev-kind' }, 'resume not possible'), ev.text);
  if (k === 'text') return h('div', { class: 'ev text' }, ev.text);
  if (k === 'thinking') return h('div', { class: 'ev thinking' }, ev.text);
  if (k === 'tool_use') {
    const inp = ev.input || {};
    const body = inp.script != null ? inp.script + (inp.message ? `\n# ${inp.message}` : '') + (inp.dry_run ? '\n# dry run' : '') : JSON.stringify(inp);
    // The timer node is kept on the card: a browser's `children` is an HTMLCollection of elements only, with no
    // array methods, so it must not be found by position.
    const time = h('span', { class: 'ev-time' });
    const card = h('div', { class: 'ev tool_use' }, h('div', { class: 'ev-kind' }, '→ ' + String(ev.name || '').replace(/^mcp__wsp__/, ''), ' ', time),
      body && body !== '{}' ? h('pre', {}, body) : null);
    card.timeNode = time;
    return card;
  }
  if (k === 'tool_result') return h('div', { class: 'ev tool_result' + (ev.error ? ' error' : '') }, h('pre', {}, ev.text));
  if (k === 'init') return h('div', { class: 'ev' }, h('div', { class: 'ev-kind' }, 'session'), `${ev.model || ''} · ${ev.n_tools ?? '?'} tools · MCP ${(ev.mcp || []).map((m) => `${m.name}: ${m.status}`).join(', ')}`);
  if (k === 'result') return h('div', { class: 'ev result' + (ev.ok ? '' : ' bad') }, h('div', { class: 'ev-kind' }, ev.ok ? 'turn complete' : 'turn stopped or failed'),
    [ev.turns != null ? `${ev.turns} turns` : '', ev.duration_ms ? `${(ev.duration_ms / 1000).toFixed(0)} s` : '', ev.input_tokens ? `${fmtInt(ev.input_tokens)} tokens in` : '', ev.output_tokens ? `${fmtInt(ev.output_tokens)} out` : '', costNote && typeof ev.cost_usd === 'number' ? `est. $${ev.cost_usd.toFixed(2)} (${costNote})` : ''].filter(Boolean).join(' · '), !ev.ok && ev.text ? h('p', {}, ev.text) : null);
  return h('div', { class: 'ev' }, h('pre', {}, ev.text || JSON.stringify(ev)));
}

function appendAgentEvent(log, event, toolNodes) {
  log.drawn = (log.drawn || 0) + 1;                // how many of the run's events this log node holds
  if (event.kind === 'text_delta' || event.kind === 'thinking_delta') {
    const key = `${event.kind}:${event.id}`;
    const streams = log.streams || (log.streams = new Map());
    const previous = streams.get(key);
    if (previous) { previous.textContent += event.text; return; }
    const node = evCard({ ...event, kind: event.kind === 'text_delta' ? 'text' : 'thinking' });
    streams.set(key, node);
    log.append(node);
  } else {
    const node = evCard(event, log.costNote);
    if (event.kind === 'tool_use' && toolNodes && event.id != null) toolNodes.set(event.id, node.timeNode);
    log.append(node);
  }
}

// Events missed while the event stream was reconnecting, and the server's clock: fetched every SYNC_S while a
// run is active (and after a reconnect), from the index of the first event not yet seen.
const SYNC_S = 10;
async function syncRun(st) {
  if (!st.run?.id || st.syncing) return false;
  st.syncing = true;
  const id = st.run.id, sent = Date.now();
  let since = 0;
  for (const ev of [...st.events].sort((a, b) => a.i - b.i)) { if (ev.i === since) since++; else if (ev.i > since) break; }
  try {
    const r = await api('/api/harness?' + q({ id, since }));
    noteServerTime(st, r.now, sent);
    if (st.run?.id !== id) return false;
    let added = false;
    for (const ev of r.events || []) if (!st.events.some((e) => e.i === ev.i)) { st.events.push(ev); added = true; if (st.logEl?.runId === id) appendAgentEvent(st.logEl, ev, st.logEl.toolNodes); }
    if (added) st.events.sort((a, b) => a.i - b.i);
    const changed = r.status !== st.run.status;
    const { events, now, ...meta } = r;
    st.run = { ...st.run, ...meta };
    return changed;
  } catch (e) {
    // The server no longer knows the run: it restarted. The run is not shown as still working.
    if (e.payload?.error !== 'Unknown run' || st.run?.id !== id) return false;
    st.run = { ...st.run, status: 'interrupted', ended: st.run.ended ?? serverNow(st),
      interrupted: 'the IDE server restarted, so this run is no longer running. Its saved conversation is listed under earlier conversations.' };
    return true;
  } finally { st.syncing = false; st.syncedAt = Date.now(); }
}

// At most one ticker in the page: once a second while the Programmer tab shows an active run. It stops when the
// run ends, the tab is left, or the workspace changes (app.keep is reset, so app.keep.prog is another object).
let ticker = null;
export function stopTicker() { if (ticker) { clearInterval(ticker.id); ticker = null; } }
export const tickerCount = () => (ticker ? 1 : 0);
function ensureTicker(app, st) {
  if (ticker && ticker.app === app && ticker.st === st) return;
  stopTicker();
  ticker = { app, st, id: setInterval(() => tickProgrammer(app), 1000) };
}
const tickerOwns = (app, st) => !!st && app.keep.prog === st && app.tab === 'Programmer' && ACTIVE.includes(st.run?.status);

// One tick of the status line.
export async function tickProgrammer(app, clientMs = Date.now()) {
  const st = app.keep.prog;
  if (ticker && (ticker.app !== app || ticker.st !== st)) { stopTicker(); return; }
  if (!st || app.tab !== 'Programmer') { stopTicker(); return; }
  refreshProgrammerStatus(st, clientMs);
  if (ACTIVE.includes(st.run?.status) && (clientMs - (st.syncedAt || 0) >= SYNC_S * 1000 || app.eventsLostAt > (st.syncedAt || 0))) {
    if (await syncRun(st)) { app.rerender(); app.runChanged?.(); if (ENDED.includes(st.run?.status)) refreshAfterEnd(app, st); }
    else { refreshProgrammerStatus(st); followLog(st); }
  }
  if (!tickerOwns(app, st)) stopTicker();
}

// After a run ends the conversation list is fetched once more a little later, even when the run's final event
// was missed, so its row (resumable, Try again) is never left as it was in the moment the run was finishing.
let endedRefreshMs = 1500;
export function setEndedRefreshMs(ms) { endedRefreshMs = ms; }     // tests
const ENDED = ['closed', 'failed', 'interrupted', 'done', 'stopped'];
function refreshAfterEnd(app, st) {
  if (st.endRefresh) return;
  st.endRefresh = setTimeout(() => { st.endRefresh = null; if (app.keep.prog === st && app.tab === 'Programmer') app.rerender(); }, endedRefreshMs);
}

export function onAgentEvent(ev, app) {
  const st = app.keep.prog || (app.keep.prog = { events: [], run: null });
  if (st.clockOffset == null && typeof ev.t === 'number') st.clockOffset = ev.t * 1000 - Date.now();   // until a response gives `now`
  if (ev.type === 'agent') {
    if (!st.run || st.run.id !== ev.run) { st.run = { id: ev.run, status: 'starting' }; st.events = []; }
    if (st.events.some(event => event.i === ev.event.i)) return;
    st.events.push(ev.event);
    // The log node is kept up to date on every tab (it is cheap, and detached), so coming back to the tab shows
    // everything that arrived meanwhile without rebuilding it.
    const log = st.logEl;
    if (log && log.runId === ev.run) {
      appendAgentEvent(log, ev.event, log.toolNodes);
      if (app.tab === 'Programmer') { refreshProgrammerStatus(st); followLog(st); }
    }
  } else if (ev.type === 'run') {
    const was = st.run?.status;
    st.run = ev.run;
    if (ENDED.includes(ev.run?.status) && !ENDED.includes(was)) refreshAfterEnd(app, st);
    if (app.tab === 'Programmer') app.rerender();
  }
  app.runChanged?.();                              // the tab bar's marker; it never redraws the current view
}

// The log keeps to its end while the person has not scrolled up; once they have, new events never move it. Whether it
// follows belongs to one run's log (log.follow, log.scroll): a log built for another run starts at its end, following.
const FOLLOW_PX = 40;
function followLog(st) {
  const log = st.logEl, box = log?.parentElement;
  if (!box || box.isConnected === false || log.follow === false) return;
  box.scrollTop = box.scrollHeight;
}

// What the Programmer tab's button says from any tab: null when no conversation is open. Text, so that a screen
// reader reads it with the tab's name; the colour (CSS class `state`) only repeats it.
export function programmerMarker(st) {
  const run = st?.run;
  if (!run) return null;
  if (run.status === 'waiting' || (ACTIVE.includes(run.status) && run.pending?.length))
    return { state: 'prompt', text: 'waiting for you', title: 'The programmer asked for your permission or an answer; it cannot go on until you respond in the Programmer tab.' };
  if (ACTIVE.includes(run.status)) return { state: 'working', text: 'working', title: 'A programmer turn is running. Open the Programmer tab to follow it.' };
  if (run.status === 'idle') return { state: 'idle', text: 'turn finished', title: 'The programmer finished its turn and waits for your next message.' };
  return null;
}

// At page load, from any tab: which run the server has, so the tab bar can say whether one is working. One GET of
// the runs' metadata (no events); the Programmer tab fetches the events itself when it is first shown.
export async function seedProgrammer(app) {
  const st = app.keep.prog || (app.keep.prog = { events: [], run: null });
  if (st.loaded || st.run) return st.run;
  const sent = Date.now();
  const hs = await api('/api/harness');
  if (app.keep.prog !== st || st.loaded || st.run) return st.run;
  noteServerTime(st, hs.now, sent);
  noteToken(st, hs);
  const last = lastRunHere(hs);
  if (last) st.run = last;
  return st.run;
}

// The last run of the open workspace. The server keeps its runs across a workspace switch; a run of another
// workspace is not this tab's (its record is in that workspace's conversations), so the tab starts without one.
export function lastRunHere(hs) {
  const here = (hs?.runs || []).filter((r) => typeof r.workspace === 'string' && r.workspace === hs.workspace);
  return here[here.length - 1] || null;
}

const PROVIDER = { claude: 'Claude Code', codex: 'Codex', omp: 'oh-my-pi', custom: 'custom command' };

// Every Programmer POST carries the per-process token the server hands out (X-WSP-Auth); GETs refresh it.
function harnessPost(st, path, body) {
  return api(path, body, { headers: { 'Content-Type': 'application/json', 'X-WSP-Auth': st.authToken || '' } });
}
const noteToken = (st, r) => { if (r && typeof r.csrf_token === 'string') st.authToken = r.csrf_token; };
const CONV_STATUS = { interrupted: 'interrupted', closed: 'closed', failed: 'failed', done: 'finished', stopped: 'stopped', idle: 'waiting for a message' };
const pad = (n) => String(n).padStart(2, '0');
export function fmtWhen(seconds) {
  if (typeof seconds !== 'number') return '';
  const d = new Date(seconds * 1000);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// The saved conversations of the open workspace and, when one is opened, its transcript (read-only) with the two
// actions. Reading never contacts a provider; Resume and New conversation are explicit clicks.
function conversationsView(el, app, st, { open, running, start, authState, again, showRun }) {
  const list = st.conversations || [];
  const openConversation = async (id) => {
    const sent = Date.now();
    try {
      const r = await api('/api/harness/conversation?' + q({ id }));
      noteServerTime(st, r.now, sent);
      st.viewing = id; st.viewed = r; st.viewError = null; st.resumeText = '';
      if (!open && !running && ['claude', 'codex'].includes(r.meta.provider)) { st.harness = r.meta.provider; st.mode = r.meta.mode || st.mode; }
    } catch (e) { st.convError = e.message; }
    again();
  };
  if (list.length || st.convError) {
    const tbl = h('table', { class: 'grid', style: { marginTop: '6px' } },
      h('tr', {}, h('th', {}, 'started'), h('th', {}, 'harness'), h('th', {}, 'task'), h('th', {}, 'status'), h('th', { class: 'num' }, 'events'), h('th', {}, '')));
    for (const c of list) {
      if (c.unreadable) {
        // One damaged entry is one row; the others are listed, and Start and Resume work as usual.
        tbl.append(h('tr', { class: 'unreadable' }, h('td', { class: 'mono' }, c.id), h('td', {}, ''),
          h('td', { colspan: 2 }, 'unreadable conversation: ' + c.unreadable), h('td', {}, ''), h('td', {}, '')));
        continue;
      }
      const status = c.live ? 'running now' : c.elsewhere ? 'held by another IDE process' : CONV_STATUS[c.status] || c.status;
      tbl.append(h('tr', { style: st.viewing === c.id ? { background: 'var(--surface-2)' } : {} },
        h('td', { class: 'mono' }, fmtWhen(c.started)), h('td', {}, (PROVIDER[c.provider] || c.provider) + (c.mode ? ` · ${c.mode}` : '')),
        h('td', {}, c.task), h('td', {}, status), h('td', { class: 'num' }, fmtInt(c.n_events)),
        h('td', {}, c.live ? h('button', { class: 'ghost', title: 'The open conversation is at the top of this tab', onclick: showRun }, 'Show at the top')
          : h('button', { class: 'ghost', onclick: () => openConversation(c.id) }, 'Open'))));
    }
    el.append(h('div', { class: 'card', id: 'agent-conversations', style: { marginTop: '10px' } },
      h('h3', { style: { marginTop: 0 } }, 'earlier conversations'),
      h('p', { class: 'hint' }, 'Saved in this workspace\'s conversations directory, beside the agent\'s working directory. The files are readable by your user account, which includes any agent you run as yourself, such as a files-mode programmer. Open one to read it; reading never contacts the provider.'),
      st.lockingNote ? h('p', { class: 'hint', id: 'conversations-locking' }, st.lockingNote) : null,
      st.convError ? h('p', { class: 'pill bad' }, st.convError) : null,
      list.length ? h('div', { class: 'scroll', style: { maxHeight: '240px' } }, tbl) : null));
  }
  if (!st.viewing || !st.viewed) return;
  const v = st.viewed, m = v.meta;
  const verdict = list.find((c) => c.id === st.viewing)?.resume || v.resume || { ok: false, reason: 'Unknown conversation.' };
  const signedIn = st.harness !== m.provider || !['claude', 'codex'].includes(m.provider) || authState?.status === 'ready';
  const label = PROVIDER[m.provider] || m.provider;
  const reopen = async () => { const id = st.viewing; await refreshConversations(st); if (id) await openConversation(id); else again(); };
  const resume = async (retry = false) => {
    st.viewError = null;
    const sent = Date.now();
    try {
      const r = await harnessPost(st, '/api/harness/resume', { conversation: st.viewing, text: st.resumeText || '', ...(retry ? { retry: true } : {}) });
      noteServerTime(st, r.now, sent);
      const { events, now, ...meta } = r;
      st.run = meta; st.events = events; st.syncedAt = Date.now();
      st.viewing = null; st.viewed = null; st.resumeText = '';
    } catch (e) { st.viewError = e.message; }
    again();
  };
  const release = async () => {
    const owner = m.owner_pid ? ` (pid ${m.owner_pid})` : '';
    if (!confirm(`Mark this conversation interrupted?\n\nOnly do this if the IDE process recorded as running it${owner} is not running it any more. If that process is still running the conversation, the two will write over each other's record.`)) return;
    try { await harnessPost(st, '/api/harness/conversation/release', { conversation: st.viewing, confirm: true }); st.viewError = null; }
    catch (e) { st.viewError = e.message; }
    await reopen();
  };
  const newConversation = async () => { st.viewing = null; st.viewed = null; await start(); };
  const reason = !verdict.ok ? verdict.reason : !signedIn ? `Sign in to ${label} above before resuming.` : null;
  const status = programmerStatus({ status: m.live ? 'running' : m.status, started: m.started, ended: m.ended, interrupted: m.interrupted }, v.events, serverNow(st));
  const busy = open || running;
  // A per-call Claude resume gets the recorded budget again, counted from zero (the provider's budget is per query).
  const budgetNote = m.provider === 'claude' && m.billing === 'per_call' && m.budget
    ? `Resuming gives this conversation its $${m.budget} budget again, counted from zero: what it spent before is not counted.` : null;
  const box = h('div', { class: 'card', id: 'agent-conversation', style: { marginTop: '10px' } },
    h('div', { class: 'row' }, h('h3', { style: { margin: 0 } }, `conversation of ${fmtWhen(m.started)}`),
      h('span', { class: 'muted small' }, [label, m.mode, m.model || 'default model', `${fmtInt(v.events.length)} events`].filter(Boolean).join(' · ')),
      h('span', { class: 'grow' }), h('button', { class: 'ghost', onclick: () => { st.viewing = null; st.viewed = null; again(); } }, 'Close transcript')),
    h('p', {}, h('b', {}, 'task: '), m.task),
    h('p', { class: 'agent-status-text' }, status ? status.live + (status.time ? ' · ' + status.time : '') : ''),
    m.closed_reason ? h('p', { class: 'hint' }, m.closed_reason) : null,
    m.last_resume_failure ? h('p', { class: 'hint', id: 'resume-last-failure' }, 'The last resume attempt failed: ' + m.last_resume_failure) : null,
    h('p', { class: 'hint' }, 'Resume continues the provider\'s own session with this conversation\'s harness, access mode and model. Its first message tells the model that the workspace may have changed since and lists the current patches.'),
    budgetNote ? h('p', { class: 'hint', id: 'resume-budget' }, budgetNote) : null,
    (verdict.ok || verdict.retry) && signedIn ? field('message to send with the resume (optional)', h('textarea', { rows: 2, value: st.resumeText || '', oninput: (e) => { st.resumeText = e.target.value; } })) : null,
    reason ? h('p', { class: 'hint', id: 'resume-refusal' }, 'Resume is not available: ' + reason) : null,
    h('div', { class: 'row', style: { marginTop: '6px' } },
      h('button', { class: 'primary', disabled: !verdict.ok || !signedIn || busy, onclick: () => resume() }, 'Resume'),
      verdict.retry ? h('button', { disabled: !signedIn || busy, onclick: () => resume(true) }, 'Try again') : null,
      verdict.release ? h('button', { class: 'danger', disabled: busy, onclick: release }, 'It is not running; mark interrupted') : null,
      h('button', { disabled: busy || (['claude', 'codex'].includes(st.harness) && authState?.status !== 'ready'), onclick: newConversation }, 'New conversation'),
      h('span', { class: 'muted small' }, `New conversation starts ${PROVIDER[st.harness] || st.harness} on the current task, as Start programmer does.`)),
    st.viewError ? h('p', { class: 'pill bad', style: { whiteSpace: 'normal' } }, st.viewError) : null);
  const log = h('div', {});
  log.costNote = !m.native ? 'harness estimate' : m.billing === 'per_call' ? 'billed per call' : null;
  const tools = new Map();
  for (const event of v.events) appendAgentEvent(log, event, tools);
  for (const c of toolCalls(v.events, false)) { const node = tools.get(c.id); if (node) { node.textContent = toolTimeText(c, 0); node.className = 'ev-time ' + c.state; } }
  box.append(h('div', { class: 'scroll', style: { maxHeight: '420px', marginTop: '8px' } }, log));
  el.append(box);
}

async function refreshConversations(st) {
  try {
    const sent = Date.now();
    const cl = await api('/api/harness/conversations');
    noteServerTime(st, cl.now, sent);
    noteToken(st, cl);
    st.conversations = cl.conversations || []; st.convError = null; st.lockingNote = cl.locking || null;
  } catch (e) { st.conversations = st.conversations || []; st.convError = e.message; }
}

export async function programmerView(el, app) {
  const st = app.keep.prog || (app.keep.prog = { events: [], run: null });
  const renderGeneration = st.renderGeneration = (st.renderGeneration || 0) + 1;
  st.harness = st.harness || (app.state.harnesses.find((x) => x.installed && x.name !== 'custom') || { name: 'claude' }).name;
  st.mode = st.mode || 'tools';
  const t = st.taskDraft || app.state.task;
  const again = () => app.rerender();
  if (!st.loaded) {
    let sent = Date.now();
    const hs = await api('/api/harness');
    noteServerTime(st, hs.now, sent);
    noteToken(st, hs);
    if (renderGeneration !== st.renderGeneration) return;
    const last = lastRunHere(hs);
    if (last) {
      sent = Date.now();
      const full = await api('/api/harness?' + q({ id: last.id }));
      noteServerTime(st, full.now, sent);
      if (renderGeneration !== st.renderGeneration) return;
      const { events, now, ...meta } = full;
      st.run = meta; st.events = events; st.syncedAt = Date.now();
    }
    st.loaded = true;
  }
  // Earlier conversations of this workspace, read from its saved files; listing them never contacts a provider.
  await refreshConversations(st);
  if (renderGeneration !== st.renderGeneration) return;
  const ended = ['closed', 'failed', 'interrupted', 'done', 'stopped'];
  if (st.run?.native && !ended.includes(st.run.status)) { st.harness = st.run.harness; st.mode = st.run.mode || st.mode; }
  const open = st.run?.native && !ended.includes(st.run.status);
  el.append(h('div', { class: 'row' }, h('h2', {}, 'Programmer'), h('span', { class: 'muted' }, 'hand the task to a model running in a standard agent harness; it reads and writes weights through the IDE\'s tools')));
  // Sign-in, task and harness: what a new conversation starts with. Placed first when no conversation is open, and
  // folded under the open conversation's run otherwise (see the end of this function).
  const setup = h('div', {});

  // Authentication has its own cached state; polling never starts a login.
  const nativeProvider = ['claude', 'codex'].includes(st.harness);
  st.auth = st.auth || {};
  st.authGeneration = st.authGeneration || {};
  clearTimeout(st.authTimer);
  const authRequest = async (provider, route = '', body = {}) => {
    const generation = st.authGeneration[provider] = (st.authGeneration[provider] || 0) + 1;
    try {
    const response = await fetch('/api/harness/auth' + (route || '?' + q({ provider })), route ? {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-WSP-Auth': st.authToken || '' },
      body: JSON.stringify({ ...body, provider }),
    } : { cache: 'no-store' });
    const data = await response.json();
    if (generation !== st.authGeneration[provider]) return st.auth[provider];
    if (!response.ok) throw new Error(data.error || data.detail || 'Authentication request failed');
    st.authToken = data.csrf_token;
    st.auth[provider] = data;
    st.authReadAt = Date.now();
    return data;
    } catch (e) {
      if (generation !== st.authGeneration[provider]) return st.auth[provider];
      throw e;
    }
  };
  if (nativeProvider && !st.authBusy && (!st.auth[st.harness] || st.authSelected !== st.harness || Date.now() - (st.authReadAt || 0) > 10000)) {
    if (st.authSelected !== st.harness && st.authCodeInput) st.authCodeInput.value = '';
    st.authSelected = st.harness;
    try { await authRequest(st.harness); } catch (e) { st.auth[st.harness] = { status: 'error', message: e.message }; }
    if (renderGeneration !== st.renderGeneration) return;
    st.authReadAt = Date.now();
  }
  const authState = st.auth[st.harness];
  if (nativeProvider && ['checking', 'signing_in', 'verifying'].includes(authState?.status)) {
    const provider = st.harness;
    const pollAuth = async () => {
      if (app.tab !== 'Programmer' || st.harness !== provider) return;
      if (st.authBusy) { st.authTimer = setTimeout(pollAuth, 1000); return; }
      const before = JSON.stringify(st.auth[provider]);
      try { await authRequest(provider); } catch (e) { st.auth[provider] = { status: 'error', message: e.message }; }
      st.authReadAt = Date.now();
      if (JSON.stringify(st.auth[provider]) !== before) again();
      else st.authTimer = setTimeout(pollAuth, 1000);
    };
    st.authTimer = setTimeout(pollAuth, 1000);
  }
  const authAction = async (route, body = {}) => {
    const provider = st.harness;
    if (st.authCodeInput) st.authCodeInput.value = '';
    st.authBusy = true;
    again();
    try { await authRequest(provider, route, body); }
    catch (e) { st.auth[provider] = { ...st.auth[provider], message: e.message }; }
    finally { st.authBusy = false; again(); }
  };
  if (nativeProvider) {
    const pending = ['signing_in', 'verifying'].includes(authState?.status);
    const label = st.harness === 'claude' ? 'Claude' : 'ChatGPT';
    const codeInput = st.authCodeInput || (st.authCodeInput = h('input', { type: 'password', autocomplete: 'off', placeholder: 'Paste the sign-in code', 'aria-label': 'Sign-in code', style: { width: '320px' } }));
    if (!authState?.needs_code || st.authCodeAttempt !== authState?.attempt_id) codeInput.value = '';
    st.authCodeAttempt = authState?.attempt_id;
    setup.append(h('div', { class: 'card', id: 'agent-auth', style: { margin: '10px 0' } },
      h('h3', { style: { marginTop: 0 } }, label + ' authentication'),
      h('p', {}, ({ ready: 'Authentication configured', signed_out: 'Sign in before starting the programmer', checking: 'Checking authentication…', signing_in: 'Waiting for sign-in…', verifying: 'Verifying sign-in…', cancelled: 'Sign-in cancelled', error: 'Authentication check failed' })[authState?.status] || 'Checking authentication…'),
      authState?.message ? h('p', { class: 'hint' }, authState.message) : null,
      h('div', { class: 'row' },
        ['signed_out', 'cancelled'].includes(authState?.status) ? h('button', { class: 'primary', disabled: open || st.authBusy, onclick: () => authAction('/login') }, 'Sign in') : null,
        authState?.url && pending ? h('a', { href: authState.url, target: '_blank', rel: 'noopener noreferrer', referrerpolicy: 'no-referrer' }, 'Open ' + label + ' sign-in') : null,
        pending ? h('button', { disabled: st.authBusy, onclick: () => authAction('/cancel', { attempt_id: authState.attempt_id }) }, 'Cancel sign-in') : null,
        !pending ? h('button', { disabled: open || st.authBusy || authState?.status === 'checking', onclick: () => authAction('/refresh') }, 'Check again') : null),
      authState?.needs_code ? h('div', { class: 'row', style: { marginTop: '10px' } }, codeInput,
        h('button', { disabled: st.authBusy, onclick: () => { const code = codeInput.value.trim(); codeInput.value = ''; if (code) authAction('/code', { attempt_id: authState.attempt_id, code }); } }, 'Submit sign-in code')) : null));
  }

  // task
  const edit = { ...t };
  const touch = () => { app.keep.editingTask = true; st.taskDraft = edit; };
  const saveTask = async () => {
    edit.scope = (typeof edit.scope === 'string' ? edit.scope.split(',') : edit.scope).map((s) => s.trim()).filter(Boolean);
    try { app.state.task = (await api('/api/task', edit)).task; app.keep.editingTask = false; st.taskDraft = null; await app.refreshStatus(); return true; } catch (e) { alert(e.message); return false; }
  };
  const suites = app.state.suites || [];
  setup.append(h('div', { class: 'card', style: { margin: '10px 0' } }, h('h3', { style: { marginTop: 0 } }, 'task'),
    h('div', { class: 'row', style: { alignItems: 'flex-end' } }, field('title', h('input', { type: 'text', value: t.title || '', style: { width: '260px' }, oninput: (e) => { edit.title = e.target.value; touch(); } })),
      field('scope — addresses the programmer may change', h('input', { type: 'text', value: typeof t.scope === 'string' ? t.scope : (t.scope || []).join(', '), style: { width: '300px', fontFamily: 'var(--mono)' }, oninput: (e) => { edit.scope = e.target.value; touch(); } })),
      field('test suite', h('select', { onchange: (e) => { edit.suite = e.target.value || null; touch(); } }, [h('option', { value: '' }, '(none)'), ...suites.map((s) => h('option', { value: s.name, selected: s.name === t.suite }, s.name))])),
      field('max drift (KL)', h('input', { type: 'number', step: 0.01, value: t.preserve ? t.preserve.max_kl : '', placeholder: 'none', style: { width: '80px' },
        oninput: (e) => { edit.preserve = e.target.value === '' ? null : { max_kl: +e.target.value }; touch(); } }))),
    h('div', { style: { marginTop: '8px' } }, field('instructions', h('textarea', { rows: 4, value: t.instructions || '', style: { fontFamily: 'var(--sans)', fontSize: '13px' }, oninput: (e) => { edit.instructions = e.target.value; touch(); } }))),
    h('div', { class: 'row', style: { marginTop: '8px' } }, h('button', { disabled: open, onclick: async () => { if (await saveTask()) again(); } }, 'Save task'),
      h('button', { class: 'ghost', onclick: async () => { st.brief = st.brief ? null : await api('/api/brief?' + q({ mode: st.mode })); again(); } }, st.brief ? 'hide brief' : 'show the brief the programmer receives'),
      st.brief ? h('button', { class: 'ghost', onclick: () => navigator.clipboard.writeText(st.brief) }, 'copy brief') : null),
    st.brief ? h('pre', { class: 'doc' }, st.brief) : null));

  // harness
  const running = st.run && ['running', 'starting', 'waiting', 'interrupting'].includes(st.run.status);
  const selected = app.state.harnesses.find(x => x.name === st.harness);
  const action = async (route, body) => { try { st.run = await harnessPost(st, route, body); st.error = null; } catch (e) { st.error = e.message; } again(); };
  const hsel = h('select', { onchange: (e) => { st.harness = e.target.value; again(); } }, app.state.harnesses.map((x) => h('option', { value: x.name, selected: x.name === st.harness }, x.label + (x.installed ? '' : ' — not installed'))));
  // The Claude budget stops a run once its estimated spend reaches it; it exists only for a sign-in billed per call.
  const budgetShown = st.harness === 'claude' && authState?.status === 'ready' && authState.billing === 'per_call';
  const budget = budgetShown ? st.budget || 0 : 0;
  const start = async () => {
    if (!await saveTask()) return;
    const sent = Date.now();
    try { st.events = []; st.run = await harnessPost(st, '/api/harness/start', { harness: st.harness, mode: st.mode, model: st.model || '', budget, extra: st.extra || '', custom: st.custom || '' }); noteServerTime(st, st.run.now, sent); st.syncedAt = Date.now(); st.prep = null; }
    catch (e) { st.prep = { error: e.message }; }
    again();
  };
  // While a conversation is open (or a legacy run runs) the run comes first on the page, with its own Interrupt /
  // Close buttons; the setup cards are folded under it.
  const runFirst = !!(open || running);
  setup.append(h('div', { class: 'card' }, h('h3', { style: { marginTop: 0 } }, 'harness'),
    h('div', { class: 'row', style: { alignItems: 'flex-end' } }, field('harness', hsel),
      field('access', tabs(['tools', 'files'], st.mode, (m) => { st.mode = m; again(); }), ),
      field('model (optional)', h('input', { type: 'text', value: st.model || '', placeholder: 'harness default', style: { width: '150px' }, onchange: (e) => { st.model = e.target.value; } })),
      budgetShown ? field('Claude budget (USD, billed per call)', h('input', { type: 'number', value: st.budget || '', placeholder: 'none', step: 0.5, style: { width: '80px' }, onchange: (e) => { st.budget = +e.target.value; } })) : null,
      !runFirst ? h('button', { class: 'primary', disabled: !selected?.installed || (nativeProvider && authState?.status !== 'ready') || st.authBusy, onclick: start }, 'Start programmer') : null,
      h('button', { disabled: open || running, onclick: async () => { st.prep = await harnessPost(st, '/api/harness/prepare', { harness: st.harness, mode: st.mode, model: st.model || '', budget, custom: st.custom || 'my-harness --prompt-file {brief} --mcp-config {mcp_config}' }).catch((e) => ({ error: e.message })); again(); } }, 'Prepare only')),
    st.harness === 'custom' ? h('div', { style: { marginTop: '8px' } }, field('command — {brief} {mcp_config} {url} {dir} are substituted; the brief is also piped to stdin',
      h('input', { type: 'text', value: st.custom || '', style: { width: '100%', fontFamily: 'var(--mono)' }, onchange: (e) => { st.custom = e.target.value; } }))) : null,
    h('p', { class: 'hint' }, st.mode === 'tools' ? (st.harness === 'codex' ? 'Codex tools mode: WSP edits plus built-in read-only filesystem access. Requests to expand file or command permissions are declined.' : 'Claude tools mode: only WSP tools. OMP and custom commands use their own permissions.') : 'files: it also gets its own file and shell tools, to edit checked-out .npy weights with numpy and commit the difference.'),
    selected?.setup_error ? h('p', { class: 'hint' }, selected.setup_error) : null,
    st.error && !runFirst ? h('p', { class: 'pill bad' }, st.error) : null,
    st.prep ? (st.prep.error ? h('p', { style: { color: 'var(--crit)' } }, st.prep.error) : h('pre', { class: 'doc' }, `working directory  ${st.prep.dir}\nMCP config         ${st.prep.mcp_config}\nbrief              ~${fmtInt(st.prep.brief_tokens)} tokens (BRIEF.md)\n\n$ ${st.prep.command.join(' ')}\n\nThe same directory works interactively: cd there and start your harness; .mcp.json declares the wsp server.`)) : null));

  // The run: its status line, the prompts that wait for the person, its log and, while the conversation is open,
  // the follow-up box and the Interrupt / Close buttons.
  const runBox = st.run ? runSection(st, { open, running, runFirst, action, again }) : null;
  if (runFirst) {
    if (open) {
      // Existing controls configure only a new conversation. Keep its running identity visible.
      for (const control of setup.querySelectorAll('input, select, textarea')) control.disabled = true;
      for (const control of setup.querySelectorAll('.seg button')) control.disabled = true;
      setup.append(h('p', { class: 'hint' }, 'This conversation keeps its task and workspace. Close it before changing either. Your existing CLI login is used.'));
    }
    const title = String(t.title || t.instructions || '').trim().split('\n')[0].slice(0, 120);
    const details = h('details', { class: 'agent-setup', open: !!st.setupOpen },
      h('summary', {}, (open ? 'Task, harness and sign-in of this conversation (fixed while it is open)' : 'Task, harness and sign-in') + (title ? ': ' + title : '')), setup);
    details.addEventListener('toggle', () => { st.setupOpen = details.open; });
    el.append(runBox.box, details);
  } else el.append(setup);

  // earlier conversations: read one, then Resume it (the provider's own session) or start a New conversation
  const showRun = () => { runBox?.box.scrollIntoView?.({ block: 'start' }); runBox?.heading.focus?.(); };
  conversationsView(el, app, st, { open, running, start, authState, again, showRun });

  if (!runFirst) {
    if (runBox) el.append(runBox.box);
    else el.append(h('p', { class: 'muted', style: { marginTop: '14px' } }, 'No run yet. Starting one writes BRIEF.md and .mcp.json into the workspace\'s agent/ directory and launches the harness there; its messages and tool calls stream in below, and its edits appear in the patch stack as they are committed.'));
  }
  if (runBox) {
    refreshProgrammerStatus(st);
    el.afterMount = runBox.afterMount;            // app.js renderView calls it once the view is in the document
    if (tickerOwns(app, st)) ensureTicker(app, st); else stopTicker();
  } else stopTicker();
  app.runChanged?.();
}

// The run's part of the tab. The log node is kept across renders (st.logEl) while it holds every event of this run,
// so a re-render never rebuilds hundreds of cards; its scroll position is restored once the view is mounted (a
// detached element has no scroll position).
function runSection(st, { open, running, runFirst, action, again }) {
  const run = st.run;
  const heading = h('h3', { style: { margin: 0 }, tabindex: -1 }, open ? 'Open conversation' : 'run');
  const buttons = !runFirst ? [] : [
    running ? h('button', { class: 'danger', onclick: () => action(run.native && run.status === 'starting' ? '/api/harness/close' : '/api/harness/stop', { id: run.id }) }, run.native ? (run.status === 'starting' ? 'Cancel startup' : 'Interrupt turn') : 'Stop') : null,
    open ? h('button', { onclick: () => action('/api/harness/close', { id: run.id }) }, 'Close conversation') : null];
  const box = h('section', { id: 'agent-run', class: 'agent-run' + (runFirst ? ' first' : '') },
    h('div', { class: 'row', style: { marginTop: runFirst ? '10px' : '14px' } }, heading,
      h('span', { class: 'pill ' + (running ? 'warn' : ['done', 'idle'].includes(run.status) ? 'good' : 'bad') }, run.status),
      h('span', { class: 'muted small mono' }, run.id), h('span', { class: 'muted small' }, `${st.events.length} events`), h('span', { class: 'grow' }), ...buttons));
  // One status line, always current: the sentence (a polite live region, the same node on every render; see
  // announce) and its ticking time.
  if (!st.statusBox) {
    st.statusLive = h('span', { class: 'agent-status-text', role: 'status', 'aria-live': 'polite' });
    st.statusTime = h('span', { class: 'agent-status-time muted' });
    st.statusBox = h('div', { class: 'agent-status' }, st.statusLive, st.statusTime);
  }
  // The status line and the log are shared nodes: they move into this view only once it is mounted (afterMount).
  // Taking them while the previous view is still on screen would shorten that page and make the browser clamp its
  // scroll position before the shell reads it; and a render that is never mounted leaves them where they are seen.
  const statusSlot = h('div', { class: 'agent-status-slot' });
  box.append(statusSlot);
  if (open) {
    for (const request of run.pending || []) {
      const answers = st.answers || (st.answers = {});
      const values = answers[request.request_id] || (answers[request.request_id] = {});
      box.append(h('div', { class: 'card agent-prompt', style: { marginTop: '10px' } },
        h('h3', {}, 'Waiting for your response: ' + request.title), request.input ? h('pre', { class: 'doc' }, JSON.stringify(request.input, null, 2)) : null,
        ...(request.questions || []).map(question => field(question.question,
          h('div', {}, question.options?.length ? h('p', { class: 'hint' }, question.options.map(o => o.label + (o.description ? ': ' + o.description : '')).join(' · ')) : null,
            h('input', { type: 'text', value: values[question.id] || '', style: { width: '100%' }, oninput: e => { values[question.id] = e.target.value; } })))),
        h('div', { class: 'row' },
          h('button', { class: 'primary', onclick: () => action('/api/harness/respond', { id: run.id, request_id: request.request_id, allow: true, answers: values }) }, request.questions?.length ? 'Send answers' : 'Allow once'),
          h('button', { onclick: () => action('/api/harness/respond', { id: run.id, request_id: request.request_id, allow: false }) }, 'Decline'))));
    }
  }
  if (runFirst && st.error) box.append(h('p', { class: 'pill bad', role: 'alert' }, st.error));

  const costNote = !run.native ? 'harness estimate' : run.billing === 'per_call' ? 'billed per call' : null;
  let log = st.logEl;
  if (!log || log.runId !== run.id || log.drawn !== st.events.length || log.costNote !== costNote) {
    // Rebuilt for the same run, the log keeps where the person had it; a new or resumed conversation's log (another
    // run id) starts at its end and follows its output, whatever the previous one's position was.
    const same = !!log && log.runId === run.id;
    const follow = same ? log.follow : true, scroll = same ? log.scroll : 0;
    log = h('div', { id: 'agent-log' });
    Object.assign(log, { runId: run.id, costNote, drawn: 0, toolNodes: new Map(), follow, scroll });
    for (const event of st.events) appendAgentEvent(log, event, log.toolNodes);
    st.logEl = log;
  }
  st.toolNodes = log.toolNodes;
  const logBox = h('div', { class: 'scroll card', id: 'agent-log-box', style: { maxHeight: '560px', marginTop: '6px' } });
  logBox.addEventListener('scroll', () => {
    if (logBox.isConnected === false || log.parentElement !== logBox) return;     // an earlier render's box
    log.follow = logBox.scrollHeight - logBox.scrollTop - logBox.clientHeight <= FOLLOW_PX;
    log.scroll = logBox.scrollTop;
  });
  box.append(logBox);
  if (open) {
    box.append(h('div', { class: 'card', style: { marginTop: '10px' } },
      field('Follow-up message', h('textarea', { rows: 3, value: st.message || '', placeholder: 'Continue this conversation…', oninput: e => { st.message = e.target.value; } })),
      h('button', { class: 'primary', disabled: run.status !== 'idle' || !!run.pending?.length, onclick: async () => {
        try { st.run = await harnessPost(st, '/api/harness/send', { id: run.id, text: st.message || '' }); st.message = ''; st.error = null; }
        catch (e) { st.error = e.message; } again();
      } }, 'Send message')));
  }
  // Mounted: take the shared nodes, put the log where the person left it (at its end unless they had scrolled up),
  // and write the status sentence now that its live region is in the document (announce defers it otherwise).
  const afterMount = () => {
    if (box.isConnected === false) return;
    statusSlot.append(st.statusBox);
    if (st.logEl !== log) return;                  // a newer render has its own log
    logBox.append(log);
    logBox.scrollTop = log.follow === false ? (log.scroll || 0) : logBox.scrollHeight;
    refreshProgrammerStatus(st);
  };
  return { box, heading, afterMount };
}
