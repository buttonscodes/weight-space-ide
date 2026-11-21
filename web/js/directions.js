// The Directions tab: measure a behaviour direction from two prompt sets, look at it per layer, and remove it from
// the weights as a patch (Heretic's methods through the IDE's one edit path). This file is the tab's frame, shared by
// every sub-view: the measurement picker, the state notes, the requests running elsewhere (B2 `requests`), this
// page's own job and its Stop, the drafts kept in the browser, and the events. Routes: docs/DIRECTIONS.md ("Routes").
//
// Opening the tab, switching sub-views and selecting things only GET metadata; every request that runs the model or
// writes anything is sent by a click. Text from the server, a dataset, a model or the person is put into the page as
// text only (children of h() / svg(), textContent, value, title), never as markup.
import { h, api, tabs } from './util.js';
import { measureView, layersView } from './directions_measure.js';
import { editView, compareView } from './directions_edit.js';
import { searchView } from './directions_search.js';
import { outputView } from './directions_output.js';
import { withDefaults } from './directions_forms.js';

// The event types the shell (app.js listen) hands to onDirectionsEvent.
export const DIRECTION_EVENTS = new Set(['directions', 'direction_scores', 'direction_prefix', 'direction_datasets', 'direction_search']);
const TERMINAL = new Set(['measured', 'scored', 'detected', 'fetched', 'finished', 'stopped', 'failed']);
const STOPPABLE = new Set(['measure', 'prefix', 'scores', 'fetch']);    // apply and dry have no stop point
const PROGRESS = ['state', 'phase', 'done', 'total', 'seconds', 'file', 'done_bytes', 'total_bytes'];

// The sub-views, in the order of the control; each draw(el, app, st, again) builds its part of the view.
export const SUBVIEWS = [{ name: 'Measure', draw: measureView }, { name: 'Layers', draw: layersView }, { name: 'Edit', draw: editView },
  { name: 'Compare', draw: compareView }, { name: 'Search', draw: searchView }, { name: 'Output', draw: outputView }];

export const DRAFT_KEY = (name) => 'wsp.directions.v1.' + name;
export const DRAFT_LIMIT = 262144;                 // characters of the drafts' JSON kept in localStorage
const DRAFTS_TOO_LARGE = 'Drafts are larger than 256 KiB and are kept only until the page is reloaded.';

// ------------------------------------------------------------------------------------------------ small helpers
export function fmtBytes(b) {
  if (b < 1024) return `${b} bytes`;
  if (b < 2 ** 20) return (b / 1024).toFixed(1) + ' KiB';
  if (b < 2 ** 30) return (b / 2 ** 20).toFixed(1) + ' MiB';
  return (b / 2 ** 30).toFixed(2) + ' GiB';
}
// Element.append would print a null as text; optional parts are left out instead.
export const put = (el, ...kids) => { el.append(...kids.flat().filter(Boolean)); return el; };
// A refusal as shown: the server's message without its leading "<Name>Error: ".
export const refusalText = (e) => String(e?.message ?? e).replace(/^[A-Za-z]+Error: /, '');
export const enc = encodeURIComponent;
const phaseText = (p) => ({ reference: "the base model's first-token passes", responses: 'responses', logits: 'first-token passes',
  compare: 'comparing the weights (tensors)', base_responses: "the base model's responses" }[p] || 'starting');
const counted = (ev, unit = '') => (ev.total == null ? '' : ` ${ev.done ?? 0} of ${ev.total}${unit}`);

function storage() {
  try { return globalThis.window?.localStorage || null; } catch { return null; }   // throws when site data is blocked
}

// ------------------------------------------------------------------------------------------------ state
// One state object per open workspace, in app.keep (dropped when another model is opened).
function fresh(name) {
  return {
    workspace: name, view: 'Measure', selected: null,
    list: null, listKey: null, searches: null, searchDescs: {}, searchDescVersion: {}, searchJob: null, chatInfo: null, loadError: '',
    search: null, searchForm: null, baselines: {},                               // the Search view; baselines by search id
    output: null,                                                                // the Output view (directions_output.js ensureOutput)
    rendering: { chat: null, system: null, prefix: '', chatOptions: '' },      // chat null: as the model allows
    sets: null,                                                                  // {good, bad, prefix}: prompt sets
    measure: null,                                                               // the measure form
    method: 'ablation', params: null,                                            // the Edit form (paramsFor)
    layers: {}, layersNote: '',                                                  // GET /api/directions/{name} by key
    maxResponseLength: '', prefix: null,                                         // shared max_response_length; prefix result
    errors: {}, job: null, lastJob: null, foreign: {}, ownStreams: new Set(), paint: {},
    draftsTooLarge: false, draftTimer: null,
  };
}

export function dirsState(app) {
  if (app.keep.dirs) return app.keep.dirs;
  const st = app.keep.dirs = fresh(app.state.open.name);
  restoreDrafts(st);
  return st;
}

export const shown = (app, st) => app.keep.dirs === st && app.tab === 'Directions';
const rerenderIfShown = (app, st) => { if (shown(app, st)) app.rerender(); };

// As a chat message: the model has its own chat template and the box is not unticked.
export const asChat = (st) => st.chatInfo?.available === true && st.rendering.chat !== false;

// The rendering fields a request sends (10.4): `chat` always; with chat, `system`, `prefix` and `chat_options`
// (null when empty); without chat none of the three (the server refuses them).
export function renderingFields(st) {
  const r = st.rendering, problems = [];
  if (!asChat(st)) return { fields: { chat: false }, problems };
  let options = null;
  if (r.chatOptions.trim()) {
    try { options = JSON.parse(r.chatOptions); } catch { options = undefined; }
    if (!isObj(options)) { problems.push('chat options: not a JSON object'); options = null; }
  }
  return { fields: { chat: true, system: r.system ? r.system : null, prefix: r.prefix ? r.prefix : null, chat_options: options }, problems };
}
const sortKeys = (v) => (Array.isArray(v) ? v.map(sortKeys) : isObj(v) ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, sortKeys(v[k])])) : v);
const sameJSON = (a, b) => JSON.stringify(sortKeys(a ?? null)) === JSON.stringify(sortKeys(b ?? null));
// Whether the rendering differs from the one a measurement was measured with (its meta's `options`).
export function renderingDiffers(st, options) {
  if (!options) return false;
  const { fields } = renderingFields(st);
  if (!!options.chat !== fields.chat) return true;
  if (!fields.chat) return false;
  return (options.system ?? null) !== fields.system || (options.prefix ?? null) !== fields.prefix || !sameJSON(options.chat_options, fields.chat_options);
}

// ------------------------------------------------------------------------------------------------ drafts
// Forms and prompt sets are kept in app.keep.dirs and, as a convenience, in localStorage (never results, streams or
// jobs). Each row: [key, read from the state, valid(stored value), restore into the state].
const isStr = (v) => typeof v === 'string';
const isObj = (v) => v != null && typeof v === 'object' && !Array.isArray(v);
export const DRAFTS = [
  ['rendering', (st) => ({ ...st.rendering }),
    (v) => isObj(v) && (v.chat === null || typeof v.chat === 'boolean') && (v.system === null || isStr(v.system)) && isStr(v.prefix) && isStr(v.chatOptions),
    (st, v) => { st.rendering = { chat: v.chat, system: v.system, prefix: v.prefix, chatOptions: v.chatOptions }; }],
  // prompt sets: their parts, never their check results
  ['sets', (st) => (st.sets ? Object.fromEntries(Object.entries(st.sets).map(([k, s]) => [k, { parts: s.parts }])) : null),
    (v) => v === null || (isObj(v) && Object.values(v).every((s) => isObj(s) && Array.isArray(s.parts) && s.parts.every(validPart))),
    (st, v) => { if (v) st.sets = Object.fromEntries(Object.entries(v).map(([k, s]) => [k, { parts: s.parts, check: null }])); }],
  // the measure form (a ticked "replace" is not kept: it is ticked again for the name it is meant for)
  ['measure', (st) => (st.measure ? { name: st.measure.name, variant: st.measure.variant, wq: st.measure.wq, moduleIo: st.measure.moduleIo } : null),
    (v) => v === null || (isObj(v) && (v.name === null || isStr(v.name)) && isStr(v.variant) && isStr(v.wq) && (v.moduleIo === null || typeof v.moduleIo === 'boolean')),
    (st, v) => { if (v) st.measure = { name: v.name, variant: v.variant, wq: v.wq, moduleIo: v.moduleIo, replace: false }; }],
  // the Edit form: the method and each method's values as typed (10.1)
  ['method', (st) => st.method, isStr, (st, v) => { st.method = v; }],
  ['maxResponseLength', (st) => st.maxResponseLength, isStr, (st, v) => { st.maxResponseLength = v; }],
  ['params', (st) => st.params || {}, (v) => isObj(v) && Object.values(v).every(validParams), (st, v) => { st.params = v; }],
  // the scorer list (Compare, the search form): plugins, names, settings as typed and prompt sets, never results
  ['scorers', (st) => (st.scorers ? st.scorers.map((s) => ({ plugin: s.plugin, instance_name: s.instance_name, optimization: s.optimization, settings: s.settings, set: { parts: s.set.parts } })) : null),
    (v) => v === null || (Array.isArray(v) && v.every(validScorer)),
    (st, v) => { if (v) st.scorers = v.map((s) => ({ ...s, set: { parts: s.set.parts, check: null } })); }],
  // Compare's two states (its results are page-only)
  ['compare', (st) => (st.compare ? { left: st.compare.left, right: st.compare.right } : null),
    (v) => v === null || (isObj(v) && isStr(v.left) && isStr(v.right)),
    (st, v) => { if (v) st.compare = { left: v.left, right: v.right, results: null, showAll: {} }; }],
  // the New search form as typed (null: the default shown); its settings per modifier
  ['searchForm', (st) => (st.searchForm ? { name: st.searchForm.name, plugin: st.searchForm.plugin, settings: st.searchForm.settings,
    nTrials: st.searchForm.nTrials, nStartup: st.searchForm.nStartup, seed: st.searchForm.seed } : null),
  (v) => v === null || (isObj(v) && [v.name, v.plugin, v.nTrials, v.nStartup].every((x) => x === null || isStr(x)) && isStr(v.seed) &&
    isObj(v.settings) && Object.values(v.settings).every((s) => isObj(s) && Object.values(s).every(isValue))),
  (st, v) => { if (v) st.searchForm = { open: null, name: v.name, plugin: v.plugin, settings: v.settings, nTrials: v.nTrials, nStartup: v.nStartup, seed: v.seed }; }],
];
const validScorer = (s) => isObj(s) && isStr(s.plugin) && isStr(s.instance_name) && isStr(s.optimization) && isObj(s.settings) &&
  Object.values(s.settings).every(isValue) && isObj(s.set) && Array.isArray(s.set.parts) && s.set.parts.every(validPart);
const isValue = (v) => isStr(v) || typeof v === 'boolean';
function validParams(p) {
  if (!isObj(p)) return false;
  return Object.entries(p).every(([k, v]) => (k === 'components'
    ? isObj(v) && Object.values(v).every((c) => isObj(c) && typeof c.on === 'boolean' && isObj(c.fields) && Object.values(c.fields).every(isValue))
    : isValue(v)));
}

// The Edit form's values of a method, initialised from the served descriptors (GET /api/directions `parameters`):
// number and integer fields as text (empty for null), booleans, choices; the ablation's components {on, fields}.
// Values restored from drafts keep what was typed; a field or component they lack gets its default.
export function paramsFor(st, method) {
  st.params ||= {};
  const desc = st.list?.parameters?.[method];
  if (desc) st.params[method] = withDefaults(desc, st.params[method]);
  return st.params[method] || null;
}
function validPart(p) {
  if (!isObj(p)) return false;
  if (p.kind === 'texts') return isStr(p.text);
  return p.kind === 'dataset' && isObj(p.spec) && Object.values(p.spec).every(isStr);
}

function restoreDrafts(st) {
  try {
    const raw = storage()?.getItem(DRAFT_KEY(st.workspace));
    if (!raw) return;
    const o = JSON.parse(raw);
    if (!isObj(o) || o.version !== 1) return;
    const rows = DRAFTS.filter(([key]) => key in o);
    if (!rows.every(([key, , valid]) => valid(o[key]))) return;              // any shape error: none of it is used
    for (const [key, , , restore] of rows) restore(st, o[key]);
  } catch { /* unreadable drafts are ignored */ }
}

export function saveDrafts(app, st) {
  clearTimeout(st.draftTimer);
  const o = { version: 1 };
  for (const [key, read] of DRAFTS) o[key] = read(st);
  const text = JSON.stringify(o);
  const was = st.draftsTooLarge;
  st.draftsTooLarge = text.length > DRAFT_LIMIT;
  if (!st.draftsTooLarge) {
    try { storage()?.setItem(DRAFT_KEY(st.workspace), text); } catch { /* full or blocked: kept in the page only */ }
  }
  if (was !== st.draftsTooLarge) paintNotes(app, st);
}
// On every `change` event at once; on `input`, 400 ms after the last keystroke.
export const draftChanged = (app, st) => saveDrafts(app, st);
export function draftTyped(app, st) {
  clearTimeout(st.draftTimer);
  st.draftTimer = setTimeout(() => saveDrafts(app, st), 400);
}

// ------------------------------------------------------------------------------------------------ loading (GETs only)
// A card's key is its request's stream; a request sent without one is keyed by its place in the list, so two such
// requests are two cards.
const mine = (st, ev) => !!ev.stream && (ev.stream === st.job?.stream || st.ownStreams.has(ev.stream));
function foreignFrom(st, list) {
  st.foreign = {};
  (list.requests || []).forEach((ev, i) => { if (!mine(st, ev)) st.foreign[ev.stream || `${ev.type}#${i}`] = ev; });
}

// The requests running elsewhere, read again (GET /api/directions): when an event without a stream cannot be told
// apart from another one, the list says which are still running. One read at a time; events meanwhile ask for one more.
async function reloadRequests(app, st) {
  if (st.requestsReading) { st.requestsAgain = true; return; }
  st.requestsReading = true;
  try {
    do {
      st.requestsAgain = false;
      const list = await api('/api/directions');
      if (app.keep.dirs !== st) return;
      const before = Object.keys(st.foreign).join('\n');
      st.list = list;
      foreignFrom(st, list);
      if (before !== Object.keys(st.foreign).join('\n')) rerenderIfShown(app, st); else paintForeign(app, st);
    } while (st.requestsAgain);
  } catch { /* the cards stay as they are */ } finally { st.requestsReading = false; }
}

export async function loadList(app, st) {
  const key = app.state.open.name;
  try {
    const list = await api('/api/directions');
    let searches = null;
    try { searches = await api('/api/directions/searches'); } catch (e) { searches = { searches: [], error: e.message }; }
    if (app.keep.dirs !== st) return;
    st.list = list; st.searches = searches; st.listKey = key; st.loadError = '';
    st.searchJob = searches.job || null;           // this IDE's search job of this workspace (running or the last one)
    foreignFrom(st, list);
    if (st.rendering.system === null) st.rendering.system = list.limits?.datasets?.heretic_defaults?.system_prompt ?? '';
    if (st.selected && !(list.directions || []).some((m) => m.name === st.selected && !m.error)) st.selected = null;
    // A running search's measurement (metadata) is needed to know which measurement it holds.
    await Promise.all(runningSearches(st).map(async (s) => {
      try { storeDesc(app, st, s.name, await api(`/api/directions/searches/${enc(s.name)}`)); } catch { /* left unknown */ }
    }));
  } catch (e) {
    if (app.keep.dirs === st) st.loadError = e.message;
  }
}

async function loadChat(app, st) {
  const key = app.state.open.name;
  try {
    const r = await api('/api/probe/chat');
    st.chatInfo = { key, available: r?.available === true, reason: r?.reason || '' };
  } catch (e) { st.chatInfo = { key, available: false, reason: e.message }; }
}

async function load(app, st) {
  const key = app.state.open.name, work = [];
  if (st.listKey !== key) work.push(loadList(app, st));
  if (st.chatInfo?.key !== key) work.push(loadChat(app, st));
  await Promise.all(work);
}

// ------------------------------------------------------------------------------------------------ states and reasons
// Where a search runs now: this IDE's job says so first (the list may not show a search that has just started, nor
// one that has just ended); otherwise the list or the description ('this IDE', 'another process' or null).
export function searchRuns(st, name) {
  const job = st.searchJob?.search === name ? st.searchJob : null;
  const desc = st.searchDescs[name];
  const listed = (desc ? desc.running : (st.searches?.searches || []).find((s) => s.name === name)?.running) ?? null;
  if (!job) return listed;
  if (job.running) return 'this IDE';
  return listed === 'this IDE' ? null : listed;
}
export const runningSearches = (st) => (st.searches?.searches || []).filter((s) => !s.error)
  .map((s) => ({ ...s, running: searchRuns(st, s.name) })).filter((s) => s.running);
// A search's description (GET /api/directions/searches/{name}), with the workspace version it was read at; its job,
// when it has one, is this IDE's job as the server has it now.
export function storeDesc(app, st, name, d) {
  st.searchDescs[name] = d;
  st.searchDescVersion[name] = app.state.status?.version;
  if (d?.job) st.searchJob = d.job;
}
// The measurement a running search was made from, when its description is known.
export const searchesOf = (st, measurement) => runningSearches(st).filter((s) => st.searchDescs[s.name]?.search?.modifier?.direction === measurement);

export function notes(app, st) {
  const s = app.state.status || {}, o = app.state.open, out = [];
  if (s.can_run === false) {
    out.push(`Measuring, scoring, prefix detection and searching run the model, and this model cannot be run here: ${s.run_error || o.capabilities?.generation?.reason || 'no reason given'}. Saved measurements, the per-layer view, plans and applying (untested) still work.`);
  } else if (s.runner_loaded === false) out.push('The model is not loaded: measuring, scoring, prefix detection and searching load it first.');
  if (st.list && st.list.parameters === null) out.push("This checkpoint's layers are not known, so directions cannot be measured or applied here.");
  if (o.kind === 'gguf') out.push('GGUF: an edit is stored as FP32 low-rank patches on the quantized base, which never changes and is never dequantized.');
  if (o.capabilities?.logical_edit?.supported === false) out.push(`Edits are not supported for this checkpoint: ${o.capabilities.logical_edit.reason || 'no reason given'}.`);
  if (st.chatInfo && st.chatInfo.available === false) out.push(`As a chat message: unavailable (${st.chatInfo.reason}).`);
  for (const r of runningSearches(st)) out.push(`Search '${r.name}' is running ${r.running === 'this IDE' ? 'in this IDE' : 'in another process'}: committing an edit would end it.`);
  if (st.draftsTooLarge) out.push(DRAFTS_TOO_LARGE);
  return out;
}

// Why an action that runs the model cannot be sent here ('' when it can).
export function runReason(app, st) {
  const s = app.state.status || {};
  if (s.can_run === false) return 'this model cannot be run here' + (s.run_error ? `: ${s.run_error}` : '');
  if (st.list && st.list.parameters === null) return "this checkpoint's layers are not known";
  return '';
}

// Why no request may be sent now ('' when one may): this page's job runs, or a request started elsewhere does.
// Under a supervised runtime the server runs one non-GET request at a time, so a second one would only wait.
export function busyReason(st) {
  if (st.job) return `wait for this page's job to end, or stop it (${st.job.label})`;
  if (Object.keys(st.foreign).length) return 'a direction request started elsewhere is running: wait for it to end, or stop it';
  return '';
}

// ------------------------------------------------------------------------------------------------ the job slot
// One request job at a time per page. fn(stream, job) sends the request and returns the job's outcome text (shown
// on the job line until the next job starts). The job ends when fn settles, not on an event.
// A request's stream id: unique per request, at most 64 characters. crypto.randomUUID exists only in a secure context
// (https, or http on localhost): a page served over plain http on another address makes one as run.js does.
export const streamId = (kind) => `dirs-${kind}-${globalThis.crypto?.randomUUID?.() || 'r' + Math.random().toString(36).slice(2) + Date.now().toString(36)}`;
// verb: what a 'scores' job says it does to its stateLabel ("Scoring" unless given: a check scores nothing).
export async function runJob(app, st, { kind, label, steps = 1, name = null, stateLabel = null, verb = null }, fn) {
  if (st.job || busyReason(st)) return null;
  const stream = STOPPABLE.has(kind) ? streamId(kind) : null;
  const wasLoaded = app.state.status?.runner_loaded;
  const job = st.job = { kind, stream, label, name, stateLabel, verb, state: null, phase: null, done: null, total: null, seconds: null,
    file: null, done_bytes: null, total_bytes: null, stopping: false, stopError: '', step: 1, steps };
  if (stream) st.ownStreams.add(stream);
  st.lastJob = null;
  paintJob(app, st);
  rerenderIfShown(app, st);
  let outcome = '';
  try { outcome = (await fn(stream, job)) || ''; }
  catch (e) { outcome = refusalText(e); console.error(e); }
  finally {
    if (st.job === job) st.job = null;
    st.lastJob = { kind, text: outcome };
    paintJob(app, st);
    rerenderIfShown(app, st);
  }
  // The job may have loaded the model: once the shell has the new status, the notes (and the view) say so.
  if (wasLoaded === false && app.keep.dirs === st) {
    try { await app.refreshStatus(); } catch { /* the status stays as it was */ }
    if (app.keep.dirs === st) { paintNotes(app, st); rerenderIfShown(app, st); }
  }
  return outcome;
}

export function jobText(j) {
  const started = j.state != null;
  switch (j.kind) {
    case 'measure': return started ? `Measuring '${j.name}': ${j.done ?? 0} of ${j.total} prompts (${j.seconds} s)` : `Measuring '${j.name}': starting; this loads the model if it is not loaded.`;
    case 'prefix': return started && j.total != null ? `Detecting the response prefix: ${j.done ?? 0} of at most ${j.total} responses` : 'Detecting the response prefix: starting; this loads the model if it is not loaded.';
    case 'scores': return `${j.verb || 'Scoring'} the ${j.stateLabel || 'weights'}${j.steps > 1 ? ` (${j.step} of ${j.steps})` : ''}: ${phaseText(j.phase)}${counted(j)}`;
    case 'fetch': return started ? `Downloading ${j.file || 'the file list'}: ${fmtBytes(j.done_bytes ?? 0)} of ${fmtBytes(j.total_bytes ?? 0)}` : 'Downloading: starting.';
    case 'dry': return 'Dry run: the script runs on the current weights, with the candidate tests and drift; this cannot be stopped.';
    case 'apply': return 'Committing: the script runs again through the edit path; this cannot be stopped.';
    default: return j.label;
  }
}

async function stopJob(app, st, job) {
  job.stopping = true; job.stopError = '';
  paintJob(app, st);
  try { await api('/api/generation/stop', { stream: job.stream }); }
  catch (e) { job.stopping = false; job.stopError = e.message; }
  paintJob(app, st);
}

function jobKids(app, st) {
  const j = st.job;
  if (!j) return st.lastJob?.text ? [h('span', { role: 'status' }, st.lastJob.text)] : [];
  return [h('span', { role: 'status' }, jobText(j)),
    j.stream ? h('button', { disabled: j.stopping, onclick: () => stopJob(app, st, j) }, j.stopping ? 'Stopping…' : 'Stop') : null,
    j.stopError ? h('span', { role: 'alert', class: 'hint err' }, ` Stop failed: ${j.stopError}. Try again.`) : null].filter(Boolean);
}
// Live regions look their container up in the page: a missing one is a region not shown, and nothing is painted.
export function paintJob(app, st) { document.getElementById('dirs-job')?.replaceChildren(...jobKids(app, st)); }

// ------------------------------------------------------------------------------------------------ requests started elsewhere
export function foreignText(ev) {
  switch (ev.type) {
    case 'directions': return `A measurement '${ev.name}' is running (not started by this page):${counted(ev, ' prompts')}.`;
    case 'direction_scores': return `A scoring is running (not started by this page): ${phaseText(ev.phase)}${counted(ev)}.`;
    case 'direction_prefix': return `A response-prefix detection is running (not started by this page):${ev.total == null ? ' starting' : ` ${ev.done ?? 0} of at most ${ev.total} responses`}.`;
    case 'direction_datasets': return `A dataset download is running (not started by this page): ${ev.file || 'the file list'}, ${fmtBytes(ev.done_bytes ?? 0)} of ${fmtBytes(ev.total_bytes ?? 0)}.`;
    default: return `A ${ev.type} request is running (not started by this page).`;
  }
}

function foreignCards(app, st) {
  return Object.values(st.foreign).map((ev) => {
    const note = h('span', { class: 'hint', role: 'status' });
    const stop = ev.stream ? h('button', { onclick: async () => {
      stop.disabled = true; stop.textContent = 'Stopping…'; note.textContent = '';
      try { await api('/api/generation/stop', { stream: ev.stream }); }
      catch (e) { stop.disabled = false; stop.textContent = 'Stop it'; note.textContent = ` Stop failed: ${e.message}. Try again.`; }
    } }, 'Stop it') : null;
    return put(h('div', { class: 'card dirs-foreign' }), h('span', {}, foreignText(ev)), ' ', stop, note);
  });
}
export function paintForeign(app, st) { document.getElementById('dirs-foreign')?.replaceChildren(...foreignCards(app, st)); }

function noteKids(app, st) { return notes(app, st).map((t) => h('p', { class: 'hint', role: 'note' }, t)); }
export function paintNotes(app, st) { document.getElementById('dirs-notes')?.replaceChildren(...noteKids(app, st)); }

// ------------------------------------------------------------------------------------------------ events
// Live updates repaint live regions only; a whole re-render happens when a job ends or the set of requests changes.
export function onDirectionsEvent(ev, app) {
  const st = app.keep.dirs;
  if (!st || !app.state?.open || ev.workspace !== app.state.open.name) return;
  if (ev.type === 'direction_search') { searchEvent(ev, app, st); return; }
  if (st.job && ev.stream && ev.stream === st.job.stream) {
    for (const k of PROGRESS) if (k in ev) st.job[k] = ev[k];
    paintJob(app, st);
    return;
  }
  if (ev.stream && st.ownStreams.has(ev.stream)) return;                   // a late event of a job that has ended
  let key = ev.stream;
  if (!key) {                                    // no stream: the one card of its kind sent without one, if it is one
    const same = Object.keys(st.foreign).filter((k) => !st.foreign[k].stream && st.foreign[k].type === ev.type && (st.foreign[k].name ?? null) === (ev.name ?? null));
    if (same.length > 1) { reloadRequests(app, st); return; }
    key = same[0] || `${ev.type}#event`;
  }
  const had = key in st.foreign;
  if (TERMINAL.has(ev.state)) delete st.foreign[key]; else st.foreign[key] = ev;
  if (ev.type === 'directions' && ev.state === 'measured') {
    st.listKey = null;
    loadList(app, st).then(() => rerenderIfShown(app, st));
  } else if (had !== (key in st.foreign)) rerenderIfShown(app, st);
  else paintForeign(app, st);
}

// A search's events (whatever their stream): the job is mirrored as the server mirrors it; its progress repaints the
// search's header, each trial its header, trials and front (the description is read again), and its end, or a run
// this page did not know of, reads both lists again and redraws the view. A trial never redraws the view.
const SEARCH_MIRRORED = ['trial', 'trials_done', 'n_trials', 'phase', 'done', 'total', 'reason', 'error', 'event'];
function searchEvent(ev, app, st) {
  const before = st.searchJob?.search === ev.search ? st.searchJob : null;
  const job = st.searchJob = { ...(before || { workspace: ev.workspace, search: ev.search, stop: null }), state: ev.state, running: !TERMINAL.has(ev.state) };
  for (const k of SEARCH_MIRRORED) if (k in ev) job[k] = ev[k];
  const desc = () => api(`/api/directions/searches/${enc(ev.search)}`).then((d) => { if (app.keep.dirs === st) storeDesc(app, st, ev.search, d); }, () => {});
  if (TERMINAL.has(ev.state) || !before?.running) {
    st.listKey = null;
    Promise.all([desc(), loadList(app, st)]).then(() => rerenderIfShown(app, st));
  } else if (ev.state === 'trial') desc().then(() => { for (const k of ['searchHead', 'trials', 'front']) st.paint[k]?.(); });
  else st.paint.searchHead?.();
}

// ------------------------------------------------------------------------------------------------ the Patches pane
// The line app.js renderPatches adds under a patch committed from a search trial or as the re-run of another patch's
// record (its source's origin), or null.
export function patchOrigin(p) {
  const o = p?.source?.origin;
  if (o?.kind === 'reproduction') return h('div', { class: 'small dirs-origin' }, `re-run of patch ${o.of}: ${o.verdict ?? '–'}`);
  if (!o || o.kind !== 'direction_search') return null;
  return h('div', { class: 'small dirs-origin' }, `from search ${o.search}, trial ${o.trial}: ${(o.displays || []).join(', ')}`,
    o.edit_equals_scored === false ? [' · ', h('span', { class: 'pill warn' }, 'the committed edit differs from the scored one')] : null);
}

// The shell read the patches (app.js renderPatches keeps them in app.keep.patchList): the Output view, which shows
// that list and reads none of its own, is redrawn when it is the one shown.
export function onPatchesLoaded(app) {
  const st = app.keep.dirs;
  if (st && st.view === 'Output') rerenderIfShown(app, st);
}

// ------------------------------------------------------------------------------------------------ the view
const otherShape = (app, m) => {
  const c = app.state.open.config;
  return !!c && !m.error && (m.n_layers !== c.n_layers || m.d_model !== c.d_model);
};
export const usableMeasurement = (app, m) => !m.error && !otherShape(app, m);
export { otherShape };

function picker(app, st, again) {
  const list = st.list?.directions || [];
  return h('label', { class: 'row small' }, 'measurement ',
    h('select', { 'aria-label': 'measurement', onchange: (e) => { st.selected = e.target.value || null; again(); } },
      h('option', { value: '', selected: !st.selected }, '(none)'),
      list.map((m) => h('option', { value: m.name, selected: st.selected === m.name, disabled: !usableMeasurement(app, m) },
        m.name + (m.error ? ' (damaged)' : otherShape(app, m) ? ' (other model shape)' : '')))));
}

export async function directionsView(el, app) {
  const st = dirsState(app);
  const again = () => rerenderIfShown(app, st);
  await load(app, st);
  if (app.keep.dirs !== st) return;
  const names = SUBVIEWS.map((v) => v.name);
  if (!names.includes(st.view)) st.view = names[0];
  el.append(h('h2', {}, 'Directions'),
    tabs(names, st.view, (v) => { st.view = v; again(); }, { label: 'Directions view' }),
    h('p', { class: 'muted' }, "Measure a behaviour from two prompt sets, look at it per layer, and remove it as a patch: Heretic's methods through the IDE's edit path."));
  if (st.loadError) { el.append(h('p', { class: 'hint err', role: 'alert' }, st.loadError)); return; }
  el.append(picker(app, st, again),
    h('div', { id: 'dirs-notes' }, noteKids(app, st)),
    h('div', { id: 'dirs-foreign' }, foreignCards(app, st)),
    h('div', { id: 'dirs-job', class: 'row' }, jobKids(app, st)));
  const sub = h('div', { id: 'dirs-sub' });
  await SUBVIEWS.find((v) => v.name === st.view).draw(sub, app, st, again);
  el.append(sub);
}
