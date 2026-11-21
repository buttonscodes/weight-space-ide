// The Directions tab's Search sub-view: the workspace's automatic searches (Heretic's optimisation over a method's
// parameters), a new search made and run, and the selected search's header. A search runs on the server in the
// background and never commits anything. Opening the view or selecting a search only GETs.
// Routes: docs/DIRECTIONS.md ("The automatic search", its "Routes", "Committing a trial").
import { h, api, field, fmt } from './util.js';
import { put, enc, refusalText, busyReason, runReason, renderingFields, draftChanged, draftTyped, loadList, storeDesc, searchRuns,
  runningSearches, runJob, streamId } from './directions.js';
import { scorerEditor, scorersBody, scorerSets, maxResponseLength, ensureScorers, settingsForm, settingsValues, readValue, pasteParameters,
  RESPONSE_SCORERS } from './directions_forms.js';
import { errorTarget, checkAfterRefusal } from './directions_prompts.js';
import { ARA_NEEDS, applyReport, responsesTable } from './directions_edit.js';
import { SEARCH_SLOWS } from './directions_measure.js';
import { frontChart } from './directions_charts.js';

const PLUGINS = ['Abliteration', 'ARA'];
const METHOD_OF = { Abliteration: 'ablation', ARA: 'ara' };
const METHOD_LABELS = { ablation: 'ablation', ara: 'ARA' };
const STATE_ORDER = ['complete', 'failed', 'stopped', 'interrupted', 'running'];
// The create form's own fields, as a refusal names them (13.2).
const FORM_FIELDS = ['name', 'n_trials', 'n_startup_trials', 'seed', 'max_response_length', 'system', 'prefix', 'chat_options', 'chat'];
export const COMMIT_RUNS_AGAIN = "Committing runs the trial's script again through the edit path (with the candidate tests and drift): this can take as long as the trial's own fit did, and cannot be stopped.";
export const OPTUNA_MISSING = 'The automatic search needs Optuna 5: python -m pip install -r requirements-search.txt. Existing searches can be read and their trials committed.';

const entriesOf = (st) => st.searches?.searches || [];
const entryOf = (st, name) => entriesOf(st).find((s) => s.name === name) || null;
const selectedMeta = (st) => (st.list?.directions || []).find((m) => m.name === st.selected && !m.error) || null;
const optunaOf = (st) => st.searches?.sampler?.optuna ?? null;

// "4 complete, 1 failed" (9.5.1): the trial counts by state, in the order of the states.
export function countsText(counts) {
  const c = counts || {};
  const keys = [...STATE_ORDER.filter((k) => c[k]), ...Object.keys(c).filter((k) => !STATE_ORDER.includes(k) && c[k])];
  return keys.length ? keys.map((k) => `${c[k]} ${k}`).join(', ') : 'no trials';
}
const finished = (counts) => (counts?.complete || 0) + (counts?.failed || 0);

function ensureSearch(st) {
  st.search ||= { selected: null, creating: false };
  return st.search;
}
// The search shown: the one picked, else the running one, else the last created.
function selectedName(st) {
  const s = ensureSearch(st), list = entriesOf(st);
  if (s.selected && list.some((e) => e.name === s.selected)) return s.selected;
  const run = runningSearches(st)[0];
  if (run) return run.name;
  return [...list].filter((e) => !e.error).sort((a, b) => b.created - a.created)[0]?.name ?? null;
}

// The selected search's description: read when it is missing or the weights changed since it was read (a GET).
async function ensureDesc(app, st, name) {
  if (st.searchDescs[name] && st.searchDescVersion[name] === app.state.status?.version) return;
  try { storeDesc(app, st, name, await api(`/api/directions/searches/${enc(name)}`)); st.search.descError = ''; }
  catch (e) { st.search.descError = refusalText(e); }
}

// Why no search may be started now ('' when one may): Optuna, the model, a request, a search running anywhere.
function startWhy(app, st) {
  if (!optunaOf(st)) return 'the automatic search needs Optuna 5: python -m pip install -r requirements-search.txt';
  const run = runningSearches(st)[0];
  return runReason(app, st) || busyReason(st) ||
    (run ? `search '${run.name}' is running ${run.running === 'this IDE' ? 'in this IDE' : 'in another process'}: one search runs at a time` : '');
}

// ------------------------------------------------------------------------------------------------ the searches
async function deleteSearch(app, st, entry, again) {
  const n = Object.values(entry.trials || {}).reduce((a, b) => a + b, 0);
  if (!confirm(`Delete search '${entry.name}' and its ${n} trial records?`)) return;
  st.errors.searchList = '';
  try {
    await api(`/api/directions/searches/${enc(entry.name)}`, undefined, { method: 'DELETE' });
    delete st.searchDescs[entry.name];
    if (st.search.selected === entry.name) st.search.selected = null;
  } catch (e) { st.errors.searchList = refusalText(e); }
  st.listKey = null;
  await loadList(app, st);
  again();
}

async function selectSearch(app, st, name, again) {
  st.search.selected = name;
  delete st.searchDescVersion[name];             // read again
  again();
}

function searchesTable(app, st, again, current) {
  const list = entriesOf(st);
  if (!list.length) return h('p', { class: 'hint' }, 'No searches in this workspace yet.');
  const busy = busyReason(st);
  const row = (e) => {
    const runs = e.error ? null : searchRuns(st, e.name);
    const delWhy = busy || (runs ? `search '${e.name}' is running: stop it first` : '');
    const del = h('button', { class: 'ghost danger', disabled: !!delWhy, title: delWhy || null, onclick: () => deleteSearch(app, st, e, again) }, 'delete');
    if (e.error) return h('tr', {}, h('td', {}, e.name), h('td', { colspan: 5, class: 'hint err' }, e.error), h('td', {}, del));
    const c = e.compatible || {};
    return h('tr', { class: e.name === current ? 'on' : '' },
      h('td', {}, e.name, c.run ? [' ', h('span', { class: 'pill warn', title: c.run }, 'cannot run')] : null,
        c.apply ? [' ', h('span', { class: 'pill warn', title: c.apply }, 'cannot commit')] : null),
      h('td', {}, METHOD_LABELS[e.method] || e.method),
      h('td', {}, countsText(e.trials)),
      h('td', {}, String((e.front || []).length)),
      h('td', {}, (e.objectives || []).map((o) => `${o.name} (${o.direction})`).join(', ')),
      h('td', {}, runs || ''),
      h('td', {}, h('button', { class: 'ghost', onclick: () => selectSearch(app, st, e.name, again) }, 'select'), del));
  };
  return h('table', { class: 'grid', id: 'dirs-searches' },
    h('tr', {}, ['name', 'method', 'trials', 'front', 'objectives', 'running', ''].map((t) => h('th', {}, t))),
    list.map(row));
}

// ------------------------------------------------------------------------------------------------ a new search
export function ensureForm(st) {
  return st.searchForm ||= { open: null, name: null, plugin: null, settings: {}, nTrials: null, nStartup: null, seed: '' };
}
const pluginDefault = (st) => (selectedMeta(st)?.module_io ? 'ARA' : 'Abliteration');
const pluginOf = (st) => ensureForm(st).plugin ?? pluginDefault(st);
// `<measurement>-<method>-<k>`, the smallest k whose name is free.
function nameDefault(st, plugin) {
  if (!st.selected) return '';
  const taken = new Set(entriesOf(st).map((e) => e.name));
  for (let k = 1; ; k++) if (!taken.has(`${st.selected}-${METHOD_OF[plugin]}-${k}`)) return `${st.selected}-${METHOD_OF[plugin]}-${k}`;
}
const nameOf = (st) => ensureForm(st).name ?? nameDefault(st, pluginOf(st));

// A modifier's settings as a form describes them: limits.search.settings[plugin] is {key: default}; each kind is the
// kind of the method's parameter field of the same name, else the JSON type of its default.
export function searchSettingsDesc(st, plugin) {
  const defaults = st.list?.limits?.search?.settings?.[plugin] || {};
  const fields = st.list?.parameters?.[METHOD_OF[plugin]]?.fields || [];
  return Object.fromEntries(Object.entries(defaults).map(([key, d]) => {
    const f = fields.find((x) => x.name === key);
    const kind = f ? f.kind : typeof d === 'boolean' ? 'boolean' : Number.isInteger(d) ? 'integer' : typeof d === 'number' ? 'number' : 'text';
    return [key, { kind, default: d, ...(f?.choices ? { choices: f.choices } : {}) }];
  }));
}

// The create request (6.1) -> {body, problems}: settings only where they differ from their default, an empty seed is
// null (drawn at random and recorded), an empty count is left to the server's default.
export function createBody(app, st) {
  const f = ensureForm(st), plugin = pluginOf(st), problems = [];
  const r = renderingFields(st), scorers = scorersBody(st), M = maxResponseLength(st);
  problems.push(...r.problems, ...scorers.problems, ...(M.problem ? [M.problem] : []));
  if (!st.selected) problems.push('select a measurement');
  const defaults = st.list?.limits?.search?.defaults || {};
  const count = (typed, key) => {
    const t = String(typed ?? (defaults[key] != null ? String(defaults[key]) : '')).trim();
    return t === '' ? {} : { [key]: readValue({ kind: 'integer' }, t, key, problems) };
  };
  const seedText = String(f.seed ?? '').trim();
  const seed = seedText === '' ? null : readValue({ kind: 'integer' }, seedText, 'seed', problems);
  const settings = settingsValues(searchSettingsDesc(st, plugin), f.settings[plugin] || {}, 'setting');
  problems.push(...settings.problems);
  const body = { name: nameOf(st), modifiers: [{ plugin, direction: st.selected, ...settings.settings }], scorers: scorers.scores,
    ...count(f.nTrials, 'n_trials'), ...count(f.nStartup, 'n_startup_trials'), seed, ...r.fields, max_response_length: M.value };
  return { body, problems };
}

function createWhy(app, st) {
  return startWhy(app, st) || (st.search.creating ? 'the search is being created' : '') || (st.selected ? '' : 'select a measurement first') ||
    (pluginOf(st) === 'ARA' && !selectedMeta(st)?.module_io ? ARA_NEEDS : '');
}

async function create(app, st, again) {
  if (createWhy(app, st)) return;                // a click that reaches a disabled button anyway
  const f = ensureForm(st), { body, problems } = createBody(app, st);
  if (problems.length) { st.errors.search = { text: problems.join(' '), target: null }; again(); return; }
  st.errors.search = null;
  st.search.creating = true;
  again();
  let refused = '';
  try {
    const d = await api('/api/directions/searches', { ...body, stream: streamId('search') });
    if (app.keep.dirs !== st) return;
    const name = d.search?.name ?? body.name;
    storeDesc(app, st, name, d);
    st.search.selected = name;
    f.name = null; f.open = false;
    draftChanged(app, st);
    st.listKey = null;
    await loadList(app, st);
    if (d.job) st.searchJob = d.job;
  } catch (e) {
    refused = refusalText(e);
    st.errors.search = { text: refused, target: errorTarget(refused, FORM_FIELDS) };
  } finally { st.search.creating = false; }
  again();
  if (refused) await checkAfterRefusal(app, st, refused, scorerSets(st, 'scorers'), again);
}

function searchForm(app, st, again) {
  const f = ensureForm(st), limits = st.list?.limits || {}, plugin = pluginOf(st), why = createWhy(app, st);
  const target = st.errors.search?.target?.field;
  const bad = (name) => (target === name ? 'dirs-bad' : null);
  const defaults = limits.search?.defaults || {};
  f.settings[plugin] ||= {};
  const text = (key, label, value, hint, placeholder = null) => field(label, h('input', { type: 'text', value: value ?? '', class: bad(label), 'aria-label': label, placeholder,
    oninput: (e) => { f[key] = e.target.value; draftTyped(app, st); }, onchange: () => draftChanged(app, st) }), hint);
  const noIo = !selectedMeta(st)?.module_io;
  return h('details', { id: 'dirs-search-form', open: f.open ?? !entriesOf(st).length, ontoggle: (e) => { f.open = !!e.target.open; } },
    h('summary', {}, 'New search'),
    h('p', { class: 'hint' }, "Heretic's optimisation: each trial draws the method's parameters, scores the edit as an uncommitted candidate on top of the current weights, and tells the scores to the sampler. A search never commits anything."),
    st.selected ? null : h('p', { class: 'hint' }, 'Select a measurement: the search removes its direction.'),
    h('div', { class: 'row' },
      text('name', 'name', nameOf(st), null),
      field('plugin', h('select', { 'aria-label': 'plugin', onchange: (e) => { f.plugin = e.target.value; draftChanged(app, st); again(); } },
        PLUGINS.map((p) => h('option', { value: p, selected: p === plugin, disabled: p === 'ARA' && noIo && !!st.selected, title: p === 'ARA' && noIo ? ARA_NEEDS : null }, p))),
      `the method: ${METHOD_LABELS[METHOD_OF[plugin]]}`),
      text('nTrials', 'n_trials', f.nTrials ?? (defaults.n_trials != null ? String(defaults.n_trials) : ''), `default ${defaults.n_trials ?? 'the server\'s'}`),
      text('nStartup', 'n_startup_trials', f.nStartup ?? (defaults.n_startup_trials != null ? String(defaults.n_startup_trials) : ''), 'trials drawn at random first'),
      text('seed', 'seed', f.seed, 'empty: random, recorded', 'random')),
    h('div', { class: target === 'modifiers[0]' ? 'dirs-bad' : null, id: 'dirs-search-settings' },
      h('span', { class: 'small muted' }, `${plugin}'s settings`),
      settingsForm(searchSettingsDesc(st, plugin), f.settings[plugin], ({ typed }) => (typed ? draftTyped(app, st) : draftChanged(app, st)), 'setting')),
    h('p', { class: 'hint' }, 'The scorers (each with its optimization: the objectives are those to minimize or maximize), max_response_length and the rendering of the Measure view are the search\'s.'),
    scorerEditor(app, st, again, { field: 'scorers', refusal: st.errors.search }),
    h('div', { class: 'row', id: 'dirs-search-go' },
      h('button', { class: 'primary', disabled: !!why, title: why || null, onclick: () => create(app, st, again) }, st.search.creating ? 'Creating…' : 'Create and run'),
      why && !st.search.creating ? h('span', { class: 'hint' }, why) : null),
    st.errors.search ? h('p', { class: 'hint err', role: 'alert', id: 'dirs-search-error' }, st.errors.search.text) : null);
}

// ------------------------------------------------------------------------------------------------ the selected search
// The state line of a search (13.1), from this IDE's job when it has one for it, else from the list.
export function stateLine(st, name) {
  const job = st.searchJob?.search === name ? st.searchJob : null, entry = entryOf(st, name);
  const runs = searchRuns(st, name);
  if (runs === 'another process') return `Search '${name}' is running in another process; it can be stopped only there.`;
  const counted = (j) => (j.total == null ? '' : ` ${j.done ?? 0} of ${j.total}`);
  const phase = (p) => ({ reference: "the base model's first-token passes", responses: 'responses', logits: 'first-token passes' }[p] || 'starting');
  if (job) {
    switch (job.state) {
      case 'preparing': return `Preparing: ${job.phase === 'reference' ? 'the KL reference' : 'checks and replay'}${counted(job)}`;
      case 'running': return `Running trial ${job.trial} of ${job.n_trials} (${job.trials_done} finished): ${phase(job.phase)}${counted(job)}`;
      case 'trial': return `Running: ${job.trials_done} of ${job.n_trials} trials finished.`;
      case 'finished': return `Finished: ${countsText(entry?.trials)}.`;
      case 'stopped': return `Stopped (${job.reason || 'stopped'}).`;
      case 'failed': return `The run ended: ${job.error}${job.event?.message ? ` ${job.event.message}` : ''}`;
      default: break;
    }
  }
  const s = st.searchDescs[name]?.search;
  return `Not running: ${countsText(entry?.trials)} of ${s?.n_trials ?? entry?.n_trials} trials.`;
}

// Stop the running search now (at its next stop point) or after the trial in progress: {"when": "now" | "after_trial"}.
async function stopSearch(app, st, name, when) {
  const job = st.searchJob?.search === name ? st.searchJob : null;
  st.search.stopError = '';
  st.search.stopping = when;
  st.paint.searchHead?.();
  try {
    const r = await api(`/api/directions/searches/${enc(name)}/stop`, { when });
    if (!r.found) st.search.stopError = 'this IDE does not run that search now';
    else if (job && (when === 'now' || !job.stop)) job.stop = when;          // as the server records it
  } catch (e) { st.search.stopError = e.message; }
  st.search.stopping = null;
  st.paint.searchHead?.();
}

function stopButtons(app, st, name) {
  const job = st.searchJob?.search === name ? st.searchJob : null;
  if (!job?.running) return null;
  const now = job.stop === 'now' || st.search.stopping === 'now', after = job.stop === 'after_trial' || st.search.stopping === 'after_trial';
  return h('div', { class: 'row' },
    h('button', { disabled: now, title: 'stops at the next stop point; the trial in progress is run again from its start next time', onclick: () => stopSearch(app, st, name, 'now') },
      now ? 'Stopping…' : 'Stop now'),
    h('button', { disabled: now || after, title: 'lets the trial in progress finish first', onclick: () => stopSearch(app, st, name, 'after_trial') },
      after ? 'stopping after this trial' : 'Stop after this trial'),
    st.search.stopError ? h('span', { class: 'hint err', role: 'alert' }, `Stop failed: ${st.search.stopError}. Try again.`) : null);
}

function headKids(app, st, name) {
  const d = st.searchDescs[name], s = d?.search;
  if (!s) return [h('p', { class: 'hint' }, st.search.descError || `Reading search '${name}'…`)];
  const t = d.transport;
  return [
    h('h3', {}, `${s.name} · ${METHOD_LABELS[s.method] || s.method} from '${s.modifier?.direction}' · seed ${s.seed} (${s.seed_from}) · Optuna ${s.sampler?.optuna}`),
    ...(s.notes || []).map((n) => h('p', { class: 'hint', role: 'note' }, n)),
    h('p', { role: 'status', id: 'dirs-search-state' }, stateLine(st, name)),
    stopButtons(app, st, name),
    t ? h('p', { class: 'small' }, `Carrying the candidates to the model worker took a median ${(t.median_share * 100).toFixed(1)}% of a trial (limit ${(t.limit * 100).toFixed(0)}%)`) : null];
}

// ------------------------------------------------------------------------------------------------ resume, more trials
// The rows of `compatible` that need no model: `run` alone non-null means only the library versions changed, the one
// case accept_changed_libraries continues (4.1.19).
const librariesOnly = (c) => !!c && c.run != null && c.apply == null;
function runWhy(app, st, name) {
  const c = st.searchDescs[name]?.compatible;
  if (c?.run != null && !librariesOnly(c)) return c.run;
  return startWhy(app, st) || (st.search.starting ? 'the run is being started' : '') ||
    (librariesOnly(c) && !st.search.acceptLibraries ? 'the library versions changed: tick the box to continue with them' : '');
}

async function runSearch(app, st, name, more, again) {
  if (runWhy(app, st, name)) return;             // a click that reaches a disabled button anyway
  const c = st.searchDescs[name]?.compatible;
  const body = { ...(more != null ? { n_additional_trials: more } : {}), ...(librariesOnly(c) && st.search.acceptLibraries ? { accept_changed_libraries: true } : {}),
    stream: streamId('search') };
  st.search.runError = '';
  st.search.starting = true;
  again();
  try {
    const d = await api(`/api/directions/searches/${enc(name)}/run`, body);
    if (app.keep.dirs !== st) return;
    storeDesc(app, st, name, d);
    st.search.acceptLibraries = false;
    st.listKey = null;
    await loadList(app, st);
    if (d.job) st.searchJob = d.job;
  } catch (e) { st.search.runError = refusalText(e); }
  finally { st.search.starting = false; }
  again();
}

function runControls(app, st, name, again) {
  const d = st.searchDescs[name], s = d?.search;
  if (!s || searchRuns(st, name)) return null;
  const c = d.compatible || {}, why = runWhy(app, st, name);
  const left = s.n_trials - finished(entryOf(st, name)?.trials);
  const more = String(st.search.more ?? '10');
  const blocked = c.run != null && !librariesOnly(c);
  return h('div', { id: 'dirs-search-actions' },
    blocked ? h('p', { class: 'hint err', role: 'alert' }, `This search cannot run here now: ${c.run}`) : null,
    librariesOnly(c) ? h('p', { class: 'small' }, h('span', { class: 'hint' }, c.run), ' ',
      h('label', {}, h('input', { type: 'checkbox', checked: !!st.search.acceptLibraries, 'aria-label': 'continue with the changed library versions',
        onchange: (e) => { st.search.acceptLibraries = !!e.target.checked; again(); } }), ' continue with the changed library versions (recorded as accept_changed_libraries)')) : null,
    blocked ? null : h('div', { class: 'row' },
      left > 0 ? h('button', { class: 'primary', disabled: !!why, title: why || null, onclick: () => runSearch(app, st, name, null, again) }, `Resume (${left} trial${left === 1 ? '' : 's'} left)`) : null,
      field('more trials', h('input', { type: 'text', inputmode: 'numeric', value: more, 'aria-label': 'more trials', style: { width: '70px' },
        oninput: (e) => { st.search.more = e.target.value; } })),
      h('button', { disabled: !!why, title: why || null, onclick: () => {
        const problems = [], k = readValue({ kind: 'integer' }, String(st.search.more ?? '10'), 'more trials', problems);
        if (problems.length) { st.search.runError = problems.join(' '); again(); return; }
        runSearch(app, st, name, k, again);
      } }, 'Run more'),
      why ? h('span', { class: 'hint' }, why) : null),
    st.search.runError ? h('p', { class: 'hint err', role: 'alert', id: 'dirs-search-run-error' }, st.search.runError) : null);
}

// ------------------------------------------------------------------------------------------------ the trials
// A trial picked in the table, the chart or the front list: its full record is read (a GET). A refusal, its "commit
// anyway" box and a report belong to the trial they were for.
const pickedIndex = (st, name) => (st.search.trial?.search === name ? st.search.trial.index : null);
const recordOf = (st, name, index) => st.search.records?.[`${name}#${index}`] || null;
async function pickTrial(app, st, name, index, again) {
  st.search.trial = { search: name, index };
  st.search.apply = null;
  st.search.recordError = '';
  try {
    const rec = await api(`/api/directions/searches/${enc(name)}/trials/${index}`);
    if (app.keep.dirs !== st) return;
    (st.search.records ||= {})[`${name}#${index}`] = rec;
  } catch (e) { st.search.recordError = refusalText(e); }
  again();
}

const SHOWN_TRIALS = 200;
function trialRows(app, st, name, again) {
  const d = st.searchDescs[name], s = d?.search;
  if (!s) return [];
  const names = s.scoring?.names || [], all = d.trials || [];
  let rows = all;
  if (all.length > SHOWN_TRIALS && !st.search.allTrials) {
    const recent = new Set([...all].sort((a, b) => b.index - a.index).slice(0, SHOWN_TRIALS).map((t) => t.index));
    rows = all.filter((t) => t.on_front || recent.has(t.index));
  }
  const picked = pickedIndex(st, name);
  const row = (t) => h('tr', { 'data-trial': t.index, class: [t.state === 'failed' ? 'muted' : '', t.index === picked ? 'on' : ''].filter(Boolean).join(' '),
    title: t.state === 'failed' ? recordOf(st, name, t.index)?.error || null : null, onclick: () => pickTrial(app, st, name, t.index, again) },
    h('td', {}, String(t.index)), h('td', {}, t.state), h('td', {}, t.kind),
    names.map((_, i) => h('td', {}, t.displays?.[i] ?? '')),
    h('td', {}, t.seconds == null ? '' : fmt(t.seconds)), h('td', {}, t.on_front ? '●' : ''));
  const b = baselineOf(app, st, s);
  const base = b?.fresh ? h('tr', { class: 'dirs-baseline-row' }, h('td', {}, 'Baseline'), h('td', {}, 'current weights'), h('td', {}, ''),
    names.map((_, i) => h('td', {}, b.result.scores?.[i]?.display ?? '')), h('td', {}, ''), h('td', {}, '')) : null;
  return [h('tr', {}, ['Trial', 'State', 'Kind', ...names, 'Seconds', 'Front'].map((x) => h('th', {}, x))), base, ...rows.map(row)];
}

// ------------------------------------------------------------------------------------------------ the baseline (10.6)
// The current weights scored with the search's own scoring request: page-only, fresh while the weights are those it
// was scored on.
function baselineOf(app, st, s) {
  const b = s ? st.baselines[s.id] : null;
  return b ? { ...b, fresh: b.version === app.state.status?.version } : null;
}
// The objective values of a fresh baseline of this search, or null.
function baselineValues(app, st, s) {
  const b = baselineOf(app, st, s);
  return b?.fresh ? (s.objectives || []).map((o) => b.result.scores?.[o.index]?.value) : null;
}
const baselineWhy = (app, st) => runReason(app, st) || busyReason(st);

async function scoreBaseline(app, st, name) {
  const s = st.searchDescs[name]?.search;
  if (!s || baselineWhy(app, st)) return;
  st.search.baselineError = '';
  const version = app.state.status?.version;
  await runJob(app, st, { kind: 'scores', label: 'Scoring the baseline', stateLabel: 'current weights' }, async (stream) => {
    try {
      const r = await api('/api/directions/scores', { state: 'current', ...s.scoring.request, stream });
      if (r.stopped) return `Stopped after ${r.done} of ${r.total}; nothing was scored.`;
      if (app.keep.dirs === st) st.baselines[s.id] = { version, result: r };
      return `Scored the baseline of '${name}': the current weights.`;
    } catch (e) { st.search.baselineError = refusalText(e); return ''; }
  });
}

function baselineControl(app, st, name) {
  const s = st.searchDescs[name]?.search;
  if (!s) return null;
  const b = baselineOf(app, st, s), why = baselineWhy(app, st);
  return h('div', { class: 'row', id: 'dirs-search-baseline' },
    h('button', { disabled: !!why, title: why || "the current weights scored with this search's scorers and rendering (nothing is committed)",
      onclick: () => scoreBaseline(app, st, name) }, b && !b.fresh ? 'Score the baseline again (the weights changed)' : 'Score the baseline'),
    why ? h('span', { class: 'hint' }, why) : runningSearches(st).length ? h('span', { class: 'hint' }, SEARCH_SLOWS) : null,
    st.search.baselineError ? h('span', { class: 'hint err', role: 'alert' }, st.search.baselineError) : null);
}

// ------------------------------------------------------------------------------------------------ a trial
// Why a trial cannot be committed (or its commit checked) now ('' when it can): a search never commits, and committing
// one of its trials ends a running search, so neither is sent while one runs.
function commitWhy(app, st, name, rec) {
  if (rec.state !== 'complete') return `only a complete trial can be committed; this one is ${rec.state}`;
  const edit = app.state.open.capabilities?.logical_edit;
  if (edit?.supported === false) return `edits are not supported for this checkpoint: ${edit.reason || 'no reason given'}`;
  const c = st.searchDescs[name]?.compatible;
  if (c?.apply) return c.apply;
  const run = runningSearches(st)[0];
  if (run) return `Search '${run.name}' is running ${run.running === 'this IDE' ? 'in this IDE' : 'in another process'}: committing an edit would end it.`;
  return busyReason(st);
}

// Commit a complete trial, or check its commit (dry_run), through the trial-apply route: Session.apply runs its script
// again. accept_different_edit is sent only once the person ticked the box a refusal for a different edit showed.
async function applyTrial(app, st, name, rec, dry) {
  if (commitWhy(app, st, name, rec)) return;     // a click that reaches a disabled button anyway
  const key = `${name}#${rec.index}`;
  const ap = st.search.apply?.key === key ? st.search.apply : (st.search.apply = { key, different: false, accept: false, report: null });
  const body = { dry_run: dry, enforce_scope: document.getElementById('enforce-scope')?.checked === true,
    accept_different_edit: ap.different === true && ap.accept === true, message: null };
  ap.report = null;
  await runJob(app, st, { kind: dry ? 'dry' : 'apply', label: dry ? `Checking the commit of trial ${rec.index}` : `Committing trial ${rec.index}` }, async () => {
    let r;
    try { r = await api(`/api/directions/searches/${enc(name)}/trials/${rec.index}/apply`, body); }
    catch (e) { ap.report = e; return `Not committed: ${refusalText(e)}`; }
    ap.report = r;
    if (r.ok === false) {
      if (r.edit_equals_scored === false) { ap.different = true; ap.accept = false; }        // the box appears, unticked
      return `Not committed: ${r.error}`;
    }
    if (dry) return `Checked the commit of trial ${rec.index}: nothing was committed.`;
    if (r.committed) {
      st.listKey = null;                       // the search's compatibility changes with the weights
      await app.weightsChanged();
      return `Committed trial ${rec.index} as patch ${r.patch}.`;
    }
    return r.note || 'No weights changed; nothing was committed.';
  });
}

// "Use in Edit": the trial's parameters and the search's settings (and its seed for ARA) in the Edit view's form.
function useInEdit(app, st, s, rec, again) {
  const desc = st.list?.parameters?.[s.method];
  if (!desc) return;
  const { print_loss: _, ...settings } = s.modifier?.settings || {};
  const r = pasteParameters(s.method, JSON.stringify({ ...rec.parameters, ...settings, ...(s.method === 'ara' ? { seed: s.seed } : {}) }), desc, null);
  if (r.error) { st.search.recordError = r.error; again(); return; }
  st.method = s.method;
  (st.params ||= {})[s.method] = r.values;
  if ((st.list?.directions || []).some((m) => m.name === s.modifier.direction && !m.error)) st.selected = s.modifier.direction;
  st.view = 'Edit';
  draftChanged(app, st);
  again();
}

const lineCount = (text) => String(text ?? '').split('\n').length;
function trialView(app, st, name, again) {
  const index = pickedIndex(st, name);
  if (index == null) return null;
  const rec = recordOf(st, name, index), d = st.searchDescs[name], s = d?.search;
  if (!rec || !s) return h('div', { id: 'dirs-trial', class: 'card' }, st.search.recordError ? h('p', { class: 'hint err', role: 'alert' }, st.search.recordError) : h('p', { class: 'hint' }, `Reading trial ${index}…`));
  const b = baselineOf(app, st, s), cand = rec.candidate, secs = rec.seconds, env = rec.environment;
  const key = `${name}#${rec.index}`, ap = st.search.apply?.key === key ? st.search.apply : null;
  const why = commitWhy(app, st, name, rec);
  const responses = (rec.scores || []).map((x, i) => {
    if (!RESPONSE_SCORERS.has(x.plugin) || rec.responses_omitted) return null;
    const prompts = s.scoring?.request?.scores?.[i]?.prompts || [];
    const columns = [{ label: 'this trial', rows: x.rows || [] }, ...(b?.fresh ? [{ label: 'baseline', rows: b.result.scores?.[i]?.responses || [] }] : [])];
    const all = st.search.allResponses?.[`${key}#${i}`];
    return h('div', {}, h('h4', {}, x.name), responsesTable(`dirs-trial-responses-${i}`, x.plugin, prompts, columns,
      { all, onShowAll: () => { (st.search.allResponses ||= {})[`${key}#${i}`] = true; again(); } }));
  });
  return put(h('div', { id: 'dirs-trial', class: 'card' }),
    h('h4', {}, `Trial ${rec.index} · ${rec.state} · ${rec.kind}`),
    rec.state === 'failed' && rec.error ? h('p', { class: 'hint err', role: 'alert' }, rec.error) : null,
    h('table', { class: 'grid', id: 'dirs-trial-parameters' }, h('tr', {}, h('th', {}, 'parameter'), h('th', {}, 'value')),
      Object.entries(rec.presented || {}).map(([k, v]) => h('tr', {}, h('td', {}, k), h('td', {}, String(v))))),
    rec.scores ? h('p', { id: 'dirs-trial-scores' }, rec.scores.map((x) => `${x.name}: ${x.display}`).join(', ')) : null,
    h('div', { class: 'row' }, h('b', { class: 'small' }, 'script'),
      h('button', { class: 'ghost', title: 'puts the script in the weight script box below', onclick: () => app.setConsole(rec.script) }, 'copy to the weight script bar')),
    h('pre', { class: 'dirs-output', id: 'dirs-trial-script' }, rec.script),
    cand?.output ? h('details', {}, h('summary', {}, `the fit's output (${lineCount(cand.output)} lines)`), h('pre', { class: 'dirs-output' }, cand.output)) : null,
    cand?.commit_check && cand.commit_check.ok === false ? h('p', { class: 'hint err', role: 'alert' }, `Committing this trial would be refused: ${cand.commit_check.error}`) : null,
    secs ? h('p', { class: 'small' }, `seconds: script ${fmt(secs.script)}, scoring ${fmt(secs.scoring)}, total ${fmt(secs.total)}`) : null,
    rec.attempts?.length ? h('p', { class: 'small' }, `${rec.attempts.length} earlier attempt${rec.attempts.length === 1 ? '' : 's'} (stopped or interrupted, then run again with the same draws)`) : null,
    env ? h('p', { class: 'small' }, `torch ${env.libraries?.torch}, ${env.torch_threads} threads`) : null,
    rec.responses_omitted ? h('p', { class: 'hint', role: 'note' }, `Responses were not kept: ${rec.responses_omitted}`) : null,
    ...responses,
    h('div', { class: 'row', id: 'dirs-trial-actions' },
      rec.parameters ? h('button', { title: "the trial's parameters and the search's settings in the Edit view's form (nothing is sent)", onclick: () => useInEdit(app, st, s, rec, again) }, 'Use in Edit') : null,
      rec.state === 'complete' ? [
        h('button', { disabled: !!why, title: why || 'a dry run: the script runs again with the candidate tests and drift and its pieces are compared with the scored ones; nothing is committed',
          onclick: () => applyTrial(app, st, name, rec, true) }, 'Check the commit'),
        h('button', { class: 'primary', disabled: !!why, title: why || 'commits the trial as a patch through the edit path', onclick: () => applyTrial(app, st, name, rec, false) }, `Commit trial ${rec.index}`)] : null),
    rec.state === 'complete' && why ? h('p', { class: 'hint', id: 'dirs-trial-why' }, why) : null,
    rec.state === 'complete' ? h('p', { class: 'hint' }, COMMIT_RUNS_AGAIN) : null,
    ap?.different ? h('label', { class: 'small' }, h('input', { type: 'checkbox', checked: ap.accept, 'aria-label': 'commit anyway',
      onchange: (e) => { ap.accept = !!e.target.checked; } }), ' commit anyway: its scores will not describe the committed edit (accept_different_edit, recorded in the patch)') : null,
    h('div', { id: 'dirs-trial-report' }, ap?.report ? applyReport(ap.report, { trial: true }) : null));
}
// The front (#dirs-front): the chart of 9.8 and Heretic's list of it (format_trial_title), in the front's order.
function frontKids(app, st, name, again) {
  const d = st.searchDescs[name], s = d?.search;
  if (!s) return [];
  const names = s.scoring?.names || [], byIndex = new Map((d.trials || []).map((t) => [t.index, t])), picked = pickedIndex(st, name);
  const pick = (i) => pickTrial(app, st, name, i, again);
  const front = d.front || [];
  return [h('h4', {}, 'The Pareto front'),
    frontChart(d, { baseline: baselineValues(app, st, s), onPick: pick }),
    front.length ? h('ol', { id: 'dirs-front-list' }, front.map((i) => h('li', { 'data-trial': i, class: i === picked ? 'on' : '', onclick: () => pick(i) },
      `[Trial ${String(i).padStart(3)}] ` + names.map((n, j) => `${n}: ${byIndex.get(i)?.displays?.[j] ?? ''}`).join(', '))))
      : h('p', { class: 'hint' }, 'No complete trial yet.')];
}
function trialsMore(app, st, name, again) {
  const n = st.searchDescs[name]?.trials?.length || 0;
  return n > SHOWN_TRIALS && !st.search.allTrials ? h('button', { class: 'ghost', onclick: () => { st.search.allTrials = true; again(); } }, `show all ${n} trials`) : null;
}

export async function searchView(el, app, st, again) {
  ensureSearch(st);
  ensureScorers(st);
  const sampler = st.searches?.sampler;
  const name = selectedName(st);
  if (name && !entryOf(st, name)?.error) await ensureDesc(app, st, name);
  if (app.keep.dirs !== st) return;
  // Live regions: an event repaints these (never the view, so no field loses its focus or its caret).
  const region = (id, kids) => () => document.getElementById(id)?.replaceChildren(...kids().filter(Boolean));
  st.paint.searchHead = region('dirs-search-head', () => headKids(app, st, name));
  st.paint.trials = region('dirs-trials', () => trialRows(app, st, name, again));
  st.paint.front = region('dirs-front', () => frontKids(app, st, name, again));
  put(el,
    h('div', { id: 'dirs-search-notes' },
      sampler && !sampler.optuna ? h('p', { class: 'hint', role: 'note' }, OPTUNA_MISSING) : null,
      sampler?.optuna && sampler.validated && sampler.validated !== sampler.optuna ? h('p', { class: 'hint', role: 'note' },
        `Optuna ${sampler.optuna} is installed; the draws were checked against Heretic's with ${sampler.validated}. A resumed search is refused if its replay does not reproduce its draws.`) : null,
      st.searches?.error ? h('p', { class: 'hint err', role: 'alert' }, st.searches.error) : null),
    h('h3', {}, 'Searches'),
    searchesTable(app, st, again, name),
    st.errors.searchList ? h('p', { class: 'hint err', role: 'alert', id: 'dirs-searches-error' }, st.errors.searchList) : null,
    searchForm(app, st, again),
    name && !entryOf(st, name)?.error ? h('div', { id: 'dirs-search', class: 'card' },
      put(h('div', { id: 'dirs-search-head' }), headKids(app, st, name)),
      runControls(app, st, name, again),
      baselineControl(app, st, name),
      h('table', { class: 'grid', id: 'dirs-trials' }, trialRows(app, st, name, again)),
      trialsMore(app, st, name, again),
      put(h('div', { id: 'dirs-front' }), frontKids(app, st, name, again)),
      trialView(app, st, name, again)) : null);
}
