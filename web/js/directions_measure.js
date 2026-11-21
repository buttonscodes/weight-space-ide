// The Directions tab's Measure sub-view: the saved measurements, the rendering shared by every request that runs
// the model (measure, prefix detection, scores, searches), the good and bad prompt sets, and the measure form.
// Routes: docs/DIRECTIONS.md ("Routes", "The measurement").
import { h, api, field, visible, fmt } from './util.js';
import { put, enc, refusalText, fmtBytes, asChat, renderingFields, renderingDiffers, busyReason, runReason, runJob, searchesOf,
  runningSearches, otherShape, usableMeasurement, loadList, draftChanged, draftTyped, paramsFor, SUBVIEWS } from './directions.js';
import { layersChart, cosineChart } from './directions_charts.js';
import { addSteerEntry } from './steer.js';
import { newSet, setEditor, setFromMeasurement, setValue, errorTarget, checkAfterRefusal, knownCount } from './directions_prompts.js';

// The measure form's own fields, as a refusal names them (13.2).
const FORM_FIELDS = ['name', 'variant', 'winsorization_quantile', 'module_io', 'replace', 'system', 'prefix', 'chat_options'];
export const SEARCH_SLOWS = 'A search is running: this will wait for or interleave with its trials, and both get slower.';

const created = (t) => new Date(t * 1000).toISOString().slice(0, 16).replace('T', ' ');
const cut = (s, n) => (s.length > n ? s.slice(0, n) + '…' : s);

function renderingOf(m) {
  const o = m.options || {};
  if (!o.chat) return 'raw';
  return ['chat', o.system ? `system: ${cut(o.system, 40)}` : null, o.prefix ? `prefix: ${visible(o.prefix)}` : null].filter(Boolean).join(', ');
}

// ------------------------------------------------------------------------------------------------ saved measurements
async function deleteMeasurement(app, st, name, again) {
  if (!confirm(`Delete measurement '${name}'? Searches made from it can no longer run or commit.`)) return;
  st.errors.list = '';
  try {
    await api(`/api/directions/${enc(name)}`, undefined, { method: 'DELETE' });
    if (st.selected === name) st.selected = null;
  } catch (e) { st.errors.list = refusalText(e); }
  st.listKey = null;
  await loadList(app, st);
  again();
}

function listTable(app, st, again, extra) {
  const list = st.list?.directions || [];
  if (!list.length) return h('p', { class: 'hint' }, 'No saved measurements in this workspace yet.');
  const busy = busyReason(st);
  const row = (m) => {
    const held = searchesOf(st, m.name);
    const delWhy = busy || (held.length ? `search '${held[0].name}' is running from this measurement` : '');
    const del = h('button', { class: 'ghost danger', disabled: !!delWhy, title: delWhy || null, onclick: () => deleteMeasurement(app, st, m.name, again) }, 'delete');
    if (m.error) return h('tr', {}, h('td', {}, m.name), h('td', { colspan: 5, class: 'hint err' }, m.error), h('td', {}, del));
    const other = otherShape(app, m);
    return h('tr', { class: st.selected === m.name ? 'on' : '' },
      h('td', {}, m.name, other ? [' ', h('span', { class: 'pill warn', title: `${m.n_layers} layers, d_model ${m.d_model}` }, 'other model shape')] : null),
      h('td', {}, `${m.counts?.good} good / ${m.counts?.bad} bad`),
      h('td', {}, m.variant),
      h('td', {}, renderingOf(m)),
      h('td', {}, m.module_io ? fmtBytes(m.module_io.bytes) : '–'),
      h('td', {}, created(m.created)),
      h('td', {}, h('button', { class: 'ghost', disabled: !usableMeasurement(app, m), onclick: () => { st.selected = m.name; again(); } }, 'select'),
        extra ? extra(m) : null, del));
  };
  return h('div', {},
    h('table', { class: 'grid', id: 'dirs-list' },
      h('tr', {}, ['name', 'prompts', 'variant', 'rendering', 'module inputs and outputs', 'created', ''].map((t) => h('th', {}, t))),
      list.map(row)),
    st.errors.list ? h('p', { class: 'hint err', role: 'alert' }, st.errors.list) : null);
}

// ------------------------------------------------------------------------------------------------ the rendering
function renderingCard(app, st, again) {
  const r = st.rendering, info = st.chatInfo, ok = info?.available === true, on = asChat(st);
  const why = ok ? "one user turn in the model's own chat template, generation prompt included" : `unavailable: ${info?.reason || ''}`;
  const problem = h('span', { class: 'hint err', role: 'alert' }, renderingFields(st).problems.join(' '));
  const typed = (key) => (e) => { r[key] = e.target.value; draftTyped(app, st); };
  const changed = () => draftChanged(app, st);
  const meta = (st.list?.directions || []).find((m) => m.name === st.selected && !m.error);
  const bad = (f) => (Object.values(st.errors).some((r) => r?.target?.field === f) ? 'dirs-bad' : null);   // the field a refusal named
  return put(h('div', { class: 'card', id: 'dirs-rendering' }),
    h('b', {}, 'Rendering'), h('span', { class: 'hint' }, ' (shared by measuring, prefix detection, scores and searches)'),
    h('div', {}, h('label', { class: 'small', title: why },
      h('input', { type: 'checkbox', checked: on, disabled: !ok, 'aria-label': 'as a chat message', onchange: (e) => { r.chat = !!e.target.checked; changed(); again(); } }),
      ok ? ' as a chat message' : ` as a chat message (${why})`)),
    field('system_prompt', h('textarea', { rows: 2, value: r.system ?? '', disabled: !on, class: bad('system'), 'aria-label': 'system_prompt', oninput: typed('system'), onchange: changed }), 'sent as system'),
    h('div', { class: 'row' },
      field('response_prefix', h('input', { type: 'text', value: r.prefix, disabled: !on, class: bad('prefix'), 'aria-label': 'response_prefix', oninput: typed('prefix'), onchange: changed }), 'sent as prefix'),
      field('chat options (JSON object)', h('input', { type: 'text', value: r.chatOptions, disabled: !on, class: bad('chat_options'), 'aria-label': 'chat options', placeholder: '{"enable_thinking": false}',
        oninput: (e) => { r.chatOptions = e.target.value; problem.textContent = renderingFields(st).problems.join(' '); draftTyped(app, st); }, onchange: changed }), 'empty: none')),
    problem,
    meta && renderingDiffers(st, meta.options) ? h('p', { class: 'hint', role: 'note' }, `The rendering differs from the one '${meta.name}' was measured with. `,
      h('button', { class: 'ghost', onclick: () => {
        const o = meta.options;
        st.rendering = { chat: !!o.chat, system: o.system ?? '', prefix: o.prefix ?? '', chatOptions: o.chat_options ? JSON.stringify(o.chat_options) : '' };
        changed(); again();
      } }, 'Use its rendering')) : null);
}

// ------------------------------------------------------------------------------------------------ prompt sets
export function ensureSets(st) {
  st.sets ||= {};
  for (const k of ['good', 'bad', 'prefix']) st.sets[k] ||= newSet();
  return st.sets;
}

// "into the form": a measurement's stored prompts (GET .../prompts) become the good and bad sets again.
async function intoForm(app, st, m, again) {
  st.errors.list = '';
  try {
    const p = await api(`/api/directions/${enc(m.name)}/prompts`);
    st.sets.good = setFromMeasurement(p.good, m.datasets?.good);
    st.sets.bad = setFromMeasurement(p.bad, m.datasets?.bad);
    draftChanged(app, st);
  } catch (e) { st.errors.list = refusalText(e); }
  again();
}

function setsSplit(app, st, again) {
  const limits = st.list?.limits || {}, heretic = limits.datasets?.heretic_defaults || {};
  const editor = (name, title) => setEditor(app, st, st.sets[name], { id: `dirs-set-${name}`, title, label: `${name} prompts`, name,
    limit: limits.prompts_per_set, heretic: heretic[`${name}_prompts`], refusal: st.errors.measure, again, onInput: () => st.paint.estimate?.() });
  return h('div', { class: 'split' }, editor('good', 'good prompts: the model behaves as wanted'), editor('bad', 'bad prompts: the behaviour to remove'));
}

// ------------------------------------------------------------------------------------------------ the measure form
export const ensureMeasure = (st) => st.measure || (st.measure = { name: null, variant: 'base', wq: '1', moduleIo: null, replace: false });
const names = (st) => new Set((st.list?.directions || []).map((m) => m.name));
// `refusal`, else `refusal-2`, `refusal-3`, ...: the first name not in the list.
export function defaultName(st) {
  const taken = names(st);
  if (!taken.has('refusal')) return 'refusal';
  for (let k = 2; ; k++) if (!taken.has(`refusal-${k}`)) return `refusal-${k}`;
}
const measureName = (st) => (st.measure.name ?? defaultName(st));

// The bytes module inputs and outputs take (10.3), or null until every dataset part of both sets is checked.
export function moduleEstimate(app, st) {
  const c = app.state.open.config;
  const good = knownCount(st, st.sets.good), bad = knownCount(st, st.sets.bad);
  if (!c || good == null || bad == null) return null;
  const dinAttn = c.n_heads * (c.d_head ?? c.d_model / c.n_heads);
  const row = (dinAttn + c.d_model) + (c.d_mlp ?? 0) + (c.d_mlp ? c.d_model : 0);
  const bytes = 4 * (good + bad) * c.n_layers * row;
  return Number.isFinite(bytes) ? bytes : null;
}
const moduleTicked = (st, bytes) => (st.measure.moduleIo ?? (bytes != null && bytes < 2 ** 30));

// Why Measure cannot be pressed now ('' when it can).
function measureWhy(app, st) {
  const name = measureName(st);
  const held = searchesOf(st, name);
  return runReason(app, st) || busyReason(st) ||
    (names(st).has(name) && !st.measure.replace ? 'the name is taken: tick replace or choose another' : '') ||
    (names(st).has(name) && held.length ? `search '${held[0].name}' is running from '${name}': replacing it would end that search's runs` : '');
}

async function measure(app, st, again) {
  if (measureWhy(app, st)) return;               // a click that reaches a disabled button anyway
  const m = st.measure, r = renderingFields(st), good = setValue(st.sets.good), bad = setValue(st.sets.bad);
  const wq = Number(String(m.wq).trim());
  const problems = [...r.problems, ...good.problems.map((p) => `good prompts ${p}`), ...bad.problems.map((p) => `bad prompts ${p}`)];
  if (!String(m.wq).trim() || !Number.isFinite(wq)) problems.push(`winsorization_quantile: "${m.wq}" is not a number`);
  if (problems.length) { st.errors.measure = { text: problems.join(' '), target: null }; again(); return; }
  const name = measureName(st), replace = names(st).has(name) && m.replace;
  st.errors.measure = null;
  let refused = '';
  await runJob(app, st, { kind: 'measure', label: `Measuring '${name}'`, name }, async (stream) => {
    const body = { name, good: good.value, bad: bad.value, ...r.fields, variant: m.variant, winsorization_quantile: wq,
      module_io: moduleTicked(st, moduleEstimate(app, st)), ...(replace ? { replace: true } : {}), stream };
    try {
      const out = await api('/api/directions/measure', body);
      if (out.stopped) return `Stopped after ${out.done} of ${out.total} prompts; nothing was saved.`;
      st.selected = out.name;
      m.name = null; m.replace = false;
      st.listKey = null;
      await loadList(app, st);
      if (SUBVIEWS.some((v) => v.name === 'Layers')) st.view = 'Layers';
      return `Measured '${out.name}' in ${out.seconds} s.`;
    } catch (e) {
      refused = refusalText(e);
      st.errors.measure = { text: refused, target: errorTarget(refused, FORM_FIELDS) };
      return '';
    }
  });
  if (refused) await checkAfterRefusal(app, st, refused, st.sets, again);
}

function measureForm(app, st, again) {
  const m = ensureMeasure(st), limits = st.list?.limits || {};
  const bad = (f) => (st.errors.measure?.target?.field === f ? 'dirs-bad' : null);
  const changed = () => draftChanged(app, st);
  const estimate = h('span', { class: 'hint' });
  const moduleBox = h('input', { type: 'checkbox', class: bad('module_io'), 'aria-label': 'module inputs and outputs',
    onchange: (e) => { m.moduleIo = !!e.target.checked; changed(); } });
  st.paint.estimate = () => {
    const bytes = moduleEstimate(app, st);
    estimate.textContent = bytes == null ? ' unknown until the sets are checked' : ` about ${fmtBytes(bytes)}`;
    moduleBox.checked = moduleTicked(st, bytes);
  };
  st.paint.estimate();
  const go = h('div', { class: 'row', id: 'dirs-measure-go' });
  const paintGo = () => {
    const name = measureName(st), why = measureWhy(app, st);
    go.replaceChildren(...[
      names(st).has(name) ? h('label', { class: 'small' }, h('input', { type: 'checkbox', checked: m.replace, 'aria-label': 'replace',
        onchange: (e) => { m.replace = !!e.target.checked; changed(); paintGo(); } }), ` replace the saved measurement '${name}'`) : null,
      h('button', { class: 'primary', disabled: !!why, title: why || null, onclick: () => measure(app, st, again) }, 'Measure'),
      why ? h('span', { class: 'hint' }, why) : null].filter(Boolean));
  };
  paintGo();
  return put(h('div', { class: 'card', id: 'dirs-measure-form' }),
    h('b', {}, 'Measure'),
    h('div', { class: 'row' },
      field('name', h('input', { type: 'text', value: measureName(st), class: bad('name'), 'aria-label': 'measurement name',
        oninput: (e) => { m.name = e.target.value; paintGo(); draftTyped(app, st); }, onchange: changed })),
      field('variant', h('select', { 'aria-label': 'variant', class: bad('variant'), onchange: (e) => { m.variant = e.target.value; changed(); } },
        (limits.variants || ['base', 'current']).map((v) => h('option', { value: v, selected: m.variant === v }, v)))),
      field('winsorization_quantile', h('input', { type: 'number', value: m.wq, min: 0, max: 1, step: 'any', class: bad('winsorization_quantile'),
        'aria-label': 'winsorization_quantile', oninput: (e) => { m.wq = e.target.value; draftTyped(app, st); }, onchange: changed }), "1 is off; Heretic's name")),
    h('label', { class: 'small' }, moduleBox, ' module inputs and outputs (needed for ARA)', estimate),
    runningSearches(st).length ? h('p', { class: 'hint', role: 'note' }, SEARCH_SLOWS) : null,
    go,
    st.errors.measure ? h('p', { class: 'hint err', role: 'alert' }, st.errors.measure.text) : null);
}

// ------------------------------------------------------------------------------------------------ response prefix
// Heretic's response-prefix detection (POST /api/directions/prefix): it always reads its prompts as a chat message,
// with the rendering's system message and chat options. The result is page-only; "Use this prefix" puts it in the
// rendering's response_prefix.
const FOUND_BY = {
  template_closed: 'The chat template already closes the thinking block: no prefix is needed.',
  template_initializer: 'The chat template opens a thinking block: its closing tag is the prefix.',
  common_prefix: 'Every response starts with it.',
  common_prefix_block: 'Every response opens a thinking block: the prefix closes it.',
  none: 'The responses have no common start: no prefix.',
};

function detectWhy(app, st) {
  if (st.chatInfo?.available !== true) return `it needs the model's own chat template: ${st.chatInfo?.reason || 'checking…'}`;
  return runReason(app, st) || busyReason(st);
}

async function detect(app, st, again) {
  if (detectWhy(app, st)) return;                // a click that reaches a disabled button anyway
  const set = st.sets.prefix, v = setValue(set), limits = st.list?.limits || {};
  const text = String(st.maxResponseLength ?? '').trim();
  const M = text === '' ? limits.scores?.max_response_length_default ?? 100 : Number(text);
  const problems = [...v.problems];
  let options = null;
  if (st.rendering.chatOptions.trim()) {
    try { options = JSON.parse(st.rendering.chatOptions); } catch { options = undefined; }
    if (!options || typeof options !== 'object' || Array.isArray(options)) { problems.push('chat options: not a JSON object'); options = null; }
  }
  if (!Number.isInteger(M)) problems.push('max_response_length: must be a whole number');
  if (problems.length) { st.errors.prefix = { text: problems.join(' '), target: null }; again(); return; }
  st.errors.prefix = null;
  let refused = '';
  await runJob(app, st, { kind: 'prefix', label: 'Detecting the response prefix' }, async (stream) => {
    const body = { prompts: v.value, system: st.rendering.system || null, ...(options ? { chat_options: options } : {}),
      max_response_length: M, print_responses: true, stream };
    try {
      const out = await api('/api/directions/prefix', body);
      if (out.stopped) return 'Stopped; no prefix was found.';
      st.prefix = { result: out };
      return `Detected the response prefix in ${out.seconds} s.`;
    } catch (e) {
      refused = refusalText(e);
      st.errors.prefix = { text: refused, target: errorTarget(refused, ['system', 'chat_options', 'max_response_length']) };
      return '';
    }
  });
  if (refused) await checkAfterRefusal(app, st, refused, { prompts: st.sets.prefix }, again);
}

async function measurementSets(app, st, again) {
  st.errors.prefix = null;
  try {
    const p = await api(`/api/directions/${enc(st.selected)}/prompts`);
    st.sets.prefix = setFromMeasurement([...p.good, ...p.bad].slice(0, st.list?.limits?.prompts_per_set ?? 1024), null);
    draftChanged(app, st);
  } catch (e) { st.errors.prefix = { text: refusalText(e), target: null }; }
  again();
}

function prefixResult(app, st, again) {
  const r = st.prefix?.result;
  if (!r) return null;
  const steps = Object.entries(r.steps || {}).filter(([, s]) => s && typeof s.responses === 'number');
  return h('div', { id: 'dirs-prefix-result' },
    h('div', { class: 'mono' }, 'response_prefix: ', r.response_prefix === '' ? '(empty)' : JSON.stringify(r.response_prefix)),
    h('p', { class: 'small' }, FOUND_BY[r.found_by] || r.found_by),
    steps.length ? h('ul', { class: 'small' }, steps.map(([name, s]) => h('li', {}, `${name}: ${s.responses} responses`))) : null,
    r.response_prefix !== '' ? h('button', { onclick: () => { st.rendering.prefix = r.response_prefix; st.rendering.chat = true; draftChanged(app, st); again(); } }, 'Use this prefix') : null);
}

function prefixCard(app, st, again) {
  const limits = st.list?.limits || {}, why = detectWhy(app, st);
  const fromMeasurement = h('button', { class: 'ghost', disabled: !st.selected, title: st.selected ? `the good then the bad prompts of '${st.selected}' (a GET)` : 'select a measurement first',
    onclick: () => measurementSets(app, st, again) }, "the measurement's sets");
  return put(h('div', { class: 'card', id: 'dirs-prefix' }),
    h('b', {}, "response_prefix detection (Heretic's)"),
    h('p', { class: 'hint' }, 'Finds where responses start to differ (a closed thinking block, or a common start) on these prompts, read as a chat message with the rendering above.'),
    setEditor(app, st, st.sets.prefix, { id: 'dirs-set-prefix', title: 'test prompts', label: 'prefix prompts', name: 'prompts', limit: limits.prompts_per_set,
      heretic: limits.datasets?.heretic_defaults?.response_prefix_test_prompts, extra: fromMeasurement, refusal: st.errors.prefix, again }),
    h('div', { class: 'row' },
      field('max_response_length', h('input', { type: 'number', min: 1, step: 1, value: st.maxResponseLength ?? '', 'aria-label': 'max_response_length',
        placeholder: String(limits.scores?.max_response_length_default ?? 100), class: st.errors.prefix?.target?.field === 'max_response_length' ? 'dirs-bad' : null,
        oninput: (e) => { st.maxResponseLength = e.target.value; draftTyped(app, st); }, onchange: () => draftChanged(app, st) }), `empty: ${limits.scores?.max_response_length_default ?? 100}`),
      h('button', { class: 'primary', disabled: !!why, title: why || null, onclick: () => detect(app, st, again) }, 'Detect'),
      why ? h('span', { class: 'hint' }, why) : null),
    st.errors.prefix ? h('p', { class: 'hint err', role: 'alert' }, st.errors.prefix.text) : null,
    prefixResult(app, st, again));
}

// ------------------------------------------------------------------------------------------------ Layers
// The selected measurement per index (GET /api/directions/{name}, with the Edit form's orthogonalize_direction for
// `usable`): the norms and the cosine as charts and a table. A click on an index sets the Edit form's direction_index.
export async function layersView(el, app, st, again) {
  const name = st.selected;
  if (!name) { el.append(h('p', { class: 'hint' }, 'Select a measurement to see it per layer.')); return; }
  const ablation = paramsFor(st, 'ablation');
  const ortho = ablation ? ablation.orthogonalize_direction !== false : true;
  const key = `${name}|${ortho}|${app.state.status.version}`;
  if (!st.layers[key]) {
    try { st.layers[key] = { meta: await api(`/api/directions/${enc(name)}?orthogonalize_direction=${ortho}`) }; }
    catch (e) { return el.append(h('p', { class: 'hint err', role: 'alert' }, refusalText(e))); }
  }
  const stats = st.layers[key].meta.stats || [];
  const n = stats.length - 1;
  const text = ablation ? String(ablation.direction_index ?? '').trim() : '';
  const d = text === '' ? null : Number(text);
  const pick = (k) => {
    const v = paramsFor(st, 'ablation');
    if (!v) return;
    v.direction_index = String(k - 1);
    st.layersNote = `direction_index set to ${k - 1} (index ${k})`;
    draftChanged(app, st);
    again();
  };
  const num = (v) => (v == null ? '–' : fmt(v, 4));
  // steer: adds direction:<name>@<k> with dose -1 (towards the good mean) to the Run tab's steering panel, pinned to
  // the measurement's means (means_sha256) and with this view's orthogonalize_direction. Nothing is sent.
  const means = st.layers[key].meta.means_sha256;
  const steerCell = (s, k) => {
    if (k >= n) return h('td', { title: `index ${n} is after the final norm; no layer reads it` }, '–');
    const addr = `direction:${name}@${k}`;
    return h('td', {}, h('button', { class: 'ghost', disabled: !s.usable, 'aria-label': `steer along index ${k}`,
      title: s.usable ? `adds ${addr} with dose -1 to the Run tab's steering panel (-1 removes the behaviour, +1 adds it); nothing runs` : `index ${k} is not usable`,
      onclick: () => {
        addSteerEntry(app, addr, 'add', '-1', 'diff', { pin: means, ortho });
        st.layersNote = `${addr} -1 added to the Run tab's steering panel; Run, Chat and the Probe tab's Next token use it.`;
        again();
      } }, 'steer -1'));
  };
  put(el,
    h('p', { class: 'hint' }, `Index k is the residual stream entering layer k; index ${n} is after the final norm. With a direction per layer, layer l uses index l + 1.`),
    h('div', { id: 'dirs-layers-note', class: 'small', role: 'status' }, st.layersNote),
    ablation ? null : h('p', { class: 'hint' }, "This checkpoint's layers are not known: an index cannot be used in an edit here."),
    layersChart(stats, { directionIndex: Number.isFinite(d) ? d : null, onPick: ablation ? pick : null }),
    h('div', { class: 'row small' }, h('span', { class: 'dirs-key good' }, 'good mean norm'), h('span', { class: 'dirs-key bad' }, 'bad mean norm'),
      h('span', { class: 'dirs-key diff' }, 'norm of the difference'), h('span', { class: 'hint' }, `· usable ${ortho ? 'with' : 'without'} orthogonalize_direction`)),
    h('div', { class: 'small muted' }, 'cosine of the good and the bad mean'),
    cosineChart(stats),
    h('table', { class: 'grid', id: 'dirs-layers-table' },
      h('tr', {}, ['index', 'enters layer', 'good_norm', 'bad_norm', 'diff_norm', 'relative difference', 'cosine', 'usable', 'steer'].map((t) => h('th', {}, t))),
      stats.map((s, k) => {
        const mean = (s.good_norm + s.bad_norm) / 2;
        return h('tr', { class: s.usable ? '' : 'muted' }, h('td', {}, String(k)), h('td', {}, k < n ? String(k) : 'after the final norm'),
          h('td', {}, num(s.good_norm)), h('td', {}, num(s.bad_norm)), h('td', {}, num(s.diff_norm)), h('td', {}, fmt(mean ? s.diff_norm / mean : 0, 4)),
          h('td', {}, num(s.cosine)), h('td', {}, s.usable ? 'yes' : 'no'), steerCell(s, k));
      })));
}

// ------------------------------------------------------------------------------------------------ the sub-view
export function measureView(el, app, st, again) {
  ensureSets(st);
  const into = (m) => h('button', { class: 'ghost', title: 'its prompts into the good and bad sets below (a GET)', onclick: () => intoForm(app, st, m, again) }, 'into the form');
  put(el, h('h3', {}, 'Saved measurements'), listTable(app, st, again, into), renderingCard(app, st, again),
    h('h3', {}, 'Prompt sets'), setsSplit(app, st, again), measureForm(app, st, again), prefixCard(app, st, again));
}
