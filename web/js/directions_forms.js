// Forms of the Directions tab generated from the descriptors the server serves (GET /api/directions `parameters` and
// `limits.scores.settings`): the method's parameter fields with their defaults, bounds and Heretic's search ranges,
// Heretic's parameters pasted in, and the scorer list shared by Compare and the search form. Values are kept as typed
// (text for number and integer fields, booleans, choices) and become numbers, booleans and nulls only when a request
// is built; the page checks no bound (the server's refusal names the field).
// Routes: docs/DIRECTIONS.md ("GET /api/directions", "The parameters", "POST /api/directions/scores").
import { h, api, field } from './util.js';
import { put, enc, refusalText, draftChanged, draftTyped } from './directions.js';
import { newSet, setEditor, setValue, setFromMeasurement } from './directions_prompts.js';

const SAFE = Number.MAX_SAFE_INTEGER;
const isObj = (v) => v != null && typeof v === 'object' && !Array.isArray(v);
// A bound a browser cannot keep exactly (above 2^53 - 1) is not shown: it would be shown wrong.
const keepable = (b) => (typeof b === 'number' && Number.isFinite(b) && Math.abs(b) <= SAFE ? b : null);
const short = (v) => String(Number(Number(v).toPrecision(6)));

// ------------------------------------------------------------------------------------------------ values as typed
const initial = (f) => (f.kind === 'boolean' ? f.default === true : f.kind === 'choice' ? f.default : f.default == null ? '' : String(f.default));
// A method's values as typed, every field and component the descriptor has filled in (missing ones get the default;
// typed ones are kept). `values` is changed in place and returned, or made when null.
export function withDefaults(desc, values = null) {
  const v = isObj(values) ? values : {};
  for (const f of desc.fields) if (!(f.name in v)) v[f.name] = initial(f);
  if (desc.components) {
    if (!isObj(v.components)) v.components = {};
    for (const c of desc.components) {
      const have = v.components[c.name];
      const comp = isObj(have) ? have : (v.components[c.name] = { on: true, fields: {} });
      if (typeof comp.on !== 'boolean') comp.on = true;
      if (!isObj(comp.fields)) comp.fields = {};
      for (const f of c.fields) if (!(f.name in comp.fields)) comp.fields[f.name] = initial(f);
    }
  }
  return v;
}
export const defaultValues = (desc) => withDefaults(desc, null);

// The names a refusal may give a field of this method (13.2): the top fields and "<component>.<field>".
export const fieldNames = (desc) => [...(desc?.fields || []).map((f) => f.name),
  ...(desc?.components || []).flatMap((c) => c.fields.map((f) => `${c.name}.${f.name}`))];

// One typed value as sent (10.1): a number, an integer, null for an empty nullable box, a boolean or a choice.
// Returns undefined and adds a problem when it cannot be sent.
export function readValue(f, v, name, problems) {
  if (f.kind === 'boolean' || f.kind === 'bool') return v === true;
  if (f.kind === 'choice') return v;
  const t = String(v ?? '').trim();
  if (t === '') {
    if (f.nullable) return null;
    problems.push(`${name}: enter a number`);
    return undefined;
  }
  const x = Number(t);
  if (!Number.isFinite(x)) { problems.push(`${name}: "${t}" is not a number`); return undefined; }
  if (f.kind === 'integer' || f.kind === 'count') {
    if (!Number.isInteger(x)) { problems.push(`${name}: must be a whole number`); return undefined; }
    if (!Number.isSafeInteger(x)) { problems.push(`${name}: at most ${SAFE} here (the largest integer a browser keeps exactly)`); return undefined; }
  }
  return x;
}

// -> {params, problems}: the request body of POST /api/directions/script for these values (params null when a
// problem stops it). Ablation: the ticked components' distributions; ARA: every field.
export function paramValues(desc, values, direction, method = desc?.components ? 'ablation' : 'ara') {
  const problems = [];
  if (!direction) problems.push('select a measurement');
  const params = { method, direction: direction || null };
  for (const f of desc.fields) params[f.name] = readValue(f, values[f.name], f.name, problems);
  if (method === 'ablation') {
    const wd = {};
    for (const c of desc.components || []) {
      const v = values.components?.[c.name];
      if (!v?.on) continue;
      wd[c.name] = Object.fromEntries(c.fields.map((f) => [f.name, readValue(f, v.fields[f.name], `${c.name}.${f.name}`, problems)]));
    }
    if (!Object.keys(wd).length) problems.push('tick at least one component');
    params.weight_distributions = wd;
  }
  return { params: problems.length ? null : params, problems };
}

// ------------------------------------------------------------------------------------------------ pasting Heretic's
// pasteParameters(method, text, desc, current) -> {values, unused, error}: a JSON object of the method's parameters
// (or a trial record: its `parameters`) as typed values, starting from `current` (or the defaults). The ablation's
// `weight_distributions` ticks the components it lists and unticks the others. Numbers become their text, null an
// empty box; every key the form has no field for is listed in `unused`.
export function pasteParameters(method, text, desc, current = null) {
  let o;
  try { o = JSON.parse(text); } catch (e) { return { values: null, unused: [], error: `not JSON: ${e.message}` }; }
  if (isObj(o) && isObj(o.parameters)) o = o.parameters;
  if (!isObj(o)) return { values: null, unused: [], error: 'not a JSON object' };
  const values = withDefaults(desc, current ? JSON.parse(JSON.stringify(current)) : null);
  const unused = [];
  const take = (f, store, v) => {
    if (f.kind === 'boolean') { if (typeof v !== 'boolean') return false; store[f.name] = v; return true; }
    if (f.kind === 'choice') { if (!(f.choices || []).includes(v)) return false; store[f.name] = v; return true; }
    if (v === null) { store[f.name] = ''; return true; }
    if (typeof v !== 'number') return false;
    store[f.name] = String(v);
    return true;
  };
  for (const [k, v] of Object.entries(o)) {
    const f = desc.fields.find((x) => x.name === k);
    if (f) { if (!take(f, values, v)) unused.push(k); continue; }
    if (method === 'ablation' && k === 'weight_distributions' && isObj(v)) {
      for (const c of desc.components || []) {
        const listed = isObj(v[c.name]);
        values.components[c.name].on = listed;
        if (!listed) continue;
        for (const [key, x] of Object.entries(v[c.name])) {
          const cf = c.fields.find((y) => y.name === key);
          if (!cf || !take(cf, values.components[c.name].fields, x)) unused.push(`weight_distributions.${c.name}.${key}`);
        }
      }
      for (const name of Object.keys(v)) if (!(desc.components || []).some((c) => c.name === name)) unused.push(`weight_distributions.${name}`);
      continue;
    }
    unused.push(k);
  }
  return { values, unused, error: null };
}

// ------------------------------------------------------------------------------------------------ inputs
// The hint of a field (9.3): what an empty box means, Heretic's search range, the bounds the server checks.
export function fieldHint(f) {
  const parts = [];
  if (f.nullable && f.null_label) parts.push(`empty: ${f.null_label}`);
  if (Array.isArray(f.heretic_range)) parts.push(`Heretic searches ${short(f.heretic_range[0])} to ${short(f.heretic_range[1])}${f.log ? ' (log scale)' : ''}`);
  const lo = keepable(f.min), hi = keepable(f.max);
  if (lo != null && f.min_exclusive) parts.push(hi != null ? `above ${lo}, at most ${hi}` : `above ${lo}`);
  else if (lo != null && hi != null) parts.push(`${lo} to ${hi}`);
  else if (lo != null) parts.push(`at least ${lo}`);
  else if (hi != null) parts.push(`at most ${hi}`);
  return parts.length ? parts.join(' · ') : null;
}

// One input for a field: a checkbox (boolean), a select (choice) or a text box (number, integer). onChange(value,
// {typed}) is called as text is typed (typed true) and when a value is changed (typed false).
export function fieldInput(desc, value, onChange, label, { disabled = false, bad = false } = {}) {
  const cls = bad ? 'dirs-bad' : null;
  if (desc.kind === 'boolean' || desc.kind === 'bool') {
    return h('input', { type: 'checkbox', checked: value === true, disabled, class: cls, 'aria-label': label,
      onchange: (e) => onChange(!!e.target.checked, { typed: false }) });
  }
  if (desc.kind === 'choice') {
    return h('select', { disabled, class: cls, 'aria-label': label, onchange: (e) => onChange(e.target.value, { typed: false }) },
      (desc.choices || []).map((c) => h('option', { value: c, selected: c === value }, c)));
  }
  return h('input', { type: 'text', inputmode: desc.kind === 'integer' || desc.kind === 'count' ? 'numeric' : 'decimal', value: value ?? '', disabled, class: cls,
    'aria-label': label, style: { width: '110px' },
    oninput: (e) => onChange(e.target.value, { typed: true }), onchange: (e) => onChange(e.target.value, { typed: false }) });
}

// The parameter form of a method (#dirs-params): its top fields, then (ablation) one fieldset per component with its
// "on" box and its fields. `values` is changed in place; onChange({typed, redraw}) follows each change (redraw: a
// box or a choice changed, which may enable other fields). bad(name): the field a refusal named.
export function paramForm(desc, values, { onChange = () => {}, bad = () => false } = {}) {
  const holds = (f) => !isObj(f.applies_when) || Object.entries(f.applies_when).every(([k, v]) => values[k] === v);
  const one = (f, store, name, off) => field(name, fieldInput(f, store[f.name], (v, { typed }) => {
    store[f.name] = v;
    onChange({ typed, redraw: !typed && (f.kind === 'boolean' || f.kind === 'choice') });
  }, name, { disabled: off || !holds(f), bad: bad(name) }), fieldHint(f));
  return h('div', { id: 'dirs-params' },
    h('div', { class: 'row' }, desc.fields.map((f) => one(f, values, f.name, false))),
    (desc.components || []).map((c) => {
      const v = values.components[c.name];
      return h('fieldset', { class: 'dirs-component' },
        h('legend', {}, h('label', { class: 'small' }, h('input', { type: 'checkbox', checked: v.on, 'aria-label': `${c.name} on`,
          onchange: (e) => { v.on = !!e.target.checked; onChange({ typed: false, redraw: true }); } }), ` ${c.name}`)),
        h('div', { class: 'row' }, c.fields.map((f) => one(f, v.fields, `${c.name}.${f.name}`, !v.on))));
    }));
}

// ------------------------------------------------------------------------------------------------ settings
// A plugin's settings: {key: {kind, default}} (kinds text64, markers, bool, count, number as limits.scores.settings
// serves them; text, boolean, integer, choice for the search form's). Typed values: text, markers one per line, booleans.
const settingDesc = (key, d) => ({ name: key, kind: d.kind, default: d.default, nullable: false, choices: d.choices ?? null });
export function settingTyped(d) {
  if (d.kind === 'markers') return (d.default || []).join('\n');
  if (d.kind === 'bool' || d.kind === 'boolean') return d.default === true;
  if (d.kind === 'choice') return d.default;
  return d.default == null ? '' : String(d.default);
}
const markers = (text) => String(text ?? '').split(/\r?\n/).filter((line) => line.trim());
const typedOf = (desc, values, key) => (key in values ? values[key] : settingTyped(desc[key]));

// -> {settings, problems}: the settings whose value differs from its default, as sent (`hide`: keys not offered).
export function settingsValues(desc, values, label, { hide = [] } = {}) {
  const settings = {}, problems = [];
  for (const [key, d] of Object.entries(desc || {})) {
    if (hide.includes(key) || !(key in values)) continue;
    const v = values[key];
    let x;
    if (d.kind === 'markers') x = markers(v);
    else if (d.kind === 'text64' || d.kind === 'text') x = String(v ?? '');
    else x = readValue(settingDesc(key, d), v, `${label} ${key}`, problems);
    if (x === undefined) continue;
    if (JSON.stringify(x) !== JSON.stringify(d.default)) settings[key] = x;
  }
  return { settings, problems };
}

// The inputs of a plugin's settings; `values` is changed in place, onChange({typed}) follows each change.
export function settingsForm(desc, values, onChange, label, { hide = [], bad = () => false } = {}) {
  return h('div', { class: 'row' }, Object.entries(desc || {}).filter(([key]) => !hide.includes(key)).map(([key, d]) => {
    const name = `${label} ${key}`, v = typedOf(desc, values, key);
    if (d.kind === 'markers') {
      return field(key, h('textarea', { rows: 3, value: v, 'aria-label': name, class: bad(key) ? 'dirs-bad' : null,
        oninput: (e) => { values[key] = e.target.value; onChange({ typed: true }); }, onchange: (e) => { values[key] = e.target.value; onChange({ typed: false }); } }),
      `one per line; Heretic's ${d.default?.length ?? 0} by default`);
    }
    const f = settingDesc(key, d.kind === 'text64' ? { ...d, kind: 'text' } : d);
    const input = f.kind === 'text'
      ? h('input', { type: 'text', value: v, 'aria-label': name, class: bad(key) ? 'dirs-bad' : null,
        oninput: (e) => { values[key] = e.target.value; onChange({ typed: true }); }, onchange: (e) => { values[key] = e.target.value; onChange({ typed: false }); } })
      : fieldInput(f, v, (x, { typed }) => { values[key] = x; onChange({ typed }); }, name, { bad: bad(key) });
    return field(key, input, d.default == null || typeof d.default === 'boolean' ? null : `default ${d.default}`);
  }));
}

// ------------------------------------------------------------------------------------------------ the scorer list
// Shared by Compare and the search form (9.7): Heretic's default scorers, each with its own prompt set.
export const DEFAULT_SCORERS = [{ plugin: 'KeywordRate', optimization: 'minimize' }, { plugin: 'KLDivergence', optimization: 'minimize' }];
export const RESPONSE_SCORERS = new Set(['KeywordRate', 'ResponseLength']);   // the tab asks for their responses
const OPTIMIZATIONS = ['minimize', 'maximize', 'none'];
const HIDDEN = ['print_responses'];                                            // the tab sets it
export const newScorer = (plugin, optimization = 'minimize') => ({ plugin, instance_name: '', optimization, settings: {}, set: newSet() });
export const ensureScorers = (st) => st.scorers || (st.scorers = DEFAULT_SCORERS.map((d) => newScorer(d.plugin, d.optimization)));

// max_response_length, shared by every request that generates (empty: the server's default) -> {value, problem}.
export function maxResponseLength(st) {
  const text = String(st.maxResponseLength ?? '').trim();
  if (text === '') return { value: st.list?.limits?.scores?.max_response_length_default ?? 100, problem: null };
  const problems = [];
  const value = readValue({ kind: 'integer' }, text, 'max_response_length', problems);
  return { value: problems.length ? null : value, problem: problems[0] || null };
}

// -> {scores, problems}: the scorer list as sent; `print_responses` for the scorers that have responses, settings
// only where they differ from their default.
export function scorersBody(st) {
  const scores = [], problems = [], all = st.list?.limits?.scores?.settings || {};
  ensureScorers(st).forEach((s, i) => {
    const v = setValue(s.set);
    problems.push(...v.problems.map((p) => `scorer ${i + 1} prompts ${p}`));
    const r = settingsValues(all[s.plugin], s.settings, `scorer ${i + 1}`, { hide: HIDDEN });
    problems.push(...r.problems);
    scores.push({ plugin: s.plugin, prompts: v.value, ...(s.instance_name ? { instance_name: s.instance_name } : {}), optimization: s.optimization,
      ...r.settings, ...(RESPONSE_SCORERS.has(s.plugin) ? { print_responses: true } : {}) });
  });
  return { scores, problems };
}
export const scorerSets = (st, field = 'scores') => Object.fromEntries(ensureScorers(st).map((s, i) => [`${field}[${i}].prompts`, s.set]));

// The scorer list's editor (#dirs-scorers) with max_response_length. opts: field ('scores', or 'scorers' for a
// search: the name refusals give the list) and refusal ({text, target} of the last request that used it).
export function scorerEditor(app, st, again, { field: list = 'scores', refusal = null } = {}) {
  const limits = st.list?.limits || {}, all = limits.scores?.settings || {};
  const plugins = limits.scores?.plugins || ['KeywordRate', 'KLDivergence', 'ResponseLength'];
  const heretic = limits.datasets?.heretic_defaults?.scorers || {};
  const scorers = ensureScorers(st);
  const changed = ({ typed }) => (typed ? draftTyped(app, st) : draftChanged(app, st));
  const fromMeasurement = async (s, side) => {
    try {
      const p = await api(`/api/directions/${enc(st.selected)}/prompts`);
      const meta = (st.list?.directions || []).find((m) => m.name === st.selected);
      s.set = setFromMeasurement(p[side], meta?.datasets?.[side]);
      s.note = '';
      draftChanged(app, st);
    } catch (e) { s.note = refusalText(e); }
    again();
  };
  const card = (s, i) => {
    const n = i + 1, side = s.plugin === 'KeywordRate' ? 'bad' : 'good';
    const marked = refusal?.target?.field === `${list}[${i}]`;
    return put(h('div', { class: 'card dirs-scorer' + (marked ? ' dirs-bad' : ''), id: `dirs-scorer-${i}` }),
      h('div', { class: 'row' }, h('b', {}, `scorer ${n}`),
        field('plugin', h('select', { 'aria-label': `scorer ${n} plugin`, onchange: (e) => { s.plugin = e.target.value; s.settings = {}; draftChanged(app, st); again(); } },
          plugins.map((p) => h('option', { value: p, selected: p === s.plugin }, p)))),
        field('instance_name', h('input', { type: 'text', value: s.instance_name, 'aria-label': `scorer ${n} instance_name`, placeholder: 'none',
          oninput: (e) => { s.instance_name = e.target.value; draftTyped(app, st); }, onchange: () => draftChanged(app, st) })),
        field('optimization', h('select', { 'aria-label': `scorer ${n} optimization`, onchange: (e) => { s.optimization = e.target.value; draftChanged(app, st); } },
          OPTIMIZATIONS.map((o) => h('option', { value: o, selected: o === s.optimization }, o)))),
        h('button', { class: 'ghost', disabled: scorers.length <= 1, title: scorers.length <= 1 ? 'a request needs at least one scorer' : null,
          onclick: () => { scorers.splice(i, 1); draftChanged(app, st); again(); } }, 'remove')),
      settingsForm(all[s.plugin], s.settings, changed, `scorer ${n}`, { hide: HIDDEN }),
      setEditor(app, st, s.set, { id: `dirs-scorer-${i}-set`, title: 'prompts', label: `scorer ${n} prompts`, name: `${list}[${i}].prompts`,
        limit: limits.prompts_per_set, heretic: heretic[s.plugin], refusal, again,
        extra: h('button', { class: 'ghost', disabled: !st.selected, title: st.selected ? `the ${side} prompts of '${st.selected}' (a GET)` : 'select a measurement first',
          onclick: () => fromMeasurement(s, side) }, `the measurement's ${side} prompts`) }),
      s.note ? h('p', { class: 'hint err', role: 'alert' }, s.note) : null);
  };
  const max = limits.scores?.scorers_per_request ?? 8;
  const dflt = limits.scores?.max_response_length_default ?? 100;
  return h('div', { id: 'dirs-scorers' },
    scorers.map(card),
    h('div', { class: 'row' },
      h('button', { class: 'ghost', disabled: scorers.length >= max, title: scorers.length >= max ? `at most ${max} scorers in a request` : null,
        onclick: () => { scorers.push(newScorer(plugins[0])); draftChanged(app, st); again(); } }, '+ scorer'),
      field('max_response_length', h('input', { type: 'text', inputmode: 'numeric', value: st.maxResponseLength ?? '', 'aria-label': 'max_response_length', placeholder: String(dflt),
        class: refusal?.target?.field === 'max_response_length' ? 'dirs-bad' : null,
        oninput: (e) => { st.maxResponseLength = e.target.value; draftTyped(app, st); }, onchange: () => draftChanged(app, st) }), `empty: ${dflt}`)));
}
