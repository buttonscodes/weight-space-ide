// Steering for Run and Chat: the panel, the recipe parser, the request fields and the labels.
// Steering pushes the edited model's activations while it generates. The base model is never
// steered and no weight changes. Contract and units: docs/STEERING.md.
import { h, api, fmt, field } from './util.js';

export const MAX_DOSE = 1e6;
const fmtInt = (n) => Number(n).toLocaleString('en-US');   // exact counts: a bound is not rounded
export const MAX_NEURONS = 131072;                       // = wsp.steering.MAX_NEURONS; /api/steer/groups says it too
const NUM = String.raw`(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?`;
const MINUS = /−/g;                                      // the typographic minus sign (U+2212), as in pasted notes
const DASH = /[‐‑‒–—―]/;                                 // hyphens and en/em dashes: a range in notes, never a minus
export const MAX_RECIPE = 20000;                         // characters of a pasted recipe read at once

// Later additions to the panel (a picker for saved selections) register here: fn(app, addEntry) -> node or null.
export const panelPickers = [];

// ------------------------------------------------------------------------------------------------ targets
// -> { addr, kind } with kind neuron | range | mlp | head | group | direction, or { error }.
// A direction is a saved measurement's index: direction:<measurement>@<k> adds a vector to the residual stream
// entering layer k (docs/STEERING.md).
const TARGET = new RegExp(String.raw`^(?:L(\d+)\s*([NH])\s*(\d+)|L(\d+)\s*(?:\.\s*)?mlp|blocks\.(\d+)\.mlp(?:\.neuron\.(\d+)|\.neurons\.(\d+)-(\d+))?|blocks\.(\d+)\.attn\.head\.(\d+)|group:([\w.\-]+)|direction:([A-Za-z0-9][A-Za-z0-9._-]{0,63})@(0|[1-9][0-9]*))`, 'i');
const TARGET_HELP = 'write a neuron L34N3838, a head L11H3, an MLP blocks.12.mlp or a group group:name, a direction direction:name@12';

function matchTarget(text) {
  const m = TARGET.exec(text);
  if (!m) return null;
  let t;
  if (m[1] != null) t = m[2].toUpperCase() === 'N' ? { addr: `L${+m[1]}N${+m[3]}`, kind: 'neuron' } : { addr: `L${+m[1]}H${+m[3]}`, kind: 'head' };
  else if (m[4] != null) t = { addr: `blocks.${+m[4]}.mlp`, kind: 'mlp' };
  else if (m[5] != null) {
    if (m[6] != null) t = { addr: `L${+m[5]}N${+m[6]}`, kind: 'neuron' };
    else if (m[7] != null) t = { addr: `blocks.${+m[5]}.mlp.neurons.${+m[7]}-${+m[8]}`, kind: 'range' };
    else t = { addr: `blocks.${+m[5]}.mlp`, kind: 'mlp' };
  } else if (m[9] != null) t = { addr: `L${+m[9]}H${+m[10]}`, kind: 'head' };
  else if (m[11] != null) t = { addr: `group:${m[11]}`, kind: 'group' };
  else t = { addr: `direction:${m[12]}@${m[13]}`, kind: 'direction' };
  return { ...t, length: m[0].length };
}

// A direction target's measurement name and index, or null.
const directionOf = (addr) => {
  const m = /^direction:(.+)@(\d+)$/.exec(String(addr ?? ''));
  return m ? { name: m[1], index: m[2] } : null;
};

export function parseTarget(text) {
  const s = String(text ?? '').trim().replace(/:$/, '').trim();
  if (!s) return { error: 'no target: ' + TARGET_HELP };
  const g = /^group:(.+)$/i.exec(s);                     // a saved selection's name may hold spaces (as saved)
  if (g) return { addr: `group:${g[1]}`, kind: 'group' };
  const t = matchTarget(s);
  if (!t || t.length !== s.length) return { error: `unknown target "${s}": ${TARGET_HELP}` };
  return { addr: t.addr, kind: t.kind };
}

// Heads and whole MLPs can only be multiplied: the backend adds to individual neurons and along directions only.
export const addAllowed = (kind) => kind === 'neuron' || kind === 'range' || kind === 'group' || kind === 'direction';

// ------------------------------------------------------------------------------------------------ doses
function parseNumber(text) {
  const s = String(text ?? '').trim().replace(MINUS, '-');
  if (!new RegExp(`^[+-]?${NUM}$`, 'i').test(s)) return null;
  return Number(s);
}

function doseProblem(n) {
  if (n == null || !Number.isFinite(n)) return 'the dose must be a number';
  if (Math.abs(n) > MAX_DOSE) return `the dose ${n} is larger than ${MAX_DOSE.toExponential(0).replace('e+', 'e')}`;
  return '';
}

// One panel entry ({ addr, mode, dose (text), unit }) -> '' or why the backend would refuse it.
export function entryProblem(e) {
  const t = parseTarget(e.addr);
  if (t.error) return t.error;
  if (e.mode !== 'add' && e.mode !== 'mul') return 'choose add or multiply';
  if (t.kind === 'direction' && e.mode !== 'add') return 'a direction is added: choose add';
  if (e.mode === 'add' && !addAllowed(t.kind)) return `add pushes neurons only; multiply ${t.kind === 'head' ? 'a head' : 'a whole MLP'} instead`;
  const p = doseProblem(parseNumber(e.dose));
  if (p) return p;
  if (t.kind === 'direction') {
    if (e.unit === 'resid') return DIRECTION_RESID;
    if (e.unit !== 'diff' && e.unit !== 'raw') return 'choose the unit: difference of the means or raw';
    return '';
  }
  if (e.mode === 'add' && e.unit !== 'resid' && e.unit !== 'raw') return 'choose the unit: residual-norm fraction or raw';
  return '';
}
const DIRECTION_RESID = 'unit resid is not available for a direction: choose difference of the means or raw';

// ------------------------------------------------------------------------------------------------ recipes
// Pasted notation, one entry per line or separated by commas or semicolons:
//   L34 N3838 +0.75 · L10 N7757 +1.3, L34 N3838 -0.5 · L34N3838 +0.75 · L24 h30 ×2 · blocks.12.mlp x0.5
//   L11H3 *0 · L5N9 mul 2 · L5N9 add 0.4 · L5N9 +3 raw · group:name -0.2
// A signed number adds (in residual-norm fractions unless followed by "raw"); x, × or * multiplies.
// direction:refusal@12 -0.5 adds along a direction (in the difference of the means, "diff", unless followed by "raw").
// Lines starting with # are notes. Items that cannot be read are reported; the rest are kept.
const DOSE = new RegExp(String.raw`^(?:(add|mul)\s*:?\s*([+-]?${NUM})|([+-])\s*(${NUM})?|([x×*])\s*([+-]?${NUM})?|(${NUM}))(?:\s+(raw|resid|diff))?$`, 'i');

// The dose stands apart from the target (a space or a colon) and its sign is against its number: in
// neuron-explorer notes "L12 N5-7" is a range and "L19 N7839−" a polarity, so a sign or dash against the target,
// a dash of any other kind, and a sign standing apart from its number are refused, never read as a push.
const AMBIGUOUS = 'ambiguous: a range or a push?';

// A dash against the target. "L3 N10-20 +0.5" (a number, then a dose of its own) is a range: say how to write
// one. "L3N10-0.5" or "L3N10-20" alone can be a push or a range: say how to write each.
function dashProblem(t, rest) {
  const n = t.kind === 'neuron' ? /^L(\d+)N(\d+)$/.exec(t.addr) : null;
  const m = /^.(\d+(?:\.\d*)?)(?:\s+(\S.*))?$/.exec(rest.trim());
  const range = (last) => `blocks.${n[1]}.mlp.neurons.${n[2]}-${last}`;
  if (n && m && m[2] && /^\d+$/.test(m[1])) {
    const dose = m[2].replace(/\s+/g, ' ');
    return `${AMBIGUOUS} "${t.addr}${rest.trim()[0]}${m[1]}" reads as the neurons ${n[2]} to ${m[1]}; write a range as ${range(m[1])} ${dose.length > 40 ? dose.slice(0, 40) + '…' : dose}`;
  }
  const push = `${AMBIGUOUS} a dash against the neuron reads as a range or a polarity note; to push, put a space before the dose: ${t.addr || 'L5N9'} ${m && !m[2] ? '-' + m[1] : '-0.5'}`;
  if (!n || !m) return push;                             // no number after the dash: a polarity note, not a range
  return `${push}; for a range, write ${range(/^\d+$/.test(m[1]) ? m[1] : '<last>')} and a dose, e.g. +0.5`;
}

function parseItem(raw) {
  const item = raw.replace(/`/g, '').replace(/^\s*(?:[-*•]\s+|\d+[.)]\s+)/, '').trim();
  if (!item) return null;
  const t = matchTarget(item);
  if (!t) return { problem: `unknown target: ${TARGET_HELP}` };
  let rest = item.slice(t.length);
  if (rest && !/^[\s:x×*]/i.test(rest)) {
    if (/^[-−]/.test(rest) || DASH.test(rest[0])) return { problem: dashProblem(t, rest) };
    if (rest[0] === '+') {
      if (!rest.slice(1).trim()) return { problem: 'a sign without a dose (a polarity note): write +0.5 or -0.5' };
      return { problem: `the dose must stand apart from the target: put a space before it, e.g. ${t.addr} +0.5` };
    }
    return { problem: `unknown target: ${TARGET_HELP}` };
  }
  if (DASH.test(rest)) return { problem: `${AMBIGUOUS} a hyphen or an en or em dash is not read as a minus; write -0.5` };
  // Runs of spaces become one before the dose is read, so no pattern below backtracks over them (linear time).
  rest = rest.trim().replace(/\s+/g, ' ').replace(/^: ?/, '').replace(MINUS, '-');
  if (!rest) return { problem: 'no dose: write +0.75 to add or x2 to multiply' };
  const apart = /^([+-])\s+(\S*)/.exec(rest);
  if (apart) return { problem: apart[2] ? `${AMBIGUOUS} the sign stands apart from its number; write ${apart[1]}${apart[2]} with no space` : 'a sign without a dose (a polarity note): write +0.5 or -0.5' };
  const m = DOSE.exec(rest);
  if (!m) return { problem: `could not read the dose "${rest.length > 40 ? rest.slice(0, 40) + '…' : rest}": write +0.75 to add or x2 to multiply` };
  let mode, dose;
  if (m[1]) { mode = m[1].toLowerCase(); dose = Number(m[2]); }
  else if (m[3]) {
    if (m[4] == null) return { problem: 'a sign without a dose (a polarity note): write +0.5 or -0.5' };
    mode = 'add'; dose = Number(m[3] + m[4]);
  } else if (m[5]) {
    if (m[6] == null) return { problem: 'no multiplier after ' + m[5] + ': write x2' };
    mode = 'mul'; dose = Number(m[6]);
  } else return { problem: `"${m[7]}" has no sign: write +${m[7]} to add or x${m[7]} to multiply` };
  const unit = m[8] ? m[8].toLowerCase() : null;
  if (mode === 'mul' && unit) return { problem: `${unit} applies to add only; a multiplier has no unit` };
  if (mode === 'mul' && t.kind === 'direction') return { problem: 'a direction is added: write a signed dose, e.g. -0.5' };
  if (mode === 'add' && !addAllowed(t.kind)) return { problem: `add pushes neurons only; multiply ${t.kind === 'head' ? 'a head' : 'a whole MLP'} instead (x2)` };
  if (t.kind === 'direction' && unit === 'resid') return { problem: DIRECTION_RESID };
  if (t.kind !== 'direction' && unit === 'diff') return { problem: 'diff (the difference of the means) is a unit of a direction only; a neuron is pushed in resid or raw' };
  const p = doseProblem(dose);
  if (p) return { problem: p };
  const dflt = t.kind === 'direction' ? 'diff' : 'resid';
  return { entry: mode === 'add' ? { addr: t.addr, mode, dose, unit: unit || dflt } : { addr: t.addr, mode, dose } };
}

export function parseRecipe(text) {
  const entries = [], problems = [];
  const all = String(text ?? '');
  if (all.length > MAX_RECIPE) {
    return { entries, problems: [{ item: `(${all.length.toLocaleString('en-US')} characters)`, reason: `a recipe is at most ${MAX_RECIPE.toLocaleString('en-US')} characters; nothing was read` }] };
  }
  for (const line of all.split(/\r?\n/)) {
    if (/^\s*#/.test(line)) continue;
    const parts = line.split(/([,;])/);                    // separators kept: a decimal comma is told apart
    let last = null;                                       // what the previous part gave: { raw, at: 'entries'|'problems' }
    for (let i = 0; i < parts.length; i += 2) {
      const raw = parts[i];
      if (last && parts[i - 1] === ',' && /^\d/.test(raw) && /\d$/.test(last.raw)) {
        // "+0,75": a decimal comma (a digit on both sides of the comma). Neither half is kept.
        (last.at === 'entries' ? entries : problems).pop();
        const num = /([+-]?[\d.]+)$/.exec(last.raw.replace(MINUS, '-'))[1];
        problems.push({ item: `${last.raw.trim()},${raw.trim()}`, reason: `decimal comma? write ${num}.${/^\d+/.exec(raw)[0]}` });
        last = null;
        continue;
      }
      const r = parseItem(raw);
      if (!r) { last = null; continue; }
      if (r.entry) { entries.push(r.entry); last = { raw, at: 'entries' }; }
      else { problems.push({ item: raw.trim(), reason: r.problem }); last = { raw, at: 'problems' }; }
    }
  }
  return { entries, problems };
}

// "12: 3.5, 13: 4.1" or {"12": 3.5} -> { map } | { error }. Empty -> { map: null }.
export function parseP50(text) {
  const s = String(text ?? '').trim();
  if (!s) return { map: null };
  if (s.startsWith('{')) {
    const keys = [...s.matchAll(/"\s*L?(\d+)\s*"\s*:/gi)].map((m) => String(+m[1]));
    const twice = keys.find((k, i) => keys.indexOf(k) !== i);  // JSON.parse would keep the last one silently
    if (twice != null) return { error: `layer ${twice} is given twice` };
    try {
      const o = JSON.parse(s);
      return Object.keys(o).length ? checkP50(Object.entries(o).map(([k, v]) => [String(k).replace(/^L/i, ''), v])) : { map: null };
    } catch { return { error: 'the per-layer p50 is not valid JSON' }; }
  }
  const pairs = [];
  for (const part of s.split(/[,;\n]/)) {
    if (!part.trim()) continue;
    const m = /^\s*L?(\d+)\s*[:=]\s*(\S+)\s*$/i.exec(part);
    if (!m) return { error: `could not read "${part.trim()}": write layer: value, e.g. 12: 3.5` };
    pairs.push([m[1], parseNumber(m[2])]);
  }
  return checkP50(pairs);
}

function checkP50(pairs) {
  const map = {};
  for (const [layer, v] of pairs) {
    if (!/^\d+$/.test(layer)) return { error: `"${layer}" is not a layer number` };
    if (String(+layer) in map) return { error: `layer ${+layer} is given twice` };
    if (typeof v !== 'number' || !Number.isFinite(v) || v <= 0) return { error: `the p50 of layer ${layer} must be a number above 0` };
    map[String(+layer)] = v;
  }
  return { map };
}

// ------------------------------------------------------------------------------------------------ labels
const trim = (n) => String(Number(Number(n).toPrecision(6)));
export function entryLabel(e) {
  const d = typeof e.dose === 'number' ? e.dose : parseNumber(e.dose);
  if (d == null || !Number.isFinite(d)) return `${e.addr} (no valid dose)`;
  if (e.mode === 'mul') return `${e.addr} ×${trim(d)}`;
  return `${e.addr} ${d < 0 ? '-' : '+'}${trim(Math.abs(d))}${e.unit === 'raw' ? ' raw' : ''}`;
}
export const steerLabel = (entries) => (entries || []).map(entryLabel).join(', ');

// ------------------------------------------------------------------------------------------------ state
const KEY = (name) => 'wsp.steer.v1.' + name;
const fresh = () => ({ on: true, entries: [], recipe: '', p50: '', advanced: false });

function storage() {
  try { return globalThis.window?.localStorage || null; } catch { return null; }   // throws when site data is blocked
}

export function loadSteer(name) {
  try {
    const raw = storage()?.getItem(KEY(name));
    if (!raw) return null;
    const o = JSON.parse(raw);
    if (!o || !Array.isArray(o.entries)) return null;
    return { ...fresh(), on: o.on !== false, recipe: String(o.recipe || ''), p50: String(o.p50 || ''),
      entries: o.entries.filter((e) => e && typeof e.addr === 'string').map((e) => ({ addr: e.addr, mode: e.mode === 'mul' ? 'mul' : 'add',
        dose: String(e.dose ?? ''), unit: e.unit === 'raw' ? 'raw' : parseTarget(e.addr).kind === 'direction' ? 'diff' : 'resid', on: e.on !== false,
        ...directionExtra(e) })) };
  } catch { return null; }
}

// A direction entry's pin (the measurement's means_sha256 when the entry was made) and orthogonalize_direction.
const directionExtra = (e) => ({ ...(typeof e.pin === 'string' && e.pin ? { pin: e.pin } : {}), ...(typeof e.ortho === 'boolean' ? { ortho: e.ortho } : {}) });

export function saveSteer(name, st) {
  try {
    const s = storage();
    if (!s) return false;
    s.setItem(KEY(name), JSON.stringify({ on: st.on, recipe: st.recipe, p50: st.p50,
      entries: st.entries.map((e) => ({ addr: e.addr, mode: e.mode, dose: e.dose, unit: e.unit, on: e.on, ...directionExtra(e) })) }));
    return true;
  } catch { return false; }
}

// The panel state of the open workspace: app.keep is cleared when another model is opened.
export function steerState(app) {
  return app.keep.steer || (app.keep.steer = loadSteer(app.state.open.name) || fresh());
}

const persist = (app) => saveSteer(app.state.open.name, steerState(app));

// extra: { pin, ortho } of a direction entry (the Layers view's button).
export function addSteerEntry(app, addr, mode = 'add', dose = '0.5', unit = 'resid', extra = {}) {
  const st = steerState(app);
  const t = parseTarget(addr);
  st.entries.push({ addr: t.addr || addr, mode: t.kind && !addAllowed(t.kind) ? 'mul' : mode, dose: String(dose), unit, on: true, ...directionExtra(extra || {}) });
  st.check = null;
  persist(app);
  return st;
}

export function activeEntries(st) {
  return st.on ? st.entries.filter((e) => e.on) : [];
}

// ------------------------------------------------------------------------------------------------ reach
// What an entry pushes -> { neurons, text, problem }. neurons: how many neurons it pushes (0 for a head or an MLP,
// null when not known here). For a group the counts come from /api/steer/groups (st.groups, loaded by the panel).
export function entryReach(e, groups) {
  const t = parseTarget(e.addr);
  if (t.error) return { neurons: null, text: '', problem: '' };
  if (t.kind === 'neuron') return { neurons: 1, text: '1 neuron', problem: '' };
  if (t.kind === 'head') return { neurons: 0, text: '1 head', problem: '' };
  if (t.kind === 'mlp') return { neurons: 0, text: 'the whole MLP', problem: '' };
  if (t.kind === 'direction') return { neurons: 0, text: `a direction entering layer ${directionOf(t.addr).index}`, problem: '' };
  if (t.kind === 'range') {
    const [a, b] = /(\d+)-(\d+)$/.exec(t.addr).slice(1).map(Number);
    if (b < a) return { neurons: 0, text: '', problem: `the range ${a}-${b} is empty` };
    return { neurons: b - a + 1, text: `${fmtInt(b - a + 1)} neurons`, problem: '' };
  }
  if (!groups || groups.error) return { neurons: null, text: groups ? '' : 'counting…', problem: '' };
  const g = groups.list.find((x) => x.addr === t.addr);
  if (!g) return { neurons: null, text: '', problem: `there is no saved selection named "${t.addr.slice(6)}"` };
  if (g.error) return { neurons: null, text: '', problem: g.error };
  const left = skippedText(g.skipped);
  if (!g.neurons) return { neurons: 0, text: '', problem: `the selection has no neurons to push${left ? ` (it holds ${left})` : ''}` };
  return { neurons: g.neurons, text: `${fmtInt(g.neurons)} neuron${g.neurons === 1 ? '' : 's'}` + (left ? `; leaves out ${left}` : ''), problem: '' };
}

// {unresolved, heads, slices} -> "2 members that no longer resolve, 1 head, 3 tensor slices" (as the server words it)
export function skippedText(k) {
  if (!k) return '';
  const part = (n, one, many) => (n ? `${fmtInt(n)} ${n === 1 ? one : many}` : null);
  return [part(k.unresolved, 'member that no longer resolves', 'members that no longer resolve'),
    part(k.heads, 'head', 'heads'), part(k.slices, 'tensor slice', 'tensor slices')].filter(Boolean).join(', ');
}

// The saved selections as targets, fetched again whenever the selections change.
export async function loadSteerGroups(app, st, paint) {
  if (app.state.groups == null) return null;             // not known here: the server checks
  const key = JSON.stringify(app.state.groups);
  if (st.groupsKey === key) return st.groups;
  st.groupsKey = key;
  if (!Object.keys(app.state.groups).length) { st.groups = { list: [], max: MAX_NEURONS }; return st.groups; }
  st.groups = null;
  let got;
  try {
    const r = await api('/api/steer/groups');
    got = { list: Array.isArray(r?.groups) ? r.groups : [], max: Number(r?.max_neurons) || MAX_NEURONS };
  } catch (e) { got = { list: [], max: MAX_NEURONS, error: e.message }; }
  if (st.groupsKey === key) { st.groups = got; paint?.(); }
  return got;
}

const asSpec = (e) => {
  const t = parseTarget(e.addr);
  const out = { addr: t.addr || e.addr, mode: e.mode, dose: parseNumber(e.dose) };
  if (e.mode === 'add') out.unit = e.unit;
  if (t.kind === 'direction') {                          // the other kinds have no such fields (the server refuses them)
    if (e.pin) out.means_sha256 = e.pin;
    if (e.ortho === false) out.orthogonalize_direction = false;
  }
  return out;
};

// The fields a Run, Chat or probe request adds -> { fields, label, problems }.
// fields is null when nothing is switched on; problems are reasons not to send.
export function steerRequest(st) {
  const active = activeEntries(st);
  const problems = [];
  active.forEach((e) => {
    const p = entryProblem(e);
    if (p) problems.push(`${e.addr || 'entry ' + (st.entries.indexOf(e) + 1)}: ${p}`);
  });
  const max = st.groups?.max || MAX_NEURONS;
  let total = 0;
  active.forEach((e) => {
    if (entryProblem(e)) return;
    const r = entryReach(e, st.groups);
    const name = e.addr || 'entry ' + (st.entries.indexOf(e) + 1);
    if (r.problem) problems.push(`${name}: ${r.problem}`);
    else if (r.neurons > max) problems.push(`${name}: pushes ${fmtInt(r.neurons)} neurons; at most ${fmtInt(max)} can be pushed at once`);
    total += r.neurons || 0;
  });
  if (total > max && !problems.some((p) => p.includes('at most'))) problems.push(`the entries push ${fmtInt(total)} neurons in all; at most ${fmtInt(max)} can be pushed at once`);
  const p50 = parseP50(st.p50);
  if (p50.error) problems.push('per-layer p50: ' + p50.error);
  if (!active.length) return { fields: null, label: '', problems: [] };
  if (problems.length) return { fields: null, label: '', problems };
  const steer = active.map(asSpec);
  return { fields: { steer, ...(p50.map ? { p50: p50.map } : {}) }, label: steerLabel(steer), problems: [] };
}

// ------------------------------------------------------------------------------------------------ events
// `steering` events: calibration and column-norm reads. Run and Chat events carry the request's
// stream id; those of Check and the probes do not.
export function steeringStatus(ev) {
  if (!ev) return '';
  if (ev.state === 'calibrating') return `measuring the base model's residual norms: one forward pass over ${ev.tokens} tokens…`;
  if (ev.state === 'calibrated') return `measured in ${fmt(ev.seconds)} s; cached for this model`;
  if (ev.state === 'calibration_unusable') return `the measurement was unusable for layer${(ev.layers || []).length === 1 ? '' : 's'} ${(ev.layers || []).join(', ')}`;
  if (ev.state === 'reading') return `reading output weights: ${ev.done} of ${ev.total} neurons…`;
  if (ev.state === 'read') return `output weights read (${ev.total} neurons)`;
  return '';
}

export function onSteeringEvent(ev, app) {
  if (ev.stream) {
    const live = Object.values(app.keep.run?.live || {}).find((l) => l && l.id === ev.stream);
    if (live) { live.steering = ev; live.paint?.(); }
    return;
  }
  const st = app.keep.steer;
  if (st?.checking) { st.progress = ev; st.paint?.(); }
}

// ------------------------------------------------------------------------------------------------ check
function checkTable(r) {
  const rows = (r.entries || []).map((e) => {
    const p50 = e.p50 ? Object.values(e.p50)[0] : null;
    const value = e.value != null ? fmt(e.value) : e.values ? `${fmt(e.values.min)} … ${fmt(e.values.max)}` : '–';
    const norm = e.norm != null ? fmt(e.norm) : e.norms ? `${fmt(e.norms.min)} … ${fmt(e.norms.max)}` : '–';
    const source = e.mode === 'mul' ? 'multiplier' : e.unit === 'raw' ? 'raw (the dose is the constant)' : e.value == null && !e.values ? 'not measured yet' : e.source || '–';
    const reach = e.neurons > 1 ? h('span', { class: 'small muted' }, ` ${fmtInt(e.neurons)} neurons`) : null;
    const left = skippedText(e.skipped);
    const label = h('td', { class: 'mono' }, entryLabel(e), reach, left ? h('div', { class: 'small muted', role: 'note' }, 'not pushed: ' + left) : null);
    // a direction: the length of the vector pushed (|dose| × scale) and what one unit is; no column or p50
    if (e.kind === 'direction') return h('tr', {}, label, h('td', {}, `vector of length ${fmt(e.norm)}`), h('td', {}, '–'), h('td', {}, '–'),
      h('td', {}, e.unit === 'raw' ? 'raw (the dose is the length)' : `unit diff: one unit is ${fmt(e.scale)}`));
    return h('tr', {}, label,
      h('td', {}, e.mode === 'mul' ? '×' + fmt(e.dose) : value),
      h('td', {}, e.mode === 'mul' ? '–' : norm), h('td', {}, e.unit === 'resid' && p50 != null ? fmt(p50) : '–'), h('td', {}, source));
  });
  return h('table', { class: 'small' }, h('tr', {}, ['entry', 'constant added', 'output column length', 'layer residual norm (p50)', 'p50 source'].map((t) => h('th', { style: { textAlign: 'left', paddingRight: '10px' } }, t))), rows);
}

async function runCheck(app, st, paint, options) {
  const req = steerRequest(st);
  if (req.problems.length) { st.check = { error: req.problems.join('; ') }; paint(); return; }
  if (!req.fields) { st.check = { error: 'No entry is switched on.' }; paint(); return; }
  st.checking = true; st.progress = null; st.check = null; paint();
  try {
    const r = await api('/api/steer/resolve', { ...req.fields, ...options });
    st.check = { result: r, label: req.label };
  } catch (e) { st.check = { error: e.message }; }
  finally { st.checking = false; st.progress = null; paint(); }
}

// ------------------------------------------------------------------------------------------------ panel
const UNITS = [['resid', 'residual-norm fraction'], ['raw', 'raw']];
const DIRECTION_UNITS = [['diff', 'difference of the means'], ['raw', 'raw (unit direction)']];

// A target typed into an entry: canonical, and its mode and unit made possible for its kind (a head or an MLP is
// multiplied; a direction is added, in diff or raw). A pin belongs to its measurement: another name drops it.
function retarget(e, was) {
  const before = directionOf(was);
  const c = parseTarget(e.addr);
  if (c.addr) e.addr = c.addr;
  if (c.kind && !addAllowed(c.kind)) e.mode = 'mul';
  if (c.kind === 'direction') {
    e.mode = 'add';
    if (e.unit !== 'raw') e.unit = 'diff';
  } else if (e.unit === 'diff') e.unit = 'resid';
  const now = directionOf(e.addr);
  if (e.pin && (!now || !before || now.name !== before.name)) delete e.pin;
}

// The panel's header, from what a request would send (steerRequest): never a label for entries that would be refused.
export function headerText(st) {
  const req = steerRequest(st);
  if (req.problems.length) {
    const n = req.problems.filter((p) => !p.startsWith('per-layer p50')).length;
    const fix = [n ? `${n} entr${n === 1 ? 'y' : 'ies'}` : null, n < req.problems.length ? 'the per-layer p50' : null].filter(Boolean).join(' and ');
    return `not steered (fix ${fix})`;
  }
  if (req.fields) return `the edited model is steered: ${req.label}`;
  return 'off: the edited model runs unsteered';
}

export function steerPanel(app) {
  const st = steerState(app);
  const box = h('div', { class: 'card', style: { margin: '8px 0' } });
  const save = () => persist(app);
  const paint = () => { box.replaceChildren(...build().flat(Infinity).filter(Boolean)); };   // replaceChildren would print a null
  st.paint = paint;
  const changed = () => { st.check = null; save(); paint(); };

  const entryRow = (e, i) => {
    const t = parseTarget(e.addr);
    const reach = e.addr && !entryProblem(e) ? entryReach(e, st.groups) : { text: '', problem: '' };
    const over = reach.neurons > (st.groups?.max || MAX_NEURONS) ? `pushes ${fmtInt(reach.neurons)} neurons; at most ${fmtInt(st.groups?.max || MAX_NEURONS)} can be pushed at once` : '';
    const problem = (e.addr ? entryProblem(e) : '') || reach.problem || over;
    const was = e.addr;                                  // the target as drawn; typing changes e.addr before change
    const target = h('input', { type: 'text', value: e.addr, 'aria-label': `steer entry ${i + 1} target`, placeholder: 'L34N3838', style: { width: '150px', fontFamily: 'var(--mono)' },
      oninput: (ev) => { e.addr = ev.target.value; save(); }, onchange: () => { retarget(e, was); changed(); } });
    const mode = h('select', { 'aria-label': `steer entry ${i + 1} mode`, onchange: (ev) => { e.mode = ev.target.value; changed(); } },
      h('option', { value: 'add', selected: e.mode === 'add', disabled: !!t.kind && !addAllowed(t.kind) }, 'add'),
      h('option', { value: 'mul', selected: e.mode === 'mul', disabled: t.kind === 'direction' }, 'multiply'));
    const dose = h('input', { type: 'text', value: e.dose, 'aria-label': `steer entry ${i + 1} dose`, style: { width: '70px' },
      oninput: (ev) => { e.dose = ev.target.value; save(); }, onchange: changed });
    const unit = h('select', { 'aria-label': `steer entry ${i + 1} unit`, disabled: e.mode !== 'add', title: e.mode === 'add' ? '' : 'a multiplier has no unit',
      onchange: (ev) => { e.unit = ev.target.value; changed(); } },
      (t.kind === 'direction' ? DIRECTION_UNITS : UNITS).map(([v, label]) => h('option', { value: v, selected: e.unit === v }, label)));
    return [h('div', { class: 'row', style: { gap: '6px', alignItems: 'center' } },
      h('input', { type: 'checkbox', checked: !!e.on, 'aria-label': `steer entry ${i + 1} on`, onchange: (ev) => { e.on = !!ev.target.checked; changed(); } }),
      target, mode, dose, unit,
      h('button', { class: 'ghost', 'aria-label': `remove steer entry ${i + 1}`, onclick: () => { st.entries.splice(i, 1); changed(); } }, 'remove'),
      reach.text && !problem ? h('span', { class: 'small muted', 'aria-label': `steer entry ${i + 1} reach` }, 'pushes ' + reach.text) : null),
      problem ? h('div', { class: 'small', role: 'alert', style: { color: 'var(--crit)' } }, problem) : null];
  };

  const build = () => {
    const active = activeEntries(st);
    const head = h('div', { class: 'row', style: { alignItems: 'center', gap: '8px' } },
      h('b', {}, 'Steering'),
      h('label', { class: 'small' }, h('input', { type: 'checkbox', checked: !!st.on, 'aria-label': 'steering on', onchange: (ev) => { st.on = !!ev.target.checked; changed(); } }), ' on'),
      h('span', { class: 'small muted', 'aria-label': 'steering header' }, headerText(st)));
    const out = [head, h('p', { class: 'hint', style: { margin: '4px 0' } }, 'Pushes the edited model\'s activations while it generates. The base model is never steered and no weight changes. Add: a constant on one neuron, as a fraction of the layer\'s residual norm or raw. Multiply: a neuron, a head (L11H3) or a whole MLP (blocks.12.mlp).')];
    st.entries.forEach((e, i) => out.push(...entryRow(e, i)));
    out.push(h('div', { class: 'row', style: { gap: '6px', marginTop: '4px' } },
      h('button', { onclick: () => { st.entries.push({ addr: '', mode: 'add', dose: '0.5', unit: 'resid', on: true }); changed(); } }, 'add entry'),
      selectionPicker(app, st, (addr) => { addSteerEntry(app, addr); paint(); }),
      panelPickers.map((fn) => fn(app, (addr) => { addSteerEntry(app, addr); paint(); })),
      h('button', { disabled: !!st.checking, onclick: () => runCheck(app, st, paint, { calibrate: false }) }, 'Check')));
    // paste a recipe
    const recipe = h('textarea', { rows: 2, value: st.recipe, 'aria-label': 'steering recipe', placeholder: 'L10 N7757 +1.3, L34 N3838 -0.5, L24 h30 ×2', style: { width: '100%', fontFamily: 'var(--mono)' },
      oninput: (ev) => { st.recipe = ev.target.value; save(); } });
    out.push(h('div', { style: { marginTop: '6px' } }, h('div', { class: 'small muted' }, 'Paste a recipe (neuron-explorer notation; one entry per line or comma-separated):'), recipe,
      h('button', { onclick: () => {
        const r = parseRecipe(st.recipe);
        st.entries.push(...r.entries.map((e) => ({ addr: e.addr, mode: e.mode, dose: String(e.dose), unit: e.unit || 'resid', on: true })));
        st.recipeReport = { added: r.entries.length, problems: r.problems };
        changed();
      } }, 'add from recipe')));
    if (st.recipeReport) {
      const { added, problems } = st.recipeReport;
      out.push(h('div', { class: 'small', role: 'status' }, `${added} entr${added === 1 ? 'y' : 'ies'} added.`,
        problems.length ? h('div', { role: 'alert', style: { color: 'var(--crit)' } }, `${problems.length} not read: `,
          problems.map((p) => h('div', {}, h('span', { class: 'mono' }, p.item), ' — ', p.reason))) : null));
    }
    // advanced: a given per-layer p50, and measuring again
    out.push(h('details', { open: !!st.advanced, ontoggle: (ev) => { st.advanced = !!ev.target.open; } },
      h('summary', { class: 'small' }, 'advanced: given residual norms'),
      field('per-layer p50 (layer: value), used instead of measuring', h('input', { type: 'text', value: st.p50, 'aria-label': 'per-layer p50', placeholder: '12: 3.5, 34: 41.2', style: { width: '260px' },
        oninput: (ev) => { st.p50 = ev.target.value; save(); }, onchange: changed })),
      parseP50(st.p50).error ? h('div', { class: 'small', role: 'alert', style: { color: 'var(--crit)' } }, parseP50(st.p50).error) : null,
      h('button', { disabled: !!st.checking, onclick: () => runCheck(app, st, paint, { calibrate: true, recalibrate: true }) }, 'measure again')));
    // the check result
    if (st.checking) out.push(h('p', { class: 'small muted', role: 'status' }, steeringStatus(st.progress) || 'checking… (waits for a running generation to finish)'));
    else if (st.check?.error) out.push(h('p', { class: 'small', role: 'alert', style: { color: 'var(--crit)' } }, st.check.error));
    else if (st.check?.result) {
      const r = st.check.result, cal = r.calibration || {};
      out.push(checkTable(r));
      if (cal.needed && !cal.ran && !cal.cached) {
        out.push(h('p', { class: 'small', role: 'status' }, 'Calibration will run before the first steered reply: one forward pass of the base model over a reference text, then cached for this model.',
          ' ', h('button', { onclick: () => runCheck(app, st, paint, { calibrate: true }) }, 'measure now')));
      } else if (cal.ran) out.push(h('p', { class: 'small muted' }, `Measured now on the base model (${cal.tokens ?? '?'} tokens); cached.`));
      else if (cal.cached) out.push(h('p', { class: 'small muted' }, 'Residual norms from the cache (measured earlier on the base model).'));
      if (r.reads?.reads) out.push(h('p', { class: 'small muted' }, `${r.reads.reads} weight read${r.reads.reads === 1 ? '' : 's'} for the column lengths.`));
    }
    return out;
  };
  paint();
  loadSteerGroups(app, st, paint);
  return box;
}

// The saved selections, with the neurons each pushes; one that cannot be pushed is listed but not offered.
function selectionPicker(app, st, add) {
  const names = Object.keys(app.state.groups || {});
  if (!names.length) return null;
  const g = st.groups;
  const max = g?.max || MAX_NEURONS;
  const pick = h('select', { 'aria-label': 'steer a saved selection' },
    h('option', { value: '' }, g ? 'a saved selection…' : 'saved selections (counting…)'),
    (g?.list || []).map((x) => {
      const why = x.error ? x.error : !x.neurons ? 'no neurons to push' : x.neurons > max ? `more than ${fmtInt(max)} neurons` : '';
      return h('option', { value: x.addr, disabled: !!why, title: why || skippedText(x.skipped) },
        `${x.name} (${fmtInt(x.neurons || 0)} neuron${x.neurons === 1 ? '' : 's'}${why ? ': ' + why : ''})`);
    }));
  return [pick, h('button', { 'aria-label': 'add the saved selection', onclick: () => { if (pick.value) add(pick.value); } }, 'add selection')];
}

// For an inspector page: "steer this selection" (a saved selection, one dose for all its neurons) or "+ steering"
// (a neuron, a neuron range, a head or a whole MLP); null for anything steering cannot push. `n`: the /api/node answer.
export function steerControl(app, n, status) {
  const t = parseTarget(n.addr || '');
  if (t.error) return null;
  let why = '';
  const s = n.selection;
  if (t.kind === 'group' && s) {
    if (!s.neurons) why = 'this selection has no neurons to push';
    else if (s.neurons > MAX_NEURONS) why = `this selection has ${fmtInt(s.neurons)} neurons; at most ${fmtInt(MAX_NEURONS)} can be pushed at once`;
  }
  const label = t.kind === 'group' ? 'steer this selection' : '+ steering';
  return h('button', { class: 'ghost', disabled: !!why, title: why || "add to the Run tab's steering panel (one dose for all its neurons); no weight changes",
    onclick: () => {
      const st = addSteerEntry(app, t.addr);
      const e = st.entries[st.entries.length - 1];
      if (status) status.textContent = `${entryLabel(e)} added to the Run tab's steering panel; set the dose there.`;
    } }, label);
}

// For a result or live card: what the edited reply ran with.
export function steeredLine(label) {
  return label ? h('div', { class: 'small', style: { color: 'var(--warn)' } }, 'steered: ' + label) : null;
}
