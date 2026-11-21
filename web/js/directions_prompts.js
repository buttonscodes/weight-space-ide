// Prompt sets of the Directions tab: an ordered list of parts, each a box of typed prompts (one per line, or a JSON
// list when the box starts with "[") or a Hugging Face dataset or local file named by Heretic's eight fields. The value
// sent is the parts' items in order: a plain list of texts when no part is a dataset (the old path, hashed as before),
// else a list mixing texts, prompt objects and specifications (docs/DIRECTIONS.md "Prompt sets from datasets").
// "Check" asks POST /api/directions/datasets what the set holds without fetching; only "Download" fetches.
import { h, api, field } from './util.js';
import { put, refusalText, fmtBytes, busyReason, runJob, renderingFields, draftChanged, draftTyped } from './directions.js';

export const FIELDS = ['dataset', 'commit', 'config', 'split', 'column', 'prefix', 'suffix', 'system_prompt'];
const ROUTE = 'POST /api/directions/datasets';

export const newSet = () => ({ parts: [{ kind: 'texts', text: '' }], check: null });
const datasetPart = (spec = {}) => ({ kind: 'dataset', spec: Object.fromEntries(FIELDS.map((f) => [f, spec[f] == null ? '' : String(spec[f])])) });

// ------------------------------------------------------------------------------------------------ values
const isPromptObject = (x) => x != null && typeof x === 'object' && !Array.isArray(x) && typeof x.user === 'string' &&
  (x.system === undefined || x.system === null || typeof x.system === 'string');

// One texts part (number p, from 1) -> {items, problem}.
export function parseTexts(text, p) {
  const t = String(text ?? '');
  if (t.trim().startsWith('[')) {
    const bad = (why) => ({ items: [], problem: `part ${p}: starts with [ but is not a JSON list of prompts (${why})` });
    let v;
    try { v = JSON.parse(t); } catch (e) { return bad(e.message); }
    if (!Array.isArray(v)) return bad('not a list');
    const k = v.findIndex((x) => typeof x !== 'string' && !isPromptObject(x));
    if (k >= 0) return bad(`item ${k + 1} is neither text nor {user, system}`);
    return { items: v, problem: null };
  }
  return { items: t.split(/\r?\n/).filter((line) => line.trim()), problem: null };
}

// A dataset part as sent: its filled fields (an empty one is left out), `dataset` always.
export function specOf(part) {
  const out = { dataset: part.spec.dataset ?? '' };
  for (const f of FIELDS.slice(1)) if (part.spec[f]) out[f] = part.spec[f];
  return out;
}

// -> {value, index, problems}: the items of every part in order; index[k] is the part (from 0) of item k.
export function setValue(set) {
  const value = [], index = [], problems = [];
  set.parts.forEach((part, i) => {
    if (part.kind === 'dataset') { value.push(specOf(part)); index.push(i); return; }
    const r = parseTexts(part.text, i + 1);
    if (r.problem) problems.push(r.problem);
    for (const x of r.items) { value.push(x); index.push(i); }
  });
  return { value, index, problems };
}
export const partOfItem = (set, k) => (k == null ? null : setValue(set).index[k] ?? null);
const typedCount = (set) => set.parts.reduce((n, part, i) => n + (part.kind === 'texts' ? parseTexts(part.text, i + 1).items.length : 0), 0);

// ------------------------------------------------------------------------------------------------ refusals
// The field a refusal names (13.2) -> {field, item} (item from 0, or null), or null. `fields` are the names of the
// active form's fields.
const SETS = String.raw`good|bad|prompts|scores\[\d+\]\.prompts|scorers\[\d+\]\.prompts`;
export function errorTarget(message, fields = []) {
  const t = refusalText(message);
  let m;
  if ((m = /^(good|bad) prompt (\d+)/.exec(t))) return { field: m[1], item: Number(m[2]) - 1 };
  if ((m = new RegExp(`^(${SETS})\\[(\\d+)\\]`).exec(t))) return { field: m[1], item: Number(m[2]) };
  if ((m = new RegExp(`^(${SETS})\\b`).exec(t))) return { field: m[1], item: null };
  if ((m = /^weight_distributions\.([\w.]+?)\.(max_weight_position|max_weight|min_weight_distance|min_weight)/.exec(t))) return { field: `${m[1]}.${m[2]}`, item: null };
  if ((m = /^(scores|scorers)\[(\d+)\]/.exec(t))) return { field: `${m[1]}[${m[2]}]`, item: null };
  if (/^modifiers\[0\]/.test(t)) return { field: 'modifiers[0]', item: null };
  const named = [...fields].sort((a, b) => b.length - a.length).find((f) => t.startsWith(f + ':') || t.startsWith(f + ' must'));
  return named ? { field: named, item: null } : null;
}
export const namesDatasetsRoute = (text) => String(text || '').includes(ROUTE);

// ------------------------------------------------------------------------------------------------ check and download
const checkKey = (value, system) => JSON.stringify([value, system]);
export const checkSystem = (st) => renderingFields(st).fields.system ?? null;
// The set's last Check (answer or refusal), when it is for the set as it is now.
export function currentCheck(st, set) {
  const c = set.check;
  return c && c.key === checkKey(setValue(set).value, checkSystem(st)) ? c : null;
}
// The prompts the set holds when that is known: every dataset part checked (or none there).
export function knownCount(st, set) {
  if (!set.parts.some((p) => p.kind === 'dataset')) return typedCount(set);
  const c = currentCheck(st, set);
  return c?.answer?.cached ? c.answer.count : null;
}

// Check (download false: nothing is fetched) or Download (a job with a stream: progress, Stop between files).
export async function checkSet(app, st, set, { download = false, again = () => {}, forced = null } = {}) {
  const { value, problems } = setValue(set);
  const system = checkSystem(st), key = checkKey(value, system);
  if (problems.length) { set.check = { key, refusal: problems.join(' '), target: null }; again(); return; }
  const refused = (e) => {
    const text = refusalText(e);
    set.check = { key, refusal: text, target: errorTarget(text), download: namesDatasetsRoute(text) || !!forced };
  };
  if (!download) {
    if (busyReason(st) || set.checking) return;
    set.checking = true; again();
    try {
      const answer = await api('/api/directions/datasets', { prompts: value, system });
      set.check = { key, answer, download: !!forced && answer.cached !== false };
    } catch (e) { refused(e); }
    finally { set.checking = false; }
    again();
    return;
  }
  await runJob(app, st, { kind: 'fetch', label: 'Downloading' }, async (stream) => {
    try {
      const answer = await api('/api/directions/datasets', { prompts: value, system, download: true, stream });
      if (answer.stopped) return `Download stopped after ${fmtBytes(answer.done_bytes)} of ${fmtBytes(answer.total_bytes)}; the ${answer.fetched.length} file(s) fetched stay in the cache.`;
      set.check = { key, answer };
      return `Downloaded ${answer.fetched.length} file(s), ${fmtBytes(answer.fetched_bytes)}.`;
    } catch (e) { refused(e); return ''; }
  });
}

// After a request (measure, detect, score, create) was refused for a set that is not in the cache: Check that set at
// once (no network), so what a download would fetch is shown without a second click (5.4, addendum B).
export async function checkAfterRefusal(app, st, text, sets, again = () => {}) {
  if (!namesDatasetsRoute(text)) return false;
  const target = errorTarget(text);
  const set = target && sets[target.field];
  if (!set) return false;
  await checkSet(app, st, set, { again, forced: text });
  return true;
}

// ------------------------------------------------------------------------------------------------ a measurement's sets
// A measurement's stored prompts as parts again: with its datasets record, an inline part becomes typed prompts and
// any other part its specification; without a record, one box of all its prompts.
const asText = (list) => (list.every((x) => typeof x === 'string' && !x.includes('\n')) && !String(list[0] ?? '').trim().startsWith('[')
  ? list.join('\n') : JSON.stringify(list, null, 1));
export function setFromMeasurement(prompts, record) {
  const list = prompts || [];
  if (!record?.parts?.length) return { parts: [{ kind: 'texts', text: asText(list) }], check: null };
  return { parts: record.parts.map((p) => (p.kind === 'inline'
    ? { kind: 'texts', text: asText(list.slice(p.first - 1, p.first - 1 + p.count)) }
    : datasetPart(p.specification || {}))), check: null };
}

// ------------------------------------------------------------------------------------------------ the editor
const promptText = (x) => (typeof x === 'string' ? x : x?.user ?? JSON.stringify(x));
function previewItem(x) {
  return h('li', {}, promptText(x), x && typeof x === 'object' && x.system != null ? h('div', { class: 'hint' }, `system: ${x.system}`) : null);
}

function wouldFetch(entries, downloadButton) {
  const known = entries.flatMap((e) => (e.files || []).map((f) => f.bytes)).filter((b) => typeof b === 'number');
  return [...entries.map((e) => h('div', { class: 'dirs-would' },
    h('div', {}, `${e.repository} at ${e.revision} (${e.commit ? e.commit.slice(0, 12) : 'latest commit unknown'})`),
    e.files ? h('ul', { class: 'small' }, e.files.map((f) => h('li', {}, `${f.path} (${f.bytes == null ? 'size unknown' : fmtBytes(f.bytes)})`)))
      : h('div', { class: 'hint' }, 'file names and sizes known after the file list'),
    h('ul', { class: 'small' }, (e.requests || []).map((r) => h('li', {}, r))))),
  downloadButton(known.length ? `Download (${fmtBytes(known.reduce((a, b) => a + b, 0))})` : 'Download')];
}

function checkResult(c, downloadButton) {
  if (!c?.answer) return null;
  const a = c.answer;
  if (a.cached === false) return h('div', { class: 'dirs-check', role: 'status' }, h('div', {}, 'Not in the cache; a download would fetch:'), wouldFetch(a.would_fetch || [], downloadButton));
  return h('div', { class: 'dirs-check', role: 'status' },
    h('div', {}, `${a.count} prompts · sha256 ${String(a.prompts_sha256 || '').slice(0, 12)}`, a.fetched?.length ? ` · fetched ${a.fetched.length} file(s), ${fmtBytes(a.fetched_bytes)}` : ''),
    h('ol', { class: 'small' }, (a.preview || []).map(previewItem)),
    c.download ? downloadButton('Download') : null);
}

// opts: id, title (shown), label (short, for the inputs' names), name (the request field a refusal names), limit,
// heretic (a specification or a list of them, for "Heretic's default"), extra (more buttons), refusal ({text, target}
// of the last request of the editor's kind that named this set), onInput (called as prompts are typed), again.
export function setEditor(app, st, set, opts) {
  const again = opts.again || (() => {});
  const busy = busyReason(st);
  const refusal = opts.refusal && opts.refusal.target?.field === opts.name ? opts.refusal : null;
  const c = currentCheck(st, set);
  const checkRefusal = c && !c.answer ? c : null;
  const badPart = (r) => (r?.target?.item != null ? partOfItem(set, r.target.item) : null);
  const changed = () => { draftChanged(app, st); };
  const count = h('span', { class: 'hint' });
  const paintCount = () => {
    const typed = typedCount(set), cur = currentCheck(st, set);
    const segs = [`${typed} prompts typed`];
    if (set.parts.some((p) => p.kind === 'dataset')) {
      if (cur?.answer?.cached) segs.push(`${cur.answer.count - typed} from datasets`);
      else set.parts.forEach((p, i) => { if (p.kind === 'dataset') segs.push(`part ${i + 1} not checked`); });
    }
    count.textContent = `${segs.join(' + ')}, of 1..${opts.limit ?? 1024}`;
  };
  paintCount();
  const downloadButton = (label) => h('button', { disabled: !!busy, title: busy || null, onclick: () => checkSet(app, st, set, { download: true, again }) }, label);
  const refusalLine = (r) => put(h('div', { class: 'hint err', role: 'alert' }, r.refusal ?? r.text), r.download ? [' ', downloadButton('Download')] : null);
  const part = (p, i) => {
    const n = i + 1, problem = h('div', { class: 'hint err', role: 'alert' });
    const bad = badPart(refusal) === i || badPart(checkRefusal) === i;
    const remove = h('button', { class: 'ghost', onclick: () => { set.parts.splice(i, 1); changed(); again(); } }, 'remove');
    let body;
    if (p.kind === 'texts') {
      const show = () => { problem.textContent = parseTexts(p.text, n).problem || ''; };
      show();
      body = h('textarea', { rows: 4, value: p.text, 'aria-label': `${opts.label} part ${n} texts`,
        oninput: (e) => { p.text = e.target.value; show(); paintCount(); opts.onInput?.(); draftTyped(app, st); }, onchange: changed });
    } else {
      body = h('div', { class: 'row' }, FIELDS.map((f) => field(f, h('input', { type: 'text', value: p.spec[f], 'aria-label': `${opts.label} part ${n} ${f}`,
        style: { width: f === 'dataset' ? '260px' : f === 'system_prompt' ? '200px' : '110px' },
        oninput: (e) => { p.spec[f] = e.target.value; draftTyped(app, st); }, onchange: changed }))));
    }
    return h('div', { class: 'dirs-part' + (bad ? ' dirs-bad' : ''), 'data-part': n },
      h('div', { class: 'row small muted' }, `part ${n}: ${p.kind === 'texts' ? 'prompts, one per line (or a JSON list)' : 'a dataset (Heretic\'s fields)'}`, remove),
      body, problem,
      checkRefusal && badPart(checkRefusal) === i ? refusalLine(checkRefusal) : null,
      refusal && badPart(refusal) === i ? refusalLine(refusal) : null);
  };
  const add = (p) => { set.parts.push(p); changed(); again(); };
  const heretic = opts.heretic ? [].concat(opts.heretic) : null;
  return put(h('div', { id: opts.id, class: 'card dirs-set' + (refusal || checkRefusal?.target?.field === 'prompts' ? ' dirs-bad' : '') },
    h('div', {}, h('b', {}, opts.title), ' ', count)),
  set.parts.map(part),
  h('div', { class: 'row' },
    h('button', { class: 'ghost', onclick: () => add({ kind: 'texts', text: '' }) }, '+ prompts'),
    h('button', { class: 'ghost', onclick: () => add(datasetPart()) }, '+ dataset'),
    heretic ? h('button', { class: 'ghost', title: 'Heretic\'s default prompt set; nothing is fetched', onclick: () => { set.parts.push(...heretic.map(datasetPart)); changed(); again(); } }, "Heretic's default") : null,
    opts.extra || null,
    h('button', { disabled: !!busy || !!set.checking, title: busy || 'what the set holds (nothing is fetched)', onclick: () => checkSet(app, st, set, { again }) }, set.checking ? 'Checking…' : 'Check')),
  checkResult(c, downloadButton),
  checkRefusal && badPart(checkRefusal) == null ? refusalLine(checkRefusal) : null,
  refusal && badPart(refusal) == null ? refusalLine(refusal) : null);
}
