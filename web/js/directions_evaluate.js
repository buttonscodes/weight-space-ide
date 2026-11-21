// Compare's third right-hand state, a checkpoint on disk (POST /api/directions/evaluate, Heretic's evaluate_model):
// a modified checkpoint's directory (or a PEFT LoRA adapter's, or another workspace's: the route detects which) is
// read, never written, compared with the base weights tensor by tensor, and
// scored as an uncommitted change on top of the loaded model against the base. "Check the checkpoint" compares the
// weights only (the model is not run); "Score" is one request answering the comparison and both sides' scores. The
// path is page-only (not in the drafts). Route: docs/DIRECTIONS.md ("POST /api/directions/evaluate").
import { h, api, fmt, field } from './util.js';
import { put, busyReason, refusalText, fmtBytes, runJob, renderingFields } from './directions.js';
import { scorersBody, scorerSets, maxResponseLength } from './directions_forms.js';
import { errorTarget, checkAfterRefusal } from './directions_prompts.js';
import { JOB_LABELS } from './directions_edit.js';

// The route's refusal for a GGUF workspace (E-H1), shown without a request.
export const EVALUATE_GGUF = "scoring a modified checkpoint needs a floating-point base: this workspace's base is GGUF (quantized) and is never dequantized to compare it with another checkpoint";
export const NO_PATH = "type the path of the modified model's directory (a checkpoint, a PEFT adapter or a workspace)";
export const EVALUATE_HINT = "A checkpoint on disk is scored against the base weights, as Heretic's evaluate_model does: its weights are read and applied as an uncommitted change on top of the loaded model, never written; the KL divergence is against the base. Give the directory of a merged model or a fine-tune with the base's tensor names and shapes, of a PEFT LoRA adapter (scored as the base plus the adapter), or of another workspace (its base with its enabled patches).";
const ROWS_SHOWN = 50;

// Why a checkpoint cannot be sent at all ('' when it can): Score's first reasons, and Check's with busyReason
// (Check runs no model, so the model's run reasons do not hold it).
export function checkpointWhy(app, st) {
  if (!String(st.compare?.path ?? '').trim()) return NO_PATH;
  if (app.state.open.kind === 'gguf') return EVALUATE_GGUF;
  return '';
}
export const checkWhy = (app, st) => checkpointWhy(app, st) || busyReason(st);

// The score request's body without its stream -> {body, problems}.
export function evaluateBody(st) {
  const r = renderingFields(st), s = scorersBody(st), M = maxResponseLength(st);
  return { problems: [...r.problems, ...s.problems, ...(M.problem ? [M.problem] : [])],
    body: { path: st.compare.path, ...r.fields, max_response_length: M.value, scores: s.scores } };
}

// A refusal as kept in st.errors.compare: a text starting with "path" names the path field.
const refusalOf = (text) => ({ text, target: /^path\b/.test(text) ? { field: 'path', item: null }
  : errorTarget(text, ['max_response_length', 'system', 'prefix', 'chat_options']) });

export async function checkCheckpoint(app, st, again) {
  if (checkWhy(app, st)) return;                 // a click that reaches a disabled button anyway
  const c = st.compare;
  st.errors.compare = null; c.check = null;
  await runJob(app, st, { kind: 'scores', label: 'Checking the checkpoint', stateLabel: 'checkpoint', verb: 'Checking' }, async (stream) => {
    let a;
    try { a = await api('/api/directions/evaluate', { path: c.path, check: true, stream }); } catch (e) { st.errors.compare = refusalOf(refusalText(e)); return ''; }
    if (a.stopped) return `Stopped after ${a.done} of ${a.total}; nothing was compared.`;
    if (app.keep.dirs === st) c.check = a;
    return "Compared the checkpoint's weights with the base's.";
  });                                            // runJob redraws the view when the job ends
}

// Score (Compare's own guard has passed): one request, both sides' scores and the comparison in its answer.
export async function scoreCheckpoint(app, st, again) {
  const c = st.compare, { body, problems } = evaluateBody(st);
  if (problems.length) { st.errors.compare = { text: problems.join(' '), target: null }; again(); return; }
  const version = app.state.status?.version;
  st.errors.compare = null;
  c.results = null; c.check = null; c.showAll = {};
  let refused = '';
  await runJob(app, st, { kind: 'scores', label: 'Scoring the base weights and the checkpoint', steps: 1, stateLabel: JOB_LABELS.checkpoint }, async (stream) => {
    const sent = { ...body, stream };
    let a;
    try { a = await api('/api/directions/evaluate', sent); } catch (e) { refused = refusalText(e); st.errors.compare = refusalOf(refused); return ''; }
    if (a.stopped) return `Stopped after ${a.done} of ${a.total}; nothing was scored.`;
    if (app.keep.dirs === st) {
      c.results = { left: { state: 'base', answer: { scores: a.base.scores }, sent },
        right: { state: 'checkpoint', answer: { scores: a.modified.scores, candidate: null }, sent },
        version, compare: a.compare, notes: a.notes };                         // the weights they describe
    }
    return 'Scored the base weights and the checkpoint.';
  });
  if (refused) await checkAfterRefusal(app, st, refused, scorerSets(st), again);
}

// ------------------------------------------------------------------------------------------------ the comparison
const commas = (n) => (typeof n === 'number' ? n.toLocaleString('en-US') : '–');
const num = (x) => (typeof x === 'number' ? fmt(x, 4) : '–');
const dtypesOf = (counts) => Object.entries(counts || {}).map(([k, n]) => `${k} ×${n}`).join(', ') || '–';

// The answer's `compare` (a check's or a score's) with its notes (#dirs-evaluate-compare): the counts, the differing
// tensors (the first 50, then "show all <n>"), the dtypes, what is held while it is scored, how exactly it applied.
export function compareCard(answer, { all = false, onShowAll = null } = {}) {
  const k = answer?.compare;
  if (!k) return null;
  const rows = k.rows || [], shown = all ? rows : rows.slice(0, ROWS_SHOWN), held = k.held, rec = k.reconstruction;
  return put(h('div', { class: 'card', id: 'dirs-evaluate-compare' }),
    h('b', {}, `${k.differ} of ${k.tensors} tensors differ from the base (${commas(k.changed)} of ${commas(k.of)} weights)`),
    rows.length ? h('table', { class: 'grid', id: 'dirs-evaluate-rows' },
      h('tr', {}, ['param', 'changed', 'of', 'norm', 'relative', 'max_abs'].map((t) => h('th', {}, t))),
      shown.map((r) => h('tr', {}, h('td', { class: 'mono' }, String(r.param ?? '')), h('td', {}, commas(r.changed)), h('td', {}, commas(r.of)),
        h('td', {}, num(r.norm)), h('td', {}, num(r.relative)), h('td', {}, num(r.max_abs))))) : null,
    shown.length < rows.length && onShowAll ? h('button', { class: 'ghost', onclick: onShowAll }, `show all ${rows.length}`) : null,
    k.dtypes ? h('div', { class: 'small' }, `base ${dtypesOf(k.dtypes.base)}; checkpoint ${dtypesOf(k.dtypes.modified)}`) : null,
    held ? (held.fits ? h('div', { class: 'small' }, `${fmtBytes(held.bytes ?? 0)} held as FP32 changes while it is scored`)
      : h('p', { class: 'hint err', role: 'note' }, held.why ?? '')) : null,
    rec ? h('div', { class: 'small' }, rec.inexact ? `${commas(rec.inexact)} weights within ${fmt(rec.max_abs, 3)} after applying` : 'applied exactly') : null,
    (answer.notes || []).map((n) => h('p', { class: 'hint', role: 'note' }, n)));
}

// Below Compare's selects when the right side is a checkpoint on disk: the path, the hint, Check and its comparison.
// Typing repaints Check and Score in place (no re-render: the caret stays where it is).
export function checkpointControls(app, st, again) {
  const c = st.compare, bad = st.errors.compare?.target?.field === 'path';
  const row = h('div', { class: 'row', id: 'dirs-evaluate-check' });
  const paint = () => {
    const why = checkWhy(app, st);
    row.replaceChildren(h('button', { disabled: !!why, title: why || 'compares its weights with the base; the model is not run',
      onclick: () => checkCheckpoint(app, st, again) }, 'Check the checkpoint'));
  };
  st.paint.checkpoint = paint;
  paint();
  return h('div', { id: 'dirs-evaluate' },
    field('checkpoint', h('input', { type: 'text', 'aria-label': 'checkpoint path', placeholder: '/path/to/the/merged/model', value: c.path,
      class: bad ? 'dirs-bad' : null, oninput: (e) => { c.path = e.target.value; paint(); st.paint.compareGo?.(); } })),
    h('p', { class: 'hint', id: 'dirs-evaluate-hint' }, EVALUATE_HINT),
    row,
    c.check ? compareCard(c.check, { all: !!c.showAll.evaluate, onShowAll: () => { c.showAll.evaluate = true; again(); } }) : null);
}
