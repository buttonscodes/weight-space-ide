// The Directions tab's Edit sub-view: the method (ablation or ARA), its parameters as a form generated from the
// served descriptors (or Heretic's parameters pasted in), the plan and weight script of the edit (POST
// /api/directions/script: nothing is read or run), its dry run and its commit; and the Compare sub-view. Every edit
// goes through the IDE's one edit path, the console's own app.runScript: the tab never posts /api/edit itself and
// never writes a weight. Routes: docs/DIRECTIONS.md ("The parameters", "POST /api/directions/script", "POST
// /api/directions/scores").
import { h, api, fmt, field } from './util.js';
import { put, paramsFor, draftChanged, draftTyped, busyReason, runReason, refusalText, fmtBytes, runJob, paintJob, renderingFields,
  runningSearches } from './directions.js';
import { paramForm, pasteParameters, paramValues, fieldNames, scorerEditor, scorersBody, scorerSets, maxResponseLength, ensureScorers,
  RESPONSE_SCORERS } from './directions_forms.js';
import { errorTarget, checkAfterRefusal } from './directions_prompts.js';
import { checkpointControls, checkpointWhy, compareCard, scoreCheckpoint } from './directions_evaluate.js';
import { steerState, steerRequest, steerLabel } from './steer.js';

const METHOD_LABELS = { ablation: 'ablation', ara: 'ARA' };
export const ARA_NEEDS = 'ARA needs a measurement with module inputs and outputs: measure again with them';

const selectedMeta = (st) => (st.list?.directions || []).find((m) => m.name === st.selected && !m.error) || null;
// The methods the open model has fields for (GET /api/directions `parameters`), in the order of the control.
export const methodsOf = (st) => ['ablation', 'ara'].filter((m) => st.list?.parameters?.[m]);
// Why the method cannot be used with the selected measurement ('' when it can).
export function methodWhy(st, method = st.method) {
  const meta = selectedMeta(st);
  return method === 'ara' && meta && !meta.module_io ? ARA_NEEDS : '';
}

// ------------------------------------------------------------------------------------------------ the plan and its key
// The parameters of the form as sent (10.1) -> {params, problems}.
export function currentParams(st) {
  const desc = st.list?.parameters?.[st.method];
  if (!desc) return { params: null, problems: ["this checkpoint's layers are not known"] };
  return paramValues(desc, paramsFor(st, st.method), st.selected, st.method);
}
// A plan is for these parameters on these weights: a commit changes the workspace's version, so a plan cannot be
// applied twice, nor after the weights changed, without a new preview.
export function planKey(app, st) {
  const { params } = currentParams(st);
  return params ? `${JSON.stringify(params)}|${app.state.status?.version}` : null;
}
export const planCurrent = (app, st) => !!st.plan && st.plan.key != null && st.plan.key === planKey(app, st);

// Why Preview plan cannot be pressed now ('' when it can). It runs nothing, but it is a request: it waits behind one.
function previewWhy(app, st) {
  return busyReason(st) || (st.previewing ? 'the plan is being read' : '') || methodWhy(st);
}

async function preview(app, st, again) {
  if (previewWhy(app, st)) return;                // a click that reaches a disabled button anyway
  const { params, problems } = currentParams(st);
  if (problems.length) { st.errors.plan = { text: problems.join('; '), target: null }; again(); return; }
  const key = `${JSON.stringify(params)}|${app.state.status?.version}`;
  st.errors.plan = null;
  st.previewing = true;
  again();
  try {
    const result = await api('/api/directions/script', params);
    if (app.keep.dirs !== st) return;
    st.plan = { key, method: params.method, direction: params.direction, result };
  } catch (e) {
    const text = refusalText(e);
    st.errors.plan = { text, target: errorTarget(text, fieldNames(st.list?.parameters?.[params.method])) };
  } finally { st.previewing = false; }
  again();
}

const STALE = 'The parameters or the weights changed since this plan: preview it again before applying it.';
const cell = (v) => h('td', {}, v == null ? '–' : String(v));
function planTable(plan, method) {
  const cols = method === 'ara'
    ? [['layer', (r) => r.layer], ['component', (r) => r.component], ['views', (r) => r.views], ['d_in', (r) => r.d_in], ['stored', (r) => r.stored],
      ['scratch', (r) => (r.scratch == null ? null : fmtBytes(r.scratch))], ['skipped', (r) => r.skipped]]
    : [['layer', (r) => r.layer], ['component', (r) => r.component], ['weight', (r) => (r.weight == null ? null : fmt(r.weight))], ['direction', (r) => r.direction],
      ['views', (r) => r.views], ['stored', (r) => r.stored], ['skipped', (r) => r.skipped]];
  return h('table', { class: 'grid', id: 'dirs-plan-table' },
    h('tr', {}, cols.map(([t]) => h('th', {}, t))),
    (plan.rows || []).map((r) => h('tr', { class: r.skipped ? 'muted' : '' }, cols.map(([, get]) => cell(get(r))))));
}
export function planTotals(plan, method) {
  const t = plan.totals || {};
  return `${t.components} components in ${t.layers} layers, ${t.stored} values stored, ${t.skipped} skipped` +
    (method === 'ara' ? `; the largest fit asks the memory admission for ${fmtBytes(t.scratch ?? 0)}` : '');
}

function planView(app, st) {
  const p = st.plan;
  if (!p) return h('div', { id: 'dirs-plan' });
  const current = planCurrent(app, st);
  return h('div', { id: 'dirs-plan', class: 'card' },
    h('div', { class: 'row' }, h('b', {}, `The plan of ${p.method === 'ara' ? 'ARA' : 'the ablation'} of '${p.direction}'`),
      h('button', { class: 'ghost', title: 'puts the script in the weight script box below, to edit or run it there', onclick: () => app.setConsole(p.result.script) }, 'copy to the weight script bar')),
    h('p', { class: 'hint', role: 'note', id: 'dirs-plan-stale' }, current ? '' : STALE),
    h('pre', { id: 'dirs-script' }, p.result.script),
    planTable(p.result.plan || {}, p.method),
    h('p', { class: 'small', id: 'dirs-plan-totals' }, planTotals(p.result.plan || {}, p.method)));
}

// ------------------------------------------------------------------------------------------------ dry run and apply
// Both run the previewed script through the shell's own call (app.runScript: POST /api/edit with the console's scope
// box and edit progress, weightsChanged() after a commit). Neither can be stopped: Session.apply has no stop point.
export function editWhy(app, st, kind) {
  const edit = app.state.open.capabilities?.logical_edit;
  if (edit?.supported === false) return `edits are not supported for this checkpoint: ${edit.reason || 'no reason given'}`;
  const busy = busyReason(st) || (st.previewing ? 'the plan is being read' : '');
  if (busy) return busy;
  if (!st.plan) return 'preview the plan first';
  if (!planCurrent(app, st)) return 'the parameters or the weights changed since the plan: preview it again';
  if (kind === 'apply') {
    if (st.plan.committed) return 'this plan is committed: preview it again to apply another';
    const running = runningSearches(st);
    if (running.length) return `Search '${running[0].name}' is running ${running[0].running === 'this IDE' ? 'in this IDE' : 'in another process'}: committing an edit would end it.`;
  }
  return '';
}

export function editMessage(plan) {
  const t = plan.result.plan?.totals || {};
  return `directions: ${plan.method === 'ara' ? 'ablate_ara' : 'ablate_direction'}('${plan.direction}'), ${t.components} components in ${t.layers} layers`;
}

async function runEdit(app, st, kind) {
  if (editWhy(app, st, kind)) return;            // a click that reaches a disabled button anyway
  const plan = st.plan, dry = kind === 'dry', message = editMessage(plan);
  st.applied = null;
  await runJob(app, st, { kind, label: dry ? 'Dry run' : 'Committing' }, async () => {
    const r = await app.runScript(plan.result.script, dry ? { dry: true, message } : { message });
    if (app.keep.dirs !== st) return '';
    st.applied = { kind, report: r };
    if (!r || r.ok === false) return `Not committed: ${r?.error ?? 'no answer'}`;
    if (dry) return 'Dry run: nothing was committed.';
    if (r.committed) { plan.committed = true; return `Committed patch ${r.patch}.`; }
    return r.note || 'No weights changed; nothing was committed.';
  });
}

const lines = (text) => String(text ?? '').split('\n').length;
// The report of a dry run or a commit (#dirs-apply-report; also a search trial's, with `trial`), or of a refused one.
export function applyReport(r, { trial = false } = {}) {
  if (r instanceof Error || (r && r.payload && r.ok === undefined)) {
    const p = r.payload || {};
    if (p.status === 'not_run') return h('p', { class: 'hint err', role: 'alert' },
      `Not committed: the candidate tests could not run: ${refusalText(r)} ${(r.event?.recommendations || p.event?.recommendations || []).join(' ')}`.trim());
    return h('p', { class: 'hint err', role: 'alert' }, `Not committed: ${refusalText(r)}`);
  }
  if (!r || r.ok === false) return h('p', { class: 'hint err', role: 'alert' }, `Not committed: ${r?.error ?? 'no answer'}`);
  if (!r.changed?.length && !r.committed) return h('p', { class: 'hint', role: 'status' }, r.note || 'no weights changed');
  const t = r.tests, d = r.drift;
  return h('div', { class: 'card' },
    h('b', {}, r.committed ? `Committed patch ${r.patch}` : 'Dry run: nothing committed'),
    r.changed?.length ? h('div', { class: 'small' }, `${r.n_changed} weights in ${r.changed.length} tensors, stored as ${r.stored} numbers`) : null,
    t ? h('div', { class: 'small' }, `tests ${t.suite}: ${t.before[0]}/${t.before[1]} → ${t.after[0]}/${t.after[1]}`,
      t.fixed?.length ? `  +${t.fixed.length} fixed` : '', t.broken?.length ? h('span', { class: 'err' }, `  −${t.broken.length} broken: ${t.broken.slice(0, 6).join(', ')}`) : '') : null,
    d && d.n ? h('div', { class: 'small' }, `drift: KL ${d.kl} (max ${d.kl_max}), top-1 agreement ${d.top1_agree}`) : null,
    r.not_run ? h('div', {}, h('span', { class: 'pill warn' }, `${r.committed ? 'Committed untested' : 'Untested'}: ${r.not_run}`)) : null,
    r.output ? (lines(r.output) > 20 ? h('details', {}, h('summary', {}, `the script's output (${lines(r.output)} lines)`), h('pre', { class: 'dirs-output' }, r.output))
      : h('pre', { class: 'dirs-output' }, r.output)) : null,
    trial && r.edit_equals_scored === true ? h('p', { class: 'small' }, `The committed edit equals the scored one (pieces ${String(r.digest || '').slice(0, 12)}).`) : null,
    trial && r.edit_equals_scored === false ? h('p', { class: 'small' }, h('span', { class: 'pill warn' }, 'The committed edit differs from the scored one (accepted; recorded in the patch).')) : null);
}

// The buttons that act on the plan, repainted as the form is typed (no re-render: the caret stays where it is).
function actions(app, st, again) {
  const box = h('div', { class: 'row', id: 'dirs-edit-actions' });
  st.paint.editActions = () => {
    const why = previewWhy(app, st), dryWhy = editWhy(app, st, 'dry'), applyWhy = editWhy(app, st, 'apply');
    box.replaceChildren(...[
      h('button', { class: 'primary', disabled: !!why, title: why || 'the script and what it will change (nothing is read or run)', onclick: () => preview(app, st, again) },
        st.previewing ? 'Previewing…' : 'Preview plan'),
      h('button', { disabled: !!dryWhy, title: dryWhy || 'runs the script on the current weights with the candidate tests and drift; nothing is committed', onclick: () => runEdit(app, st, 'dry') }, 'Dry run'),
      h('button', { disabled: !!applyWhy, title: applyWhy || 'commits the script as a patch through the edit path', onclick: () => runEdit(app, st, 'apply') }, 'Apply'),
      why && !st.previewing ? h('span', { class: 'hint' }, why) : applyWhy && st.plan ? h('span', { class: 'hint' }, applyWhy) : null].filter(Boolean));
    const stale = document.getElementById('dirs-plan-stale');
    if (stale) stale.textContent = planCurrent(app, st) ? '' : STALE;
  };
  st.paint.editActions();
  return box;
}

function methodControl(app, st, again) {
  return h('div', { class: 'seg', role: 'group', 'aria-label': 'method' }, methodsOf(st).map((m) => {
    const why = methodWhy(st, m);
    return h('button', { class: m === st.method ? 'on' : '', 'aria-pressed': m === st.method ? 'true' : 'false', disabled: !!why, title: why || null,
      onclick: () => { st.method = m; draftChanged(app, st); again(); } }, METHOD_LABELS[m]);
  }));
}

// "paste Heretic's parameters": a trial's or a config's parameters fill the form (10.2).
function pasteBox(app, st, again) {
  const paste = st.paste ||= { text: '', note: '', error: false };
  const fill = () => {
    const desc = st.list.parameters[st.method];
    const r = pasteParameters(st.method, paste.text, desc, paramsFor(st, st.method));
    if (r.error) { paste.note = r.error; paste.error = true; again(); return; }
    st.params[st.method] = r.values;
    paste.note = r.unused.length ? `not used: ${r.unused.join(', ')}` : 'the form is filled';
    paste.error = false;
    draftChanged(app, st);
    again();
  };
  return h('details', { id: 'dirs-paste', open: !!(paste.text || paste.note) },
    h('summary', {}, "paste Heretic's parameters"),
    h('textarea', { rows: 4, value: paste.text, 'aria-label': 'Heretic parameters', placeholder: '{"direction_index": 12.3, "weight_distributions": {...}}',
      oninput: (e) => { paste.text = e.target.value; } }),
    h('div', { class: 'row' }, h('button', { onclick: fill }, 'Fill the form'),
      paste.note ? h('span', { class: paste.error ? 'hint err' : 'hint', role: paste.error ? 'alert' : 'status' }, paste.note) : null));
}

export function editView(el, app, st, again) {
  const P = st.list?.parameters;
  if (!P) { el.append(h('p', { class: 'hint' }, "This checkpoint's layers are not known, so no edit can be planned here.")); return; }
  const methods = methodsOf(st);
  if (!methods.includes(st.method)) st.method = methods[0];
  const desc = P[st.method], values = paramsFor(st, st.method), why = methodWhy(st);
  const onChange = ({ typed, redraw }) => {
    if (typed) { draftTyped(app, st); st.paint.editActions?.(); return; }
    draftChanged(app, st);
    if (redraw) again(); else st.paint.editActions?.();
  };
  const bad = (name) => st.errors.plan?.target?.field === name;
  put(el,
    h('div', { class: 'row' }, h('span', { class: 'small muted' }, 'method'), methodControl(app, st, again)),
    why ? h('p', { class: 'hint', role: 'note' }, why) : null,
    h('p', { class: 'hint' }, 'The change is applied on top of the current weights (the base plus the enabled patches).'),
    st.selected ? null : h('p', { class: 'hint' }, 'Select a measurement: the edit removes its direction.'),
    paramForm(desc, values, { onChange, bad }),
    pasteBox(app, st, again),
    actions(app, st, again),
    st.errors.plan ? h('p', { class: 'hint err', role: 'alert' }, st.errors.plan.text) : null,
    planView(app, st),
    h('div', { id: 'dirs-apply-report' }, st.applied ? applyReport(st.applied.report) : null));
}

// ------------------------------------------------------------------------------------------------ Compare
// Two states of the weights scored with the same scorers, one request after the other (POST /api/directions/scores;
// nothing is committed): the scores side by side, each response scorer's responses, the KL per prompt. The candidate
// is the Edit view's previewed script, scored on top of the current weights and never committed. A checkpoint on disk
// (directions_evaluate.js) is scored against the base weights in one request. "steered" is the current weights with
// the Run tab's steering panel (its active entries, sent as `steer` and `p50`); the KL reference stays the base.
export const STATE_LABELS = { current: 'current weights', steered: "current weights, steered by the Run tab's panel", base: 'base weights',
  candidate: "the Edit view's candidate", checkpoint: 'a checkpoint on disk' };
export const JOB_LABELS = { current: 'current weights', steered: 'steered current weights', base: 'base weights', candidate: 'candidate',
  checkpoint: 'base weights and the checkpoint' };     // "Scoring the <label>"
const LEFT = ['current', 'base'], RIGHT = ['candidate', 'current', 'steered', 'base', 'checkpoint'];
const RESPONSES_SHOWN = 50;
export const COMPARE_STALE = 'Scored before the weights changed: score again to compare the current weights.';
const LEFT_IS_BASE = 'a checkpoint on disk is scored against the base weights';
function ensureCompare(st) {
  const c = st.compare ||= { left: 'current', right: 'candidate', results: null };
  c.showAll ||= {};
  if (typeof c.path !== 'string') c.path = '';                       // page-only, never in the drafts
  c.check ??= null;
  if (c.right === 'checkpoint') c.left = 'base';
  return c;
}

// Why the candidate cannot be scored now ('' when it can).
export function candidateWhy(app, st) {
  const edit = app.state.open.capabilities?.logical_edit;
  if (edit?.supported === false) return `edits are not supported for this checkpoint: ${edit.reason || 'no reason given'}`;
  const running = runningSearches(st);
  if (running.length) return `search '${running[0].name}' is running: score the candidate once it has ended`;
  if (!planCurrent(app, st)) return 'preview the plan in the Edit view first';
  return '';
}
// Why the Run tab's steering panel cannot steer a side now ('' when it can).
export function steeredWhy(app) {
  const s = steerRequest(steerState(app));
  if (s.problems.length) return `Steering: ${s.problems.join('; ')}`;
  return s.fields ? '' : "the Run tab's steering panel has no active entries";
}
function scoreWhy(app, st) {
  const c = ensureCompare(st);
  if (c.right === 'checkpoint') { const why = checkpointWhy(app, st); if (why) return why; }
  if (c.left === c.right) return 'pick two different states';
  return runReason(app, st) || busyReason(st) || (c.left === 'candidate' || c.right === 'candidate' ? candidateWhy(app, st) : '')
    || (c.left === 'steered' || c.right === 'steered' ? steeredWhy(app) : '');
}

async function score(app, st, again) {
  if (scoreWhy(app, st)) return;                 // a click that reaches a disabled button anyway
  if (st.compare.right === 'checkpoint') return scoreCheckpoint(app, st, again);     // one request (directions_evaluate.js)
  const c = ensureCompare(st), r = renderingFields(st), body = scorersBody(st), M = maxResponseLength(st);
  const problems = [...r.problems, ...body.problems, ...(M.problem ? [M.problem] : [])];
  if (problems.length) { st.errors.compare = { text: problems.join(' '), target: null }; again(); return; }
  const script = st.plan?.result?.script, sides = [c.left, c.right], version = app.state.status?.version;
  const steering = sides.includes('steered') ? steerRequest(steerState(app)).fields : null;     // the panel as it is at the click
  st.errors.compare = null;
  c.results = null; c.showAll = {};
  let refused = '';
  await runJob(app, st, { kind: 'scores', label: 'Scoring two states', steps: 2, stateLabel: JOB_LABELS[sides[0]] }, async (stream, job) => {
    const done = [];
    for (const [k, state] of sides.entries()) {
      if (k === 1) {
        if (job.stopping) return `Stopped before scoring the ${JOB_LABELS[state]}; nothing was scored.`;
        // the second request: its own stream (a late event of the first is not taken for it), the same Stop button
        job.step = 2; job.stateLabel = JOB_LABELS[state]; job.stream = `${stream}-2`;
        st.ownStreams.add(job.stream);
        for (const key of ['state', 'phase', 'done', 'total', 'seconds']) job[key] = null;
        paintJob(app, st);
      }
      const sent = state === 'steered'
        ? { state: 'current', ...r.fields, max_response_length: M.value, scores: body.scores, stream: job.stream, ...steering }
        : { state, ...(state === 'candidate' ? { script } : {}), ...r.fields, max_response_length: M.value, scores: body.scores, stream: job.stream };
      let answer;
      try { answer = await api('/api/directions/scores', sent); } catch (e) {
        refused = refusalText(e);
        st.errors.compare = { text: k ? `The ${JOB_LABELS[state]}: ${refused}` : refused,
          target: errorTarget(refused, ['max_response_length', 'system', 'prefix', 'chat_options', 'script']) };
        return '';
      }
      if (answer.stopped) return `Stopped after ${answer.done} of ${answer.total}; nothing was scored.`;
      done.push({ state, answer, sent });
    }
    if (app.keep.dirs === st) c.results = { left: done[0], right: done[1], version };      // the weights they describe
    return `Scored the ${JOB_LABELS[sides[0]]} and the ${JOB_LABELS[sides[1]]}.`;
  });
  if (refused) await checkAfterRefusal(app, st, refused, scorerSets(st), again);
}

// A prompt as shown: its user text, and its system message on a muted line when it is a prompt object.
function promptCell(x, k) {
  if (typeof x === 'string') return h('td', {}, h('div', { class: 'dirs-text' }, x));
  if (x && typeof x === 'object' && typeof x.user === 'string') {
    return h('td', {}, h('div', { class: 'dirs-text' }, x.user), x.system != null ? h('div', { class: 'hint' }, `system: ${x.system}`) : null);
  }
  return h('td', { class: 'muted' }, `prompt ${k + 1}`);
}
function responseCell(row, plugin) {
  if (!row) return h('td', { class: 'muted' }, '–');
  return h('td', {}, h('div', { class: 'dirs-text' }, row.response),
    plugin === 'KeywordRate' && row.match ? h('span', { class: 'pill bad' }, 'refusal') : null,
    plugin === 'ResponseLength' ? h('div', { class: 'hint' }, `${row.length} characters`) : null);
}
// Responses side by side (Compare's two states; a search trial's beside its baseline): a row per prompt, a column per
// side, a row whose two texts differ marked; the first 50 rows, then "show all <N>". -> [the table, the button or null]
export function responsesTable(id, plugin, prompts, columns, { all = false, onShowAll = null } = {}) {
  const n = Math.max(prompts.length, ...columns.map((c) => c.rows.length)), shown = all ? n : Math.min(n, RESPONSES_SHOWN);
  const differs = (k) => columns.length > 1 && columns[0].rows[k]?.response !== columns[1].rows[k]?.response;
  return [h('table', { class: 'grid', id },
    h('tr', {}, ['prompt', ...columns.map((c) => c.label)].map((t) => h('th', {}, t))),
    Array.from({ length: shown }, (_, k) => h('tr', { class: differs(k) ? 'dirs-differs' : '' }, promptCell(prompts[k], k), columns.map((c) => responseCell(c.rows[k], plugin))))),
  shown < n && onShowAll ? h('button', { class: 'ghost', onclick: onShowAll }, `show all ${n}`) : null];
}
const change = (a, b) => {
  if (typeof a?.value !== 'number' || typeof b?.value !== 'number') return '–';
  const d = b.value - a.value;
  return (d > 0 ? '+' : '') + fmt(d, 4);
};

function compareResults(app, st, again) {
  const c = st.compare, res = c.results;
  if (!res) return null;
  const { left, right } = res, labels = [STATE_LABELS[left.state], STATE_LABELS[right.state]];
  const L = left.answer.scores || [], R = right.answer.scores || [];
  const notes = [left, right].filter((x) => x.answer.candidate).flatMap((x) => {
    const cand = x.answer.candidate;
    return [cand.commit_check?.ok === false ? h('p', { class: 'hint err', role: 'alert' }, `Applying this candidate would be refused: ${cand.commit_check.error}`) : null,
      cand.note ? h('p', { class: 'hint', role: 'note' }, cand.note) : null];
  });
  const tables = L.map((a, i) => {
    const b = R[i];
    if (RESPONSE_SCORERS.has(a.plugin)) {
      const rows = a.responses || [], other = b?.responses || [];
      const prompts = (rows.length >= other.length ? rows : other).map((r) => r.prompt);
      return h('div', {}, h('h4', {}, a.name),
        responsesTable(`dirs-responses-${i}`, a.plugin, prompts, [{ label: labels[0], rows }, { label: labels[1], rows: other }],
          { all: !!c.showAll[i], onShowAll: () => { c.showAll[i] = true; again(); } }));
    }
    if (a.plugin === 'KLDivergence') {
      const sentPrompts = left.sent.scores[i]?.prompts || [];
      const per = (x) => (x?.per_prompt == null ? null : x.per_prompt);
      const lp = per(a), rp = per(b), n = Math.max(lp?.length ?? 0, rp?.length ?? 0) || sentPrompts.length;
      const plain = sentPrompts.length === n && sentPrompts.every((x) => typeof x === 'string' || typeof x?.user === 'string');
      const kl = (list, k) => h('td', {}, list == null ? '0 (by definition)' : fmt(list[k], 4));
      return h('div', {}, h('h4', {}, a.name),
        h('table', { class: 'grid', id: `dirs-kl-${i}` },
          h('tr', {}, ['prompt', `${labels[0]} KL`, `${labels[1]} KL`].map((t) => h('th', {}, t))),
          Array.from({ length: n }, (_, k) => h('tr', {}, promptCell(plain ? sentPrompts[k] : null, k), kl(lp, k), kl(rp, k)))));
    }
    return null;
  });
  const stale = res.version !== app.state.status?.version;            // a commit (or a patch change) since they were scored
  // a checkpoint's comparison above the table, unless a newer Check's is shown under the path
  const card = res.compare && !(c.check && c.right === 'checkpoint')
    ? compareCard(res, { all: !!c.showAll.evaluate, onShowAll: () => { c.showAll.evaluate = true; again(); } }) : null;
  return h('div', { id: 'dirs-compare-results' },
    stale ? h('p', { class: 'hint', role: 'note', id: 'dirs-compare-stale' }, COMPARE_STALE) : null,
    card,
    h('table', { class: 'grid', id: 'dirs-compare-table' },
      h('tr', {}, ['score', labels[0], labels[1], 'change'].map((t) => h('th', {}, t))),
      L.map((a, i) => h('tr', {}, h('td', {}, a.name), h('td', {}, a.display), h('td', {}, R[i]?.display ?? '–'), h('td', {}, change(a, R[i]))))),
    steeredNotes(left, right), notes, tables);
}
// What a steered side was steered with, from the server's report (the panel may have changed since).
function steeredNotes(...sides) {
  return sides.filter((x) => x.state === 'steered' && x.answer.steering)
    .map((x) => h('div', {}, h('span', { class: 'pill warn' }, `steered: ${steerLabel(x.answer.steering.entries)}`)));
}

export function compareView(el, app, st, again) {
  ensureScorers(st);
  const c = ensureCompare(st), checkpoint = c.right === 'checkpoint';
  const pick = (key, states, label, attrs = {}) => h('select', { 'aria-label': label, ...attrs,
    onchange: (e) => { c[key] = e.target.value; ensureCompare(st); draftChanged(app, st); again(); } },
  states.map((s) => h('option', { value: s, selected: c[key] === s }, STATE_LABELS[s])));
  const selects = [field('left', pick('left', LEFT, 'left state', checkpoint ? { disabled: true, title: LEFT_IS_BASE } : {})),
    field('right', pick('right', RIGHT, 'right state'))];
  // the Score button and its reason, repainted as the checkpoint's path is typed
  const go = h('div', { class: 'row', id: 'dirs-compare-go' });
  st.paint.compareGo = () => {
    const why = scoreWhy(app, st);
    go.replaceChildren(...selects, h('button', { class: 'primary', disabled: !!why, title: why || null, onclick: () => score(app, st, again) }, 'Score'),
      ...(why ? [h('span', { class: 'hint' }, why)] : []));
  };
  st.paint.compareGo();
  put(el,
    h('p', { class: 'hint' }, 'Scores two states of the weights with the same scorers, one after the other; nothing is committed. The candidate is the script previewed in the Edit view, scored on top of the current weights.'),
    scorerEditor(app, st, again, { field: 'scores', refusal: st.errors.compare }),
    go,
    checkpoint ? checkpointControls(app, st, again) : null,
    st.errors.compare ? h('p', { class: 'hint err', role: 'alert' }, st.errors.compare.text) : null,
    compareResults(app, st, again));
}
