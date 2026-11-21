// The Directions tab's Output sub-view: the workspace's patches exported as a PEFT LoRA adapter (POST
// /api/export_adapter), each patch's reproduction record with what changed since (GET /api/patches/{pid}/record),
// and the re-run of an edit from its record (POST /api/patches/{pid}/reproduce, Session.apply on the server; a commit
// makes a new patch). Opening the view sends no request: its patches are the Patches pane's own list (app.js
// renderPatches keeps it in app.keep.patchList and calls onPatchesLoaded); a record is read, a plan made, a file
// written or an edit re-run on a click. Routes: docs/DIRECTIONS.md ("A LoRA adapter (PEFT)", "The reproduction record").
import { h, api, fmt, field } from './util.js';
import { put, enc, refusalText, busyReason, runJob, patchOrigin, loadList, fmtBytes } from './directions.js';
import { applyReport } from './directions_edit.js';

// The export's refusals for a workspace kind (adapter_export A1, A2), shown without a request.
export const NO_WSP = "a PEFT adapter is for a transformers checkpoint; this is one of this project's own models: use the existing export (GET /api/export)";
export const NO_GGUF = "this workspace's base is a quantized GGUF file: a PEFT adapter applies to a floating-point transformers checkpoint, and these patches were computed against the quantized weights, so the adapter's base would be the dequantized model, which the IDE never builds. Export the WSP companion bundle (GET /api/export), or apply the edit to the floating-point checkpoint the GGUF was made from";
export const NO_TICK = 'tick at least one patch';
export const NO_RECORD = 'No reproduction record: this patch was not made by ablate_direction, ablate_ara, direction or direction_means, or it was made before records existed.';
const LAYOUTS = [['auto', 'auto (uniform when every module has the same rank)'], ['uniform', 'uniform'], ['per_module', 'per_module']];
const EXISTS = /^an adapter named '.*' exists in this workspace/;          // A8: the replace box appears

export function ensureOutput(st) {
  return st.output || (st.output = { ticks: {}, name: '', layout: 'auto', base: '', replace: false,
    replaceFor: null, exporting: false, result: null, refusal: '', records: {}, reruns: {} });
}

// The patches in stack order: the Patches pane's list (undefined until the shell has read it).
const patchesOf = (app) => (Array.isArray(app.keep.patchList) ? app.keep.patchList : null);
const patchOf = (app, pid) => (patchesOf(app) || []).find((p) => p.id === pid) || null;

// ------------------------------------------------------------------------------------------------ the export
const ticked = (o, p) => (p.id in o.ticks ? o.ticks[p.id] : p.enabled === true);       // by default the enabled patches
const nameOf = (o) => (o.name === '' ? null : o.name);
// The request's body: `patches` null when exactly the enabled patches are ticked, else the ticked ids in stack order;
// `replace` only once an "exists" refusal for this name showed its box and the person ticked it.
export function exportBody(app, st, dryRun = true) {
  const o = ensureOutput(st), list = patchesOf(app) || [];
  const on = list.filter((p) => ticked(o, p)).map((p) => p.id), enabled = list.filter((p) => p.enabled === true).map((p) => p.id);
  const same = on.length === enabled.length && on.every((id, i) => id === enabled[i]);
  return { name: nameOf(o), patches: same ? null : on, layout: o.layout, base_model_name_or_path: o.base.trim() ? o.base : null,
    replace: o.replaceFor != null && o.replaceFor === nameOf(o) && o.replace === true, dry_run: dryRun };
}

// Why Preview and Export cannot be pressed ('' when they can).
export function exportWhy(app, st) {
  const o = ensureOutput(st), kind = app.state.open.kind;
  if (kind === 'wsp') return NO_WSP;
  if (kind === 'gguf') return NO_GGUF;
  if (!(patchesOf(app) || []).some((p) => ticked(o, p))) return NO_TICK;
  return busyReason(st) || (o.exporting ? 'the export request is running' : '');
}

async function sendExport(app, st, dry, again) {
  if (exportWhy(app, st)) return;                // a click that reaches a disabled button anyway
  const o = st.output, body = exportBody(app, st, dry);
  o.exporting = true; o.refusal = ''; o.result = null;
  again();
  try {
    const r = await api('/api/export_adapter', body);
    if (app.keep.dirs !== st) return;
    o.result = r;
    if (r.written) { o.replace = false; o.replaceFor = null; }
  } catch (e) {
    o.refusal = refusalText(e);
    if (EXISTS.test(o.refusal)) { o.replaceFor = body.name; o.replace = false; }
  } finally { o.exporting = false; }
  again();
}

function exportForm(app, st, again) {
  const o = st.output, why = exportWhy(app, st);
  const replaceShown = o.replaceFor != null && o.replaceFor === nameOf(o);
  return put(h('div', { class: 'card', id: 'dirs-export' }),
    h('b', {}, 'A LoRA adapter (PEFT)'),
    h('p', { class: 'hint' }, 'The ticked patches, each at its strength, as adapter_config.json and adapter_model.safetensors (with wsp_adapter.json, its provenance) in <workspace>/export/adapters/<name>/. No weight, patch or workspace file changes.'),
    h('div', { class: 'row' },
      field('name', h('input', { type: 'text', value: o.name, 'aria-label': 'adapter name', oninput: (e) => { o.name = e.target.value; }, onchange: () => again() })),
      field('layout', h('select', { 'aria-label': 'layout', onchange: (e) => { o.layout = e.target.value; } },
        LAYOUTS.map(([v, t]) => h('option', { value: v, selected: o.layout === v }, t)))),
      field('base_model_name_or_path', h('input', { type: 'text', value: o.base, 'aria-label': 'base model', oninput: (e) => { o.base = e.target.value; } }),
        "a Hub id; left empty, the workspace's repository or none")),
    replaceShown ? h('label', { class: 'small' }, h('input', { type: 'checkbox', checked: o.replace, 'aria-label': 'replace the adapter',
      onchange: (e) => { o.replace = !!e.target.checked; } }), ` replace the adapter '${o.replaceFor}'`) : null,
    h('div', { class: 'row', id: 'dirs-export-go' },
      h('button', { disabled: !!why, title: why || 'what would be written (a dry run: nothing is written)', onclick: () => sendExport(app, st, true, again) }, 'Preview (dry run)'),
      h('button', { class: 'primary', disabled: !!why, title: why || 'writes the adapter into the workspace', onclick: () => sendExport(app, st, false, again) }, 'Export'),
      why ? h('span', { class: 'hint' }, why) : null),
    o.refusal ? h('p', { class: 'hint err', role: 'alert' }, o.refusal) : null);
}

const hex12 = (s) => String(s ?? '').slice(0, 12);
const td = (v) => h('td', {}, v == null ? '–' : String(v));
const headRow = (names) => h('tr', {}, names.map((t) => h('th', {}, t)));
function exportResult(o) {
  const r = o.result;
  if (!r) return null;
  return h('div', { class: 'card', id: 'dirs-export-result' },
    h('b', {}, r.written ? `Adapter written to ${r.dir}` : 'Preview: nothing written'),
    h('div', { class: 'small' }, `layout ${r.layout}, r ${r.r}, factors ${fmtBytes(r.bytes ?? 0)}; base_model_name_or_path ${r.base_model_name_or_path ?? 'null'}` +
      (r.revision ? ` at ${r.revision}` : '') + `; model type ${r.family}${r.validated_family === false ? ' (not validated with PEFT)' : ''}`),
    (r.files || []).length ? h('table', { class: 'grid', id: 'dirs-export-files' }, headRow(['file', 'size', 'sha256']),
      r.files.map((f) => h('tr', {}, td(f.name), td(fmtBytes(f.bytes)), h('td', { class: 'mono', title: f.sha256 || null }, hex12(f.sha256))))) : null,
    h('table', { class: 'grid', id: 'dirs-export-modules' }, headRow(['module', 'rank', 'exact', 'pieces']),
      (r.modules || []).map((m) => h('tr', {}, td(m.module), td(m.rank), td(m.exact ? 'exact' : 'rounded once'),
        td((m.pieces || []).map((p) => `${p.patch}:${p.kind}×${p.rank}`).join(', '))))),
    (r.notes || []).map((n) => h('p', { class: 'hint', role: 'note' }, n)));
}

// ------------------------------------------------------------------------------------------------ the record
async function showRecord(app, st, pid, again) {
  const o = st.output;
  o.records[pid] = { loading: true };
  again();
  try {
    const a = await api(`/api/patches/${enc(pid)}/record`);
    if (app.keep.dirs !== st) return;
    o.records[pid] = { answer: a };
  } catch (e) { o.records[pid] = { error: refusalText(e) }; }
  again();
}

// "Switch patch <id> off": the patch route, then the shell's weightsChanged (which reads the patches again and
// redraws this view through onPatchesLoaded) and the record read again.
async function switchOff(app, st, pid, again) {
  if (busyReason(st)) return;
  const o = st.output;
  try {
    await api('/api/patch', { action: 'set', id: pid, enabled: false });
    if (app.keep.dirs !== st) return;
  } catch (e) { o.records[pid] = { ...o.records[pid], switchError: refusalText(e) }; again(); return; }
  await app.weightsChanged();
  if (app.keep.dirs === st) await showRecord(app, st, pid, again);
}

const HEX = /^[0-9a-f]{32,}$/;
// A check's or an environment row's value as one line: long hex digests cut to 12 characters.
export const brief = (v) => (v == null ? '–' : typeof v === 'string' ? (HEX.test(v) ? v.slice(0, 12) : v)
  : JSON.stringify(v, (k, x) => (typeof x === 'string' && HEX.test(x) ? x.slice(0, 12) : x)));
// A patch fingerprint ([[id, strength, created, stored], ...]) as the ids, or the base.
const madeOnText = (fp) => (Array.isArray(fp) && fp.length ? fp.map((e) => (Array.isArray(e) ? e[0] : e)).join(', ') : 'the base');

function callsTable(rec) {
  return h('table', { class: 'grid dirs-calls' }, headRow(['function', 'measurement', 'method', 'parameters']),
    (rec.calls || []).map((c) => h('tr', {}, td(c.function), td(c.measurement?.name), td(c.method),
      h('td', {}, h('details', {}, h('summary', {}, 'parameters'), h('pre', { class: 'dirs-output' }, JSON.stringify(c.parameters, null, 1)))))));
}
function measurementLines(rec) {
  const seen = new Set(), out = [];
  for (const c of rec.calls || []) {
    const m = c.measurement;
    if (!m || seen.has(m.name)) continue;
    seen.add(m.name);
    out.push(h('li', {}, `${m.name}: meta ${hex12(m.meta_sha256)}, means ${hex12(m.means_sha256)}, module inputs and outputs ${m.io_sha256 ? hex12(m.io_sha256) : 'none'}, prompts ${brief(m.prompts_sha256)}`));
  }
  return out.length ? h('ul', { class: 'small dirs-measurements' }, out) : null;
}
function checksTable(checks) {
  return h('table', { class: 'grid dirs-checks' }, headRow(['check', 'then', 'now', 'effect']),
    (checks || []).flatMap((c) => {
      const show = (v) => (c.key === 'made_on' ? madeOnText(v) : brief(v));
      return [h('tr', { class: c.same === false ? 'dirs-bad' : '' }, td(c.key), td(show(c.then)), td(show(c.now)),
        td(c.same === null ? `${c.effect} (not known yet)` : c.effect)),
      c.message ? h('tr', {}, h('td', { colspan: 4, class: 'hint' }, c.message)) : null].filter(Boolean);
    }));
}
function environmentTable(rows) {
  if (!rows?.length) return null;
  return h('table', { class: 'grid dirs-environment' }, headRow(['library or setting', 'then', 'now', 'severity']),
    // same null: not known yet (the runner while the model is not loaded), which is not a difference
    rows.map((r) => h('tr', { class: r.same === false ? 'dirs-differs' : '' }, td(r.name), td(brief(r.then)), td(r.same === null ? 'not loaded yet' : brief(r.now)),
      h('td', {}, h('span', { class: r.severity === 'high' || r.severity === 'critical' ? 'pill warn' : 'pill' }, r.severity)))));
}
// GET /api/patches/{pid}/record/heretic, with the Hub repository (and its commit) the person typed, each encoded.
export const hereticHref = (pid, model = '', commit = '') => `/api/patches/${enc(pid)}/record/heretic` +
  (model ? `?model=${enc(model)}${commit ? `&model_commit=${enc(commit)}` : ''}` : '');
const hereticLink = (href) => h('a', { href, download: 'reproduce.json' }, 'Heretic reproduce.json');
// The record as Heretic's reproduce.json: the link when Heretic can run it; when only the Hub repository id is
// missing (`needs_model`), the reasons and two fields, the link appearing once a model id is typed; else the reasons.
function hereticPart(st, hr, pid) {
  if (!hr) return null;
  if (hr.runnable) return h('p', { class: 'small' }, hereticLink(hereticHref(pid)));
  const reasons = h('div', { class: 'small dirs-heretic' }, 'Heretic cannot run this record:', h('ul', {}, (hr.reasons || []).map((t) => h('li', {}, t))));
  if (hr.needs_model !== true) return reasons;
  const hubs = st.output.hub || (st.output.hub = {});
  const hub = hubs[pid] || (hubs[pid] = { model: '', commit: '' });
  const box = h('p', { class: 'small dirs-heretic-link' });
  // repainted as the fields are typed (no re-render: the caret stays where it is)
  const paint = () => {
    const model = hub.model.trim(), commit = hub.commit.trim();
    box.replaceChildren(...(model ? [hereticLink(hereticHref(pid, model, commit))] : []));
  };
  paint();
  return h('div', {}, reasons,
    h('div', { class: 'row' },
      field('Hub model id', h('input', { type: 'text', value: hub.model, 'aria-label': 'Hub model id', oninput: (e) => { hub.model = e.target.value; paint(); } }),
        "the repository this checkpoint came from, for Heretic's settings.model"),
      field('Hub commit', h('input', { type: 'text', value: hub.commit, 'aria-label': 'Hub commit', oninput: (e) => { hub.commit = e.target.value; paint(); } }))),
    box);
}

export function recordView(app, st, pid, again) {
  const o = ensureOutput(st), R = o.records[pid];
  if (!R) return null;
  if (R.loading) return h('p', { class: 'hint' }, 'Reading the record…');
  if (R.error) return h('p', { class: 'hint err', role: 'alert' }, R.error);
  const a = R.answer;
  if (!a?.record) return h('p', { class: 'hint', role: 'note' }, NO_RECORD);
  const rec = a.record, p = patchOf(app, pid), offWhy = busyReason(st);
  const madeOn = (a.checks || []).find((c) => c.key === 'made_on');
  return h('div', { class: 'dirs-record' },
    callsTable(rec), measurementLines(rec),
    h('div', { class: 'small' }, `made on: ${madeOnText(rec.made_on)} · pieces ${hex12(rec.pieces?.digest)}`),
    checksTable(a.checks),
    madeOn?.same === false && p?.enabled === true
      ? h('div', { class: 'row' }, h('button', { disabled: !!offWhy, title: offWhy || null, onclick: () => switchOff(app, st, pid, again) }, `Switch patch ${pid} off`)) : null,
    R.switchError ? h('p', { class: 'hint err', role: 'alert' }, R.switchError) : null,
    environmentTable(a.environment),
    hereticPart(st, a.heretic, pid),
    rerunPart(app, st, pid, rec, again));
}

// ------------------------------------------------------------------------------------------------ the re-run
const largest = (r) => {
  const xs = (r.differences || []).map((d) => d.relative).filter((x) => typeof x === 'number');
  return xs.length ? fmt(Math.max(...xs), 2) : '–';
};
export function verdictText(r) {
  switch (r?.verdict) {
    case 'identical': return `The re-run made the same edit (pieces ${hex12(r.digest)}).`;
    case 'within_rounding': return `The re-run differs from the recorded edit by rounding only (largest relative difference ${largest(r)}, bound ${fmt(r.bound, 2)}).`;
    case 'different': {
      // a tensor without a ratio (changed by one side only, or a recorded change of zero) comes first
      const lone = (r.differences || []).find((d) => d.relative == null);
      if (!lone) return `The re-run made a different edit (largest relative difference ${largest(r)}).`;
      const side = lone.only_in === 'recorded' ? 'is changed by the recorded edit only' : lone.only_in === 'rerun' ? 'is changed by the re-run only' : 'has a recorded change of zero';
      return `The re-run made a different edit (${lone.tensor} ${side}).`;
    }
    default: return '';
  }
}
export function meansText(m) {
  if (m == null) return 'The recorded means are not in the workspace.';
  if (m.equal) return "The new measurement's means equal the recorded ones.";
  const xs = Object.values(m.relative || {}).filter((x) => typeof x === 'number');
  return `The new measurement's means differ by at most ${xs.length ? fmt(Math.max(...xs), 2) : '–'} (relative).`;
}

// The reproduce request (dry run by default). accept_different is true only on a commit after an answer whose verdict
// was `different` showed the box that names it and the person ticked it.
async function rerun(app, st, pid, { dry, measurement }, again) {
  if (busyReason(st)) return;
  const o = st.output, prev = o.reruns[pid];
  const accept = !dry && prev?.answer?.verdict === 'different' && prev.accept === true;
  const body = { measurement, dry_run: dry, message: null, enforce_scope: false, accept_different: accept };
  const run = o.reruns[pid] = { measurement, dry, accept: false, answer: null, error: '', committed: false };
  await runJob(app, st, { kind: 'rerun', label: dry ? `Re-running patch ${pid} (dry run); this cannot be stopped.` : `Committing the re-run of patch ${pid}; this cannot be stopped.` }, async () => {
    let r;
    try { r = await api(`/api/patches/${enc(pid)}/reproduce`, body); } catch (e) { run.error = refusalText(e); return `Not re-run: ${run.error}`; }
    if (app.keep.dirs !== st) return '';
    run.answer = r;
    if (r.committed) {
      run.committed = true;
      await app.weightsChanged();                    // a new patch: the shell reads the list again
      return `Committed the re-run of patch ${pid} as patch ${r.patch}.`;
    }
    if (r.ok === false) return `Not committed: ${r.error}`;
    return verdictText(r);
  });
}

// "Measure again from the record": the record's measure request (GET ...?remeasure=true), sent through the job slot
// with its own stream; once measured, the dry run of the re-run with that measurement.
async function measureAgain(app, st, pid, again) {
  if (busyReason(st)) return;
  const o = st.output, run = o.reruns[pid] = { measurement: null, dry: true, accept: false, answer: null, error: '', committed: false };
  let a;
  try { a = await api(`/api/patches/${enc(pid)}/record?remeasure=true`); } catch (e) { run.error = refusalText(e); again(); return; }
  if (app.keep.dirs !== st) return;
  if (a.record) o.records[pid] = { answer: a };
  const list = a.remeasure || [], reasons = a.remeasure_unavailable || [];
  if (list.length !== 1 || !list[0].body) {
    run.error = reasons.length ? reasons.join(' ') : list.length > 1 ? `patch ${pid} reads ${list.length} measurements; measurement mode takes one` : 'the record has no measurement to make again';
    again();
    return;
  }
  const body = list[0].body;
  let measured = null;
  await runJob(app, st, { kind: 'measure', label: `Measuring '${body.name}'`, name: body.name }, async (stream) => {
    try {
      const out = await api('/api/directions/measure', { ...body, stream });
      if (out.stopped) return `Stopped after ${out.done} of ${out.total} prompts; nothing was saved.`;
      measured = out.name;
      st.listKey = null;
      await loadList(app, st);
      return `Measured '${out.name}' in ${out.seconds} s.`;
    } catch (e) { run.error = refusalText(e); return ''; }
  });
  if (measured && app.keep.dirs === st) await rerun(app, st, pid, { dry: true, measurement: measured }, again);
}

function rerunResult(app, st, pid, run, again) {
  if (run.error) return h('p', { class: 'hint err', role: 'alert' }, run.error);
  const r = run.answer;
  if (!r) return null;
  const busy = busyReason(st), verdict = verdictText(r), notes = Array.isArray(r.notes) ? r.notes : [];
  // the answer's notes (e.g. that the recorded means are missing) under the verdict; the fixed text for missing
  // means only when the answer has no notes
  const meansLine = run.measurement != null && 'means' in r && !(r.means == null && notes.length);
  return h('div', { class: 'dirs-rerun-result' },
    verdict ? h('p', { class: 'small', role: 'status' }, verdict) : null,
    notes.map((n) => h('p', { class: 'hint', role: 'note' }, n)),
    meansLine ? h('p', { class: 'small dirs-means' }, meansText(r.means)) : null,
    applyReport(r),
    r.committed && r.record_error ? h('p', { class: 'hint', role: 'note' }, h('span', { class: 'pill warn' }, `Committed, but its reproduction record could not be written: ${r.record_error}`)) : null,
    !r.committed && r.verdict === 'different' ? h('label', { class: 'small' }, h('input', { type: 'checkbox', checked: run.accept, 'aria-label': 'accept a different edit',
      onchange: (e) => { run.accept = !!e.target.checked; } }), ' commit it anyway') : null,
    !r.committed && r.verdict ? h('div', { class: 'row' }, h('button', { class: 'primary', disabled: !!busy, title: busy || 'commits the re-run as a new patch through the edit path',
      onclick: () => rerun(app, st, pid, { dry: false, measurement: run.measurement }, again) }, 'Commit the re-run')) : null);
}

function rerunPart(app, st, pid, rec, again) {
  const run = st.output.reruns[pid], why = busyReason(st);
  return h('div', { class: 'dirs-rerun' },
    h('div', { class: 'row' },
      h('button', { disabled: !!why, title: why || 'runs the recorded script again on the weights the edit was made on; nothing is committed', onclick: () => rerun(app, st, pid, { dry: true, measurement: null }, again) }, 'Re-run (dry run)'),
      rec.rebuildable ? h('button', { disabled: !!why, title: why || "measures the record's prompts again under a new name, then re-runs the edit with that measurement",
        onclick: () => measureAgain(app, st, pid, again) }, 'Measure again from the record') : null,
      why ? h('span', { class: 'hint' }, why) : null),
    run ? rerunResult(app, st, pid, run, again) : null);
}

// ------------------------------------------------------------------------------------------------ the sub-view
function patchList(app, st, again) {
  const o = st.output, list = [...(patchesOf(app) || [])].reverse();     // newest first
  if (!list.length) return h('p', { class: 'hint' }, 'No patches in this workspace yet.');
  return h('div', { id: 'dirs-output-patches' }, list.map((p) => h('div', { class: 'card' },
    h('div', { class: 'row' },
      h('label', { class: 'small' }, h('input', { type: 'checkbox', checked: ticked(o, p), 'aria-label': `export ${p.id}`,
        onchange: (e) => { o.ticks[p.id] = !!e.target.checked; again(); } }), ` ${p.id}`),
      h('span', {}, p.message ?? ''), h('span', { class: 'small muted' }, p.author ?? ''),
      p.enabled === true ? null : h('span', { class: 'pill' }, 'switched off'),
      p.strength !== 1 ? h('span', { class: 'pill' }, `strength ${p.strength}`) : null,
      h('button', { class: 'ghost', 'aria-label': `record ${p.id}`, onclick: () => showRecord(app, st, p.id, again) }, 'Show record')),
    patchOrigin(p),
    h('div', { id: `dirs-record-${p.id}` }, recordView(app, st, p.id, again)))));
}

export const READING = 'Reading the patches…';
// Sends no request: the patches are the shell's list (the shell reads them when a workspace opens and after every
// change of the weights, then calls onPatchesLoaded, which redraws this view).
export function outputView(el, app, st, again) {
  const o = ensureOutput(st);
  put(el,
    h('p', { class: 'hint' }, "The workspace's patches as a LoRA adapter for PEFT, and each direction edit's reproduction record: what it was made from, what changed since, and its re-run."),
    patchesOf(app) ? patchList(app, st, again) : h('p', { class: 'hint', role: 'status', id: 'dirs-output-reading' }, READING),
    exportForm(app, st, again),
    exportResult(o));
}
