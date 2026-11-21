// Shell: top bar, explorer tree, inspector, patch stack, weight-script console, live events.
import { h, $, clear, api, q, fmt, fmtInt, toScript, vbars, tokList, tabs, tip, esc, generationStarted, generationSettled } from './util.js';
import { inspection, inspectionScope, calculation } from './inspection.js';
import { settingsView } from './settings.js';
import { exportView } from './export.js';
import { weightsView } from './weights.js';
import { systemsView } from './systems.js';
import { runView, traceView, onGenerationEvent } from './run.js';
import { testsView, chunksView, programmerView, onAgentEvent, programmerMarker, seedProgrammer } from './work.js';
import { selectNode, tabBar, mountView } from './shell.js';
import { atlasView, onAtlasEvent } from './atlas.js';
import { onSteeringEvent, steerControl } from './steer.js';
import { probeView, onProbeSteeringEvent } from './probe.js';
import { directionsView, onDirectionsEvent, DIRECTION_EVENTS, patchOrigin, onPatchesLoaded } from './directions.js';
import { actionsRow, breadcrumb, groupSummary, saveSelection, savedStatus, groupsEvent, patchAction, trackGeneration, trackEditProgress, settleGeneration, startGeneration, eventsLost, beginEdit, endEdit, compensatesNote } from './selection.js';

const VIEWS = { Export: exportView, Weights: weightsView, Atlas: atlasView, Systems: systemsView, Run: runView, Probe: probeView, Directions: directionsView, Trace: traceView, Tests: testsView, Chunks: chunksView, Programmer: programmerView, Settings: settingsView };

export const app = {
  state: null, sel: 'model', variant: 'current', tab: 'Weights', expanded: new Set(['model']), treeCache: new Map(),
  keep: {},                         // per-view state that should survive re-renders (prompts, plans, ...)

  async load() {
    try {
      const capability = await api('/api/inspection-capabilities');
      if (capability.version !== 1 || !capability.passive_metadata) throw new Error('unsupported inspection API');
    } catch (e) {
      clear($('#view')).append(h('p', { class: 'muted' }, 'The updated inspection controls require a server restart. Restart the IDE server before browsing weights. No inspection requests were sent.'));
      return false;
    }
    if (!this.state) clear($('#view')).append(h('p', { class: 'muted' }, 'loading the workspace…'));
    this.state = await api('/api/state');
    renderTop();
    if (!this.state.open) {
      if (this.tab === 'Settings') renderView();
      else clear($('#view')).append(h('p', { class: 'muted' }, 'Open a model to begin, or use Settings to change the runtime settings.'));
      return true;
    }
    await renderTree();
    renderGroups();
    renderView();
    renderInspector();
    renderPatches();
    // Whether a programmer run is working, for the tab bar, whichever tab is shown first (read-only).
    seedProgrammer(this).then(updateTabMarker, () => {});
    return true;
  },
  async refreshStatus() {
    this.state.status = await api('/api/status');
    renderTop();
  },
  async weightsChanged() {          // after any edit / patch change
    ++viewSeq; ++inspectorSeq; ++treeSeq;
    this.treeCache.clear();
    await this.refreshStatus();
    await Promise.all([renderTree(), renderPatches()]);
    renderInspector();
    renderView();
  },
  select(addr, tab) {               // a view that ignores the selection stays as it is (shell.js selectNode)
    selectNode(this, addr, tab, { markSelected, renderTabs, renderView, renderInspector, shownView: () => $('#view').dataset.view });
  },
  setConsole(text, append) {
    const ta = $('#console textarea');
    ta.value = append && ta.value.trim() ? ta.value.replace(/\s*$/, '\n') + text : text;
    ta.focus();
  },
  async runScript(script, { dry = false, message = '', weights = null } = {}) {
    const out = $('#console .out');
    out.className = 'out';
    out.textContent = 'running…';
    const request = beginEdit(this, { weights });         // names this edit's progress events; weights: for the time left
    try {
      const r = await api('/api/edit', { script, message, dry_run: dry, enforce_scope: $('#enforce-scope').checked, request });
      out.textContent = '';
      out.append(report(r));
      if (!r.ok) out.classList.add('err');
      if (r.ok && r.committed) await this.weightsChanged();
      return r;
    } catch (e) { out.classList.add('err'); out.textContent = e.message; return { ok: false, error: e.message }; }
    finally { endEdit(this, request); }
  },
  rerender() { renderView(); },
  runChanged() { updateTabMarker(); },
  groupsChanged() { renderGroups(); },
};
window.app = app;

export function report(r) {
  if (!r.ok) return h('div', {}, 'error: ' + r.error, r.output ? '\n' + r.output : '');
  const parts = [];
  if (r.event) parts.push(h('p',{role:'status'},r.event.message+' '+(r.event.recommendations || []).join(' ')));
  if (r.output) parts.push(h('div', {}, r.output));
  if (!r.changed || !r.changed.length) { parts.push(h('div', { class: 'muted' }, r.committed ? `committed ${r.patch}; detailed change summary unavailable` : r.note || 'no weights changed')); return h('div', {}, parts); }
  parts.push(h('div', {}, h('span', { class: 'pill ' + (r.committed ? 'good' : 'warn') }, r.committed ? `committed ${r.patch}` : 'dry run'),
    ' ', r.changed.map((c) => `${c.param} ${c.changed}/${c.of} (|Δ| ${fmt(c.norm)})`).join('; ')));
  if (r.rome) parts.push(h('div', {}, `rome: layer ${r.rome.layer}, ${r.rome.at} position; p(target) ${fmt(r.rome.p_before)} → ${fmt(r.rome.p_after)}${r.rome.success ? '' : ' (not yet top prediction)'}`));
  if (r.tests) {
    const t = r.tests;
    parts.push(h('div', {}, `tests ${t.suite}: ${t.before[0]}/${t.before[1]} → `, h('b', {}, `${t.after[0]}/${t.after[1]}`),
      t.fixed.length ? `  +${t.fixed.length} fixed` : '', t.broken.length ? h('span', { style: { color: 'var(--crit)' } }, `  −${t.broken.length} broken: ${t.broken.slice(0, 6).join(', ')}`) : ''));
  }
  if (r.drift && r.drift.n) parts.push(h('div', {}, `drift: KL ${r.drift.kl} (max ${r.drift.kl_max}), top-1 agreement ${r.drift.top1_agree}` +
    (r.drift.limit != null ? ` — limit ${r.drift.limit} ${r.drift.within_limit ? 'ok' : 'EXCEEDED'}` : '')));
  return h('div', {}, parts);
}

// ---------------------------------------------------------------- top bar
function renderTop() {
  const s = app.state, o = s.open, st = s.status;
  const top = clear($('#topbar'));
  const openModel = async (name, pull) => {
    if (app.opening) return;
    app.opening = true; ++viewSeq; ++inspectorSeq; ++treeSeq;
    top.append(h('span', { class: 'muted' }, ' opening…'));
    try { await api('/api/open', { name, pull }); } catch (e) { alert(e.message); app.opening = false; await app.load(); return; }
    app.opening = false;
    app.sel = 'model'; app.expanded = new Set(['model']); app.treeCache.clear(); app.keep = {};
    await app.load();
  };
  const sel = h('select', { onchange: async (e) => {
    const name = e.target.value;
    if (name === '@open') {
      const spec = prompt('Open a Hub safetensors repo, local checkpoint directory, local .gguf file, or .wsppatch.zip companion bundle.\n\nGGUF weights are reconstructed quantized values. Inference requires a configured packed CPU/Metal backend and external memory watchdog; opening does not load a runner.');
      if (!spec) { e.target.value = o ? o.name : ''; return; }
      return openModel(spec.trim(), false);
    }
    const opt = s.workspaces.find((w) => w.name === name);
    if (!opt.built && !confirm(`Build ${name}? (${opt.description}) This may download or train for a few minutes.`)) { e.target.value = o ? o.name : ''; return; }
    return openModel(name, false);
  } }, [...s.workspaces.map((w) => h('option', { value: w.name, selected: o && o.name === w.name, title: w.description }, w.name + (w.built ? '' : '  (build)'))),
    h('option', { value: '@open' }, 'open another model…')]);
  top.append(h('span', { class: 'brand' }, 'Weight-space IDE'), sel);
  if (!o) return;
  const c = o.config;
  if (o.capabilities) {
    const cap = o.capabilities;
    top.append(h('span', {class: 'pill warn', title: cap.representation}, 'GGUF · reconstructed weights + FP32 patches'));
    top.append(h('button', {onclick: () => alert(Object.entries(cap).filter(([,v]) => v && typeof v === 'object' && 'supported' in v)
      .map(([k,v]) => `${k}: ${v.supported ? 'supported' : 'unsupported'} — ${v.reason}`).join('\n'))}, 'Capabilities'));
  }
  const stat = (v, label, title) => h('div', { class: 'stat', title }, h('b', {}, v), h('span', {}, label));
  top.append(stat(fmtInt(o.n_params), 'weights', `${o.n_tensors} tensors, ${(o.bytes / 1e9).toFixed(2)} GB on disk` + (o.repo ? `\n${o.repo}` : '')));
  if (o.recognised) {
    top.append(stat(`${c.n_layers} × ${c.n_heads}`, 'layers × heads', c.n_kv_heads && c.n_kv_heads !== c.n_heads ? `keys and values shared in ${c.n_kv_heads} groups` : ''),
      stat(`${c.d_model} / ${c.d_mlp || '–'}`, 'd_model / d_mlp'), stat(fmtInt(c.d_vocab), 'vocab'), stat(o.family, 'family', s.describe));
  } else top.append(h('span', { class: 'pill warn', title: s.describe }, 'architecture not recognised: tensors by name'));
  if (o.remote) {
    top.append(h('span', { class: 'pill warn', title: 'Weights are read from the Hub on demand. Viewing and editing work; running the model needs the files on disk.' }, 'weights on the Hub'),
      h('button', { title: `download ${(o.bytes / 1e9).toFixed(2)} GB so the model can be run`, onclick: async (e) => {
        e.target.disabled = true; e.target.textContent = 'pulling…';
        try { await api('/api/pull', {}); } catch (err) { alert(err.message); }
        app.treeCache.clear(); await app.load();
      } }, 'Pull to run'));
  } else if (!st.can_run) top.append(h('span', { class: 'pill warn', title: st.run_error || 'no configuration that says how to run this checkpoint' }, 'view / edit only'));
  else if (!st.runner_loaded) top.append(h('button', { title: 'The model is loaded into memory the first time something has to be run (a completion, a test, an edit to be tested). Viewing and editing weights does not need it.',
    onclick: async (e) => { e.target.disabled = true; e.target.textContent = 'loading…'; try { await api('/api/load', {}); } catch (err) { alert(err.message); } await app.refreshStatus(); renderView(); } },
    st.suite ? 'load model & run tests' : 'load model'));
  if (st.tests) {
    const t = st.tests, ok = t.passed === t.total;
    top.append(h('div', { class: 'stat clickable', onclick: () => { app.tab = 'Tests'; renderTabs(); renderView(); }, title: `suite ${t.suite}; base model passes ${t.base_passed}` },
      h('b', { style: { color: ok ? 'var(--good)' : 'var(--warn)' } }, `${ok ? '✓' : '△'} ${t.passed}/${t.total}`), h('span', {}, `tests · base ${t.base_passed}`)));
  }
  top.append(stat(st.drift.n ? fmt(st.drift.kl) : '–', 'drift (KL)', 'mean KL from the base model on calibration text'),
    stat(fmtInt(st.changed), 'weights edited', `stored as ${fmtInt(st.stored)} numbers`), stat(st.n_patches, 'patches'));
  top.append(h('div', { class: 'grow' }));
  top.append(h('div', { class: 'field' }, h('span', { class: 'field-label' }, 'showing weights'),
    tabs(['current', 'base', 'delta'], app.variant, (v) => { app.variant = v; renderTop(); renderView(); })));
  top.append(h('a', { href: '/api/export', class: 'small', title: o.kind === 'wsp' ? 'download the edited weights as safetensors' : 'download a patch file with the tensors that changed (python -m wsp export writes the full checkpoint)' }, 'export'),
    h('button', { class: 'ghost', title: 'light / dark', onclick: () => {
      const d = document.documentElement; d.dataset.theme = d.dataset.theme === 'light' ? 'dark' : 'light'; renderView();
    } }, '◐'));
  $('#level-legend').textContent = 'levels 0–6';
}

// ---------------------------------------------------------------- explorer
function markSelected(addr) {
  document.querySelectorAll('.node.sel').forEach((n) => n.classList.remove('sel'));
  const el = document.querySelector(`.node[data-addr="${CSS.escape(addr)}"]`);
  if (el) el.classList.add('sel');
}
async function kids(addr) { return inspection.read(app, '/api/tree', { addr }); }
function nodeRow(n, depth) {
  const open = app.expanded.has(n.addr);
  const row = h('div', { class: 'node' + (app.sel === n.addr ? ' sel' : ''), 'data-addr': n.addr, style: { paddingLeft: 6 + depth * 12 + 'px' },
    title: `${n.addr}\nlevel ${n.level} (${n.kind}) · ${n.n_params} weights${n.norm != null ? ' · |w| ' + fmt(n.norm) : ''}` + (n.changed ? `\n${n.changed} edited, |Δ| ${fmt(n.dnorm)}` : ''),
    onclick: () => app.select(n.addr) },
    h('span', { class: 'twist', onclick: async (e) => {
      e.stopPropagation();
      if (!n.has_children) return;
      if (open) app.expanded.delete(n.addr); else app.expanded.add(n.addr);
      await renderTree();
    } }, n.has_children ? (open ? '▾' : '▸') : ''),
    h('span', { class: 'lvl' }, n.level), h('span', { class: 'name' }, n.label),
    n.changed ? h('span', { class: 'dot', title: `${n.changed} weights edited` }) : null,
    h('span', { class: 'meta' }, n.shape ? n.shape.join('×') : fmtInt(n.n_params)));
  return row;
}
let treeSeq = 0;
async function renderTree() {
  const mine = ++treeSeq, scope = inspectionScope(app);
  const valid = () => mine === treeSeq && scope === inspectionScope(app);
  const root = $('#tree');
  const frag = document.createDocumentFragment();
  async function walk(addr, depth) {
    if (!valid()) return;
    const t = await kids(addr);
    if (!valid()) return;
    for (const c of t.children) {
      frag.append(nodeRow(c, depth));
      if (c.has_children && app.expanded.has(c.addr)) await walk(c.addr, depth + 1);
    }
  }
  const top = await kids('model');
  if (!valid()) return;
  frag.append(nodeRow({ ...top.node, label: app.state.open.name, has_children: true, kind: 'model', level: 0 }, 0));
  if (app.expanded.has('model')) await walk('model', 1);
  if (!valid()) return;
  const y = root.scrollTop;
  clear(root).append(frag);
  root.scrollTop = y;
}
function renderGroups() {
  const g = clear($('#groups'));
  const groups = app.state.groups || {};
  for (const [name, v] of Object.entries(groups)) {
    const shown = v.members.slice(0, 20).join('\n') + (v.members.length > 20 ? `\n… and ${v.members.length - 20} more` : '');
    g.append(h('div', { class: 'node' + (app.sel === 'group:' + name ? ' sel' : ''), 'data-addr': 'group:' + name, title: shown + (v.note ? '\n' + v.note : ''),
      onclick: () => app.select('group:' + name) }, h('span', { class: 'lvl' }, 4), h('span', { class: 'name' }, name),
      h('span', { class: 'meta' }, v.members.length + ' parts'),
      h('button', { class: 'ghost', title: 'delete group', onclick: async (e) => { e.stopPropagation(); app.state.groups = (await api('/api/groups', { name, delete: true })).groups; renderGroups(); } }, '×')));
  }
  if (!Object.keys(groups).length) g.append(h('div', { class: 'muted small' }, 'Named sets of heads, neurons or tensors. Add the selection to a group from the inspector; groups can be viewed, chunked, scoped and scaled as one.'));
}

// ---------------------------------------------------------------- inspector
let inspectorSeq = 0;
async function renderInspector() {
  const mine = ++inspectorSeq, scope = inspectionScope(app);
  const valid = () => mine === inspectorSeq && scope === inspectionScope(app);
  const el = $('#inspector');
  const addr = app.sel;
  let n;
  try { n = await inspection.read(app, '/api/node', { addr }); } catch (e) { if (!valid()) return; clear(el).append(h('div', { class: 'muted' }, e.message)); return; }
  if (!valid() || addr !== app.sel) return;
  const c = n.stats.current, d = n.stats.delta;
  clear(el);
  el.append(breadcrumb(app, n), h('h2', { class: 'mono', style: { fontSize: '13px', overflowWrap: 'anywhere' } }, n.addr),
    h('div', { class: 'row', style: { margin: '4px 0 8px' } }, h('span', { class: 'pill' }, `level ${n.level} · ${n.kind}`),
      n.stats.changed ? h('span', { class: 'pill edit' }, `${fmtInt(n.stats.changed)} edited`) : null));
  el.append(h('dl', { class: 'kv' },
    h('dt', {}, 'weights'), h('dd', {}, fmtInt(n.n_params)),
    n.stats.deferred ? [h('dt', {}, 'statistics'), h('dd', {}, 'available on request')] : [
    h('dt', {}, 'norm'), h('dd', {}, fmt(c.norm) + (n.stats.sampled ? ' (est.)' : '') + (n.stats.changed ? `   Δ ${fmt(d.norm)}` : '')),
    h('dt', {}, 'mean ± std'), h('dd', {}, `${fmt(c.mean)} ± ${fmt(c.std)}`),
    h('dt', {}, 'range'), h('dd', {}, `${fmt(c.min)} … ${fmt(c.max)}`)],
    h('dt', {}, 'as text'), h('dd', { title: 'estimated tokens to hand these weights to the programmer model' },
      `q ${fmtInt(n.est_tokens.q)} · g3 ${fmtInt(n.est_tokens.g3)} · svd ${fmtInt(n.est_tokens.svd)} tok`)));
  calculation(el, app, { path: '/api/node', params: { addr, statistics: 1 }, label: 'Calculate statistics',
    description: 'Reads the selected weights (samples large regions). Remote weights may be fetched.', active: valid, tabBound: false,
    render: (box, data) => { const c = data.stats.current;
      box.append(h('p', {}, `Norm ${fmt(c.norm)} · mean ${fmt(c.mean)} · std ${fmt(c.std)} · range ${fmt(c.min)} … ${fmt(c.max)}`));
      if (data.stats.sampled) box.append(h('p', { class: 'hint' }, `Sampled ${fmtInt(data.stats.sample)} weights`));
      if (c.hist) box.append(vbars(c.hist, { width: 296, height: 46, signedX: true, lim: c.hist_lim, cls: 'bar plain' }));
    } });
  if (n.stats.sampled) el.append(h('p', { class: 'hint' }, `Statistics from a sample of ${fmtInt(n.stats.sample)} weights: nothing this large is read whole.`));
  if (c.hist) el.append(h('h3', {}, 'distribution'), vbars(c.hist, { width: 296, height: 46, signedX: true, lim: c.hist_lim, cls: 'bar plain',
    label: (i, v) => `${v} weights near ${fmt(-c.hist_lim + (i + 0.5) * 2 * c.hist_lim / c.hist.length)}` }));
  // actions: what each does, and whether it can be done, comes from the server (contract C2)
  if (n.kind === 'group' || n.kind === 'unit group' || n.addr.startsWith('group:')) { const g = groupSummary(app, n); if (g) el.append(h('h3', {}, 'selection'), g); }
  const saved = savedStatus(app, 'node:' + n.addr, 'div');
  el.append(h('h3', {}, 'edit'), actionsRow(app, n),
    h('div', { class: 'row', style: { marginTop: '6px' } },
      h('button', { class: 'ghost', title: 'save this as a named selection, or add it to one', onclick: () => saveSelection(app, { members: [n.addr] }, { status: saved, key: 'node:' + n.addr }) }, '+ selection'),
      h('button', { class: 'ghost', onclick: async () => {
        const t = app.state.task; const scope = [...new Set([...(t.scope || []).filter((a) => a !== 'model'), n.addr])];
        app.state.task = (await api('/api/task', { ...t, scope })).task; app.tab = 'Programmer'; renderTabs(); renderView();
      } }, '+ task scope'),
      h('button', { class: 'ghost', onclick: () => { app.keep.chunkSel = n.addr; app.tab = 'Chunks'; renderTabs(); renderView(); } }, 'as text'),
      steerControl(app, n, saved)), saved);
  if (n.direction) {
    el.append(h('h3', {}, 'as a residual direction'), h('div', { class: 'small muted' }, 'promotes'), tokList(n.direction.promotes.top, 'up'),
      h('div', { class: 'small muted', style: { marginTop: '4px' } }, 'suppresses'), tokList(n.direction.promotes.bottom, 'down'),
      h('div', { class: 'small muted', style: { marginTop: '4px' } }, 'token embeddings along it'), tokList(n.direction.embeddings.top));
  }
  if (n.neuron) {
    const k = n.neuron;
    el.append(h('h3', {}, 'neuron' + (k.gated ? ' (gated)' : '')), h('dl', { class: 'kv' },
      k.bias != null ? [h('dt', {}, 'bias'), h('dd', {}, fmt(k.bias))] : null,
      h('dt', {}, 'read · write'), h('dd', {}, `cos ${fmt(k.cos_in_out)} (${k.mechanism})`)));
    k.reads.forEach((r) => el.append(h('div', { class: 'small muted', style: { marginTop: '4px' } }, `${r.kind} read: embeddings along it`), tokList(r.embeddings.top)));
    el.append(h('div', { class: 'small muted', style: { marginTop: '4px' } }, 'promotes'), tokList(k.writes.promotes.top, 'up'),
      h('div', { class: 'small muted', style: { marginTop: '4px' } }, 'suppresses'), tokList(k.writes.promotes.bottom, 'down'));
  }
  if (n.slices.length > 1) {
    el.append(h('h3', {}, `tensors (${n.n_slices})`));
    n.slices.slice(0, 24).forEach((s) => el.append(h('div', { class: 'node', title: s.doc, onclick: () => app.select(s.addr) },
      h('span', { class: 'name mono' }, s.addr), h('span', { class: 'meta' }, s.shape.join('×')))));
  } else if (n.slices.length === 1) el.append(h('p', { class: 'hint', style: { marginTop: '10px' } }, n.slices[0].doc));
}

// ---------------------------------------------------------------- patches
async function renderPatches() {
  const el = $('#patches');
  let patches, waits;
  try { ({ patches, edit_waits_for_generation: waits } = await api('/api/patches')); } catch (e) { clear(el).append(h('div', { class: 'hint err' }, e.message)); return; }
  if (waits !== undefined) app.editWaitsForGeneration = !!waits;
  app.keep.patchList = patches;                  // the Directions tab's Output view shows this list (it reads none)
  onPatchesLoaded(app);
  clear(el);
  if (app.keep.patchNote) { el.append(h('div', { class: 'hint' + (app.keep.patchNote.bad ? ' err' : ''), role: 'status' }, app.keep.patchNote.text)); app.keep.patchNote = null; }
  if (!patches.length) { el.append(h('div', { class: 'muted small' }, 'No patches: these are the base weights. Every edit — yours or the programmer model\'s — lands here as an additive delta that can be switched off, scaled, negated or deleted.')); return; }
  for (const p of [...patches].reverse()) {
    const row = h('div', { class: 'patch' + (p.enabled ? '' : ' off') }), status = h('div', { class: 'hint', role: 'status' });
    const act = patchAction(app, row, status, () => app.weightsChanged());
    const detail = h('pre', { style: { display: 'none' } },
      (p.source.script || JSON.stringify(p.source, null, 1)) + '\n\n' + p.stats.map((s) => `${s.param}: ${s.changed} changed, |Δ| ${fmt(s.norm)}${s.rank != null ? ', rank ' + s.rank : ''}, stored as ${s.kinds.join('+')} (${s.stored} numbers)`).join('\n'));
    const strength = h('input', { type: 'range', min: -1, max: 2, step: 0.05, value: p.strength, title: 'strength: scales this patch\'s delta (−1 negates it)',
      onchange: (e) => act({ id: p.id, action: 'set', strength: parseFloat(e.target.value) }) });
    el.append(row);
    row.append(
      h('div', { class: 'p-head' },
        h('input', { type: 'checkbox', checked: p.enabled, title: 'enabled', onchange: (e) => act({ id: p.id, action: 'set', enabled: e.target.checked }) }),
        h('span', { class: 'p-msg', title: p.message }, p.message),
        h('button', { class: 'ghost', title: 'source and statistics', onclick: () => { detail.style.display = detail.style.display === 'none' ? 'block' : 'none'; } }, '⋯'),
        h('button', { class: 'ghost danger', title: 'delete patch', onclick: () => act({ id: p.id, action: 'delete' }) }, '×')),
      h('div', { class: 'row small muted' }, h('span', { class: 'pill' + (p.author === 'programmer' ? ' edit' : '') }, p.author), p.source.kind,
        `· ${fmtInt(p.changed)} weights${p.stored < p.changed ? ' as ' + fmtInt(p.stored) + ' numbers' : ''}`, p.info && p.info.tests ? `· tests ${p.info.tests[0]}/${p.info.tests[1]}` : '', h('span', { class: 'grow' }), `×${fmt(p.strength)}`),
      ...[compensatesNote(p, patches), patchOrigin(p)].filter(Boolean), strength, status, detail);
  }
}

// ---------------------------------------------------------------- tabs, view, console
// The Programmer button says, from every tab, what its conversation is doing; updating it redraws only the tab bar.
const tabsBar = tabBar(app, Object.keys(VIEWS), { host: () => $('#tabs'), marker: () => programmerMarker(app.keep.prog),
  renderView: () => renderView(), scrollToTop: () => { $('#view').scrollTop = 0; } });
function renderTabs() { tabsBar.render(); }
function updateTabMarker() { tabsBar.update(); }
let viewSeq = 0;
async function renderView() {
  const el = $('#view');
  $('#shell').classList.toggle('settings-layout', app.tab === 'Settings');
  if (!app.state) return;
  if (!app.state.open && app.tab !== 'Settings') { ++viewSeq; clear(el).append(h('p', {class:'muted'}, 'Open a model to use this view.')); el.dataset.view = ''; return; }
  const mine = ++viewSeq, scope = inspectionScope(app);
  const active = () => mine === viewSeq && scope === inspectionScope(app);
  const box = h('div', { class: app.tab === 'Settings' ? 'settings-page' : '' });
  const busy = setTimeout(() => { if (mine === viewSeq) el.classList.add('busy'); }, 250);
  try { await VIEWS[app.tab](box, app, active); } catch (e) { box.append(h('p', { style: { color: 'var(--crit)' } }, e.message)); console.error(e); }
  clearTimeout(busy);
  if (!active()) return;
  el.classList.remove('busy');
  mountView(el, box, app.tab, app.sel);          // mounted, afterMount, then the scroll position (shell.js)
  tip(null);
  if (app.state.status?.runner_loaded === false && ['Run', 'Trace', 'Tests'].includes(app.tab)) app.refreshStatus();   // that view loaded the model
}
function renderConsole() {
  const el = clear($('#console'));
  const ta = h('textarea', { spellcheck: false, placeholder: 'weight script — e.g.\nblocks[0].mlp.W_out[12] += 2 * unit(U("Paris"))        # see the function reference →' });
  ta.addEventListener('keydown', (e) => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') app.runScript(ta.value, { dry: e.shiftKey }); });
  el.append(h('div', { class: 'row', style: { marginBottom: '4px' } },
    h('b', { class: 'small' }, 'Weight script'), h('span', { class: 'muted small' }, 'Ctrl+Enter apply · Ctrl+Shift+Enter dry run'), h('span', { class: 'grow' }),
    h('label', { class: 'small muted', title: 'reject edits outside the task scope (always on for the programmer model)' }, h('input', { type: 'checkbox', id: 'enforce-scope' }), ' enforce task scope'),
    h('button', { class: 'ghost', onclick: async () => {
      const b = await api('/api/brief'); const i = b.indexOf('## Weight scripts');
      const out = $('#console .out'); out.className = 'out'; out.textContent = b.slice(i);
    } }, 'reference'),
    h('button', { onclick: () => app.runScript(ta.value, { dry: true }) }, 'Dry run'),
    h('button', { class: 'primary', onclick: () => app.runScript(ta.value) }, 'Apply')), ta, h('div', { class: 'out' }));
}

// ---------------------------------------------------------------- live events (edits by the programmer model show up as they happen)
function listen() {
  const es = new EventSource('/api/events');
  let pending = null;
  es.onerror = () => eventsLost(app);                     // events may be missed until it reconnects
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    if (ev.type === 'wiring_progress') window.dispatchEvent(new CustomEvent('wsp-wiring-progress',{detail:ev}));
    else if (ev.type === 'edit_progress') trackEditProgress(app, ev);
    else if (ev.type === 'patch') { clearTimeout(pending); pending = setTimeout(() => app.weightsChanged(), 150); }
    else if (ev.type === 'groups') groupsEvent(app, { renderGroups, renderInspector }).catch((e) => console.error(e));
    else if (ev.type === 'task') api('/api/state').then((s) => { app.state = s; renderGroups(); if (app.tab === 'Programmer' && ev.type === 'task' && !app.keep.editingTask) renderView(); });
    else if (ev.type === 'generation') { trackGeneration(app, ev); onGenerationEvent(ev, app); }
    else if (ev.type === 'atlas') onAtlasEvent(ev, app);
    else if (ev.type === 'steering') { onSteeringEvent(ev, app); onProbeSteeringEvent(ev, app); }
    else if (ev.type === 'agent' || ev.type === 'run' || ev.type === 'tool' || ev.type === 'finish') onAgentEvent(ev, app);
    else if (DIRECTION_EVENTS.has(ev.type)) onDirectionsEvent(ev, app);
  };
}

generationStarted.add((path, body) => startGeneration(app, body));    // this tab's own run/chat has begun
generationSettled.add((path, body) => settleGeneration(app, body));   // this tab's own run/chat has ended
renderTabs();
renderConsole();
app.load().then((ready) => { if (ready) listen(); });
