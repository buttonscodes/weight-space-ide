// Inspector selections: the actions row (scale / ablate / revert / console), the breadcrumb across levels,
// a named selection's summary, saving selections, the before/after view of edited weights, and the busy state of
// actions that wait behind a running generation. The server decides what each action does and whether it can be
// done (POST /api/selection/script, the node's "actions"); this module only shows that and applies the script it
// is given through the one edit path (/api/edit, via app.runScript).
import { h, api, fmt, fmtInt, decodeF32, heatmap, legend, tip, esc } from './util.js';

// ---------------------------------------------------------------- is a generation running?
// With a runtime worker, /api/run and /api/chat hold the workspace lock while they generate, so an edit or patch
// change waits for them; without one they do not, and nothing is said about waiting (the server says which, as
// `edit_waits_for_generation` on /api/node and /api/patches). A stream counts as running until it is known to have
// ended: this tab's own /api/run or /api/chat request counts until it settles (however it ends), and another
// client's stream from its first event until its final one (plus a short grace period for the gap between the
// edited and the base model), however long it is silent: reading a long prompt sends no event. Only when the event
// connection dropped after a stream's last event (`eventsLost`), so its final event may have been missed, does the
// stream stop counting after STALE_MS of silence. And once this tab's edit reports progress of its own, it holds the
// lock and is no longer waiting, whatever the streams say.
const FINAL = new Set(['complete', 'interrupted', 'failed', 'stopped']);
const GRACE_MS = 3000;
export const STALE_MS = 30_000;

const streams = (app) => app.generating || (app.generating = new Map());

export function trackGeneration(app, ev, now = Date.now()) {
  if (!ev || ev.type !== 'generation' || !ev.stream) return;
  const g = streams(app);
  const old = g.get(ev.stream);
  g.set(ev.stream, { done: FINAL.has(ev.state), at: now, own: !!(old && old.own && !old.done) });
  for (const [k, v] of g) if (v.done && now - v.at > 60_000) g.delete(k);
}

// This tab is starting its own /api/run or /api/chat request for `body.stream`.
export function startGeneration(app, body, now = Date.now()) {
  const id = body && body.stream;
  if (id) streams(app).set(id, { done: false, at: now, own: true });
}

// This tab's own /api/run or /api/chat request has settled: its stream is over, whatever events were missed.
export function settleGeneration(app, body, now = Date.now()) {
  const g = app.generating;
  const id = body && body.stream;
  if (!g || !id || !g.has(id)) return;
  const v = g.get(id);
  if (!v.done) g.set(id, { done: true, at: now });
}

// The event connection dropped: a stream's final event may be among those missed.
export function eventsLost(app, now = Date.now()) { app.eventsLostAt = now; }

export function generationRunning(app, now = Date.now()) {
  for (const v of (app.generating || new Map()).values()) {
    if (v.done) { if (now - v.at < GRACE_MS) return true; continue; }
    if (v.own) return true;
    const missed = app.eventsLostAt !== undefined && app.eventsLostAt >= v.at;
    if (!missed || now - v.at < STALE_MS) return true;
  }
  return false;
}

// A long edit (thousands of neurons on a quantized model) publishes how far it has got ("edit_progress" events),
// tagged with the id the browser gave its /api/edit request; another client's progress is not this tab's edit.
// `weights`: how many weights the edit writes, when known (a selection edit's cost), for the time left.
export function beginEdit(app, { weights = null, now = Date.now() } = {}) {
  const id = 'edit-' + now.toString(36) + '-' + Math.random().toString(36).slice(2, 10);
  (app.editRequests || (app.editRequests = new Set())).add(id);
  app.editProgress = null;
  (app.editStarts || (app.editStarts = new Map())).set(id, { at: now, weights });
  return id;
}

export function endEdit(app, id) {
  if (app.editRequests) app.editRequests.delete(id);
  if (app.editStarts) app.editStarts.delete(id);
  if (app.editProgress && app.editProgress.request === id) app.editProgress = null;
}

export function trackEditProgress(app, ev, now = Date.now()) {
  if (!ev || ev.type !== 'edit_progress' || !ev.request || !app.editRequests || !app.editRequests.has(ev.request)) return;
  app.editProgress = { request: ev.request, line: ev.line, lines: ev.lines, window: ev.window, windows: ev.windows, stored: ev.stored, at: now };
}

// Time left, worked out from the weights written so far against the edit's total and the time it has taken:
// an estimate (the reading of base weights is not even), shown once a tenth of the edit is done.
function remaining(app, p, now) {
  const start = app.editStarts && app.editStarts.get(p.request);
  if (!start || !start.weights || !p.stored) return '';
  const done = Math.min(1, p.stored / start.weights);
  if (done < 0.1 || done >= 1) return '';
  const left = (now - start.at) * (1 - done) / done / 1000;
  const t = left < 90 ? `${Math.max(1, Math.round(left))} s` : `${Math.round(left / 60)} min`;
  return `${Math.round(done * 100)}% written, about ${t} left`;
}

function progressText(app, now) {
  const p = app.editProgress;
  if (!p || now - p.at > 2000) return '';
  const parts = [];
  if (p.lines > 1) parts.push(`line ${fmtInt(p.line)} of ${fmtInt(p.lines)}`);
  if (p.windows) parts.push(`read ${fmtInt(p.window)} of ${fmtInt(p.windows)}`);
  const left = remaining(app, p, now);
  if (left) parts.push(left);
  return parts.length ? ` (${parts.join(', ')})` : '';
}

export const busyText = (app, now = Date.now()) =>
  (app.editWaitsForGeneration && !app.editProgress && generationRunning(app, now) ? 'waiting for the running generation…' : 'working…' + progressText(app, now));

// Run `work` with `controls` disabled; after `delay` ms say what it is waiting for, and keep that up to date.
export async function busy(app, { controls = [], status, work, delay = 300, onStart }) {
  const before = controls.map((c) => c.disabled);
  controls.forEach((c) => { c.disabled = true; });
  let shown = false;
  const show = () => { shown = true; if (status) { status.className = 'hint busy'; status.textContent = busyText(app); } };
  const timer = setTimeout(show, delay);
  const poll = setInterval(() => { if (shown) show(); }, 500);
  if (onStart) onStart();
  try { return await work(); }
  finally {
    clearTimeout(timer); clearInterval(poll);
    controls.forEach((c, i) => { c.disabled = before[i]; });
    if (status && status.className === 'hint busy') { status.className = 'hint'; status.textContent = ''; }
  }
}

const say = (el, text, bad = false) => { el.className = 'hint' + (bad ? ' err' : ' ok'); el.textContent = text; };

// ---------------------------------------------------------------- actions row
// n: the /api/node answer. Every button's state and reason come from n.actions, worked out on the server from the
// same plan the script is built from, so an enabled button always changes weights.
export function actionsRow(app, n) {
  const acts = n.actions || {};
  if ('edit_waits_for_generation' in n) app.editWaitsForGeneration = !!n.edit_waits_for_generation;
  const unitLike = n.kind !== 'tensor';
  const zeroLabel = unitLike ? 'ablate' : 'zero';          // a plain tensor or slice is zeroed, not "ablated"
  const status = h('div', { class: 'hint', role: 'status' });
  const factor = h('input', { type: 'number', value: String(app.keep.scaleFactor ?? 0.5), step: '0.1', style: { width: '64px' },
    oninput: (e) => { app.keep.scaleFactor = e.target.value; } });
  const controls = [];
  const remembered = app.keep.lastAction;
  // shown again on every redraw for a short while: an edit redraws the inspector more than once
  if (remembered && remembered.addr === n.addr && Date.now() - remembered.at < 15_000) say(status, remembered.text, remembered.bad);

  const run = async (action) => {
    const body = { addr: n.addr, action };
    if (action === 'scale') body.factor = parseFloat(factor.value);
    app.keep.lastAction = null;
    return busy(app, { controls, status, work: async () => {
      let s;
      try { s = await api('/api/selection/script', body); } catch (e) { say(status, e.message, true); return; }
      const what = action === 'scale' ? `scale ${n.addr} by ${body.factor}` : action === 'ablate' ? `${zeroLabel} ${n.addr}` : `revert ${n.addr} to base`;
      // A large edit (more than cost.confirm_above weights) says what it costs and is applied only when confirmed.
      if (s.cost && s.cost.confirm) {
        const ask = app.confirm || globalThis.confirm;
        if (typeof ask !== 'function' || !ask(s.cost.message)) { say(status, 'Not applied: nothing was changed.'); return; }
      }
      const r = await app.runScript(s.script, s.cost ? { message: what, weights: s.cost.weights } : { message: what });
      if (!r) { say(status, 'The edit did not go through; see the console for details.', true); return; }
      if (!r.ok) { say(status, r.error || 'The edit was refused.', true); return; }
      if (!r.committed) { say(status, r.note || r.error || 'Nothing was changed.', true); return; }
      const text = `Done: ${fmtInt(r.n_changed ?? s.weights)} weights changed, as patch ${r.patch}.` + (s.other && s.other.length ? ` Skipped ${s.other.length}: ${s.other[0].reason}.` : '');
      app.keep.lastAction = { addr: n.addr, text, at: Date.now() };        // the inspector redraws after an edit
      say(status, text);
    } });
  };
  const toConsole = () => busy(app, { controls, status, work: async () => {
    const f = parseFloat(factor.value);
    try {
      const s = await api('/api/selection/script', { addr: n.addr, action: 'scale', factor: f === 1 ? 0.5 : f });
      app.setConsole(s.script, true);
      say(status, 'The script is in the console: edit it, then Apply.');
    } catch (e) { say(status, e.message, true); }
  } });
  const button = (key, label, onclick, cls) => {
    const a = acts[key] || { enabled: false, reason: 'not available here' };
    const b = h('button', { class: cls, disabled: !a.enabled, title: a.reason, onclick }, label);
    controls.push(b);
    return b;
  };
  const scale = button('scale', 'scale ×', () => run('scale'));
  controls.push(factor);
  const row = h('div', { class: 'actions' },
    h('div', { class: 'row' },
      button('ablate', zeroLabel, () => run('ablate')),
      scale, factor,
      button('revert', 'revert to base', () => run('revert')),
      button('console', '→ console', toConsole, 'ghost')),
    reasons(acts, zeroLabel), status);
  factor.disabled = !(acts.scale && acts.scale.enabled);
  return row;
}

function reasons(acts, zeroLabel = 'ablate') {
  const s = acts.scale || {};
  const out = [];
  if (!s.enabled && s.reason) out.push(h('p', { class: 'hint err' }, 'Cannot edit: ' + s.reason));
  else if (s.reason) out.push(h('p', { class: 'hint' }, `"${zeroLabel}" ${(acts.ablate || {}).reason || ''}; "scale" ${s.reason}.`));
  if (acts.n_other) out.push(h('p', { class: 'hint' }, `${acts.n_other} part${acts.n_other === 1 ? '' : 's'} left out: ` +
    (acts.other || []).slice(0, 3).map((o) => `${o.member} (${o.reason})`).join('; ') + (acts.n_other > 3 ? '; …' : '')));
  return out;
}

// ---------------------------------------------------------------- breadcrumb
// model › block › mlp/attn › tensor › slice, each part clickable; then shortcuts across levels (a slice that is one
// neuron's weights → that neuron; a neuron → its tensors).
export function breadcrumb(app, n) {
  const crumbs = n.crumbs || [];
  const parts = [];
  crumbs.forEach((c, i) => {
    if (i) parts.push(h('span', { class: 'crumb-sep' }, '›'));
    const last = i === crumbs.length - 1;
    parts.push(last ? h('b', { class: 'crumb here', title: c.addr }, c.label)
      : h('button', { class: 'crumb', title: c.addr, onclick: () => app.select(c.addr) }, c.label));
  });
  const jumps = (n.jumps || []).map((j) => h('button', { class: 'crumb jump', title: j.addr, onclick: () => app.select(j.addr) }, '→ ' + j.label));
  return h('nav', { class: 'crumbs', 'aria-label': 'location' }, parts, jumps.length ? h('span', { class: 'crumb-jumps' }, jumps) : null);
}

// ---------------------------------------------------------------- a selection's size
export function groupSummary(app, n) {
  const s = n.selection;
  if (!s) return null;
  const bits = [];
  if (s.neurons) bits.push(`${fmtInt(s.neurons)} neuron${s.neurons === 1 ? '' : 's'}`);
  if (s.heads) bits.push(`${s.heads} head${s.heads === 1 ? '' : 's'}`);
  if (s.tensors) bits.push(`${s.tensors} tensor slice${s.tensors === 1 ? '' : 's'}`);
  const layers = s.layers || [];
  const box = h('div', { class: 'selection' },
    h('p', {}, h('b', {}, bits.join(' · ') || 'empty'),
      layers.length ? h('span', { class: 'muted small' }, ` in ${layers.length} layer${layers.length === 1 ? '' : 's'} (${layerList(layers)})`) : null,
      s.weights == null ? null : h('span', { class: 'muted small', title: 'every weight of every part (what neurons and heads read and write), each counted once' },
        ` · ${fmtInt(s.weights)} weights in all parts`)));
  const by = Object.entries(s.by_layer || {});
  if (by.length > 1) box.append(h('p', { class: 'small muted' }, 'neurons per layer: ' + by.map(([l, c]) => `L${l} ${fmtInt(c)}`).join(' · ')));
  if (s.members && s.members.length) {
    box.append(h('div', { class: 'members' }, s.members.map((m) => h('button', { class: 'crumb', onclick: () => app.select(m) }, m)),
      s.n_members > s.members.length ? h('span', { class: 'muted small' }, ` and ${fmtInt(s.n_members - s.members.length)} more`) : null));
  }
  if (s.n_other) box.append(h('p', { class: 'hint' }, `${s.n_other} member${s.n_other === 1 ? '' : 's'} not understood: ` + s.other.slice(0, 3).map((o) => `${o.member} (${o.reason})`).join('; ')));
  return box;
}

function layerList(ls) {
  const out = [];
  for (let i = 0; i < ls.length; i++) {
    let j = i;
    while (j + 1 < ls.length && ls[j + 1] === ls[j] + 1) j++;
    out.push(j > i ? `${ls[i]}–${ls[j]}` : String(ls[i]));
    i = j;
  }
  return out.join(', ');
}

// ---------------------------------------------------------------- saving a selection
// The name offered is the suggested one, made unique: never an existing selection, so accepting the default never
// adds to (or puts a selection inside) another one. Typing an existing name still adds to it.
export function defaultName(suggest, existing) {
  const base = String(suggest || 'selection').trim() || 'selection';
  if (!existing.includes(base)) return base;
  let i = 2;
  while (existing.includes(`${base} ${i}`)) i++;
  return `${base} ${i}`;
}

// what: { members } | { neurons: [[layer, neuron]...] } | { selector } (an atlas Find selector, expanded on the server).
// `key` names where the result is shown (a table, a row, a node), so a redraw shows it again (savedStatus).
export async function saveSelection(app, what, { suggest = 'selection', ask = (m, d) => globalThis.prompt(m, d), status, key } = {}) {
  const existing = Object.keys(app.state.groups || {});
  const name = (ask('Save as selection (a new name, or an existing one to add to it):', defaultName(suggest, existing)) || '').trim();
  if (!name) return null;
  const body = { name, ...what, mode: existing.includes(name) ? 'add' : 'replace' };
  try {
    const r = await api('/api/groups', body);
    app.state.groups = r.groups;
    if (app.groupsChanged) app.groupsChanged();
    if (app.refreshStatus) await app.refreshStatus();          // saving changes the workspace identity reads are scoped to
    const s = r.summary || {};
    const text = `${body.mode === 'add' ? 'Added to' : 'Saved as'} selection "${name}": ${fmtInt(s.neurons || 0)} neurons` + (s.heads ? `, ${s.heads} heads` : '') + '.';
    remember(app, key, text, false);
    if (status) say(status, text);
    return r;
  } catch (e) { remember(app, key, e.message, true); if (status) say(status, e.message, true); return null; }
}

function remember(app, key, text, bad) {
  if (!key) return;
  app.keep.saveNotes = { ...(app.keep.saveNotes || {}), [key]: { text, bad, at: Date.now() } };
}

// A status element showing the last save made from `key`, for a few minutes: redraws after a save keep it.
export function savedStatus(app, key, tag = 'span') {
  const el = h(tag, { class: 'hint', role: 'status' });
  const m = (app.keep.saveNotes || {})[key];
  if (m && Date.now() - m.at < 300_000) say(el, m.text, m.bad);
  return el;
}

// A `groups` event (a selection saved or deleted, here or elsewhere): the list and the inspector are redrawn, never
// the view, so a Find query is not run again and ticked rows stay ticked.
export async function groupsEvent(app, { renderGroups, renderInspector }) {
  const s = await api('/api/state');
  app.state = s;
  renderGroups();
  renderInspector();
}

// ---------------------------------------------------------------- before / after
// m: the /api/matrix/compare answer. Base and current share one colour scale; the difference has its own.
export function compareView(app, m, { width = 300, height = 220, onSingle } = {}) {
  const [gr, gc] = m.grid;
  const base = decodeF32(m.base), cur = decodeF32(m.current), delta = decodeF32(m.delta);
  const exact = m.factor[0] === 1 && m.factor[1] === 1;
  const [r0, , c0] = m.region;
  const where = (r, c) => {
    const rr = m.row_offset + r0 + r * m.factor[0], cc = m.col_offset + c0 + c * m.factor[1];
    const span = (x, f) => (f > 1 ? `${x}–${x + f - 1}` : `${x}`);
    return `${m.row_role} ${span(rr, m.factor[0])}, ${m.col_role} ${span(cc, m.factor[1])}`;
  };
  const hover = (r, c, v, e) => {
    if (r == null) return tip(null);
    const i = r * gc + c;
    tip(`${esc(where(r, c))}<br><b>${fmt(base[i], 4)} → ${fmt(cur[i], 4)}</b>${exact ? '' : ' (' + m.pool + ')'}<br>Δ ${fmt(delta[i], 4)}${exact ? '' : ' (largest in cell)'}`, e);
  };
  const w = Math.max(120, Math.min(width, gc * Math.max(1, Math.floor(width / gc))));
  const hh = Math.max(28, Math.min(height, gr * Math.max(1, Math.min(22, Math.floor(height / gr)))));
  const panel = (title, data, scale, note) => h('div', { class: 'cmp-panel' },
    h('div', { class: 'row small' }, h('b', {}, title), h('span', { class: 'muted' }, note), h('span', { class: 'grow' }), legend(scale)),
    heatmap({ rows: gr, cols: gc, data, scale, width: w, height: hh, onHover: hover }));
  const sb = m.stats.base, sc = m.stats.current, sd = m.stats.delta;
  const stats = h('dl', { class: 'kv' },
    h('dt', {}, 'norm'), h('dd', {}, `${fmt(sb.norm, 5)} → ${fmt(sc.norm, 5)}`),
    h('dt', {}, 'range'), h('dd', {}, `${fmt(sb.min, 4)} … ${fmt(sb.max, 4)} → ${fmt(sc.min, 4)} … ${fmt(sc.max, 4)}`),
    h('dt', {}, 'change'), h('dd', {}, `${fmtInt(sd.changed)} of ${fmtInt(sd.n)} weights · |Δ| ${fmt(sd.norm, 4)} · largest ${fmt(sd.max_abs, 4)}`));
  const quant = /^Q|^IQ|^TQ/.test(m.dtype || '') ? h('p', { class: 'hint' }, `Stored as ${m.dtype}: "base" is the reconstructed values the IDE reads and edits; edits are kept as FP32 patches over it.`) : null;
  return h('div', { class: 'compare' },
    h('div', { class: 'row', style: { margin: '6px 0' } }, h('b', {}, 'before and after'),
      h('span', { class: 'muted small' }, 'base and current share one colour scale; the difference has its own'),
      h('span', { class: 'grow' }), onSingle ? h('button', { class: 'ghost', onclick: onSingle }, 'single view') : null),
    h('div', { class: 'cmp-row' },
      panel('base', base, m.scale, ''), panel('current', cur, m.scale, ''),
      panel('difference', delta, m.delta_scale, exact ? '' : 'largest change in each cell')),
    stats, quant);
}

// ---------------------------------------------------------------- patch panel actions
// One patch row's action: the row is disabled while it runs, says when it waits behind a generation, shows a
// failure in the row, and the list is refreshed either way.
export function patchAction(app, row, status, refresh) {
  return async (body) => {
    const controls = (row.querySelectorAll ? [...row.querySelectorAll('button, input')] : []);
    row.classList?.add('busy');
    let failed = null;
    await busy(app, { controls, status, work: async () => {
      try {
        const r = await api('/api/patch', body);
        if (r.missing) app.keep.patchNote = { text: 'That patch was already gone; the list is up to date.', bad: false };
        else if (r.warning) app.keep.patchNote = { text: r.warning, bad: true };     // done, but a revert depended on it
      } catch (e) { failed = e.message; }
    } });
    row.classList?.remove('busy');
    if (failed) { say(status, failed, true); app.keep.patchNote = { text: failed, bad: true }; }
    await refresh();
  };
}

// ---------------------------------------------------------------- what a revert patch cancels
// p: one /api/patches entry. A revert is a compensating patch worked out from the patches as they were; this says
// which, and whether they have changed since (then it cancels an edit that is no longer there).
export function compensatesNote(p, patches) {
  const deps = (p.source && p.source.compensates) || [];
  if (!deps.length) return null;
  const now = new Map(patches.map((x) => [x.id, x]));
  const changed = [];
  const parts = deps.map((d) => {
    const x = now.get(d.id);
    const why = !x ? 'deleted since' : !x.enabled ? 'disabled since' : x.strength !== d.strength ? `now at ×${fmt(x.strength)}` : '';
    if (why) changed.push(`${d.id} (${why})`);
    return `${d.id}${d.message ? ` "${d.message}"` : ''} at ×${fmt(d.strength)}`;
  });
  const shown = parts.slice(0, 3).join(', ') + (parts.length > 3 ? `, and ${parts.length - 3} more` : '');
  return h('div', { class: 'hint' + (changed.length ? ' err' : '') },
    `cancels the edits of ${shown}` + (changed.length ? ` — but ${changed.slice(0, 3).join(', ')}${changed.length > 3 ? ', …' : ''}: this patch now cancels an edit that is no longer there.` : ''));
}
