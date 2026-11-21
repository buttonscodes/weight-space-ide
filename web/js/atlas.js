// Atlas view: the map of the model's units. Build it once, then use it to decide what to edit:
// stages and layers, a grid of every neuron in a layer, one neuron's card, token concepts,
// a finder that turns "the neurons that carry X" into a weight script, and attribution for a prompt.
import { h, api, q, fmt, fmtInt, tabs, tip, esc } from './util.js';
import { inspection, inspectionScope } from './inspection.js';
import { saveSelection, savedStatus } from './selection.js';

const MECH_COL = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--muted)'];
const ROLE_COL = ['var(--surface-3)', 'var(--muted)', 'var(--s1)', 'var(--s2)', 'var(--s3)'];
const TRIG = ['context only', 'current token', 'predicted token', 'both'];
const TRIG_COL = ['var(--surface-3)', 'var(--s1)', 'var(--s2)', 'var(--s3)'];
const SIGN = ['both signs', 'positive', 'negative'];
const SIGN_COL = ['var(--surface-3)', 'var(--s1)', 'var(--s2)'];
const VERDICT = { 'holds': 'good', 'holds weakly': 'warn', 'fails': 'bad', 'too few to tell': '' };
const MARK = { 'holds': '✓ ', 'holds weakly': '≈ ', 'fails': '✗ ', 'too few to tell': '' };
const show = (s) => (s === '' ? '∅' : String(s).replace(/\n/g, '⏎').replace(/ /g, '·').replace(/\t/g, '⇥'));
const toks = (xs, cls = '') => h('div', { class: 'toks ' + cls }, xs.map((t) => typeof t === 'string'
  ? h('span', { class: 'tok' }, show(t))
  : h('span', { class: 'tok' + (t.outlier ? '' : ' faint'), title: `z ${t.z} · id ${t.id}` + (t.outlier ? '' : ' · no stronger than a random direction') }, show(t.token))));
const pct = (x, d = 0) => (x == null ? '–' : (100 * x).toFixed(d) + '%');
const legend = (names, cols) => h('div', { class: 'row small', style: { gap: '12px' } }, names.map((n, i) => h('span', {}, h('i', { class: 'swatch', style: { background: cols[i] } }), ' ' + n)));

export async function atlasView(el, app) {
  const st = app.keep.atlas || (app.keep.atlas = { tab: 'Overview', layer: null, colour: 'mechanism', unit: null, find: { word: '', scale: 0 }, attr: { prompt: '', target: '', versus: '', k: 5, scale: 0 } });
  const scope = inspectionScope(app), view = app._atlasView = {};
  const a = await api('/api/atlas');
  if (scope !== inspectionScope(app) || app.tab !== 'Atlas' || app._atlasView !== view) return;
  st.info = a;
  el.append(h('h2', {}, 'Atlas'), h('p', { class: 'muted' }, 'A map of every unit of the model, built from its weights and, where the model can be run, checked against a forward pass. Consult it before editing.'));
  if (a.job && a.job.running) { el.append(buildLog(a.job)); return; }
  // A build this browser did not start (an agent's MCP call, the command line, another process): no Build or
  // Continue here, which would only be refused.  A build of ours that failed: its error, wherever it ended.
  const elsewhere = !!a.build_running;
  if (a.job && !a.job.running && a.job.error) el.append(h('p', { class: 'card', style: { color: 'var(--crit)' } }, 'The last atlas build failed: ' + a.job.error));
  if (!a.built) { el.append(elsewhere ? runningCard(a, app) : buildPanel(a, app)); return; }
  el.append(h('div', { class: 'row', style: { margin: '8px 0 12px' } }, tabs(['Overview', 'Layers', 'Find', 'Concepts', 'Attribute'], st.tab, (t) => { st.tab = t; app.rerender(); }),
    h('span', { class: 'grow' }), a.stale_layers.length ? h('span', { class: 'pill warn', title: 'the atlas describes the weights it was built from' }, `${a.stale_layers.length} layer(s) edited since`) : null));
  const body = h('div', {});
  el.append(body);
  await ({ Overview: overview, Layers: layers, Find: find, Concepts: concepts, Attribute: attribute })[st.tab](body, a, st, app);
}

// ---------------------------------------------------------------- build
// 'about 1 h 50 min', 'about 12 min', 'about 40 s' (the same rules as wsp/atlas/wiring.py: duration).
export function duration(s) {
  if (s == null) return 'an unknown time';
  if (s < 1) return 'under a second';
  if (s < 90) return `about ${Math.round(s)} s`;
  if (Math.round(s / 60) < 60) return `about ${Math.round(s / 60)} min`;
  const m = Math.round(s / 300) * 5, hours = Math.floor(m / 60), mins = m % 60;
  return `about ${hours} h` + (mins ? ` ${mins} min` : '');
}
const gb = (b) => `${(b / 1e9).toFixed(b >= 1e10 ? 0 : 1)} GB`;
const list = (xs) => (xs.length > 1 ? xs.slice(0, -1).join(', ') + ' and ' + xs.at(-1) : xs[0]);
// What the wiring box says: the estimate (appended to the label), then one line on memory and waiting.
// w: the weights stage's rebuild (a continue of edited weights), its time counted when known.
export function estimateLabel(e, w = null) {
  let t = e.read_known ? e.total_s : e.search_s;
  const known = [], unknown = [];
  if (e.read_known) known.push('reading the weights'); else unknown.push(`reading the weights, not yet known: ${e.read_note}`);
  if (w) { if (w.seconds != null) { t += w.seconds; known.push('rebuilding the weights stage'); } else unknown.push('rebuilding the weights stage, time not known'); }
  return ` (estimate: ${duration(t)}` + (known.length ? `, including ${known.join(' and ')}` : '') + unknown.map((u) => `, plus ${u}`).join('') + ')';
}
export function estimateNote(e) {
  return `While it runs it holds about ${gb(e.memory_bytes)} of memory, and ${list(e.waits)} wait until it finishes.`;
}
export function continuingNote(c) {
  return c.next_layer == null ? `An earlier wiring build finished all ${c.total} layers but was not completed; this completes it.`
    : `An earlier wiring build stopped after ${c.done} of ${c.total} layers; this continues from layer ${c.next_layer}.`;
}
export function progressText(p) {
  if (!p || p.stage !== 'wiring') return '';
  const from = p.resumed_from != null ? ` (continued from layer ${p.resumed_from})` : '';
  if (p.phase === 'reading') return `Wiring: reading the weights of layer ${p.layer} (${p.read} of ${p.read_total})${from}`;
  if (p.phase === 'finishing') return `Wiring: all ${p.total} layers done; saving`;
  return `Wiring: ${p.done} of ${p.total} layers done` + (p.remaining_s != null ? `, ${duration(p.remaining_s)} left` : '') + from;
}
// A build of this atlas started elsewhere: what is known of it, and what waits.
export function runningText(a) {
  const who = a.build_running === 'this IDE' ? 'in this IDE, started by an agent or another window' : 'in another process (for example the command line)';
  const p = a.wiring_progress, sofar = p && p.total ? ` Its wiring stage has ${p.done} of ${p.total} layers done so far.` : '';
  return `A build of this atlas is running ${who}.${sofar} Run, Chat, edits and inspector reads wait until it finishes; building here is not offered until then.`;
}
function runningCard(a, app) {
  return h('div', { class: 'card', style: { marginTop: '16px', maxWidth: '640px' } }, h('p', { class: 'small', style: { margin: 0 } }, runningText(a)),
    h('button', { class: 'ghost', onclick: () => app.rerender() }, 'Check again'));
}
function buildLog(job) {
  return h('div', { class: 'card' }, h('b', {}, 'Building the atlas…'), h('p', { class: 'small', id: 'atlas-progress', style: { margin: '6px 0 0' } }, progressText(job.progress)),
    h('pre', { class: 'small', id: 'atlas-log', style: { maxHeight: '260px', overflow: 'auto', margin: '6px 0 0' } }, job.log.slice(-40).join('\n')));
}
// What a build of the edited weights does first: rebuild the weights stage, which starts a new atlas record.
export function weightsRebuildNote(w) {
  if (!w) return '';
  return 'It first rebuilds the weights stage for the edited weights (' + (w.seconds != null ? `${duration(w.seconds)} last time` : 'time not known')
    + (w.drops_observed ? '), and the observed stage then has to be built again.' : ').');
}
// Fills a wiring estimate into `label` (appended to the box's text) and `note` (memory, waiting, continuing).
// withWeights: the build also rebuilds the weights stage (the continue card for edited weights), so its time counts.
async function wiringEstimate(label, note, variant, restart, withWeights) {
  label.replaceChildren(' (estimating the time…)'); note.replaceChildren();
  try {
    const e = await api('/api/atlas/estimate?' + q({ variant, restart: restart ? 1 : 0 }));
    if (!e || e.search_s == null || !e.waits) throw new Error('no estimate came back');
    const w = withWeights ? e.weights_rebuild : null;
    label.replaceChildren(estimateLabel(e, w));
    note.replaceChildren(estimateNote(e) + (e.continuing ? ' ' + continuingNote(e.continuing) : '') + (w ? ' ' + weightsRebuildNote(w) : ''));
  } catch (error) { label.replaceChildren(` (no time estimate: ${error.message})`); }
}
function buildPanel(a, app, again) {
  if (!a.can_build) return h('p', { class: 'muted' }, 'This checkpoint\'s architecture was not recognised, so it has no neurons to map. Its tensors are still available by name.');
  const stages = { weights: true, wiring: false, observed: false };
  // The server says why the observed stage cannot be built (a model that cannot be run); the box is then disabled and
  // never sent. For a model that runs in the model worker it says instead what building the stage involves.
  const noObserved = a.observed_unavailable
    || (a.can_observe ? null : 'This model cannot be run here, so the map will be weights-only and firing signs will be predictions.');
  const scope = inspectionScope(app);
  let busy = false;
  const box = (k, label, hint, disabled) => h('label', { class: 'small', title: hint, style: { display: 'block', opacity: disabled ? 0.5 : 1 } },
    h('input', { type: 'checkbox', checked: stages[k] && !disabled, disabled: disabled || k === 'weights', onchange: (e) => { stages[k] = e.target.checked; } }), ' ', label);
  const estimate = h('span', {}), estNote = h('p', { class: 'small muted', style: { margin: '0 0 4px 20px' } });
  const variant = h('select', { onchange: () => wiringEstimate(estimate, estNote, variant.value, again) }, h('option', { value: 'base' }, 'base weights'), h('option', { value: 'current' }, 'current weights (with patches)'));
  wiringEstimate(estimate, estNote, variant.value || 'base', again);
  return h('div', { class: 'card', style: { maxWidth: '640px' } }, h('b', {}, again ? 'Rebuild' : `No atlas yet for ${a.model}`),
    h('p', { class: 'small muted' }, `${fmtInt(a.neurons)} neurons to map.` + (a.fetch_bytes ? ` The weights are still on the Hub: building reads ${(a.fetch_bytes / 1e9).toFixed(1)} GB of them into the cache.` : '')),
    box('weights', 'Weights: what every neuron reads and writes, classes, token concepts', 'needs no forward pass; works for models too large to run'),
    box('wiring', ['Wiring: which neurons read what which other neurons write', estimate], 'pairwise search between layers; saved after every layer, so a stopped build continues where it stopped'), estNote,
    box('observed', 'Observed: run probe text, record how neurons fire, test the map', 'needs a model that can be run', !!noObserved),
    noObserved ? h('p', { class: 'small muted', id: 'atlas-observed-unavailable' }, noObserved)
      : a.observed_note ? h('p', { class: 'small muted', id: 'atlas-observed-note', style: { margin: '0 0 4px 20px' } }, a.observed_note) : null,
    h('div', { class: 'row', style: { marginTop: '8px' } }, h('span', { class: 'small muted' }, 'map the'), variant,
      h('button', { class: 'primary', onclick: async (event) => {
        if (busy || scope !== inspectionScope(app)) return;
        busy = true; event.target.disabled = true;
        try {
          await api('/api/atlas/build', { stages: Object.keys(stages).filter((k) => stages[k] && !(k === 'observed' && noObserved)), fetch: true, force: !!again, variant: variant.value, workspace_id: app.state.status.workspace_id, version: app.state.status.version });
          if (scope === inspectionScope(app) && app.tab === 'Atlas') app.rerender();
        } catch (error) { event.target.textContent = 'Retry build: ' + error.message; }
        finally { busy = false; event.target.disabled = false; }
      } }, again ? 'Rebuild' : 'Build atlas')));
}

// ---------------------------------------------------------------- overview
async function overview(el, a, st, app) {
  const s = a.stages, v = a.validation;
  el.append(h('p', {}, `${fmtInt(s.weights.neurons)} neurons in ${s.weights.layers.length} layers. Built: ${Object.keys(s).join(', ')}. It describes the ${a.variant} weights.`));
  el.append(h('h3', {}, 'Stages of the stack'), h('table', { class: 'grid' }, h('tbody', {}, a.layers.stages.map((x) => h('tr', { class: 'clickable', onclick: () => { st.tab = 'Layers'; st.layer = x.from; app.rerender(); } },
    h('td', { class: 'mono', style: { whiteSpace: 'nowrap' } }, x.from === x.to ? `L${x.from}` : `L${x.from}–L${x.to}`), h('td', {}, x.desc))))));
  el.append(h('h3', {}, 'How well the map held up'));
  if (v) {
    el.append(h('p', { class: 'small muted' }, `Forward pass over ${fmtInt(v.tokens)} tokens of ${v.source}; next-token loss ${fmt(v.loss)} nats. Verdicts follow fixed rules, not judgement.`));
    el.append(h('div', {}, v.checks.map((c) => h('div', { class: 'check' }, h('div', { class: 'row' }, h('b', {}, c.claim), h('span', { class: 'pill ' + VERDICT[c.verdict] }, MARK[c.verdict] + c.verdict)), h('p', { class: 'small' }, c.text)))));
    el.append(h('p', { class: 'small muted' }, `${pct(v.peak_signed_share)} of neurons fire hard with one clear sign; the rest use both. Read a neuron's direction from its strong activations, not from its average.`));
  } else el.append(h('p', { class: 'muted' }, 'Not checked: the observed stage has not run, so firing signs are predictions from weights.'));
  if (a.shift) el.append(h('h3', {}, 'Shifting a token everywhere'), h('p', { class: 'small' }, a.shift.note + '.'), h('p', { class: 'small muted mono' }, 'shift_tokens(word_tokens("red"), 4)'));
  if (a.residual && a.residual.sinks.length) el.append(h('h3', {}, 'Residual stream'), h('p', { class: 'small' }, `Coordinate${a.residual.sinks.length > 1 ? 's' : ''} ${a.residual.sinks.join(', ')} ${a.residual.sinks.length > 1 ? 'are' : 'is'} written far more than read (a sink).`));
  if (a.build_running) { el.append(runningCard(a, app)); return; }
  if (a.wiring_progress && !a.wiring_built) el.append(continueWiring(a, app));
  if (!s.observed && a.can_observe && !a.observed_unavailable) el.append(addObserved(a, app));
  const again = h('div', { style: { marginTop: '16px' } });
  again.append(h('button', { class: 'ghost', onclick: () => { again.replaceChildren(buildPanel(a, app, true)); } }, 'Rebuild…'));
  el.append(again);
}

// An unfinished wiring stage: continue it without rebuilding anything else (an edited-weights atlas rebuilds its
// weights stage first, as every edited-weights build does).  Whether it can be continued is the server's decision
// (wiring.assess), the same one the build makes: progress it cannot use is described with the build's reason.
function continueWiring(a, app) {
  const p = a.wiring_progress, scope = inspectionScope(app), variant = p.variant || 'base';
  const label = h('span', {}), note = h('p', { class: 'small muted', style: { margin: '4px 0 0' } });
  let busy = false;
  const button = h('button', { class: 'primary', onclick: async () => {
    if (busy || scope !== inspectionScope(app)) return;
    busy = true; button.disabled = true;
    try {
      await api('/api/atlas/build', { stages: variant === 'base' ? ['wiring'] : ['weights', 'wiring'], fetch: true, force: false, variant, workspace_id: app.state.status.workspace_id, version: app.state.status.version });
      if (scope === inspectionScope(app) && app.tab === 'Atlas') app.rerender();
    } catch (error) { button.textContent = 'Retry: ' + error.message; }
    finally { busy = false; button.disabled = false; }
  } }, p.usable ? 'Continue the wiring stage' : 'Build the wiring stage again');
  const text = p.usable ? continuingNote({ done: p.done, total: p.total, next_layer: p.next_layer })
    : `An earlier wiring build was stopped, but its saved progress cannot be used: ${p.reason}. Building the stage again starts over.`;
  wiringEstimate(label, note, variant, false, !!p.rebuilds_weights);
  return h('div', { class: 'card', style: { marginTop: '16px', maxWidth: '640px' } }, h('p', { class: 'small', style: { margin: 0 } }, text, label), note, button);
}

// A built atlas without the observed stage: add it alone, for the weights the atlas maps, without rebuilding the
// weights stage (the server refuses if those weights have changed since).
function addObserved(a, app) {
  const scope = inspectionScope(app), variant = a.variant || 'base';
  const card = (...kids) => h('div', { class: 'card', id: 'atlas-add-observed', style: { marginTop: '16px', maxWidth: '640px' } }, ...kids);
  if (variant !== 'base' && a.stale_layers.length) {
    return card(h('p', { class: 'small', style: { margin: 0 } }, 'The observed stage has not been built. The edited weights this atlas maps have changed since it was built, so adding it needs a rebuild (Rebuild… below).'));
  }
  let busy = false;
  const button = h('button', { class: 'primary', onclick: async () => {
    if (busy || scope !== inspectionScope(app)) return;
    busy = true; button.disabled = true;
    try {
      await api('/api/atlas/build', { stages: ['observed'], fetch: true, force: false, variant, workspace_id: app.state.status.workspace_id, version: app.state.status.version });
      if (scope === inspectionScope(app) && app.tab === 'Atlas') app.rerender();
    } catch (error) { button.textContent = 'Retry: ' + error.message; }
    finally { busy = false; button.disabled = false; }
  } }, 'Add the observed stage');
  return card(h('p', { class: 'small', style: { margin: 0 } }, 'The observed stage has not been built, so firing signs are predictions from weights. Adding it runs probe text through the model and checks the map against it; the stages already built are kept as they are.'),
    a.observed_note ? h('p', { class: 'small muted', id: 'atlas-observed-note', style: { margin: '4px 0 0' } }, a.observed_note) : null, button);
}

// ---------------------------------------------------------------- layers, neuron grid, unit card
async function layers(el, a, st, app) {
  const L = a.layers.layers;
  if (st.layer == null || !L.some((x) => x.layer === st.layer)) st.layer = L[0].layer;
  const bar = (x) => h('div', { class: 'stackbar', title: a.mech.map((m, i) => `${m} ${pct(x.mech[m] / x.neurons, 1)}`).join(' · ') }, a.mech.map((m, i) => h('i', { style: { flex: x.mech[m], background: MECH_COL[i] } })));
  const meter = (v) => h('div', { class: 'row', style: { gap: '6px', flexWrap: 'nowrap' } }, h('div', { class: 'meter' }, h('i', { style: { width: Math.min(100, 100 * (v || 0)) + '%' } })), h('span', { class: 'small', style: { minWidth: '36px' } }, pct(v, 1)));
  const table = h('table', { class: 'grid' }, h('thead', {}, h('tr', {}, ['Layer', 'Mechanism', 'Vocab-shaped', 'Token trigger', 'Predicted trigger'].map((t) => h('th', {}, t)))),
    h('tbody', {}, L.map((x) => h('tr', { class: 'clickable' + (x.layer === st.layer ? ' on' : ''), onclick: () => { st.layer = x.layer; st.unit = null; app.rerender(); } },
      h('td', { class: 'mono' }, 'L' + x.layer + (a.stale_layers.includes(x.layer) ? ' •' : '')), h('td', { style: { minWidth: '140px' } }, bar(x)), h('td', {}, meter(x.vocab)), h('td', {}, meter(x.trig_E)), h('td', {}, meter(x.trig_U))))));
  const left = h('div', { style: { flex: '1 1 380px', minWidth: 0, maxHeight: '520px', overflow: 'auto' } }, legend(a.mech, MECH_COL), table);
  const right = h('div', { style: { flex: '1 1 420px', minWidth: 0 } });
  el.append(h('div', { class: 'row', style: { alignItems: 'flex-start', gap: '18px' } }, left, right));
  const d = await api('/api/atlas/layer?' + q({ l: st.layer }));
  const modes = { mechanism: [d.mech, a.mech, MECH_COL], 'vocabulary role': [d.role, a.role, ROLE_COL], trigger: [d.trig, TRIG, TRIG_COL],
    'firing sign': [d.sign.map((x) => (x > 0 ? 1 : x < 0 ? 2 : 0)), SIGN, SIGN_COL] };
  const pick = h('select', { onchange: (e) => { st.colour = e.target.value; app.rerender(); } }, Object.keys(modes).map((m) => h('option', { value: m, selected: m === st.colour }, m)));
  const [vals, names, cols] = modes[st.colour] || modes.mechanism;
  const card = h('div', { style: { marginTop: '10px' } });
  right.append(h('div', { class: 'row' }, h('b', {}, `Layer ${st.layer}`), h('span', { class: 'muted small' }, `${fmtInt(d.n)} neurons` + (d.stale ? ' · edited since the atlas was built' : '')), h('span', { class: 'grow' }), h('span', { class: 'small muted' }, 'colour by'), pick),
    legend(names, cols), grid(d.n, vals, cols, st, async (i) => { st.unit = i; card.replaceChildren(await unitCard(st.layer, i, app)); }, (i) => `L${st.layer}N${i} · ${names[vals[i]]}`),
    st.colour === 'firing sign' && !d.observed ? h('p', { class: 'small muted' }, 'Predicted from weights; the observed stage has not run.') : '', card);
  if (st.unit != null && st.unit < d.n) card.replaceChildren(await unitCard(st.layer, st.unit, app));
  else card.append(h('p', { class: 'muted small' }, 'One square per neuron. Select one to open its card.'));
}

function grid(n, vals, cols, st, onPick, label) {
  const per = Math.ceil(Math.sqrt(n * 1.6)), rows = Math.ceil(n / per), cell = Math.max(3, Math.min(9, Math.floor(430 / per)));
  const cv = h('canvas', { style: { width: per * cell + 'px', height: rows * cell + 'px', cursor: 'crosshair', margin: '6px 0' } });
  const dpr = window.devicePixelRatio || 1;
  cv.width = per * cell * dpr; cv.height = rows * cell * dpr;
  const g = cv.getContext('2d');
  g.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement);
  const rgb = cols.map((c) => css.getPropertyValue(c.slice(4, -1)).trim());
  for (let i = 0; i < n; i++) { g.fillStyle = rgb[vals[i]]; g.fillRect((i % per) * cell, Math.floor(i / per) * cell, cell - (cell > 3 ? 1 : 0), cell - (cell > 3 ? 1 : 0)); }
  const mark = (i) => { g.strokeStyle = css.getPropertyValue('--text').trim(); g.lineWidth = 1.5; g.strokeRect((i % per) * cell - 0.5, Math.floor(i / per) * cell - 0.5, cell, cell); };
  if (st.unit != null && st.unit < n) mark(st.unit);
  const at = (e) => { const b = cv.getBoundingClientRect(); const i = Math.floor((e.clientY - b.top) / cell) * per + Math.floor((e.clientX - b.left) / cell); return i >= 0 && i < n ? i : -1; };
  cv.addEventListener('mousemove', (e) => { const i = at(e); tip(i < 0 ? null : esc(label(i)), e); });
  cv.addEventListener('mouseleave', () => tip(null));
  cv.addEventListener('click', (e) => { const i = at(e); if (i >= 0) { mark(i); onPick(i); } });
  return cv;
}

async function unitCard(l, n, app) {
  const u = await api('/api/atlas/unit?' + q({ l, n })), g = u.geometry, o = u.observed;
  const sign = o ? SIGN[o.peak > 0 ? 1 : o.peak < 0 ? 2 : 0] : (u.sign_w > 0 ? 'positive (predicted)' : u.sign_w < 0 ? 'negative (predicted)' : 'either sign (predicted)');
  const info = (await api('/api/atlas')), doc = info.mech_doc[u.mech] + '; ' + info.role_doc[u.role] + '.';
  const row = (label, node) => (node ? [h('div', { class: 'small muted', style: { marginTop: '6px' } }, label), node] : null);
  const con = Object.entries(u.concepts).map(([k, c]) => h('div', { class: 'small' }, `${k.replace(/_/g, ' ')}: `, h('a', { href: '#', onclick: (e) => { e.preventDefault(); app.keep.atlas.tab = 'Concepts'; app.keep.atlas.concept = c.id; app.rerender(); } }, `C${c.id}`), ` ${c.name} · purity ${fmt(c.purity, 2)}`));
  const wires = (u.wiring || []).slice(0, 6).map((w) => h('div', { class: 'small' }, h('a', { href: '#', onclick: (e) => { e.preventDefault(); const s = app.keep.atlas; s.layer = w.layer; s.unit = w.neuron; app.rerender(); } }, `L${w.layer}N${w.neuron}`), ` ${w.direction} (${w.via} read, cos ${fmt(w.cos, 2)})`));
  const ctx = (c) => h('div', { class: 'ctx small mono' }, h('span', { class: 'muted' }, `${c.value > 0 ? '+' : ''}${fmt(c.value)} `), show(c.before.slice(-8).join('')), h('mark', {}, show(c.token)), c.next != null ? h('u', {}, show(c.next)) : null);
  return h('div', { class: 'card' }, h('div', { class: 'row' }, h('b', { class: 'mono' }, `L${l}N${n}`), h('span', { class: 'pill' }, u.mech), h('span', { class: 'pill' }, u.role.replace('_', ' ')), h('span', { class: 'pill' }, 'fires ' + sign),
    u.stale ? h('span', { class: 'pill warn' }, 'edited since') : null, h('span', { class: 'grow' }),
    h('button', { class: 'ghost', onclick: () => app.select(u.addr, 'Weights') }, 'weights'),
    h('button', { class: 'ghost', title: 'put a script that switches this neuron off into the console', onclick: () => app.setConsole(`scale_neuron((${l}, ${n}), 0)`, true) }, 'script')),
    h('p', { class: 'small' }, doc.charAt(0).toUpperCase() + doc.slice(1)),
    row('Writes, + end (pushes these up when it fires positive)', toks(u.writes.pos, 'up')), row('Writes, − end', toks(u.writes.neg, 'down')),
    u.gate_tokens ? row('Gate opens on current tokens' + (u.trig_E ? '' : ' (not above random)'), toks(u.gate_tokens.pos)) : null,
    row('Gate opens when predicting' + (u.trig_U ? '' : ' (not above random)'), toks(u.gate_predicted.pos)),
    con.length ? row('Concepts', h('div', {}, con)) : null,
    row('Geometry', h('div', { class: 'small mono' }, `cos(gate,value) ${fmt(g.cos_gu, 2)} · cos(gate,write) ${fmt(g.cos_gd, 2)} · cos(value,write) ${fmt(g.cos_ud, 2)} · |write| ${fmt(g.d_norm)} · visibility ${fmt(g.out_vis, 2)}× random`)),
    wires.length ? row('Wiring', h('div', {}, wires)) : null,
    o ? row(`Observed: mean ${fmt(o.mean)}, sd ${fmt(o.std)}` + (o.sign_matches_weights == null ? '' : o.sign_matches_weights ? ' · sign as the weights predict' : ' · sign opposite to the weight prediction') + '. Strongest firing (position highlighted, next token underlined):',
      h('div', {}, o.top.slice(0, 4).map(ctx), o.bottom.slice(0, 2).map(ctx))) : row('Observed', h('span', { class: 'small muted' }, 'the observed stage has not run')));
}

// ---------------------------------------------------------------- find
async function find(el, a, st, app) {
  const f = st.find;
  const inp = (key, ph, w = '120px') => h('input', { type: 'text', value: f[key] || '', placeholder: ph, style: { width: w }, oninput: (e) => { f[key] = e.target.value; } });
  const sel = (key, opts) => h('select', { onchange: (e) => { f[key] = e.target.value; } }, ['', ...opts].map((o) => h('option', { value: o, selected: (f[key] || '') === o }, o || 'any')));
  const out = h('div', { style: { marginTop: '12px' } });
  const run = async () => {
    const s = {};
    if (f.word) s.word = f.word.split(',').map((x) => x.trim()).filter(Boolean);
    if (f.concept) s.concept = f.concept.split(',').map((x) => +x.trim().replace(/^C/i, ''));
    for (const k of ['mech', 'role', 'effect', 'trigger']) if (f[k]) s[k] = f[k];
    if (f.from !== undefined && f.from !== '' && f.to !== '' && f.to !== undefined) s.layers = [+f.from, +f.to];
    if (f.top) s.top = +f.top;
    out.replaceChildren(h('p', { class: 'muted' }, 'searching…'));
    try {
      const r = await api('/api/atlas/select', { selector: s, scale: +f.scale || 0 });
      out.replaceChildren(h('p', {}, h('b', {}, `${fmtInt(r.n)} neurons`), h('span', { class: 'muted small' }, `  ${Object.entries(r.by_layer).map(([l, c]) => `L${l}:${c}`).join(' ')}` + (s.effect ? ` · firing sign from ${r.sign_source}` : ''))),
        r.n ? h('div', { class: 'row' }, h('span', { class: 'small muted' }, 'scale what they write by'), h('input', { type: 'number', step: '0.5', value: f.scale, style: { width: '70px' }, onchange: (e) => { f.scale = +e.target.value; run(); } }),
          h('button', { class: 'primary', onclick: () => app.setConsole(`# atlas: ${JSON.stringify(s)}\n` + r.script, true) }, 'Send script to console'), h('span', { class: 'small muted' }, '0 switches them off, above 1 pushes them up')) : null,
        r.n ? saveRow(app, `Save all ${fmtInt(r.n)} as a selection`, { selector: s }, f.word ? f.word.split(',')[0].trim() : 'found', 'find-all') : null,
        rowsTable(r.rows, st, app));
    } catch (e) { out.replaceChildren(h('p', { style: { color: 'var(--crit)' } }, e.message)); }
  };
  el.append(h('p', { class: 'small muted' }, 'Find the neurons that carry a word or concept, then turn the selection into an edit. "Effect" keeps only those that raise (promote) or lower (suppress) the target when they fire hard.'),
    h('div', { class: 'row', style: { alignItems: 'flex-end' } },
      h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'word(s)'), inp('word', 'red, rouge')),
      h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'concept id(s)'), inp('concept', 'C12', '70px')),
      h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'effect'), sel('effect', ['promote', 'suppress'])),
      h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'mechanism'), sel('mech', a.mech)),
      h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'role'), sel('role', a.role)),
      h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'layers'), h('span', { class: 'row', style: { gap: '4px' } }, inp('from', 'from', '48px'), inp('to', 'to', '48px'))),
      h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'top'), inp('top', 'all', '48px')),
      h('button', { class: 'primary', onclick: run }, 'Find')), out);
  if (f.word || f.concept || f.mech || f.role) await run();
}

// One click saves a result as a named selection; the server expands it (an atlas selector is never listed here).
function saveRow(app, label, what, suggest, key) {
  const status = savedStatus(app, 'atlas:' + key);
  const b = h('button', { onclick: async () => { b.disabled = true; await saveSelection(app, what, { suggest, status, key: 'atlas:' + key }); b.disabled = false; } }, label);
  return h('div', { class: 'row', style: { marginTop: '6px' } }, b, status);
}

// Ticked rows and the last save's result live in `st` (the view's kept state), so a redraw after a patch or
// groups event keeps them; ticks for rows no longer listed are dropped.
export function rowsTable(rows, st, app, third = 'Fires') {
  const all = st.checked || (st.checked = {});
  const keys = new Set(rows.map((r) => `${r.layer},${r.neuron}`));
  const checked = all[third] || (all[third] = new Set());
  for (const k of [...checked]) if (!keys.has(k)) checked.delete(k);
  const status = savedStatus(app, 'atlas:rows:' + third);
  const label = () => `Save checked (${checked.size})`;
  const save = h('button', { class: 'ghost', disabled: !checked.size, onclick: async () => {
    save.disabled = true;
    await saveSelection(app, { neurons: [...checked].map((k) => k.split(',').map(Number)) }, { suggest: 'picked', status, key: 'atlas:rows:' + third });
    save.disabled = !checked.size;
  } }, label());
  const tick = (r) => h('input', { type: 'checkbox', title: 'pick for a selection', checked: checked.has(`${r.layer},${r.neuron}`),
    onclick: (e) => { e.stopPropagation?.(); },
    onchange: (e) => { const k = `${r.layer},${r.neuron}`; if (e.target.checked) checked.add(k); else checked.delete(k); save.textContent = label(); save.disabled = !checked.size; } });
  const table = h('table', { class: 'grid', style: { marginTop: '8px' } }, h('thead', {}, h('tr', {}, ['', 'Neuron', 'Class', third, '+ end', '− end'].map((t) => h('th', {}, t)))),
    h('tbody', {}, rows.map((r) => h('tr', { class: 'clickable', onclick: () => { st.tab = 'Layers'; st.layer = r.layer; st.unit = r.neuron; app.rerender(); } },
      h('td', {}, tick(r)),
      h('td', { class: 'mono' }, `L${r.layer}N${r.neuron}`), h('td', { class: 'small' }, r.mech || ''), h('td', { class: 'small' }, r.peak == null ? (r.effect != null ? fmt(r.effect) : '') : r.peak > 0 ? '+' : r.peak < 0 ? '−' : '±'),
      h('td', {}, r.plus ? toks(r.plus, 'up') : ''), h('td', {}, r.minus ? toks(r.minus, 'down') : '')))));
  return h('div', {}, rows.length ? h('div', { class: 'row', style: { marginTop: '6px' } }, save, status) : null, table);
}

// ---------------------------------------------------------------- concepts
async function concepts(el, a, st, app) {
  const c = await api('/api/atlas/concepts');
  const detail = h('div', { style: { flex: '1 1 380px', minWidth: 0 } });
  const open = async (id) => {
    st.concept = id;
    const d = await api('/api/atlas/concept?' + q({ id }));
    detail.replaceChildren(h('div', { class: 'card' }, h('div', { class: 'row' }, h('b', {}, `C${d.id}`), h('span', { class: 'muted small' }, `${fmtInt(d.size)} tokens · cohesion ${fmt(d.cohesion, 2)}`), h('span', { class: 'grow' }),
      h('button', { class: 'ghost', onclick: () => { st.tab = 'Find'; st.find = { concept: 'C' + d.id, scale: 0 }; app.rerender(); } }, 'find its neurons')),
    toks(d.text), h('h3', {}, 'Neurons with this concept at an end'),
    h('div', { class: 'small muted' }, Object.entries(d.by_layer).map(([k, v]) => `${k}: ${Object.values(v).reduce((x, y) => x + y, 0)}`).join(' · ') || 'none'),
    h('table', { class: 'grid', style: { marginTop: '6px' } }, h('tbody', {}, d.neurons.slice(0, 24).map((r) => h('tr', { class: 'clickable', onclick: () => { st.tab = 'Layers'; st.layer = r.layer; st.unit = r.neuron; app.rerender(); } },
      h('td', { class: 'mono' }, `L${r.layer}N${r.neuron}`), h('td', { class: 'small' }, r.end), h('td', {}, toks(r.plus, 'up')), h('td', {}, toks(r.minus, 'down'))))))));
  };
  const list = h('div', { style: { flex: '1 1 380px', minWidth: 0, maxHeight: '560px', overflow: 'auto' } }, c.groups.map((g) => h('div', {}, h('h3', {}, `Group ${g.id} · ${fmtInt(g.size)} tokens`),
    c.nodes.filter((n) => n.group === g.id).map((n) => h('div', { class: 'crow clickable', onclick: () => open(n.id) }, h('span', { class: 'mono small muted', style: { minWidth: '44px' } }, 'C' + n.id), toks(n.text.slice(0, 7)))))));
  el.append(h('p', { class: 'small muted' }, `${c.nodes.length} token concepts: clusters of the ${c.space}. They are named by their central tokens, and some are held together by spelling rather than meaning.`),
    h('div', { class: 'row', style: { alignItems: 'flex-start', gap: '18px' } }, list, detail));
  if (st.concept != null) await open(st.concept);
}

// ---------------------------------------------------------------- attribute
async function attribute(el, a, st, app) {
  const scope = inspectionScope(app), variant = app.variant, view = app._atlasView;
  const valid = () => scope === inspectionScope(app) && app.variant === variant && app.tab === 'Atlas' && app._atlasView === view;
  const f = st.attr, out = h('div', { style: { marginTop: '12px' } });
  if (!a.can_observe) { el.append(h('p', { class: 'muted' }, 'Attribution runs the model on a prompt, and this model cannot be run here.')); return; }
  const inp = (key, ph, w) => h('input', { type: 'text', value: f[key], placeholder: ph, style: { width: w }, oninput: (e) => { f[key] = e.target.value; } });
  const run = async () => {
    if (f.busy || !valid()) return;
    f.busy = true; button.disabled = true;
    out.replaceChildren(h('p', { class: 'muted' }, 'Running attribution… Leaving this view does not cancel it.'));
    try {
      const r = await inspection.read(app, '/api/atlas/attribute', { prompt: f.prompt, target: f.target, versus: f.versus, k: +f.k, scale: +f.scale, variant }, true);
      if (!valid()) return;
      const send = h('button', { class: 'primary', disabled: !r.script, onclick: () => app.setConsole(`# attribution: ${JSON.stringify(f.prompt)} -> ${JSON.stringify(r.target)}\n` + r.script, true) }, 'Send script to console');
      const max = Math.max(...Object.values(r.by_layer).map(Math.abs), 1e-9);
      out.replaceChildren(h('p', {}, `Score of `, h('span', { class: 'tok' }, show(r.target)), r.versus ? [' over ', h('span', { class: 'tok' }, show(r.versus))] : ' relative to the average token', `: `, h('b', {}, `${fmt(r.margin)} nats`), `. All MLP neurons add ${fmt(r.mlp_total)} directly; the rest comes from the mixers and the embedding.`),
        h('div', { class: 'lbars', title: 'direct contribution of each layer\'s MLP' }, Object.entries(r.by_layer).map(([l, v]) => h('div', { class: 'lbar', title: `L${l}: ${fmt(v)}` }, h('i', { class: v >= 0 ? 'p' : 'n', style: { height: Math.max(1, 28 * Math.abs(v) / max) + 'px' } }), h('span', {}, l)))),
        h('div', { class: 'row', style: { margin: '10px 0 4px' } }, h('span', { class: 'small muted' }, 'scale the'), h('input', { type: 'number', min: 1, value: f.k, style: { width: '52px' }, onchange: (e) => { f.k = +e.target.value; button.textContent = 'Attribute with updated script options'; send.disabled = true; } }),
          h('span', { class: 'small muted' }, 'neurons pushing hardest for the target by'), h('input', { type: 'number', step: '0.5', value: f.scale, style: { width: '60px' }, onchange: (e) => { f.scale = +e.target.value; button.textContent = 'Attribute with updated script options'; send.disabled = true; } }),
          send),
        r.top.length ? saveRow(app, `Save these ${r.top.length} as a selection`, { neurons: r.top.map((x) => [x.layer, x.neuron]) }, 'attributed') : null,
        rowsTable(r.top, st, app, 'Effect (nats)'), h('p', { class: 'small muted' }, 'Direct path only: what a neuron does by feeding later layers is not counted.'));
    } catch (e) { if (valid()) out.replaceChildren(h('p', { style: { color: 'var(--crit)' } }, e.message)); }
    finally {
      f.busy = false; button.disabled = false;
      if (!valid() && scope === inspectionScope(app) && app.tab === 'Atlas' && st.tab === 'Attribute') app.rerender();
    }
  };
  const button = h('button', { class: 'primary', disabled: !!f.busy, onclick: run }, f.busy ? 'Attribution running…' : 'Attribute');
  el.append(h('p', { class: 'small muted' }, 'Which neurons push an answer in a given prompt. A handful chosen this way usually moves an answer more cleanly than hundreds chosen by signature.'),
    h('div', { class: 'row', style: { alignItems: 'flex-end' } },
      h('label', { class: 'field grow' }, h('span', { class: 'field-label' }, 'prompt (the answer is the next token)'), inp('prompt', 'The colour of a ripe tomato is', '100%')),
      h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'target token'), inp('target', '" red"', '90px')),
      h('label', { class: 'field' }, h('span', { class: 'field-label' }, 'versus (optional)'), inp('versus', '" green"', '90px')),
      button), out);

}

export function onAtlasEvent(ev, app) {
  if (app.tab !== 'Atlas' || ev.workspace_id !== app.state.status.workspace_id || ev.version !== app.state.status.version) return;
  if (ev.done) { app.rerender(); return; }
  if (ev.progress) { const p = document.getElementById('atlas-progress'); if (p) p.textContent = progressText(ev.progress); return; }
  const log = document.getElementById('atlas-log');
  if (log && ev.line) { log.textContent += '\n' + ev.line; log.scrollTop = log.scrollHeight; } else if (!log) app.rerender();
}
