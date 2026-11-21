// Probe: read neurons along a text, find what fires at one token, check a push on the next token,
// and contrast two sets of texts. Every probe runs the model and writes nothing.
// Routes and limits: docs/STEERING.md ("Probes").
import { h, api, fmt, tabs, field, diverge, visible } from './util.js';
import { thinkingOptions } from './run.js';
import { saveSelection, savedStatus } from './selection.js';
import { addSteerEntry, steerState, steerRequest, parseRecipe, parseTarget, parseP50, steerLabel, steeringStatus } from './steer.js';

export const PROBE_CHARS = 20000;
export const BOUNDS = { tokens: 256, nextTokens: 512, neurons: 16, k: 200, texts: 16, top: 50 };
const VIEWS = ['reads', 'fires', 'next token', 'contrast'];

// Later tools for the neuron rows of Fires and Contrast register here (for example "save as
// selection"): fn(rows, app) -> node or null, where rows are [{ layer, neuron, addr, ... }].
export const rowTools = [];

const WAIT = 'running… If a reply is being generated, this waits until it finishes.';
const chars = (s) => `${(s || '').length.toLocaleString('en-US')} of ${PROBE_CHARS.toLocaleString('en-US')} characters`;
// The browser checks characters and counts; tokens are counted by the server's tokenizer, so that bound is the server's.
const tokenBound = (n, chat) => ` · up to ${n} tokens, counted by the server${chat ? ' (as a chat message the template\'s tokens count too)' : ''}`;
const count = (n, max, what) => `${n} of ${max} ${what}`;
const lines = (s) => String(s || '').split(/\r?\n/).filter((t) => t.trim());
const neuronAddr = (l, n) => `blocks.${l}.mlp.neuron.${n}`;
// Element.append would print a null as text; optional parts are left out instead.
const put = (el, ...kids) => el.append(...kids.filter(Boolean));

// ------------------------------------------------------------------------------------------------ as a chat message
// Whether the open model has its own chat template (GET /api/probe/chat: from its files, the model is not loaded).
function chatInfo(app, st, again) {
  const key = app.state.open?.name;
  if (st.chatInfo?.key === key) return st.chatInfo;
  const info = st.chatInfo = { key, loading: true };
  api('/api/probe/chat').then((r) => { if (st.chatInfo === info) { st.chatInfo = { key, available: r?.available === true, reason: r?.reason || '' }; again(); } },
    (e) => { if (st.chatInfo === info) { st.chatInfo = { key, available: false, reason: e.message }; again(); } });
  return info;
}

// The fields that make a probe read its text as one user turn, rendered as Chat renders it (the Run tab's thinking choice).
const chatFields = (app) => ({ chat: true, chat_options: thinkingOptions(app.keep.run?.thinking) });

function chatBox(app, st, v, name, again) {
  const info = chatInfo(app, st, again);
  const ok = info.available === true;
  const why = ok ? 'one user turn in the model\'s own chat template, generation prompt included, as Chat renders it'
    : info.loading ? 'checking whether this model has a chat template…' : `unavailable: ${info.reason}`;
  return h('label', { class: 'small', title: why },
    h('input', { type: 'checkbox', checked: ok && !!v.chat, disabled: !ok, 'aria-label': `${name} as a chat message`, onchange: (e) => { v.chat = !!e.target.checked; again(); } }),
    ok ? ' as a chat message' : ` as a chat message (${why})`);
}
const asChat = (app, st, v) => st.chatInfo?.available === true && !!v.chat;

// Token chips: invisible characters written out; with a chat rendering, the template's tokens drawn apart from the text's.
const TEMPLATE_STYLE = { border: '1px dashed currentColor', opacity: '0.6' };
const isTemplate = (r, j) => !!r.chat?.template?.[j];
function chatNote(r) {
  if (!r.chat) return null;
  if (r.chat.rendered) return h('p', { class: 'small muted', 'aria-label': 'chat rendering' }, 'Each text was read as a chat message: one user turn in the model\'s chat template, generation prompt included.');
  return h('p', { class: 'small muted', 'aria-label': 'chat rendering' }, r.chat.template
    ? 'Read as a chat message: tokens with a dashed border are the chat template\'s (role markers and the generation prompt); the rest are your text.'
    : 'Read as a chat message. The template\'s tokens could not be told apart from your text at their boundary, so none is marked.');
}

function state(app) {
  return app.keep.probe || (app.keep.probe = { view: 'fires', reads: { text: '', neurons: '' }, fires: { text: '', k: 20, sign: 'both' },
    next: { prompt: '', source: 'panel', typed: '', top: 10 }, contrast: { a: '', b: '', k: 20 } });
}

// `steering` events without a stream id (calibration, column reads) for the probe that is waiting.
export function onProbeSteeringEvent(ev, app) {
  const st = app.keep.probe;
  if (ev.stream || !st?.pending) return;
  st.pending.progress = ev;
  st.pending.paint?.();
}

async function send(app, st, v, path, body) {
  v.busy = true; v.error = ''; v.progress = null; st.pending = v;
  v.paint?.();
  try { v.result = await api(path, body); v.sent = body; v.ticks = {}; }   // ticks belong to one result
  catch (e) { v.error = e.message; v.result = null; }
  finally { v.busy = false; if (st.pending === v) st.pending = null; }
}

// The button, then right under it: the waiting state or the error of this request.
function action(v, label, go, disabled = false) {
  const status = h('div', { class: 'small', role: 'status' });
  const paint = () => {
    status.textContent = v.busy ? (steeringStatus(v.progress) || WAIT) : '';
    if (v.error && !v.busy) status.replaceChildren(h('span', { role: 'alert', style: { color: 'var(--crit)' } }, v.error));
  };
  v.paint = paint;
  paint();
  return h('div', {}, h('button', { class: 'primary', disabled: !!v.busy || disabled, onclick: go }, label), status);
}

// save: { key, suggest, ticks (a Set of row addresses kept across redraws) } -> "save as selection" for every row
// and for the ticked rows, through the inspector's saveSelection (POST /api/groups with neurons).
function neuronTable(rows, app, columns, notice, save) {
  const ticks = save?.ticks;
  const tick = (r) => h('td', {}, h('input', { type: 'checkbox', checked: ticks.has(r.addr), 'aria-label': `tick ${r.addr}`,
    onchange: (e) => { if (e.target.checked) ticks.add(r.addr); else ticks.delete(r.addr); if (saveTicked) saveTicked.disabled = !rows.some((x) => ticks.has(x.addr)); } }));
  let saveTicked = null;
  const table = h('table', { class: 'small' },
    h('tr', {}, ['neuron', 'layer', 'index', ...columns.map((c) => c[0]), '', ...(ticks ? ['tick'] : [])].map((t) => h('th', { style: { textAlign: 'left', paddingRight: '10px' } }, t))),
    rows.map((r) => h('tr', {}, h('td', { class: 'mono' }, r.addr), h('td', {}, r.layer), h('td', {}, r.neuron),
      columns.map(([, key]) => h('td', { title: String(r[key]) }, fmt(r[key], 4))),
      h('td', {}, h('button', { class: 'ghost', title: 'open this neuron in the inspector', onclick: () => app.select(neuronAddr(r.layer, r.neuron), 'Weights') }, 'inspect'),
        h('button', { class: 'ghost', title: 'add to the Run tab\'s steering panel', onclick: () => { addSteerEntry(app, r.addr); notice(`${r.addr} +0.5 added to the Run tab's steering panel.`); } }, '+ steer')), ticks ? tick(r) : null)));
  let saving = null;
  if (save) {
    const status = savedStatus(app, 'probe:' + save.key);
    const go = async (list, b) => {
      b.disabled = true;
      await saveSelection(app, { neurons: list.map((r) => [r.layer, r.neuron]) }, { suggest: save.suggest, status, key: 'probe:' + save.key });
      b.disabled = false;
    };
    const all = h('button', { class: 'ghost', title: 'save every neuron of this table as a named selection', onclick: () => go(rows, all) }, 'save all as selection');
    saveTicked = h('button', { class: 'ghost', disabled: !rows.some((x) => ticks.has(x.addr)), title: 'save the ticked neurons as a named selection',
      onclick: () => go(rows.filter((x) => ticks.has(x.addr)), saveTicked) }, 'save ticked as selection');
    saving = h('div', { class: 'row', style: { gap: '6px', marginTop: '4px' } }, all, saveTicked, status);
  }
  return h('div', {}, table, saving, rowTools.map((fn) => fn(rows, app)));
}

const ticksOf = (v, key) => { v.ticks = v.ticks || {}; return v.ticks[key] || (v.ticks[key] = new Set()); };

// ------------------------------------------------------------------------------------------------ reads
function readsProblems(v) {
  const out = [], addrs = [];
  if (!v.text.trim()) out.push('Enter a text.');
  if (v.text.length > PROBE_CHARS) out.push(`The text is longer than ${PROBE_CHARS} characters.`);
  const items = String(v.neurons || '').split(/[,;\n]/).map((s) => s.trim()).filter(Boolean);
  if (!items.length) out.push('Enter one or more neurons, e.g. L3N17, L5N2.');
  if (items.length > BOUNDS.neurons) out.push(`At most ${BOUNDS.neurons} neurons at once.`);
  for (const item of items) {
    const t = parseTarget(item);
    if (t.error) out.push(t.error);
    else if (t.kind === 'head' || t.kind === 'mlp' || t.kind === 'direction') out.push(`${t.addr} is not a neuron.`);
    else addrs.push(t.addr);
  }
  return { problems: out, addrs };
}

function readsResult(r, app) {
  const { addrs, token_strs: toks, values } = r;
  let max = 0;
  for (const rows of Object.values(values)) for (const row of rows) for (const x of row) max = Math.max(max, Math.abs(x));
  const out = [chatNote(r), h('p', { class: 'small muted' }, `Shade: activation, red above 0 and blue below, scaled to the largest magnitude in this table (${fmt(max)}). Hover a token for its exact value.`)];
  addrs.forEach((addr, i) => {
    const cur = values.current?.[i], base = values.base?.[i];
    const differs = base && cur && base.some((x, j) => Math.abs(x - cur[j]) > 1e-6 * Math.max(1, max));
    const row = (label, vals) => h('div', { class: 'row', style: { alignItems: 'baseline', gap: '6px', flexWrap: 'wrap' } },
      h('span', { class: 'small muted', style: { width: '150px' } }, `${addr} · ${label}`),
      h('span', { class: 'toks' }, toks.map((t, j) => h('span', { class: 'tok', style: { background: diverge(max ? vals[j] / max : 0), ...(isTemplate(r, j) ? TEMPLATE_STYLE : {}) },
        title: `${addr} ${label}, ${isTemplate(r, j) ? 'chat template ' : ''}token ${j} ${visible(JSON.stringify(t))}: ${vals[j]}` }, visible(t)))));
    out.push(h('div', { style: { marginTop: '6px' } }, cur ? row(r.steered ? 'edited, steered' : 'edited', cur) : null,
      differs ? row('base', base) : null,
      base && cur && !differs ? h('div', { class: 'small muted' }, 'base: the same') : null));
  });
  return out;
}

function readsView(el, app, st, again) {
  const v = st.reads;
  const steer = steerRequest(steerState(app));
  const bound = () => chars(v.text) + tokenBound(BOUNDS.tokens, asChat(app, st, v));
  const size = h('span', { class: 'small muted', 'aria-label': 'reads text bound' }, bound());
  const items = () => String(v.neurons || '').split(/[,;\n]/).filter((x) => x.trim()).length;
  const many = h('span', { class: 'small muted', 'aria-label': 'reads neurons bound' }, count(items(), BOUNDS.neurons, 'neurons (a range counts each neuron; the server counts those)'));
  el.append(h('p', { class: 'hint' }, `Each neuron's activation at every token of the text, on the edited model and (where it differs) the base model. Up to ${BOUNDS.tokens} tokens and ${BOUNDS.neurons} neurons.`),
    h('textarea', { rows: 3, value: v.text, 'aria-label': 'reads text', style: { width: '100%' }, oninput: (e) => { v.text = e.target.value; size.textContent = bound(); } }), size,
    h('div', { class: 'row', style: { alignItems: 'flex-end', gap: '8px', marginTop: '6px' } },
      field('neurons (comma-separated)', h('input', { type: 'text', value: v.neurons, 'aria-label': 'reads neurons', placeholder: 'L3N17, L5N2', style: { width: '260px', fontFamily: 'var(--mono)' },
        oninput: (e) => { v.neurons = e.target.value; many.textContent = count(items(), BOUNDS.neurons, 'neurons (a range counts each neuron; the server counts those)'); } }), many),
      chatBox(app, st, v, 'reads', again),
      h('label', { class: 'small', title: steer.fields ? steer.label : 'the Run tab\'s steering panel has no active entries' },
        h('input', { type: 'checkbox', checked: !!v.steer, disabled: !steer.fields, 'aria-label': 'reads with steering', onchange: (e) => { v.steer = !!e.target.checked; } }),
        steer.fields ? ` steer the edited row (${steer.label})` : ' steer the edited row (no active steering entries)')),
    action(v, 'Read', async () => {
      const { problems, addrs } = readsProblems(v);
      if (problems.length) { v.error = problems.join(' '); v.paint(); return; }
      const s = steerRequest(steerState(app));
      if (v.steer && s.problems.length) { v.error = 'Steering: ' + s.problems.join('; '); v.paint(); return; }
      await send(app, st, v, '/api/probe/reads', { prompt: v.text, neurons: addrs, variants: ['base', 'current'], ...(v.steer && s.fields ? s.fields : {}), ...(asChat(app, st, v) ? chatFields(app) : {}) });
      again();
    }));
  if (v.result && !v.busy) put(el, ...readsResult(v.result, app));
}

// ------------------------------------------------------------------------------------------------ fires
function firesView(el, app, st, again) {
  const v = st.fires;
  const bound = () => chars(v.text) + tokenBound(BOUNDS.tokens, asChat(app, st, v));
  const size = h('span', { class: 'small muted', 'aria-label': 'fires text bound' }, bound());
  // A click on a token asks again about the text that was sent, rendered as it was (its positions are that rendering's).
  const go = async (position = -1, text = v.text, chat = asChat(app, st, v) ? chatFields(app) : {}) => {
    if (!text.trim()) { v.error = 'Enter a text.'; v.paint(); return; }
    if (text.length > PROBE_CHARS) { v.error = `The text is longer than ${PROBE_CHARS} characters.`; v.paint(); return; }
    const k = Number(v.k);
    if (!Number.isInteger(k) || k < 1 || k > BOUNDS.k) { v.error = `Show between 1 and ${BOUNDS.k} neurons.`; v.paint(); return; }
    v.notice = '';
    await send(app, st, v, '/api/probe/fires', { prompt: text, position, k, sign: v.sign, ...chat });
    again();
  };
  el.append(h('p', { class: 'hint' }, `The neurons with the largest activation at one token (the last, until you click another). Up to ${BOUNDS.tokens} tokens; 1 to ${BOUNDS.k} neurons.`),
    h('textarea', { rows: 3, value: v.text, 'aria-label': 'fires text', style: { width: '100%' }, oninput: (e) => { v.text = e.target.value; size.textContent = bound(); } }), size,
    h('div', { class: 'row', style: { alignItems: 'flex-end', gap: '8px', marginTop: '6px' } },
      field('neurons', h('input', { type: 'number', value: v.k, min: 1, max: BOUNDS.k, 'aria-label': 'fires count', style: { width: '70px' }, oninput: (e) => { v.k = e.target.value; } }), `1 to ${BOUNDS.k}`),
      field('sign', h('select', { 'aria-label': 'fires sign', onchange: (e) => { v.sign = e.target.value; } },
        [['both', 'largest either way'], ['+', 'most positive'], ['-', 'most negative']].map(([x, label]) => h('option', { value: x, selected: v.sign === x }, label)))),
      chatBox(app, st, v, 'fires', again)),
    action(v, 'Find', () => go()));
  const r = v.result;
  if (!r || v.busy) return;
  const sentChat = v.sent?.chat ? { chat: true, chat_options: v.sent.chat_options } : {};
  put(el, chatNote(r), h('div', { class: 'small muted', style: { marginTop: '8px' } }, 'Tokens (click one to rank the neurons there):'),
    h('div', { class: 'toks' }, r.token_strs.map((t, j) => h('span', { class: 'tok clickable', title: `${isTemplate(r, j) ? 'chat template ' : ''}token ${j}`,
      style: { ...(isTemplate(r, j) ? TEMPLATE_STYLE : {}), ...(j === r.position ? { outline: '2px solid var(--accent)' } : {}) },
      'aria-current': j === r.position ? 'true' : null, onclick: () => go(j, v.sent.prompt, sentChat) }, visible(t)))),
    h('p', { class: 'small' }, `At token ${r.position} ${visible(JSON.stringify(r.token_str))}, ranked by raw activation: a large value is not the same as a large effect on the output.`),
    v.notice ? h('p', { class: 'small', role: 'status' }, v.notice) : null,
    neuronTable(r.neurons, app, [['activation', 'value']], (m) => { v.notice = m; again(); },
      { key: 'fires', suggest: `fires at token ${r.position}`, ticks: ticksOf(v, 'fires') }));
}

// ------------------------------------------------------------------------------------------------ next token
function nextSteer(app, v) {
  const panel = steerState(app);
  const p50 = parseP50(panel.p50);
  if (v.source === 'panel') {
    const s = steerRequest(panel);
    if (s.problems.length) return { error: 'Steering: ' + s.problems.join('; ') };
    if (!s.fields) return { error: 'The Run tab\'s steering panel has no active entries. Switch one on there, or type one entry here.' };
    return { fields: s.fields, label: s.label };
  }
  const r = parseRecipe(v.typed);
  if (r.problems.length) return { error: `${r.problems[0].item}: ${r.problems[0].reason}` };
  if (p50.error) return { error: `Steering: per-layer p50 (the Run tab's panel): ${p50.error}` };   // never sent without it
  if (r.entries.length !== 1) return { error: 'Type one entry, e.g. L34N3838 +0.75.' };
  return { fields: { steer: r.entries, ...(p50.map ? { p50: p50.map } : {}) }, label: steerLabel(r.entries) };
}

function probRows(rows, cols) {
  return h('table', { class: 'small' }, h('tr', {}, cols.map(([t]) => h('th', { style: { textAlign: 'left', paddingRight: '10px' } }, t))),
    rows.map((r) => h('tr', {}, cols.map(([, f]) => h('td', {}, visible(f(r)))))));
}
const pct = (p) => (100 * p).toFixed(p >= 0.001 ? 1 : 3) + '%';

function nextView(el, app, st, again) {
  const v = st.next;
  const panel = steerRequest(steerState(app));
  const bound = () => chars(v.prompt) + tokenBound(BOUNDS.nextTokens, asChat(app, st, v));
  const size = h('span', { class: 'small muted', 'aria-label': 'next token prompt bound' }, bound());
  el.append(h('p', { class: 'hint' }, `How a push moves the edited model's next-token distribution: KL divergence, the top tokens with and without it, and the tokens that rose and fell most. Up to ${BOUNDS.nextTokens} tokens.`),
    h('textarea', { rows: 2, value: v.prompt, 'aria-label': 'next token prompt', style: { width: '100%' }, oninput: (e) => { v.prompt = e.target.value; size.textContent = bound(); } }), size,
    h('div', { class: 'row', style: { alignItems: 'flex-end', gap: '8px', marginTop: '6px' } },
      field('push', h('select', { 'aria-label': 'next token source', onchange: (e) => { v.source = e.target.value; again(); } },
        h('option', { value: 'panel', selected: v.source === 'panel' }, panel.fields ? `the Run panel's entries (${panel.label})` : 'the Run panel\'s entries (none active)'),
        h('option', { value: 'typed', selected: v.source === 'typed' }, 'one entry typed here'))),
      v.source === 'typed' ? field('entry', h('input', { type: 'text', value: v.typed, 'aria-label': 'next token entry', placeholder: 'L34N3838 +0.75', style: { width: '180px', fontFamily: 'var(--mono)' }, oninput: (e) => { v.typed = e.target.value; } })) : null,
      field('top tokens', h('input', { type: 'number', value: v.top, min: 1, max: BOUNDS.top, 'aria-label': 'next token top', style: { width: '60px' }, oninput: (e) => { v.top = e.target.value; } }), `1 to ${BOUNDS.top}`),
      chatBox(app, st, v, 'next token', again)),
    action(v, 'Check the next token', async () => {
      if (!v.prompt.trim()) { v.error = 'Enter a prompt.'; v.paint(); return; }
      if (v.prompt.length > PROBE_CHARS) { v.error = `The prompt is longer than ${PROBE_CHARS} characters.`; v.paint(); return; }
      const top = Number(v.top);
      if (!Number.isInteger(top) || top < 1 || top > BOUNDS.top) { v.error = `Show between 1 and ${BOUNDS.top} tokens.`; v.paint(); return; }
      const s = nextSteer(app, v);
      if (s.error) { v.error = s.error; v.paint(); return; }
      v.label = s.label;
      await send(app, st, v, '/api/probe/next', { prompt: v.prompt, top, ...s.fields, ...(asChat(app, st, v) ? chatFields(app) : {}) });
      again();
    }));
  const r = v.result;
  if (!r || v.busy) return;
  put(el, chatNote(r), h('p', { class: 'small' }, `steered: ${v.label}. `, r.kl == null ? 'KL(unsteered ‖ steered) is infinite: the steered model gives zero probability to a token the unsteered model can produce.'
    : `KL(unsteered ‖ steered) = ${fmt(r.kl, 4)} nats.`),
  h('div', { class: 'split' },
    h('div', { class: 'card' }, h('b', {}, 'without'), probRows(r.top_unsteered, [['token', (x) => JSON.stringify(x.token)], ['p', (x) => pct(x.p)]])),
    h('div', { class: 'card' }, h('b', {}, 'with the push'), probRows(r.top_steered, [['token', (x) => JSON.stringify(x.token)], ['p', (x) => pct(x.p)]]))),
  h('div', { class: 'split' },
    ...[['rose most', r.rose], ['fell most', r.fell]].map(([title, rows]) => h('div', { class: 'card' }, h('b', {}, title),
      rows.length ? probRows(rows, [['token', (x) => JSON.stringify(x.token)], ['before', (x) => pct(x.p)], ['after', (x) => pct(x.q)], ['Δ log p', (x) => fmt(x.delta_logp, 3)]])
        : h('p', { class: 'small muted' }, 'none')))));
}

// ------------------------------------------------------------------------------------------------ contrast
function contrastView(el, app, st, again) {
  const v = st.contrast;
  const texts = (key) => count(lines(v[key]).length, BOUNDS.texts, 'texts');
  const box = (key, label) => {
    const n = h('span', { class: 'small muted', 'aria-label': `contrast set ${key.toUpperCase()} bound` }, texts(key));
    return h('div', {}, h('div', { class: 'small muted' }, label),
      h('textarea', { rows: 4, value: v[key], 'aria-label': `contrast set ${key.toUpperCase()}`, style: { width: '100%' }, oninput: (e) => { v[key] = e.target.value; n.textContent = texts(key); } }), n);
  };
  el.append(h('p', { class: 'hint' }, `Neurons that fire on set A and not on set B, and the reverse (mean over texts of each neuron's largest activation). One text per line; 1 to ${BOUNDS.texts} texts per set, each up to ${BOUNDS.tokens} tokens.`),
    h('div', { class: 'split' }, box('a', 'set A'), box('b', 'set B')),
    h('p', { class: 'small muted' }, `Each text: up to ${PROBE_CHARS.toLocaleString('en-US')} characters (checked here)${tokenBound(BOUNDS.tokens, asChat(app, st, v))}.`),
    h('div', { class: 'row', style: { alignItems: 'flex-end', gap: '8px', marginTop: '6px' } },
      field('neurons per list', h('input', { type: 'number', value: v.k, min: 1, max: BOUNDS.k, 'aria-label': 'contrast count', style: { width: '70px' }, oninput: (e) => { v.k = e.target.value; } }), `1 to ${BOUNDS.k}`),
      chatBox(app, st, v, 'contrast', again)),
    action(v, 'Contrast', async () => {
      const a = lines(v.a), b = lines(v.b), k = Number(v.k);
      const bad = [['A', a], ['B', b]].find(([, t]) => t.length < 1 || t.length > BOUNDS.texts);
      if (bad) { v.error = `Set ${bad[0]} needs 1 to ${BOUNDS.texts} texts, one per line.`; v.paint(); return; }
      if ([...a, ...b].some((t) => t.length > PROBE_CHARS)) { v.error = `Each text is at most ${PROBE_CHARS} characters.`; v.paint(); return; }
      if (!Number.isInteger(k) || k < 1 || k > BOUNDS.k) { v.error = `Show between 1 and ${BOUNDS.k} neurons.`; v.paint(); return; }
      v.notice = '';
      await send(app, st, v, '/api/probe/contrast', { a, b, k, ...(asChat(app, st, v) ? chatFields(app) : {}) });
      again();
    }));
  const r = v.result;
  if (!r || v.busy) return;
  const cols = [['A minus B', 'diff'], ['A', 'a'], ['B', 'b']];
  const notice = (m) => { v.notice = m; again(); };
  put(el, chatNote(r), h('p', { class: 'small muted' }, `${r.a} text${r.a === 1 ? '' : 's'} in A, ${r.b} in B. ${r.measure || ''}`),
    v.notice ? h('p', { class: 'small', role: 'status' }, v.notice) : null,
    h('div', { class: 'split' },
      h('div', {}, h('b', {}, 'fire on A, not B'), r.up.length ? neuronTable(r.up, app, cols, notice, { key: 'contrast:a', suggest: 'fires on A not B', ticks: ticksOf(v, 'up') }) : h('p', { class: 'small muted' }, 'none')),
      h('div', {}, h('b', {}, 'fire on B, not A'), r.down.length ? neuronTable(r.down, app, cols, notice, { key: 'contrast:b', suggest: 'fires on B not A', ticks: ticksOf(v, 'down') }) : h('p', { class: 'small muted' }, 'none'))));
}

// ------------------------------------------------------------------------------------------------ view
export async function probeView(el, app) {
  const st = state(app);
  const again = () => { if (app.keep.probe === st && app.tab === 'Probe') app.rerender(); };
  el.append(h('div', { class: 'row' }, h('h2', {}, 'Probe'), tabs(VIEWS, st.view, (m) => { st.view = m; again(); }),
    h('span', { class: 'muted' }, 'read the model on your own texts; nothing is written')));
  if (!app.state.status.can_run) {
    el.append(h('p', { class: 'hint' }, 'Probes run the model, and this model cannot be run here' + (app.state.status.run_error ? ': ' + app.state.status.run_error : '.')));
    return;
  }
  const view = { reads: readsView, fires: firesView, 'next token': nextView, contrast: contrastView }[st.view] || firesView;
  view(el, app, st, again);
}
