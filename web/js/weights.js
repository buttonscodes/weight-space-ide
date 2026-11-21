// Weights view: whatever is selected in the explorer, at its own level.
//   tensor  -> zoomable, editable heatmap + spectrum          unit   -> head / neuron card + its slices
//   3-D     -> one facet per head                             higher -> tiles for the children (small multiples)
import { h, api, q, fmt, fmtInt, decodeF32, heatmap, legend, tip, esc, hbars, vbars, tokList, tabs, toScript } from './util.js';

import { inspection, calculation } from './inspection.js';
import { breadcrumb, compareView, groupSummary } from './selection.js';

function tile(app, c, extra) {
  const t = h('div', { class: 'tile' + (c.changed ? ' edited' : ''), title: c.addr, onclick: () => app.select(c.addr) },
    h('div', { class: 't-head' }, h('b', {}, c.label), h('span', { class: 'muted small' }, c.shape ? c.shape.join('×') : fmtInt(c.n_params))));
  if (extra) t.append(extra);
  t.append(h('div', { class: 'small muted' }, (c.norm != null ? `|w| ${fmt(c.norm)}` : `${fmtInt(c.n_params)} weights`) + (c.changed ? `  ·  Δ ${fmt(c.dnorm)}` : '')));
  return t;
}

export async function weightsView(el, app, active = () => true) {
  const addr = app.sel;
  const [n, t] = await Promise.all([inspection.read(app, '/api/node', { addr }), inspection.read(app, '/api/tree', { addr })]);
  if (!active()) return;
  if (n.analysis) calculation(el, app, { path: '/api/node', params: { addr, analysis: 1, fetch: n.needs_fetch ? 1 : 0 },
    label: n.needs_fetch ? 'Fetch vocabulary and calculate readings' : 'Calculate vocabulary readings',
    description: 'Projects weights through the vocabulary. This may decode large embedding matrices.' + (n.needs_fetch ? ` Downloads about ${(n.needs_fetch / 1e9).toFixed(1)} GB from the Hub.` : ''),
    active, render: (box, data) => {
      if (data.head) headCard(box, app, data.head);
      if (data.neuron) neuronCard(box, app, data.neuron);
      if (data.direction) box.append(h('h3', {}, 'Residual direction'), h('p', {}, 'Promotes'), tokList(data.direction.promotes.top, 'up'), h('p', {}, 'Suppresses'), tokList(data.direction.promotes.bottom, 'down'));
    } });
  el.append(breadcrumb(app, n), h('div', { class: 'row' }, h('h2', { class: 'mono' }, n.addr), h('span', { class: 'pill' }, `level ${n.level} · ${n.kind}`),
    h('span', { class: 'muted small' }, `${fmtInt(n.n_params)} weights`), n.stats.changed ? h('span', { class: 'pill edit' }, `${fmtInt(n.stats.changed)} edited`) : null,
    app.variant !== 'current' ? h('span', { class: 'pill warn' }, `showing ${app.variant}`) : null));
  if (n.kind === 'tensor' && n.slices.length === 1) {
    const s = n.slices[0];
    if (s.shape.length === 3) return facets(el, app, n, s);
    return matrixEditor(el, app, n, s, active);
  }
  if (n.head) headCard(el, app, n.head);
  if (n.neuron) neuronCard(el, app, n.neuron);
  if (n.addr.startsWith('group:')) { const g = groupSummary(app, n); if (g) el.append(g); }
  const kids = t.children;
  if (!kids.length) return;
  const tensors = kids.filter((c) => c.kind === 'tensor'), others = kids.filter((c) => c.kind !== 'tensor');
  if (kids.length > 1 && (n.kind === 'model' || n.kind === 'block')) {      // where the weights are: one segment per part, width = share of weights, fill = share edited
    const total = kids.reduce((a, c) => a + c.n_params, 0);
    const bar = h('div', { style: { display: 'flex', gap: '2px', height: '26px', margin: '10px 0 2px' } });
    for (const c of kids) {
      const share = c.n_params / total;
      if (share < 0.002) continue;
      bar.append(h('div', { class: 'clickable', title: `${c.addr}\n${fmtInt(c.n_params)} weights (${(100 * share).toFixed(1)}%)` + (c.changed ? `\n${fmtInt(c.changed)} edited` : ''),
        onclick: () => app.select(c.addr),
        style: { flex: `${share} 0 0`, minWidth: '3px', background: c.changed ? 'var(--s2)' : 'var(--surface-3)', borderRadius: '3px', overflow: 'hidden', fontSize: '10.5px', padding: '5px 4px', whiteSpace: 'nowrap', color: 'var(--text-2)' } },
        share > 0.05 ? c.label : ''));
    }
    el.append(bar, h('div', { class: 'hint' }, 'share of weights per part' + (kids.some((c) => c.changed) ? ' · orange = contains edited weights' : '')));
  }
  if (others.length) {
    el.append(h('h3', {}, others[0].kind === 'unit' ? (others[0].addr.includes('.head.') ? 'heads' : 'neurons') : 'parts'));
    const grid = h('div', { class: 'cards', style: others[0].kind === 'unit' && !others[0].addr.includes('.head.') ? { gridTemplateColumns: 'repeat(auto-fill, minmax(120px, 1fr))' } : {} });
    for (const c of others) {
      const slot = h('div', {});
      grid.append(tile(app, c, slot));

    }
    el.append(grid);
  }
  if (tensors.length) {
    el.append(h('h3', {}, 'tensors'));
    const grid = h('div', { class: 'cards' });
    for (const c of tensors) {
      const slot = h('div', {});
      grid.append(tile(app, c, slot));
      slot.append(h('div', { class: 'small muted' }, 'open to view (streamed)'));
    }
    el.append(grid);
  }
}

function facets(el, app, n, s) {
  el.append(h('p', { class: 'muted' }, `${s.doc}. One facet per head; click a facet to open it.`));
  const grid = h('div', { class: 'cards' });
  for (let i = 0; i < s.shape[0]; i++) {
    const a = `${s.param}[${i}]`;
    const slot = h('div', {});
    grid.append(h('div', { class: 'tile', onclick: () => app.select(a) }, h('div', { class: 't-head' }, h('b', {}, `head ${i}`), h('span', { class: 'muted small' }, s.shape.slice(1).join('×'))), slot));

  }
  el.append(grid);
}

// Edited weights open side by side (base, current, difference); "single view" goes back to the editable heatmap.
export const showCompare = (app, n, st) => Boolean(n.stats && n.stats.changed) && app.variant === 'current' && st.compare !== false;

async function compareEditor(el, app, n, s, st, active) {
  const args = { addr: n.addr, pool: st.pool, max_h: 220, max_w: 300 };
  if (st.region) Object.assign(args, { r0: st.region[0], r1: st.region[1], c0: st.region[2], c1: st.region[3] });
  const m = await inspection.read(app, '/api/matrix/compare', args);
  if (!active()) return;
  const availW = Math.max(360, Math.min(1100, (document.querySelector('#view')?.clientWidth || 980) - 36));
  el.append(h('p', { class: 'muted' }, s.doc),
    compareView(app, m, { width: Math.floor((availW - 24) / 3), height: 300, onSingle: () => { st.compare = false; app.rerender(); } }));
  if (st.region) el.append(h('button', { onclick: () => { st.region = null; app.rerender(); } }, 'reset zoom'));
}

async function matrixEditor(el, app, n, s, active) {
  const st = app.keep['mx:' + n.addr] || (app.keep['mx:' + n.addr] = { pool: 'mean', sr: '', sc: '', region: null, svd: false });
  if (showCompare(app, n, st)) return compareEditor(el, app, n, s, st, active);
  const args = { addr: n.addr, variant: app.variant, pool: st.pool, sort_rows: st.sr, sort_cols: st.sc, max_h: 280, max_w: 460 };
  if (st.region) Object.assign(args, { r0: st.region[0], r1: st.region[1], c0: st.region[2], c1: st.region[3] });
  const m = await inspection.read(app, '/api/matrix', args);
  if (!active()) return;
  const data = decodeF32(m.data);
  const [R, C] = m.shape, [gr, gc] = m.grid, [r0, r1, c0, c1] = m.region;
  const again = () => app.rerender();
  const sortSel = (key, label) => h('label', { class: 'small muted' }, label + ' ', h('select', { onchange: (e) => { st[key] = e.target.value; st.region = null; again(); } },
    [['', 'as stored'], ['norm', 'by norm'], ['svd', 'by 1st singular vector'], ['cluster', 'by similarity']].map(([v, l]) => h('option', { value: v, selected: st[key] === v, disabled: v && !m.sortable }, l))));
  el.append(h('p', { class: 'muted' }, s.doc));
  el.append(h('div', { class: 'row', style: { margin: '8px 0' } },
    h('span', { class: 'small muted' }, `rows = ${m.row_role} (${R}) · columns = ${m.col_role} (${C})`),
    m.factor[0] * m.factor[1] > 1 ? h('span', { class: 'pill warn', title: 'more cells than pixels: each drawn cell pools several weights' }, `pooled ${m.factor[0]}×${m.factor[1]}`) : h('span', { class: 'pill' }, 'cell-exact'),
    m.factor[0] * m.factor[1] > 1 ? tabs(['mean', 'absmax', 'rms'], st.pool, (p) => { st.pool = p; again(); }) : null,
    sortSel('sr', 'rows'), sortSel('sc', 'cols'),
    st.region ? h('button', { onclick: () => { st.region = null; again(); } }, `reset zoom (rows ${r0}–${r1 - 1}, cols ${c0}–${c1 - 1})`) : h('span', { class: 'small muted' }, 'drag to zoom · click a cell to edit'),
    n.stats.changed && app.variant === 'current' ? h('button', { class: 'ghost', onclick: () => { st.compare = true; again(); } }, 'before / after') : null,
    h('span', { class: 'grow' }), legend(m.scale)));
  const edit = h('div', { class: 'card', style: { display: 'none', margin: '8px 0' } });
  const realRow = (r) => (m.row_perm ? m.row_perm[r0 + r * m.factor[0]] : r0 + r * m.factor[0]);
  const realCol = (c) => (m.col_perm ? m.col_perm[c0 + c * m.factor[1]] : c0 + c * m.factor[1]);
  const lab = (labels, i, role, off) => (labels ? `${role} ${off + i}${role === 'token' ? ' ' + JSON.stringify(labels[i]) : ''}` : null);
  const exact = m.factor[0] === 1 && m.factor[1] === 1;
  const availW = Math.max(320, Math.min(1100, (document.querySelector('#view').clientWidth || 980) - 36));
  const cv = heatmap({ rows: gr, cols: gc, data, scale: m.scale,
    width: Math.max(160, Math.min(availW, gc * Math.max(1, Math.floor(availW / gc)))), height: Math.max(28, Math.min(520, gr * Math.max(1, Math.min(22, Math.floor(520 / gr))))),
    onHover: (r, c, v, e) => {
      if (r == null) return tip(null);
      const rr = realRow(r), cc = realCol(c);
      const rl = (m.row_labels && exact && lab(m.row_labels, r, m.row_role, m.row_offset + r0)) || `${m.row_role} ${m.row_offset + rr}${exact ? '' : '–' + (m.row_offset + rr + m.factor[0] - 1)}`;
      const cl = (m.col_labels && exact && lab(m.col_labels, c, m.col_role, m.col_offset + c0)) || `${m.col_role} ${m.col_offset + cc}${exact ? '' : '–' + (m.col_offset + cc + m.factor[1] - 1)}`;
      tip(`${esc(rl)}<br>${esc(cl)}<br><b>${fmt(v, 4)}</b>${exact ? '' : ' (' + m.pool + ')'}`, e);
    },
    onRegion: (a, b, c, d) => {
      if (st.sr || st.sc) return;
      st.region = [r0 + a * m.factor[0], Math.min(R, r0 + b * m.factor[0]), c0 + c * m.factor[1], Math.min(C, c0 + d * m.factor[1])];
      again();
    },
    onClick: async (r, c) => {
      if (!exact) { st.region = [Math.max(0, realRow(r) - 8), Math.min(R, realRow(r) + 8), Math.max(0, realCol(c) - 12), Math.min(C, realCol(c) + 12)]; if (!st.sr && !st.sc) again(); return; }
      const cell = await api('/api/cell?' + q({ addr: n.addr, r: realRow(r), c: realCol(c) }));
      const inp = h('input', { type: 'number', step: 'any', value: Number(cell.current.toPrecision(6)), style: { width: '120px' } });
      edit.style.display = 'block';
      edit.replaceChildren(h('div', { class: 'row' }, h('code', {}, cell.script), '=', inp,
        h('button', { class: 'primary', onclick: () => app.runScript(`${cell.script} = ${inp.value}`) }, 'set'),
        h('button', { onclick: () => app.runScript(`${cell.script} = 0`) }, 'zero'),
        h('button', { class: 'ghost', onclick: () => app.setConsole(`${cell.script} = ${inp.value}`, true) }, '→ console'),
        h('span', { class: 'muted small' }, `base ${fmt(cell.base, 5)}`)));
      inp.focus(); inp.select();
    } });
  el.append(cv, edit);
  if (Math.min(R, C) < 2) return;
  calculation(el, app, { path: '/api/svd', params: { addr: n.addr, variant: app.variant, k: 6 },
    label: 'Calculate SVD and vocabulary readings', active,
    description: 'Computes singular vectors and their vocabulary projections; may read large weight matrices, including vocabulary weights from the Hub.',
    render: (box, sv) => spectrum(box, app, n, sv) });
}

function spectrum(box, app, n, sv) {
  const X = toScript(n.addr);
  box.append(h('h3', {}, sv.rank != null ? `singular value decomposition — rank ${sv.rank}, effective rank ${fmt(sv.effective_rank)}` : 'singular value decomposition — top 64, randomised (the matrix is too large for an exact one)'),
    vbars(sv.singular_values.slice(0, 96), { width: Math.min(700, 14 * Math.min(96, sv.singular_values.length)), height: 64, cls: 'bar plain',
      label: (i, v) => `σ<sub>${i}</sub> = ${fmt(v, 4)}<br>${fmt(100 * sv.energy[i])}% of energy up to here` }));
  const tbl = h('table', { class: 'grid', style: { marginTop: '8px' } }, h('tr', {}, h('th', {}, '#'), h('th', { class: 'num' }, 'σ'), h('th', {}, `left · ${sv.row_role}`), h('th', {}, `right · ${sv.col_role}`), h('th', {}, '')));
  const side = (d) => {
    if (!d) return h('span', { class: 'muted' }, '–');
    if (d.kind === 'resid') return h('div', {}, tokList(d.promotes.top.slice(0, 6), 'up'), h('div', { style: { height: '3px' } }), tokList(d.promotes.bottom.slice(0, 6), 'down'));
    return h('div', {}, tokList(d.top.slice(0, 6), 'up'), h('div', { style: { height: '3px' } }), tokList(d.bottom.slice(0, 6), 'down'));
  };
  sv.components.forEach((c) => tbl.append(h('tr', {}, h('td', {}, c.i), h('td', { class: 'num' }, fmt(c.s, 4)), h('td', {}, side(c.left)), h('td', {}, side(c.right)),
    h('td', {}, n.slices[0].shape.length === 2 ? h('button', { class: 'ghost', title: 'put a script in the console that rescales this singular component',
      onclick: () => app.setConsole(`Uc, Sc, Vc = svd(${X})\n${X} -= 1.0 * Sc[${c.i}] * outer(Uc[:, ${c.i}], Vc[${c.i}])   # 1.0 removes component ${c.i}; 0.5 halves it; -1.0 doubles it`) }, 'rescale…') : null))));
  box.append(tbl, h('p', { class: 'hint' }, 'For axes in the residual stream, a singular vector is read through the unembedding: tokens it promotes (red edge) and suppresses (blue edge). For token / neuron axes the largest entries are listed.'));
}

function headCard(el, app, c) {
  const pref = c.pos_offset_preference;
  const left = pref ? h('div', { class: 'card' },
    h('h3', { style: { marginTop: 0 } }, 'where it looks (from position alone)'),
    hbars(pref.slice(0, 10).map((v, i) => ({ label: i === 0 ? 'self' : `−${i}`, value: v })), { width: 290, labelW: 40 }),
    h('p', { class: 'hint' }, 'Attention score for the token `offset` back, relative to an average earlier token, using position embeddings only.'),
    h('h3', {}, 'position × position (QK)'),
    heatmap({ rows: c.pos_qk.grid[0], cols: c.pos_qk.grid[1], data: decodeF32(c.pos_qk.data), scale: c.pos_qk.scale, width: 180, height: 180,
      onHover: (r, cc, v, e) => (r == null ? tip(null) : tip(`destination ${r} → source ${cc}<br><b>${fmt(v)}</b>`, e)) }))
    : h('div', { class: 'card' }, h('h3', { style: { marginTop: 0 } }, 'where it looks'),
      h('p', { class: 'muted' }, 'This model has no position-embedding tensor (positions enter inside attention, e.g. rotary), so positional preference cannot be read from weights. Use the Trace tab to see attention on a prompt.'),
      c.kv_group != null ? h('p', {}, `Shares its keys and values with the other heads of group ${c.kv_group}.`) : null);
  const ovt = h('table', { class: 'grid' }, h('tr', {}, h('th', {}, 'source token'), h('th', {}, 'promotes')));
  c.ov_examples.forEach((e) => ovt.append(h('tr', {}, h('td', {}, h('span', { class: 'tok' }, e.src)), h('td', {}, tokList(e.out, 'up')))));
  const mid = h('div', { class: 'card' },
    h('h3', { style: { marginTop: 0 } }, 'what it moves (OV circuit)'),
    h('dl', { class: 'kv' }, h('dt', {}, 'copies own token'), h('dd', {}, `top-1 ${fmt(100 * c.copying.top1)}% · top-10 ${fmt(100 * c.copying.top10)}% of ${c.copying.sample} tokens`),
      h('dt', {}, 'norms'), h('dd', {}, `Q ${fmt(c.norms.Q)} · K ${fmt(c.norms.K)} · V ${fmt(c.norms.V)} · O ${fmt(c.norms.O)}`)), ovt);
  const qkt = h('table', { class: 'grid' }, h('tr', {}, h('th', {}, 'query'), h('th', {}, 'prefers key'), h('th', { class: 'num' }, 'score')));
  c.qk_examples.forEach((e) => qkt.append(h('tr', {}, h('td', {}, h('span', { class: 'tok' }, e.query)), h('td', {}, h('span', { class: 'tok' }, e.key)), h('td', { class: 'num' }, fmt(e.score)))));
  const right = h('div', { class: 'card' }, h('h3', { style: { marginTop: 0 } }, 'which tokens it matches (QK circuit)'), qkt,
    h('h3', {}, 'OV spectrum'), vbars(c.ov_svd.singular_values, { width: 240, height: 40, cls: 'bar plain', label: (i, v) => `σ${i} = ${fmt(v)}` }),
    c.ov_svd.components.slice(0, 2).map((k, i) => h('div', { class: 'small', style: { marginTop: '4px' } }, h('span', { class: 'muted' }, `component ${i} reads `), tokList(k.reads.top.slice(0, 5)), h('span', { class: 'muted' }, ' writes '), tokList(k.writes.top.slice(0, 5), 'up'))));
  el.append(h('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '10px', margin: '10px 0' } }, left, mid, right),
    h('p', { class: 'hint' }, 'All of this is read from the weights with LayerNorm and earlier layers ignored; the Trace tab shows what the head does on a real prompt.'));
}

function neuronCard(el, app, k) {
  const reads = h('div', { class: 'card' }, h('h3', { style: { marginTop: 0 } }, k.gated ? 'reads — activation = act(x · gate) × (x · up)' : `reads — fires when x · read + ${fmt(k.bias ?? 0)} > 0`));
  k.reads.forEach((r) => reads.append(h('div', {},
    h('div', { class: 'small', style: { marginTop: '6px' } }, h('b', {}, r.kind), h('span', { class: 'muted' }, `  |w| ${fmt(r.norm)} · cos with write ${fmt(r.cos_write)}` + (r.bias != null ? ` · bias ${fmt(r.bias)}` : ''))),
    h('div', { class: 'small muted' }, 'token embeddings along it'), tokList(r.embeddings.top, 'up'),
    h('div', { class: 'small muted', style: { marginTop: '3px' } }, 'against it'), tokList(r.embeddings.bottom, 'down'),
    r.positions ? [h('div', { class: 'small muted', style: { marginTop: '3px' } }, 'positions'), tokList(r.positions.top)] : null)));
  el.append(h('div', { class: 'split', style: { margin: '10px 0' } }, reads,
    h('div', { class: 'card' }, h('h3', { style: { marginTop: 0 } }, 'writes — adds activation × its write vector'),
      h('div', { class: 'small muted' }, 'promotes'), tokList(k.writes.promotes.top, 'up'),
      h('div', { class: 'small muted', style: { marginTop: '6px' } }, 'suppresses'), tokList(k.writes.promotes.bottom, 'down'),
      h('p', { class: 'hint' }, `cos(read, write) = ${fmt(k.cos_in_out)} → ${k.mechanism}: ${k.mechanism === 'amplifier' ? 'strengthens the direction it detects' : k.mechanism === 'damper' ? 'writes against the direction it detects' : 'writes a direction unrelated to what it reads'}.`))));
}
