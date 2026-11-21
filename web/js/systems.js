// Systems view: the model as a graph of components, at three levels of organisation.
// Nodes are heads / MLPs (or whole sublayers, or whole blocks); edges are composition scores read from the
// weights: how strongly a later component's reader is aligned with what an earlier one writes.
import { h, api, fmt, svg, tip, esc, seq, diverge, tabs } from './util.js';

import { inspection, calculation } from './inspection.js';

const KINDS = [['Q', 'query', 'var(--s1)'], ['K', 'key', 'var(--s2)'], ['V', 'value', 'var(--s3)'], ['in', 'MLP input', 'var(--s4)'], ['gate', 'MLP gate', 'var(--s4)'], ['U', 'unembed', 'var(--s5)']];
const KCOL = Object.fromEntries(KINDS.map(([k, , c]) => [k, c]));

export async function systemsView(el, app, active = () => true) {
  const st = app.keep.sys || (app.keep.sys = { level: 'units', thr: 1.6, kinds: new Set(['Q', 'K', 'V', 'in', 'gate', 'U']), overlay: 'write norm', focus: null });
  if (!app.state.open.recognised) {
    el.append(h('h2', {}, 'Systems'), h('p', { class: 'muted' }, 'The architecture of this checkpoint was not recognised, so it has no heads or MLPs to connect. Its tensors are in the explorer under their own names.'));
    return;
  }
  const cfg = app.state.open.config;
  const m = /^blocks\.(\d+)/.exec(app.sel);
  const want = st.window != null ? st.window : (m ? +m[1] : -1);
  const large = app.state.open.n_params > 400000000;
  const first = Math.max(0, Math.min(cfg.n_layers-1, want < 0 ? 0 : want));
  const count = Math.min(3, Math.max(1, st.count || 1), cfg.n_layers-first);
  const params = large ? {top:4000,layers:Array.from({length:count},(_,i)=>first+i).join(',')} : {top:4000,focus:want};
  if (large) {
    el.append(h('p',{class:'hint'},'Explicit window: at most 3 layers, 512M logical values, 128 MiB reduction budget and 120 seconds. Recurrent/hybrid attention composition is omitted; mapped MLPs and full-attention heads are shown. No inference.'));
    el.append(h('div',{class:'row'},'First layer',h('input',{type:'number',min:0,max:cfg.n_layers-1,value:first,onchange:e=>{st.window=+e.target.value;app.rerender();}}),
      'Layer count',h('input',{type:'number',min:1,max:3,value:count,onchange:e=>{st.count=+e.target.value;app.rerender();}})));
  }
  calculation(el, app, { path: '/api/wiring', params, label: large ? 'Analyze window' : 'Calculate connections', active,
    cancellable: true, progressEvent: 'wsp-wiring-progress',
    description: 'Reads bounded logical slabs. Matching requests share work; leaving the view does not cancel it. Cancel explicitly to stop the shared calculation.',
    render: (box, w) => drawSystems(box, app, w, st, cfg, active) });
}

function drawSystems(el, app, w, st, cfg, active) {
  for (const note of w.limitations || []) el.append(h('p',{class:'hint'},note));
  const win = w.window, L0 = win[0], NL = win.length, partial = NL < cfg.n_layers;
  const again = () => app.rerender();
  // ---- collapse to the chosen level
  const groupOf = (n) => {
    if (n.kind === 'embed' || n.kind === 'unembed' || st.level === 'units') return n.id;
    if (st.level === 'components') return n.kind === 'head' ? `L${n.layer}ATTN` : n.id;
    return `B${n.layer}`;
  };
  const nodes = new Map();
  for (const n of w.nodes) {
    const g = groupOf(n);
    if (!nodes.has(g)) {
      const base = g === n.id ? { ...n } : st.level === 'components'
        ? { id: g, kind: 'attn', layer: n.layer, label: `attn ${n.layer}`, addr: `blocks.${n.layer}.attn`, norm: 0, edited: false }
        : { id: g, kind: 'block', layer: n.layer, label: `block ${n.layer}`, addr: `blocks.${n.layer}`, norm: 0, edited: false };
      nodes.set(g, base);
    }
    const t = nodes.get(g);
    if (g !== n.id) { t.norm = Math.hypot(t.norm, n.norm || 0); t.edited = t.edited || n.edited; }
  }
  const gid = Object.fromEntries(w.nodes.map((n) => [n.id, groupOf(n)]));
  const emap = new Map();
  for (const e of w.edges) {
    const a = gid[e.src], b = gid[e.dst];
    if (a === b || !st.kinds.has(e.kind)) continue;
    const key = a + '>' + b + '>' + e.kind;
    if (!emap.has(key) || emap.get(key).score < e.score) emap.set(key, { ...e, src: a, dst: b, from: e.src, to: e.dst });
  }
  let edges = [...emap.values()].filter((e) => e.score >= st.thr).sort((a, b) => b.score - a.score).slice(0, 500);

  // ---- overlay values
  let val = {}, vmax = 1, vmode = 'seq', note = '';
  if (st.overlay === 'write norm') {
    for (const n of nodes.values()) val[n.id] = n.norm || 0;
    vmax = Math.max(1e-9, ...Object.values(val));
    note = 'fill = norm of what the component writes';
  } else if (st.overlay === 'ablation') {
    const params = { suite: app.state.task.suite, variant: app.variant };
    const cached = inspection.peek(app, '/api/ablation', params, true);
    if (cached?.state === 'complete') {
      const a = cached.value;
      a.heads.forEach((row, l) => row.forEach((x, hh) => { const g = gid[`L${l}H${hh}`]; if (g) val[g] = Math.max(val[g] ?? -1e9, x); }));
      a.mlps.forEach((x, l) => { const g = gid[`L${l}MLP`]; if (g && x != null) val[g] = Math.max(val[g] ?? -1e9, x); });
      vmax = Math.max(1e-9, ...Object.values(val).map(Math.abs)); vmode = 'div';
      note = `fill = loss increase when zero-ablated (base loss ${fmt(a.base_loss)}, n = ${a.n})`;
    } else {
      note = 'ablation has not been calculated for these weights';
      calculation(el, app, { path: '/api/ablation', params, post: true, label: 'Run ablation scan', active,
        description: 'Loads the runner and performs repeated inference on the selected test suite.',
        render: () => { if (active()) app.rerender(); } });
    }
  } else if (st.overlay === 'attribution') {
    const t = app.keep.lastTrace;
    if (t) {
      for (const d of t.dla) { const g = gid[d.id]; if (g) val[g] = (val[g] || 0) + d.value; }
      vmax = Math.max(1e-9, ...Object.values(val).map(Math.abs)); vmode = 'div';
      note = `fill = direct logit attribution to ${JSON.stringify(t.target)} on the last traced prompt`;
    } else note = 'run a trace first (Trace tab); its direct logit attribution is shown here';
  }

  // ---- layout: one column group per layer
  const NW = st.level === 'units' ? 34 : 70, NH = st.level === 'units' ? 17 : 26, GAPY = 4;
  const colW = st.level === 'units' ? 150 : st.level === 'components' ? 150 : 110;
  const rowsMax = st.level === 'units' ? cfg.n_heads : st.level === 'components' ? 2 : 1;
  const H = Math.max(120, rowsMax * (NH + GAPY) + 60);
  const pos = {};
  const midY = H / 2;
  pos.embed = { x: 10, y: midY - NH - 3 }; pos.pos = { x: 10, y: midY + 3 };
  if (!w.ends) { delete pos.embed; delete pos.pos; }
  for (const n of nodes.values()) {
    if (n.kind === 'embed' || n.kind === 'unembed') continue;
    const x0 = 10 + 46 + 44 + (n.layer - L0) * colW;
    if (n.kind === 'head') pos[n.id] = { x: x0, y: 30 + n.index * (NH + GAPY) };
    else if (n.kind === 'mlp') pos[n.id] = st.level === 'units' ? { x: x0 + NW + 30, y: midY - NH / 2 } : { x: x0, y: midY + 4 };
    else if (n.kind === 'attn') pos[n.id] = { x: x0, y: midY - NH - 4 };
    else pos[n.id] = { x: x0, y: midY - NH / 2 };
  }
  const W = 10 + NW + 50 + NL * colW + 46 + 30;
  if (w.ends) pos.unembed = { x: W - 46 - 10, y: midY - NH / 2 };

  // ---- controls
  el.append(h('div', { class: 'row' }, h('h2', {}, 'Systems'), h('span', { class: 'muted' }, 'composition between components, read from the weights (1.0 = what unrelated random matrices give)')));
  el.append(h('div', { class: 'row', style: { margin: '8px 0' } },
    h('div', { class: 'field' }, h('span', { class: 'field-label' }, 'level of organisation'), tabs(['blocks', 'components', 'units'], st.level, (v) => { st.level = v; st.focus = null; again(); })),
    h('div', { class: 'field' }, h('span', { class: 'field-label' }, `edges with score ≥ ${st.thr.toFixed(1)}`),
      h('input', { type: 'range', min: 1, max: 6, step: 0.1, value: st.thr, style: { width: '150px' }, onchange: (e) => { st.thr = parseFloat(e.target.value); again(); } })),
    h('div', { class: 'field' }, h('span', { class: 'field-label' }, 'reader'),
      h('div', { class: 'row', style: { gap: '4px' } }, KINDS.map(([k, name, col]) => h('button', { class: 'ghost', style: { opacity: st.kinds.has(k) ? 1 : 0.35 }, title: `edges into a ${name} reader`,
        onclick: () => { if (st.kinds.has(k)) st.kinds.delete(k); else st.kinds.add(k); again(); } }, h('span', { style: { display: 'inline-block', width: '14px', height: '3px', background: col, verticalAlign: 'middle', marginRight: '4px' } }), name)))),
    h('div', { class: 'field' }, h('span', { class: 'field-label' }, 'node fill'), tabs(['write norm', 'ablation', 'attribution'], st.overlay, (v) => { st.overlay = v; again(); }))));
  if (partial) {
    const move = (d) => { st.window = Math.max(0, Math.min(cfg.n_layers - 1, (win[1] ?? win[0]) + d)); again(); };
    el.append(h('div', { class: 'row card', style: { margin: '6px 0' } }, h('b', {}, `layers ${L0}–${win[NL - 1]} of ${cfg.n_layers}`),
      h('button', { onclick: () => move(-NL) }, '◂ earlier'), h('button', { onclick: () => move(NL) }, 'later ▸'),
      h('span', { class: 'muted small' }, 'A large model is read a window of layers at a time; select a block in the explorer to jump there.' + (w.sketched ? ` Scores are estimates through a ${w.rank}-dimensional random projection.` : ''))));
  }
  el.append(h('p', { class: 'hint' }, `${edges.length} edges shown of ${w.n_edges} writer → reader pairs. ${note}. Orange outline = edited. Click a node to select it; hover to isolate its edges.` + (w.sketched && !partial ? ` Scores are estimates through a ${w.rank}-dimensional random projection.` : '')));

  // ---- draw
  const root = svg('svg', { class: 'systems', viewBox: `0 0 ${W} ${H}`, style: `width: ${W > 1150 ? W + 'px' : 'min(100%, ' + Math.round(W * 1.7) + 'px)'}; height: auto; display: block` });
  for (let l = 0; l < NL; l++) root.append(svg('text', { x: 10 + 46 + 44 + l * colW, y: 16 }, st.level === 'blocks' ? '' : `block ${L0 + l}`));
  const gE = svg('g', {}), gN = svg('g', {});
  root.append(gE, gN);
  const edgeEls = [];
  for (const e of edges) {
    const a = pos[e.src], b = pos[e.dst];
    if (!a || !b) continue;
    const x1 = a.x + (/MLP|ATTN|^B|embed|pos/.test(e.src) ? Math.max(NW, 46) : NW), y1 = a.y + NH / 2, x2 = b.x, y2 = b.y + NH / 2, dx = Math.max(24, (x2 - x1) * 0.45);
    const p = svg('path', { d: `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`, stroke: KCOL[e.kind], 'stroke-width': Math.min(5, 0.6 + (e.score - 1) * 0.9),
      'stroke-opacity': Math.min(0.9, 0.18 + (e.score - 1) * 0.16),
      onmousemove: (ev) => tip(`<b>${esc(e.from)} → ${esc(e.to)}</b> (${e.kind} reader)<br>composition ${fmt(e.score)}× chance`, ev), onmouseleave: () => tip(null) });
    p.dataset.src = e.src; p.dataset.dst = e.dst;
    edgeEls.push(p); gE.append(p);
  }
  const focus = (id) => edgeEls.forEach((p) => { p.style.opacity = !id || p.dataset.src === id || p.dataset.dst === id ? '' : '0.06'; });
  for (const n of nodes.values()) {
    const p = pos[n.id];
    if (!p) continue;
    const v = val[n.id];
    const fill = v == null ? 'var(--surface-3)' : vmode === 'seq' ? seq(v / vmax) : diverge(v / vmax);
    const g = svg('g', { class: 'n' + (n.edited ? ' edited' : '') + (app.sel === n.addr ? ' sel' : ''), transform: `translate(${p.x},${p.y})`,
      onclick: () => app.select(n.addr), onmouseenter: () => focus(n.id), onmouseleave: () => { focus(null); tip(null); },
      onmousemove: (ev) => tip(`<b>${esc(n.addr)}</b>${v != null ? '<br>' + fmt(v) : ''}${n.edited ? '<br>edited' : ''}`, ev) });
    const nw = n.kind === 'head' ? NW : Math.max(NW, 46);
    g.append(svg('rect', { width: nw, height: NH, rx: 3, fill }));
    g.append(svg('text', { x: nw / 2, y: NH / 2 + 3.5, 'text-anchor': 'middle', style: v != null && Math.abs(v / vmax) > 0.55 ? 'fill:#fff' : '' }, n.label));
    gN.append(g);
  }
  el.append(h('div', { class: 'card scroll', style: { padding: '8px', overflowX: 'auto' } }, root));

  // ---- strongest edges as a table (the same data, sortable by eye, and identity not carried by colour alone)
  const tbl = h('table', { class: 'grid', style: { marginTop: '12px', maxWidth: '620px' } }, h('tr', {}, h('th', {}, 'writer'), h('th', {}, 'reader'), h('th', {}, 'through'), h('th', { class: 'num' }, '× chance')));
  edges.slice(0, 24).forEach((e) => tbl.append(h('tr', { class: 'clickable', onclick: () => app.select(nodes.get(e.dst).addr) }, h('td', { class: 'mono' }, e.from), h('td', { class: 'mono' }, e.to),
    h('td', {}, KINDS.find((k) => k[0] === e.kind)[1]), h('td', { class: 'num' }, fmt(e.score)))));
  el.append(h('h3', {}, 'strongest connections'), tbl);
}
