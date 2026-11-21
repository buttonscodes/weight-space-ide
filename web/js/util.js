// DOM, API, colour scales, canvas heatmap and small SVG charts shared by every view.

export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'html') el.innerHTML = v;
    else if (k in el && k !== 'list') { try { el[k] = v; } catch { el.setAttribute(k, v); } }
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

export const $ = (sel, root = document) => root.querySelector(sel);
export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

// Called with (path, body) when this tab's own /api/run or /api/chat request starts, and when it settles,
// however it ends.
export const generationStarted = new Set();
export const generationSettled = new Set();
const GENERATING = new Set(['/api/run', '/api/chat']);

export async function api(path, body, options = {}) {
  if (!GENERATING.has(path)) return request(path, body, options);
  for (const fn of generationStarted) { try { fn(path, body); } catch (e) { console.error(e); } }
  try { return await request(path, body, options); }
  finally { for (const fn of generationSettled) { try { fn(path, body); } catch (e) { console.error(e); } } }
}

async function request(path, body, options) {
  const opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const r = await fetch(path, {...opt, ...options});
  const ct = r.headers.get('content-type') || '';
  const data = ct.includes('json') ? await r.json() : await r.text();
  if (!r.ok) {
    const error = new Error((data && data.error) || (data && data.detail) || r.statusText);
    error.payload = data; error.event = data?.event; throw error;
  }
  return data;
}
export const q = (o) => Object.entries(o).filter(([, v]) => v !== undefined && v !== null && v !== '').map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&');

export function fmt(x, d = 3) {
  if (x == null || Number.isNaN(x)) return '–';
  if (x === 0) return '0';
  const a = Math.abs(x);
  if (a >= 1e5 || a < 1e-3) return x.toExponential(Math.max(0, d - 1));
  return Number(x.toPrecision(d)).toString();
}
export function fmtInt(n) {
  if (n == null) return '–';
  if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e4) return (n / 1e3).toFixed(1) + 'k';
  return String(n);
}
export function decodeF32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}
export function debounce(fn, ms = 200) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}
export function toScript(addr) { return addr.replace(/^blocks\.(\d+)/, 'blocks[$1]'); }

// ---------------------------------------------------------------- colour
// Diverging: blue (negative) <-> neutral gray <-> red (positive). Sequential: surface -> blue.
const THEMES = {
  dark: { neg: [78, 152, 240], mid: [38, 38, 36], pos: [236, 96, 96], seq0: [30, 30, 29], seq1: [96, 166, 246] },
  light: { neg: [28, 92, 171], mid: [240, 239, 236], pos: [200, 50, 50], seq0: [246, 246, 244], seq1: [28, 92, 171] },
};
export function theme() { return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark'; }
function mix(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }
export function divergeRGB(t) {          // t in [-1, 1]
  const T = THEMES[theme()];
  t = Math.max(-1, Math.min(1, t));
  const e = Math.pow(Math.abs(t), 0.7);
  return mix(T.mid, t < 0 ? T.neg : T.pos, e);
}
export function seqRGB(t) {              // t in [0, 1]
  const T = THEMES[theme()];
  return mix(T.seq0, T.seq1, Math.pow(Math.max(0, Math.min(1, t)), 0.8));
}
export const css = (rgb) => `rgb(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0})`;
export const diverge = (t) => css(divergeRGB(t));
export const seq = (t) => css(seqRGB(t));

// ---------------------------------------------------------------- heatmap
// data: Float32Array (rows*cols, row-major); mode 'div' (signed, scale = value at full colour) or 'seq' (0..scale)
export function heatmap(opts) {
  const { rows, cols, data, scale = 1, mode = 'div', width, height, onHover, onClick, onRegion, cell } = opts;
  const canvas = h('canvas', { class: 'heat' + (onClick || onRegion ? ' clickable' : '') });
  const off = document.createElement('canvas');
  off.width = cols; off.height = rows;
  const octx = off.getContext('2d');
  const img = octx.createImageData(cols, rows);
  const f = mode === 'div' ? divergeRGB : seqRGB;
  for (let i = 0; i < rows * cols; i++) {
    const v = data[i];
    const c = Number.isNaN(v) ? [128, 128, 128] : f(v / scale);
    img.data[4 * i] = c[0]; img.data[4 * i + 1] = c[1]; img.data[4 * i + 2] = c[2]; img.data[4 * i + 3] = 255;
  }
  octx.putImageData(img, 0, 0);
  let W = width, H = height;
  if (cell) { W = cols * cell; H = rows * cell; }
  if (!W) W = Math.max(60, Math.min(900, cols * Math.max(1, Math.floor(900 / cols))));
  if (!H) H = Math.max(24, Math.min(560, rows * Math.max(1, Math.floor(560 / rows))));
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  const draw = (sel) => {
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
    if (sel) {
      ctx.strokeStyle = theme() === 'dark' ? '#fff' : '#000';
      ctx.lineWidth = 1.5 * dpr;
      ctx.strokeRect(sel.x0 * dpr, sel.y0 * dpr, (sel.x1 - sel.x0) * dpr, (sel.y1 - sel.y0) * dpr);
    }
  };
  draw();
  const at = (e) => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    return { x, y, row: Math.max(0, Math.min(rows - 1, Math.floor(y / r.height * rows))), col: Math.max(0, Math.min(cols - 1, Math.floor(x / r.width * cols))) };
  };
  let drag = null;
  canvas.addEventListener('mousemove', (e) => {
    const p = at(e);
    if (drag) { draw({ x0: Math.min(drag.x, p.x), y0: Math.min(drag.y, p.y), x1: Math.max(drag.x, p.x), y1: Math.max(drag.y, p.y) }); }
    if (onHover) onHover(p.row, p.col, data[p.row * cols + p.col], e);
  });
  canvas.addEventListener('mouseleave', () => { if (onHover) onHover(null); if (drag) { drag = null; draw(); } });
  canvas.addEventListener('mousedown', (e) => { if (onRegion) drag = at(e); });
  canvas.addEventListener('mouseup', (e) => {
    const p = at(e);
    if (drag && (Math.abs(p.x - drag.x) > 6 || Math.abs(p.y - drag.y) > 6)) {
      const r0 = Math.min(drag.row, p.row), r1 = Math.max(drag.row, p.row) + 1, c0 = Math.min(drag.col, p.col), c1 = Math.max(drag.col, p.col) + 1;
      drag = null; draw();
      onRegion(r0, r1, c0, c1);
      return;
    }
    drag = null; draw();
    if (onClick) onClick(p.row, p.col, data[p.row * cols + p.col], e);
  });
  return canvas;
}

let tipEl;
export function tip(html, e) {
  if (!tipEl) { tipEl = h('div', { class: 'tip' }); document.body.append(tipEl); }
  if (html == null) { tipEl.style.display = 'none'; return; }
  tipEl.innerHTML = html;
  tipEl.style.display = 'block';
  const w = tipEl.offsetWidth, hh = tipEl.offsetHeight;
  tipEl.style.left = Math.min(window.innerWidth - w - 8, e.clientX + 14) + 'px';
  tipEl.style.top = Math.min(window.innerHeight - hh - 8, e.clientY + 14) + 'px';
}
export const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
// A token string with its invisible characters written out: control characters, bidirectional controls (U+202E
// would reverse the text after it), zero-width characters and line separators become \u{XXXX}.
const INVISIBLE = /[\u0000-\u001f\u007f-\u009f\u00ad\u061c\u180e\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff\ufff9-\ufffb]/g;
export const visible = (s) => String(s).replace(INVISIBLE, (c) => `\\u{${c.codePointAt(0).toString(16).toUpperCase().padStart(4, '0')}}`);

export function legend(scale, mode = 'div') {
  const stops = mode === 'div' ? [-1, -0.5, 0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1];
  const grad = stops.map((t, i) => `${mode === 'div' ? diverge(t) : seq(t)} ${i * 25}%`).join(',');
  return h('div', { class: 'legend' },
    h('span', {}, mode === 'div' ? '−' + fmt(scale) : '0'),
    h('span', { class: 'legend-bar', style: { background: `linear-gradient(90deg, ${grad})` } }),
    h('span', {}, (mode === 'div' ? '+' : '') + fmt(scale)));
}

// ---------------------------------------------------------------- small SVG charts
const NS = 'http://www.w3.org/2000/svg';
export function svg(tag, attrs, ...kids) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null) continue;
    if (k.startsWith('on')) el.addEventListener(k.slice(2), v); else el.setAttribute(k, v);
  }
  for (const kid of kids.flat(Infinity)) if (kid != null) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return el;
}

// horizontal signed bars: items [{label, value, title?, onClick?}]
export function hbars(items, { width = 300, rowH = 18, labelW = 86, signed = true, max, valueFmt = fmt, cls } = {}) {
  const m = max || Math.max(1e-12, ...items.map((d) => Math.abs(d.value)));
  const H = items.length * rowH + 2;
  const plotW = width - labelW - 50;
  const zero = signed ? labelW + plotW / 2 : labelW;
  const unit = (signed ? plotW / 2 : plotW) / m;
  const root = svg('svg', { width, height: H, viewBox: `0 0 ${width} ${H}`, class: 'chart',
    style: 'max-width:100%;height:auto' });
  root.append(svg('line', { x1: zero, x2: zero, y1: 0, y2: H, class: 'axis' }));
  items.forEach((d, i) => {
    const y = i * rowH + 2, w = Math.max(1, Math.abs(d.value) * unit);
    const g = svg('g', { class: d.onClick ? 'clickable' : '', onclick: d.onClick,
      onmousemove: (e) => tip(`<b>${esc(d.title || d.label)}</b><br>${valueFmt(d.value)}`, e), onmouseleave: () => tip(null) });
    g.append(svg('rect', { x: 0, y: y - 1, width, height: rowH, fill: 'transparent' }));
    g.append(svg('text', { x: labelW - 6, y: y + rowH / 2 + 3, 'text-anchor': 'end', class: 'lbl' }, d.label));
    g.append(svg('rect', { x: d.value < 0 ? zero - w : zero, y: y + 2, width: w, height: rowH - 6, rx: 2,
      class: cls || (d.value < 0 ? 'bar neg' : 'bar pos') }));
    g.append(svg('text', { x: width - 2, y: y + rowH / 2 + 3, 'text-anchor': 'end', class: 'val' }, valueFmt(d.value)));
    root.append(g);
  });
  return root;
}

// vertical bars for a histogram or spectrum: values [..]; labels(i) for tooltip
export function vbars(values, { width = 280, height = 70, signedX = false, label = (i, v) => fmt(v), log = false, cls = 'bar pos', lim } = {}) {
  const root = svg('svg', { width, height: height + 14, class: 'chart' });
  const vals = log ? values.map((v) => Math.log10(Math.max(v, 1e-12))) : values;
  const lo = log ? Math.min(...vals) - 0.2 : 0;
  const hi = Math.max(...vals, lo + 1e-12);
  const bw = width / values.length;
  values.forEach((v, i) => {
    const hh = Math.max(1, (vals[i] - lo) / (hi - lo) * height);
    root.append(svg('rect', { x: i * bw + 0.5, y: height - hh, width: Math.max(1, bw - 1), height: hh, rx: bw > 5 ? 1.5 : 0, class: cls,
      onmousemove: (e) => tip(label(i, v), e), onmouseleave: () => tip(null) }));
  });
  root.append(svg('line', { x1: 0, x2: width, y1: height + 0.5, y2: height + 0.5, class: 'axis' }));
  if (signedX && lim != null) {
    root.append(svg('line', { x1: width / 2, x2: width / 2, y1: 0, y2: height, class: 'axis' }));
    root.append(svg('text', { x: 0, y: height + 12, class: 'val' }, '−' + fmt(lim)));
    root.append(svg('text', { x: width / 2, y: height + 12, class: 'val', 'text-anchor': 'middle' }, '0'));
    root.append(svg('text', { x: width, y: height + 12, class: 'val', 'text-anchor': 'end' }, '+' + fmt(lim)));
  }
  return root;
}

export function tokList(pairs, cls = '') {
  return h('div', { class: 'toks ' + cls }, pairs.map((p) => h('span', { class: 'tok', title: `${p[1]}` + (p[2] != null ? ` · id ${p[2]}` : '') }, p[0] === '' ? '∅' : p[0])));
}

// A segmented control: one button per name, the active one pressed. Options: `labels` (name -> shown text, default the
// name), `label` (the group's accessible name) and `disabled` (every button).
export function tabs(names, active, onPick, { labels = {}, label = null, disabled = false } = {}) {
  return h('div', { class: 'seg', role: label ? 'group' : null, 'aria-label': label },
    names.map((n) => h('button', { class: n === active ? 'on' : '', 'aria-pressed': n === active ? 'true' : 'false', disabled,
      onclick: () => { if (!disabled) onPick(n); } }, labels[n] ?? n)));
}

export function field(label, input, hint) {
  return h('label', { class: 'field' }, h('span', { class: 'field-label' }, label), input, hint ? h('span', { class: 'hint' }, hint) : null);
}
