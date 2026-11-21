// SVG charts of the Directions tab, built with util.js's svg() (no library): the per-layer norms and the cosine of a
// measurement's two means (GET /api/directions/{name} `stats`). Every coordinate is written with toFixed(1). Labels
// and tooltips are text nodes (an SVG <title> child), never markup.
import { svg, fmt } from './util.js';

export const LAYOUT = { width: 640, height: 220, left: 52, right: 12, top: 12, bottom: 30 };
const plotW = LAYOUT.width - LAYOUT.left - LAYOUT.right;           // 576
const plotH = LAYOUT.height - LAYOUT.top - LAYOUT.bottom;          // 178
const x = (k, n) => LAYOUT.left + (n === 0 ? plotW / 2 : k * plotW / n);
const at = (v) => v.toFixed(1);
const points = (list) => list.map(([px, py]) => `${at(px)},${at(py)}`).join(' ');

function frame(id, n, yLabels, title) {
  const { width, height, left, top } = LAYOUT, bottom = top + plotH;
  const root = svg('svg', { id, class: 'chart', width, height, viewBox: `0 0 ${width} ${height}`, style: 'max-width:100%;height:auto' },
    svg('title', {}, title),
    svg('line', { class: 'axis', x1: left, x2: left, y1: top, y2: bottom }),
    svg('line', { class: 'axis', x1: left, x2: width - LAYOUT.right, y1: bottom, y2: bottom }));
  const step = n <= 32 ? 1 : Math.ceil(n / 16);
  for (let k = 0; k <= n; k++) {
    if (k % step === 0 || k === n) root.append(svg('text', { class: 'val', x: at(x(k, n)), y: at(bottom + 24), 'text-anchor': 'middle' }, String(k)));
  }
  for (const [label, y] of yLabels) root.append(svg('text', { class: 'val', x: at(left - 6), y: at(y + 3), 'text-anchor': 'end' }, label));
  return root;
}

// The three mean norms per index, a point per index (click: onPick(k)), "×" under an unusable index, and a marker at
// the index the Edit form's direction_index (a number d) uses: d + 1.
export function layersChart(stats, { directionIndex = null, onPick = null } = {}) {
  const n = stats.length - 1, { top } = LAYOUT;
  let yMax = 0;
  for (const s of stats) yMax = Math.max(yMax, s.good_norm, s.bad_norm, s.diff_norm);
  if (!(yMax > 0)) yMax = 1;
  const y = (v) => top + plotH * (1 - v / yMax);
  const root = frame('dirs-chart-norms', n, [['0', top + plotH], [fmt(yMax), top]], 'good mean norm, bad mean norm and the norm of the difference per index');
  for (const [key, cls] of [['good_norm', 'good'], ['bad_norm', 'bad'], ['diff_norm', 'diff']]) {
    root.append(svg('polyline', { class: `dirs-line ${cls}`, fill: 'none', points: points(stats.map((s, k) => [x(k, n), y(s[key])])) }));
  }
  stats.forEach((s, k) => {
    root.append(svg('circle', { class: 'dirs-point', cx: at(x(k, n)), cy: at(y(s.diff_norm)), r: 3, 'data-index': k, onclick: onPick ? () => onPick(k) : null },
      svg('title', {}, `index ${k}: difference ${fmt(s.diff_norm, 4)}${s.usable ? '' : ' (unusable)'}`)));
    if (!s.usable) root.append(svg('text', { class: 'dirs-unusable', 'data-index': k, x: at(x(k, n)), y: at(top + plotH + 12), 'text-anchor': 'middle' }, '×'));
  });
  if (typeof directionIndex === 'number' && Number.isFinite(directionIndex)) {
    const mx = at(x(directionIndex + 1, n));
    root.append(svg('line', { class: 'dirs-marker', x1: mx, x2: mx, y1: at(top), y2: at(top + plotH) }));
  }
  return root;
}

// The cosine of the good and the bad mean per index, from -1 to 1; an index without one (a zero norm) breaks the line.
export function cosineChart(stats) {
  const n = stats.length - 1, { top } = LAYOUT;
  const y = (c) => top + plotH * (1 - (c + 1) / 2);
  const root = frame('dirs-chart-cosine', n, [['-1', y(-1)], ['0', y(0)], ['1', y(1)]], 'cosine of the good and the bad mean');
  let run = [];
  const flush = () => { if (run.length) root.append(svg('polyline', { class: 'dirs-line cos', fill: 'none', points: points(run) })); run = []; };
  stats.forEach((s, k) => { if (s.cosine == null) flush(); else run.push([x(k, n), y(s.cosine)]); });
  flush();
  return root;
}

// ------------------------------------------------------------------------------------------------ a search's front
// One objective's axis: lo and hi of the finite values; a log scale when they are positive and span two decades;
// t(v) from 0 at lo to 1 at hi (0.5 when every value is the same).
export function axisScale(values) {
  const vs = values.filter((v) => Number.isFinite(v));
  const lo = vs.length ? Math.min(...vs) : 0, hi = vs.length ? Math.max(...vs) : 0;
  const log = lo > 0 && hi / lo >= 100;
  const t = (v) => (hi === lo ? 0.5 : log ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v - lo) / (hi - lo));
  return { lo, hi, log, t };
}

// The complete trials of a search on its objectives (desc: GET /api/directions/searches/{name}), the Pareto front
// marked and joined in its order, and a fresh baseline (its objective values) as a hollow mark. One or two
// objectives: a strip or a scatter of circles; three or more: a line per trial on parallel axes. onPick(index) on a
// click of a trial.
export function frontChart(desc, { baseline = null, onPick = null } = {}) {
  const objectives = desc?.search?.objectives || [], k = objectives.length;
  if (!k) return null;
  const onFront = new Set(desc.front || []);
  const done = (desc.trials || []).filter((t) => t.state === 'complete' && Array.isArray(t.values) && objectives.every((_, i) => Number.isFinite(t.values[i])));
  const base = Array.isArray(baseline) && objectives.every((_, i) => Number.isFinite(baseline[i])) ? baseline : null;
  const scales = objectives.map((_, i) => axisScale([...done.map((p) => p.values[i]), ...(base ? [base[i]] : [])]));
  const axisLabel = (i) => `${objectives[i].name} (${objectives[i].direction})${scales[i].log ? ', log scale' : ''}`;
  const caption = (t) => `Trial ${t.index}: ` + objectives.map((o, i) => `${o.name} ${t.displays?.[o.index] ?? fmt(t.values[i], 4)}`).join(', ');
  const cls = (t) => (onFront.has(t.index) ? 'dirs-trial front' : 'dirs-trial');
  const pick = (t) => (onPick ? () => onPick(t.index) : null);
  const text = (attrs, label) => svg('text', { class: 'val', ...attrs }, label);
  if (k >= 3) {
    const width = 120 * k + 60, height = 260, ax = (i) => 40 + 120 * i, ay = (i, v) => 12 + (1 - scales[i].t(v)) * 208;
    const root = svg('svg', { id: 'dirs-front-chart', class: 'chart', width, height, viewBox: `0 0 ${width} ${height}`, style: 'max-width:100%;height:auto' },
      svg('title', {}, 'the complete trials on parallel axes, one per objective; the front highlighted'));
    objectives.forEach((_, i) => {
      root.append(svg('line', { class: 'axis', x1: at(ax(i)), x2: at(ax(i)), y1: at(12), y2: at(220) }),
        text({ x: at(ax(i)), y: at(236), 'text-anchor': 'middle', 'data-axis': i }, axisLabel(i)),
        text({ x: at(ax(i) + 4), y: at(20), 'text-anchor': 'start' }, fmt(scales[i].hi)),
        text({ x: at(ax(i) + 4), y: at(218), 'text-anchor': 'start' }, fmt(scales[i].lo)));
    });
    const line = (vs) => points(vs.map((v, i) => [ax(i), ay(i, v)]));
    for (const t of done) {
      root.append(svg('polyline', { class: cls(t), fill: 'none', points: line(t.values.slice(0, k)), 'data-trial': t.index, onclick: pick(t) }, svg('title', {}, caption(t))));
    }
    if (base) root.append(svg('polyline', { class: 'dirs-baseline', fill: 'none', points: line(base) }, svg('title', {}, 'Baseline: the current weights')));
    return root;
  }
  const one = k === 1;
  const width = 440, height = one ? 80 : 320, left = 60, right = 16, top = 12, bottom = 40, pw = width - left - right, ph = height - top - bottom;
  const cx = (v) => left + scales[0].t(v) * pw;
  const cy = (v) => (one ? 40 : top + (1 - scales[1].t(v)) * ph);
  const yAxis = one ? 40 : top + ph;
  const root = svg('svg', { id: 'dirs-front-chart', class: 'chart', width, height, viewBox: `0 0 ${width} ${height}`, style: 'max-width:100%;height:auto' },
    svg('title', {}, 'the complete trials on the objectives; the front highlighted and joined'),
    svg('line', { class: 'axis', x1: left, x2: width - right, y1: at(yAxis), y2: at(yAxis) }),
    text({ x: at(left), y: at(yAxis + 14), 'text-anchor': 'start' }, fmt(scales[0].lo)),
    text({ x: at(width - right), y: at(yAxis + 14), 'text-anchor': 'end' }, fmt(scales[0].hi)),
    text({ x: at(left + pw / 2), y: at(yAxis + 30), 'text-anchor': 'middle', 'data-axis': 0 }, axisLabel(0)));
  if (!one) {
    root.append(svg('line', { class: 'axis', x1: left, x2: left, y1: top, y2: at(top + ph) }),
      text({ x: at(left - 6), y: at(top + ph), 'text-anchor': 'end' }, fmt(scales[1].lo)),
      text({ x: at(left - 6), y: at(top + 8), 'text-anchor': 'end' }, fmt(scales[1].hi)),
      text({ x: at(14), y: at(top + ph / 2), 'text-anchor': 'middle', transform: `rotate(-90 14 ${at(top + ph / 2)})`, 'data-axis': 1 }, axisLabel(1)));
  }
  const xy = (vs) => [cx(vs[0]), cy(vs[1])];
  const byIndex = new Map(done.map((t) => [t.index, t]));
  const front = (desc.front || []).map((i) => byIndex.get(i)).filter(Boolean);
  if (front.length > 1) root.append(svg('polyline', { class: 'dirs-front-line', fill: 'none', points: points(front.map((t) => xy(t.values))) }));
  for (const t of done) {
    const [px, py] = xy(t.values);
    root.append(svg('circle', { class: cls(t), cx: at(px), cy: at(py), r: onFront.has(t.index) ? 5 : 4, 'data-trial': t.index, onclick: pick(t) }, svg('title', {}, caption(t))));
  }
  if (base) {
    const [px, py] = xy(base);
    root.append(svg('circle', { class: 'dirs-baseline', cx: at(px), cy: at(py), r: 6, fill: 'none' }, svg('title', {}, 'Baseline: the current weights')));
  }
  return root;
}
