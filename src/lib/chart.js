/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 그래프 그리기 — 외부 라이브러리 없이 SVG 를 직접 만든다.
 *
 * seaborn 의 barplot / boxplot / scatterplot / histplot / heatmap 에 해당하는
 * 그림들을 브라우저에서 그린다. 교실 TV 에서도 보이도록 글씨를 크게 잡았다.
 */

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/** 눈금 숫자를 짧게 — 1234567 → 1.2M */
export function fmtNum(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1e8) return `${(v / 1e8).toFixed(1)}억`;
  if (a >= 1e4) return `${(v / 1e4).toFixed(1)}만`;
  if (a >= 1000) return v.toFixed(0);
  if (a >= 1) return v.toFixed(Math.min(digits, 2));
  if (a === 0) return '0';
  return v.toPrecision(2);
}

/** 눈금 위치를 보기 좋은 값으로 고른다 */
function ticks(lo, hi, count = 5) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0];
  if (lo === hi) return [lo];
  const span = hi - lo;
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const start = Math.ceil(lo / step) * step;
  const out = [];
  for (let v = start; v <= hi + step * 1e-6; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

const PALETTE = ['#2563eb', '#dc2626', '#059669', '#d97706', '#7c3aed',
  '#0891b2', '#db2777', '#65a30d', '#c2410c', '#4f46e5'];
export const color = (i) => PALETTE[i % PALETTE.length];

const W = 720;

function frame({
  height = 380, padL = 78, padR = 20, padT = 26, padB = 62,
  xLabel = '', yLabel = '', title = '',
}) {
  return {
    w: W, h: height, padL, padR, padT, padB, xLabel, yLabel, title,
    ix: W - padL - padR,
    iy: height - padT - padB,
  };
}

function wrap(f, inner) {
  const label = (f.xLabel || f.yLabel || f.title) ? `
    ${f.title ? `<text x="${W / 2}" y="16" class="c-title">${esc(f.title)}</text>` : ''}
    ${f.xLabel ? `<text x="${f.padL + f.ix / 2}" y="${f.h - 8}" class="c-axlab">${esc(f.xLabel)}</text>` : ''}
    ${f.yLabel ? `<text transform="translate(16,${f.padT + f.iy / 2}) rotate(-90)" class="c-axlab">${esc(f.yLabel)}</text>` : ''}` : '';
  return `<svg viewBox="0 0 ${W} ${f.h}" class="chart" preserveAspectRatio="xMidYMid meet"
    role="img">${label}${inner}</svg>`;
}

function yAxis(f, lo, hi, { pct = false } = {}) {
  const tk = ticks(lo, hi, 5);
  const sy = (v) => f.padT + f.iy - ((v - lo) / (hi - lo || 1)) * f.iy;
  const grid = tk.map((t) => `
    <line x1="${f.padL}" x2="${f.padL + f.ix}" y1="${sy(t)}" y2="${sy(t)}" class="c-grid"/>
    <text x="${f.padL - 8}" y="${sy(t) + 5}" class="c-tick c-right">${pct ? `${(t * 100).toFixed(0)}%` : fmtNum(t)}</text>`).join('');
  return { sy, grid };
}

/* ── 막대그래프 (seaborn barplot / countplot) ─────────────── */
export function barChart(items, opts = {}) {
  const f = frame({ height: 360, ...opts });
  const vals = items.map((d) => d.value).filter(Number.isFinite);
  if (!vals.length) return emptyChart('그릴 값이 없습니다');
  let lo = Math.min(0, ...vals);
  let hi = Math.max(0, ...vals);
  if (lo === hi) { hi = lo + 1; }
  const pad = (hi - lo) * 0.08;
  hi += pad;
  if (lo < 0) lo -= pad;
  const { sy, grid } = yAxis(f, lo, hi, opts);
  const bw = f.ix / items.length;
  const zero = sy(0);
  const bars = items.map((d, i) => {
    const x = f.padL + i * bw + bw * 0.15;
    const w = bw * 0.7;
    const y = sy(d.value);
    const top = Math.min(y, zero);
    const h = Math.max(1, Math.abs(zero - y));
    const highlight = d.highlight ? ' c-bar-hi' : '';
    return `<rect x="${x}" y="${top}" width="${w}" height="${h}"
        fill="${d.color || color(0)}" class="c-bar${highlight}"/>
      <text x="${x + w / 2}" y="${top - 6}" class="c-val">${opts.pct ? `${(d.value * 100).toFixed(1)}%` : fmtNum(d.value)}</text>
      <text x="${x + w / 2}" y="${f.padT + f.iy + 20}" class="c-cat">${esc(shorten(d.label, Math.max(6, Math.floor(90 / items.length) + 4)))}</text>`;
  }).join('');
  return wrap(f, `${grid}
    <line x1="${f.padL}" x2="${f.padL + f.ix}" y1="${zero}" y2="${zero}" class="c-axis"/>
    ${bars}`);
}

/* ── 가로 막대 (특성 중요도) ──────────────────────────────── */
export function hBarChart(items, opts = {}) {
  const rows = items.length;
  const rowH = 26;
  const f = frame({
    height: Math.max(140, rows * rowH + 60), padL: 180, padB: 34, padT: 20, ...opts,
  });
  const hi = Math.max(1e-9, ...items.map((d) => d.value));
  const sx = (v) => (v / hi) * f.ix;
  const bars = items.map((d, i) => {
    const y = f.padT + i * rowH;
    return `<text x="${f.padL - 10}" y="${y + rowH * 0.68}" class="c-tick c-right">${esc(shorten(d.label, 24))}</text>
      <rect x="${f.padL}" y="${y + 4}" width="${Math.max(1, sx(d.value))}" height="${rowH - 10}"
        fill="${d.color || color(0)}" class="c-bar"/>
      <text x="${f.padL + sx(d.value) + 6}" y="${y + rowH * 0.68}" class="c-val c-left">${opts.pct ? `${(d.value * 100).toFixed(1)}%` : fmtNum(d.value, 3)}</text>`;
  }).join('');
  return wrap(f, bars);
}

/* ── 히스토그램 (histplot) ───────────────────────────────── */
export function histChart(series, opts = {}) {
  const f = frame({ height: 340, ...opts });
  const all = series.flatMap((s) => s.counts);
  if (!all.length) return emptyChart('그릴 값이 없습니다');
  const hi = Math.max(...all) * 1.1 || 1;
  const { sy, grid } = yAxis(f, 0, hi);
  const lo = Math.min(...series.map((s) => s.edges[0]));
  const up = Math.max(...series.map((s) => s.edges[s.edges.length - 1]));
  const sx = (v) => f.padL + ((v - lo) / (up - lo || 1)) * f.ix;
  const bars = series.map((s, k) => s.counts.map((c, i) => {
    const x0 = sx(s.edges[i]);
    const x1 = sx(s.edges[i + 1]);
    return `<rect x="${x0}" y="${sy(c)}" width="${Math.max(1, x1 - x0 - 1)}"
      height="${f.padT + f.iy - sy(c)}" fill="${color(k)}"
      opacity="${series.length > 1 ? 0.55 : 0.85}"/>`;
  }).join('')).join('');
  const xt = ticks(lo, up, 6).map((t) => `
    <text x="${sx(t)}" y="${f.padT + f.iy + 20}" class="c-cat">${fmtNum(t)}</text>`).join('');
  return wrap(f, `${grid}${bars}
    <line x1="${f.padL}" x2="${f.padL + f.ix}" y1="${f.padT + f.iy}" y2="${f.padT + f.iy}" class="c-axis"/>
    ${xt}${legend(f, series.map((s) => s.label))}`);
}

/* ── 산점도 (scatterplot / regplot) ──────────────────────── */
export function scatterChart(groups, opts = {}) {
  const f = frame({ height: 400, ...opts });
  const pts = groups.flatMap((g) => g.points);
  if (!pts.length) return emptyChart('그릴 점이 없습니다');
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const xlo = Math.min(...xs);
  const xhi = Math.max(...xs);
  const ylo = Math.min(...ys);
  const yhi = Math.max(...ys);
  const padX = (xhi - xlo) * 0.05 || 1;
  const padY = (yhi - ylo) * 0.05 || 1;
  const sx = (v) => f.padL + ((v - (xlo - padX)) / ((xhi + padX) - (xlo - padX))) * f.ix;
  const { sy, grid } = yAxis(f, ylo - padY, yhi + padY);
  const dots = groups.map((g, k) => g.points.map((p) =>
    `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="${opts.r || 2.6}"
      fill="${g.color || color(k)}" opacity="0.6"/>`).join('')).join('');
  const xt = ticks(xlo - padX, xhi + padX, 6).map((t) => `
    <line x1="${sx(t)}" x2="${sx(t)}" y1="${f.padT}" y2="${f.padT + f.iy}" class="c-grid"/>
    <text x="${sx(t)}" y="${f.padT + f.iy + 20}" class="c-cat">${fmtNum(t)}</text>`).join('');
  // 기준선(예: 실제값=예측값) 이나 회귀선
  let extra = '';
  if (opts.diagonal) {
    const a = Math.max(xlo - padX, ylo - padY);
    const b = Math.min(xhi + padX, yhi + padY);
    extra += `<line x1="${sx(a)}" y1="${sy(a)}" x2="${sx(b)}" y2="${sy(b)}" class="c-refline"/>`;
  }
  if (opts.line) {
    const { slope, intercept } = opts.line;
    const a = xlo - padX;
    const b = xhi + padX;
    extra += `<line x1="${sx(a)}" y1="${sy(slope * a + intercept)}"
      x2="${sx(b)}" y2="${sy(slope * b + intercept)}" class="c-fitline"/>`;
  }
  return wrap(f, `${grid}${xt}${dots}${extra}
    <line x1="${f.padL}" x2="${f.padL + f.ix}" y1="${f.padT + f.iy}" y2="${f.padT + f.iy}" class="c-axis"/>
    ${legend(f, groups.map((g) => g.label))}`);
}

/* ── 상자그림 (boxplot / violinplot 자리) ────────────────── */
export function boxChart(groups, opts = {}) {
  const f = frame({ height: 380, ...opts });
  const valid = groups.filter((g) => g.stats);
  if (!valid.length) return emptyChart('그릴 값이 없습니다');
  const lo = Math.min(...valid.map((g) => Math.min(g.stats.whiskerLow, ...g.stats.outliers)));
  const hi = Math.max(...valid.map((g) => Math.max(g.stats.whiskerHigh, ...g.stats.outliers)));
  const pad = (hi - lo) * 0.06 || 1;
  const { sy, grid } = yAxis(f, lo - pad, hi + pad);
  const bw = f.ix / valid.length;
  const boxes = valid.map((g, i) => {
    const s = g.stats;
    const cx = f.padL + i * bw + bw / 2;
    const w = Math.min(72, bw * 0.5);
    const out = s.outliers.map((v) =>
      `<circle cx="${cx}" cy="${sy(v)}" r="2.4" fill="${color(i)}" opacity="0.5"/>`).join('');
    return `
      <line x1="${cx}" x2="${cx}" y1="${sy(s.whiskerHigh)}" y2="${sy(s.q3)}" class="c-axis"/>
      <line x1="${cx}" x2="${cx}" y1="${sy(s.q1)}" y2="${sy(s.whiskerLow)}" class="c-axis"/>
      <line x1="${cx - w / 3}" x2="${cx + w / 3}" y1="${sy(s.whiskerHigh)}" y2="${sy(s.whiskerHigh)}" class="c-axis"/>
      <line x1="${cx - w / 3}" x2="${cx + w / 3}" y1="${sy(s.whiskerLow)}" y2="${sy(s.whiskerLow)}" class="c-axis"/>
      <rect x="${cx - w / 2}" y="${sy(s.q3)}" width="${w}" height="${Math.max(1, sy(s.q1) - sy(s.q3))}"
        fill="${color(i)}" opacity="0.35" stroke="${color(i)}" stroke-width="1.5"/>
      <line x1="${cx - w / 2}" x2="${cx + w / 2}" y1="${sy(s.q2)}" y2="${sy(s.q2)}"
        stroke="${color(i)}" stroke-width="3"/>
      ${out}
      <text x="${cx}" y="${f.padT + f.iy + 20}" class="c-cat">${esc(shorten(g.label, Math.max(6, Math.floor(80 / valid.length) + 4)))}</text>
      <text x="${cx}" y="${f.padT + f.iy + 38}" class="c-cat c-dim">n=${s.n}</text>`;
  }).join('');
  return wrap(f, `${grid}
    <line x1="${f.padL}" x2="${f.padL + f.ix}" y1="${f.padT + f.iy}" y2="${f.padT + f.iy}" class="c-axis"/>
    ${boxes}`);
}

/* ── 상관계수 히트맵 (heatmap) ───────────────────────────── */
export function heatmapChart({ cols, m }, opts = {}) {
  const n = cols.length;
  const cell = Math.max(30, Math.min(58, Math.floor(520 / Math.max(1, n))));
  const padL = 150;
  const padT = 130;
  const h = padT + n * cell + 24;
  const f = frame({ height: h, padL, padT, padB: 20, padR: 20, ...opts });
  let cells = '';
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const v = m[i][j];
      const x = padL + j * cell;
      const y = padT + i * cell;
      cells += `<rect x="${x}" y="${y}" width="${cell}" height="${cell}"
          fill="${corrColor(v)}" stroke="#fff" stroke-width="1"/>
        <text x="${x + cell / 2}" y="${y + cell / 2 + 4}" class="c-heatval"
          fill="${Math.abs(v) > 0.55 ? '#fff' : '#1f2937'}">${Number.isFinite(v) ? v.toFixed(2) : '—'}</text>`;
    }
  }
  const labels = cols.map((c, i) => `
    <text x="${padL - 8}" y="${padT + i * cell + cell / 2 + 4}" class="c-tick c-right">${esc(shorten(c, 18))}</text>
    <text transform="translate(${padL + i * cell + cell / 2},${padT - 8}) rotate(-45)"
      class="c-tick c-left">${esc(shorten(c, 16))}</text>`).join('');
  return wrap(f, `${cells}${labels}`);
}

/** -1(파랑) — 0(흰색) — +1(빨강) */
function corrColor(v) {
  if (!Number.isFinite(v)) return '#e5e7eb';
  const t = Math.max(-1, Math.min(1, v));
  if (t >= 0) {
    const k = 1 - t;
    return `rgb(${Math.round(220 + 35 * k)},${Math.round(38 + 217 * k)},${Math.round(38 + 217 * k)})`;
  }
  const k = 1 + t;
  return `rgb(${Math.round(37 + 218 * k)},${Math.round(99 + 156 * k)},${Math.round(235 + 20 * k)})`;
}

/* ── 꺾은선 (학습 곡선 · ROC) ───────────────────────────── */
export function lineChart(series, opts = {}) {
  const f = frame({ height: 340, ...opts });
  const all = series.flatMap((s) => s.points);
  if (!all.length) return emptyChart('아직 그릴 값이 없습니다');
  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);
  const xlo = opts.xMin ?? Math.min(...xs);
  const xhi = opts.xMax ?? Math.max(...xs);
  const ylo = opts.yMin ?? Math.min(...ys);
  const yhi = opts.yMax ?? Math.max(...ys);
  const sx = (v) => f.padL + ((v - xlo) / (xhi - xlo || 1)) * f.ix;
  const { sy, grid } = yAxis(f, ylo, yhi === ylo ? ylo + 1 : yhi);
  const paths = series.map((s, k) => {
    const d = s.points.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join('');
    return `<path d="${d}" fill="none" stroke="${s.color || color(k)}" stroke-width="2.4"
      ${s.dash ? 'stroke-dasharray="6 4"' : ''}/>`;
  }).join('');
  const xt = ticks(xlo, xhi, 6).map((t) => `
    <text x="${sx(t)}" y="${f.padT + f.iy + 20}" class="c-cat">${fmtNum(t)}</text>`).join('');
  const diag = opts.diagonal
    ? `<line x1="${sx(xlo)}" y1="${sy(ylo)}" x2="${sx(xhi)}" y2="${sy(yhi)}" class="c-refline"/>` : '';
  return wrap(f, `${grid}${diag}${paths}
    <line x1="${f.padL}" x2="${f.padL + f.ix}" y1="${f.padT + f.iy}" y2="${f.padT + f.iy}" class="c-axis"/>
    ${xt}${legend(f, series.map((s) => s.label), series.map((s, k) => s.color || color(k)))}`);
}

function legend(f, labels, colors = null) {
  const shown = labels.filter(Boolean);
  if (shown.length < 2 && !colors) return '';
  if (!shown.length) return '';
  let x = f.padL + 4;
  return shown.map((lab, i) => {
    const c = colors ? colors[labels.indexOf(lab)] : color(labels.indexOf(lab));
    const item = `<rect x="${x}" y="${f.padT - 16}" width="12" height="12" fill="${c}"/>
      <text x="${x + 17}" y="${f.padT - 6}" class="c-cat c-left">${esc(lab)}</text>`;
    x += 30 + String(lab).length * 8.4;
    return item;
  }).join('');
}

export function emptyChart(msg) {
  return `<svg viewBox="0 0 ${W} 120" class="chart"><text x="${W / 2}" y="64"
    class="c-empty">${esc(msg)}</text></svg>`;
}

function shorten(s, n) {
  const t = String(s);
  return t.length > n ? `${t.slice(0, n - 1)}…` : t;
}

export { esc, shorten };
