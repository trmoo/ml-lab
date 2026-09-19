/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 데이터프레임(표) 다루기 — pandas 의 DataFrame 을 아주 조금만 흉내 낸 것.
 *
 * pandas 에서 자주 쓰는 read_csv / info / describe / groupby / corr 를
 * 브라우저에서도 똑같이 볼 수 있게 만들었다.
 *
 * 값을 담는 규칙 하나만 기억하면 된다.
 *   · 수치형 컬럼 → 숫자 배열, 결측치는 NaN
 *   · 범주형 컬럼 → 문자열 배열, 결측치는 null
 * 이렇게 두면 pandas 의 NaN 과 성격이 비슷해져 평균·최빈값 계산이 자연스럽다.
 */

/** CSV 글자를 표로 바꾼다. (따옴표로 감싼 칸도 처리) */
export function parseCsv(text) {
  const clean = text.replace(/^﻿/, '').replace(/\r\n?/g, '\n');
  const rows = [];
  let row = [];
  let cell = '';
  let quoted = false;
  for (let i = 0; i < clean.length; i += 1) {
    const ch = clean[i];
    if (quoted) {
      if (ch === '"') {
        if (clean[i + 1] === '"') { cell += '"'; i += 1; } else { quoted = false; }
      } else { cell += ch; }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ',') {
      row.push(cell); cell = '';
    } else if (ch === '\n') {
      row.push(cell); rows.push(row); row = []; cell = '';
    } else {
      cell += ch;
    }
  }
  if (cell !== '' || row.length) { row.push(cell); rows.push(row); }
  return rows.filter((r) => r.length > 1 || (r[0] ?? '') !== '');
}

const isBlank = (s) => s === undefined || s === null || String(s).trim() === '';

/* 숫자가 아닌 값이 이 비율 이하로 섞여 있으면 「숫자 컬럼인데 오염된 것」으로 본다. */
const JUNK_TOLERANCE = 0.05;

/**
 * 컬럼의 값들을 보고 수치형인지 범주형인지 정한다.
 *
 * 실제 데이터에는 숫자 자리에 '-' 나 '미확인' 같은 글자가 몇 개 섞여 들어오는 일이 흔하다.
 * pandas 는 그런 컬럼을 통째로 object 로 읽고, 사람이 pd.to_numeric(errors='coerce') 로
 * 고쳐 쓴다. 여기서도 거의 다 숫자면 수치형으로 보고 나머지는 결측치로 바꾼다
 * (to_numeric 과 같은 처리). 그러지 않으면 값이 수천 가지인 컬럼이 범주형이 되어
 * 원-핫 인코딩에서 컬럼이 폭발한다.
 *
 * @returns {{numeric: boolean, junk: number}} junk = 숫자로 못 바꾼 값의 개수
 */
function inferNumeric(values) {
  let seen = 0;
  let junk = 0;
  for (const v of values) {
    if (isBlank(v)) continue;
    seen += 1;
    if (Number.isNaN(Number(v))) junk += 1;
  }
  if (!seen) return { numeric: false, junk: 0 };
  return { numeric: junk / seen <= JUNK_TOLERANCE, junk };
}

export class Frame {
  /** @param {string[]} columns  @param {Map<string, Array>} cols */
  constructor(columns, cols) {
    this.columns = columns.slice();
    this.cols = cols;
    /** 숫자로 못 바꿔 결측치로 만든 값의 개수 (컬럼 → 개수) */
    this.coerced = new Map();
  }

  static fromCsv(text) {
    const rows = parseCsv(text);
    const header = rows[0].map((h) => h.trim());
    const body = rows.slice(1);
    const cols = new Map();
    const coerced = new Map();
    header.forEach((name, j) => {
      const raw = body.map((r) => r[j]);
      const { numeric, junk } = inferNumeric(raw);
      if (numeric) {
        // 숫자로 못 바꾸는 값은 결측치로 — pandas 의 to_numeric(errors='coerce') 와 같다
        cols.set(name, raw.map((v) => {
          if (isBlank(v)) return NaN;
          const n = Number(v);
          return Number.isNaN(n) ? NaN : n;
        }));
        if (junk) coerced.set(name, junk);
      } else {
        cols.set(name, raw.map((v) => (isBlank(v) ? null : String(v).trim())));
      }
    });
    const frame = new Frame(header, cols);
    frame.coerced = coerced;
    return frame;
  }

  get length() {
    return this.columns.length ? this.cols.get(this.columns[0]).length : 0;
  }

  col(name) { return this.cols.get(name); }

  /** 'number' | 'object' — pandas info() 가 보여 주는 타입 이름을 그대로 쓴다 */
  dtype(name) {
    const a = this.cols.get(name);
    return typeof a.find((v) => v !== null && !Number.isNaN(v)) === 'number' ? 'float64' : 'object';
  }

  isNumeric(name) { return this.dtype(name) === 'float64'; }

  numericColumns() { return this.columns.filter((c) => this.isNumeric(c)); }

  categoricalColumns() { return this.columns.filter((c) => !this.isNumeric(c)); }

  /** 결측치 개수 — pandas 의 isnull().sum() */
  missing(name) {
    const a = this.cols.get(name);
    let n = 0;
    for (const v of a) if (v === null || (typeof v === 'number' && Number.isNaN(v))) n += 1;
    return n;
  }

  /** info() 에 해당하는 요약 */
  info() {
    return this.columns.map((name) => ({
      name,
      dtype: this.dtype(name),
      nonNull: this.length - this.missing(name),
      missing: this.missing(name),
      unique: this.isNumeric(name) ? null : new Set(this.values(name)).size,
      coerced: this.coerced.get(name) || 0,
    }));
  }

  /** 결측치를 뺀 값들 */
  values(name) {
    const a = this.cols.get(name);
    return a.filter((v) => v !== null && !(typeof v === 'number' && Number.isNaN(v)));
  }

  head(n = 5) {
    return Array.from({ length: Math.min(n, this.length) }, (_, i) =>
      this.columns.map((c) => this.cols.get(c)[i]));
  }

  /** describe() — 수치형 컬럼의 기초 통계 */
  describe(cols = this.numericColumns()) {
    return cols.map((name) => {
      const v = this.values(name).slice().sort((a, b) => a - b);
      return { name, ...summarize(v) };
    });
  }

  /**
   * groupby(keys)[valueCols].agg()
   * pandas 는 기준 컬럼이 결측치인 행을 그룹에서 빼 버린다(dropna=True 가 기본값).
   * pandas 로 구한 답과 같은 값이 나오도록 여기서도 똑같이 뺀다.
   */
  groupby(keys, valueCols, agg = 'mean') {
    const groups = new Map();
    const n = this.length;
    let skipped = 0;
    for (let i = 0; i < n; i += 1) {
      const parts = keys.map((k) => this.cols.get(k)[i]);
      if (parts.some((v) => v === null || (typeof v === 'number' && Number.isNaN(v)))) {
        skipped += 1;
        continue;
      }
      const label = parts.map(String).join(' · ');
      if (!groups.has(label)) groups.set(label, []);
      groups.get(label).push(i);
    }
    const rows = [];
    for (const [label, idx] of groups) {
      const cells = valueCols.map((c) => {
        const arr = idx.map((i) => this.cols.get(c)[i])
          .filter((v) => v !== null && !Number.isNaN(v));
        return aggregate(arr, agg);
      });
      rows.push({ label, n: idx.length, cells });
    }
    // 첫 번째 집계값이 큰 것부터 — 「가장 ~한 것」을 찾는 문제가 많기 때문
    rows.sort((a, b) => (b.cells[0] ?? -Infinity) - (a.cells[0] ?? -Infinity));
    return { keys, valueCols, agg, rows, skipped };
  }

  /** 교차표 — pandas 의 crosstab. normalize 를 켜면 행마다 비율로 바꾼다 */
  crosstab(rowKey, colKey, normalize = false) {
    const rowVals = uniqueSorted(this.values(rowKey));
    const colVals = uniqueSorted(this.values(colKey));
    const table = rowVals.map(() => colVals.map(() => 0));
    const n = this.length;
    for (let i = 0; i < n; i += 1) {
      const r = rowVals.indexOf(fmtKey(this.cols.get(rowKey)[i]));
      const c = colVals.indexOf(fmtKey(this.cols.get(colKey)[i]));
      if (r >= 0 && c >= 0) table[r][c] += 1;
    }
    if (normalize) {
      table.forEach((r) => {
        const s = r.reduce((a, b) => a + b, 0);
        if (s) r.forEach((v, j) => { r[j] = v / s; });
      });
    }
    return { rowVals, colVals, table, normalize };
  }

  /** 상관계수 행렬 — corr(). 두 변수가 함께 커지면 +1, 반대로 움직이면 -1 */
  corr(cols = this.numericColumns()) {
    const m = cols.map(() => cols.map(() => NaN));
    for (let i = 0; i < cols.length; i += 1) {
      for (let j = i; j < cols.length; j += 1) {
        const r = pearson(this.cols.get(cols[i]), this.cols.get(cols[j]));
        m[i][j] = r;
        m[j][i] = r;
      }
    }
    return { cols, m };
  }

  copy() {
    const cols = new Map();
    this.columns.forEach((c) => cols.set(c, this.cols.get(c).slice()));
    return this.carry(new Frame(this.columns, cols));
  }

  /** 새로 만든 Frame 에 「숫자로 못 바꾼 값」 기록을 물려준다 */
  carry(next) {
    next.coerced = new Map([...this.coerced].filter(([c]) => next.columns.includes(c)));
    return next;
  }

  drop(names) {
    const remove = new Set([].concat(names));
    const keep = this.columns.filter((c) => !remove.has(c));
    const cols = new Map();
    keep.forEach((c) => cols.set(c, this.cols.get(c)));
    return this.carry(new Frame(keep, cols));
  }

  select(names) {
    const cols = new Map();
    names.forEach((c) => cols.set(c, this.cols.get(c)));
    return this.carry(new Frame(names, cols));
  }

  withColumn(name, arr) {
    const cols = new Map(this.cols);
    cols.set(name, arr);
    const columns = this.columns.includes(name) ? this.columns : [...this.columns, name];
    return this.carry(new Frame(columns, cols));
  }

  /** 지정한 행만 남긴다 */
  take(idx) {
    const cols = new Map();
    this.columns.forEach((c) => {
      const a = this.cols.get(c);
      cols.set(c, idx.map((i) => a[i]));
    });
    return this.carry(new Frame(this.columns, cols));
  }

  /** 결측치가 있는 행을 지운다 — dropna() */
  dropna(subset = this.columns) {
    const keep = [];
    for (let i = 0; i < this.length; i += 1) {
      const bad = subset.some((c) => {
        const v = this.cols.get(c)[i];
        return v === null || (typeof v === 'number' && Number.isNaN(v));
      });
      if (!bad) keep.push(i);
    }
    return this.take(keep);
  }

  /** number 컬럼을 기준으로 두 표를 나란히 붙인다 — 2회차의 파일 두 개 합치기 */
  static mergeOn(a, b, key) {
    const bIndex = new Map();
    b.col(key).forEach((v, i) => bIndex.set(String(v), i));
    const bCols = b.columns.filter((c) => c !== key);
    const rowsA = [];
    const picked = [];
    a.col(key).forEach((v, i) => {
      const j = bIndex.get(String(v));
      if (j !== undefined) { rowsA.push(i); picked.push(j); }
    });
    const left = a.take(rowsA);
    const cols = new Map(left.cols);
    bCols.forEach((c) => {
      const arr = b.col(c);
      cols.set(c, picked.map((j) => arr[j]));
    });
    return new Frame([...left.columns, ...bCols], cols);
  }
}

const fmtKey = (v) => (v === null || Number.isNaN(v) ? '(결측)' : String(v));

function uniqueSorted(arr) {
  const set = new Set(arr.map(fmtKey));
  return [...set].sort((a, b) => {
    const na = Number(a);
    const nb = Number(b);
    if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
    return a.localeCompare(b, 'ko');
  });
}

export function aggregate(arr, agg) {
  if (!arr.length) return NaN;
  if (agg === 'count') return arr.length;
  if (agg === 'sum') return arr.reduce((a, b) => a + b, 0);
  if (agg === 'mean') return arr.reduce((a, b) => a + b, 0) / arr.length;
  if (agg === 'max') return Math.max(...arr);
  if (agg === 'min') return Math.min(...arr);
  if (agg === 'std') return std(arr);
  if (agg === 'median') return quantile(arr.slice().sort((a, b) => a - b), 0.5);
  return NaN;
}

export function mean(arr) {
  const v = arr.filter((x) => typeof x === 'number' && !Number.isNaN(x));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
}

export function std(arr) {
  const v = arr.filter((x) => typeof x === 'number' && !Number.isNaN(x));
  if (v.length < 2) return 0;
  const m = mean(v);
  // pandas 와 같게 표본 표준편차(n-1로 나눔)를 쓴다
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1));
}

/** 정렬된 배열에서 분위수 — numpy 의 기본 방식(선형 보간) */
export function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function summarize(sortedValues) {
  const v = sortedValues;
  return {
    count: v.length,
    mean: mean(v),
    std: std(v),
    min: v.length ? v[0] : NaN,
    q1: quantile(v, 0.25),
    median: quantile(v, 0.5),
    q3: quantile(v, 0.75),
    max: v.length ? v[v.length - 1] : NaN,
  };
}

/** 최빈값 — mode()[0]. 범주형 결측치를 채울 때 쓴다 */
export function mode(arr) {
  const count = new Map();
  arr.forEach((v) => {
    if (v === null || (typeof v === 'number' && Number.isNaN(v))) return;
    count.set(v, (count.get(v) || 0) + 1);
  });
  let best = null;
  let bestN = -1;
  for (const [v, n] of count) if (n > bestN) { best = v; bestN = n; }
  return best;
}

/** 피어슨 상관계수. 둘 다 값이 있는 행만 쓴다 */
export function pearson(xs, ys) {
  const px = [];
  const py = [];
  for (let i = 0; i < xs.length; i += 1) {
    const a = xs[i];
    const b = ys[i];
    if (typeof a !== 'number' || Number.isNaN(a)) continue;
    if (typeof b !== 'number' || Number.isNaN(b)) continue;
    px.push(a); py.push(b);
  }
  if (px.length < 2) return NaN;
  const mx = mean(px);
  const my = mean(py);
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < px.length; i += 1) {
    const a = px[i] - mx;
    const b = py[i] - my;
    num += a * b; dx += a * a; dy += b * b;
  }
  return dx && dy ? num / Math.sqrt(dx * dy) : NaN;
}

/** 히스토그램용 구간 나누기 */
export function histogram(values, bins = 20) {
  const v = values.filter((x) => typeof x === 'number' && !Number.isNaN(x));
  if (!v.length) return { edges: [], counts: [] };
  const lo = Math.min(...v);
  const hi = Math.max(...v);
  const width = (hi - lo) / bins || 1;
  const counts = new Array(bins).fill(0);
  v.forEach((x) => {
    let k = Math.floor((x - lo) / width);
    if (k >= bins) k = bins - 1;
    if (k < 0) k = 0;
    counts[k] += 1;
  });
  const edges = Array.from({ length: bins + 1 }, (_, i) => lo + i * width);
  return { edges, counts };
}

/** 상자그림에 필요한 다섯 숫자와 이상치 */
export function boxStats(values) {
  const v = values.filter((x) => typeof x === 'number' && !Number.isNaN(x))
    .slice().sort((a, b) => a - b);
  if (!v.length) return null;
  const q1 = quantile(v, 0.25);
  const q2 = quantile(v, 0.5);
  const q3 = quantile(v, 0.75);
  const iqr = q3 - q1;
  const lowFence = q1 - 1.5 * iqr;
  const highFence = q3 + 1.5 * iqr;
  const inside = v.filter((x) => x >= lowFence && x <= highFence);
  return {
    q1, q2, q3, iqr,
    whiskerLow: inside.length ? inside[0] : q1,
    whiskerHigh: inside.length ? inside[inside.length - 1] : q3,
    outliers: v.filter((x) => x < lowFence || x > highFence),
    n: v.length,
  };
}
