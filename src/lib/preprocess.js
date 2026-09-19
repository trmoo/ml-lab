/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 전처리 — 결측치 채우기, 인코딩, 데이터 분리, 스케일링.
 *
 * 핵심 원칙 하나만 지키면 된다.
 *   「규칙은 훈련 데이터에서만 배운다(fit). 검증 데이터에는 적용(transform)만 한다.」
 * 이것을 어기면 검증 점수가 실제보다 좋게 나온다 → 데이터 누수(Data Leakage).
 */
import { mean, std, quantile, mode } from './frame.js';
import { shuffledIndex } from './rng.js';

const isMissing = (v) => v === null || v === undefined
  || (typeof v === 'number' && Number.isNaN(v));

/** 결측치를 채우는 값 하나를 구한다 (평균/중앙값/최빈값/0) */
export function fillValue(values, how) {
  const clean = values.filter((v) => !isMissing(v));
  if (how === 'mean') return mean(clean);
  if (how === 'median') return quantile(clean.slice().sort((a, b) => a - b), 0.5);
  if (how === 'mode') return mode(clean);
  if (how === 'zero') return 0;
  return null;
}

/**
 * 결측치 처리 계획을 표에 적용한다.
 * @param {Frame} frame
 * @param {Object<string,string>} plan  컬럼 → 'mean'|'median'|'mode'|'zero'|'dropRow'|'dropCol'|'keep'
 * @returns {{frame: Frame, log: Array}}
 */
export function applyImpute(frame, plan) {
  let out = frame.copy();
  const log = [];

  // ① 컬럼을 아예 지우는 것부터
  const dropCols = Object.keys(plan).filter((c) => plan[c] === 'dropCol' && out.columns.includes(c));
  if (dropCols.length) {
    out = out.drop(dropCols);
    dropCols.forEach((c) => log.push({ col: c, how: 'dropCol', detail: '컬럼 삭제' }));
  }

  // ② 값으로 채우기
  out.columns.forEach((c) => {
    const how = plan[c];
    if (!how || how === 'keep' || how === 'dropRow' || how === 'dropCol') return;
    const arr = out.col(c);
    const v = fillValue(arr, how);
    if (v === null || (typeof v === 'number' && Number.isNaN(v))) return;
    let filled = 0;
    for (let i = 0; i < arr.length; i += 1) {
      if (isMissing(arr[i])) { arr[i] = v; filled += 1; }
    }
    if (filled) {
      log.push({
        col: c, how, filled,
        detail: `${filled}개를 ${typeof v === 'number' ? round(v, 3) : v} 로 채움`,
      });
    }
  });

  // ③ 행을 지우기 (남은 결측치를 지우는 경우도 포함)
  const rowDropCols = out.columns.filter((c) => plan[c] === 'dropRow');
  if (rowDropCols.length) {
    const before = out.length;
    out = out.dropna(rowDropCols);
    log.push({
      col: rowDropCols.join(', '), how: 'dropRow',
      detail: `행 ${before - out.length}개 삭제 (${before} → ${out.length})`,
    });
  }
  return { frame: out, log };
}

/** 원-핫 인코딩 — pd.get_dummies(). 값 하나마다 0/1 컬럼을 만든다 */
export function oneHot(frame, cols, { naLabel = 'unknown' } = {}) {
  let out = frame;
  const created = [];
  cols.forEach((c) => {
    if (!out.columns.includes(c)) return;
    const arr = out.col(c).map((v) => (isMissing(v) ? naLabel : String(v)));
    const levels = [...new Set(arr)].sort((a, b) => a.localeCompare(b, 'ko'));
    out = out.drop([c]);
    levels.forEach((lv) => {
      const name = `${c}_${lv}`;
      out = out.withColumn(name, arr.map((v) => (v === lv ? 1 : 0)));
      created.push(name);
    });
  });
  return { frame: out, created };
}

/** 레이블 인코딩 — LabelEncoder(). 값을 사전순으로 0,1,2… 로 바꾼다 */
export function labelEncode(frame, cols) {
  let out = frame;
  const maps = {};
  cols.forEach((c) => {
    if (!out.columns.includes(c)) return;
    const arr = out.col(c);
    const levels = [...new Set(arr.filter((v) => !isMissing(v)).map(String))]
      .sort((a, b) => a.localeCompare(b, 'ko'));
    const map = new Map(levels.map((lv, i) => [lv, i]));
    maps[c] = Object.fromEntries(map);
    out = out.withColumn(c, arr.map((v) => (isMissing(v) ? NaN : map.get(String(v)))));
  });
  return { frame: out, maps };
}

/** 직접 매핑 — map()/replace(). 순서가 있는 값('고졸'<'학사'<'석사')에 쓴다 */
export function mapEncode(frame, col, mapping) {
  const arr = frame.col(col);
  return frame.withColumn(col, arr.map((v) => {
    if (isMissing(v)) return NaN;
    const k = String(v);
    return Object.prototype.hasOwnProperty.call(mapping, k) ? mapping[k] : NaN;
  }));
}

/**
 * 훈련/검증 데이터 분리 — train_test_split().
 * stratify 를 주면 분류 문제에서 각 정답의 비율을 훈련·검증에 똑같이 유지한다.
 */
export function trainTestSplit(n, { testSize = 0.3, seed = 100, stratify = null } = {}) {
  if (!stratify) {
    const idx = shuffledIndex(n, seed);
    const nTest = Math.round(n * testSize);
    return { train: idx.slice(nTest), valid: idx.slice(0, nTest) };
  }
  // 정답별로 따로 섞어서 같은 비율로 잘라 낸다
  const buckets = new Map();
  for (let i = 0; i < n; i += 1) {
    const k = String(stratify[i]);
    if (!buckets.has(k)) buckets.set(k, []);
    buckets.get(k).push(i);
  }
  const train = [];
  const valid = [];
  let salt = 0;
  for (const [, list] of [...buckets].sort((a, b) => a[0].localeCompare(b[0]))) {
    const order = shuffledIndex(list.length, seed + salt * 7919);
    salt += 1;
    const nTest = Math.round(list.length * testSize);
    order.forEach((pos, rank) => {
      (rank < nTest ? valid : train).push(list[pos]);
    });
  }
  return { train, valid };
}

/* ── 스케일러 ────────────────────────────────────────────────
 * 세 가지 모두 fit(훈련 데이터로 규칙 배우기) → transform(적용) 순서로 쓴다.
 */
class BaseScaler {
  constructor() { this.fitted = false; }

  transform(X) {
    if (!this.fitted) throw new Error('먼저 fit() 을 해야 합니다.');
    return X.map((row) => row.map((v, j) => this.scaleOne(v, j)));
  }

  fitTransform(X) { this.fit(X); return this.transform(X); }
}

/** MinMaxScaler — 최솟값 0, 최댓값 1 로 눌러 준다 */
export class MinMaxScaler extends BaseScaler {
  fit(X) {
    const d = X[0]?.length ?? 0;
    this.min = new Array(d).fill(Infinity);
    this.max = new Array(d).fill(-Infinity);
    X.forEach((row) => row.forEach((v, j) => {
      if (v < this.min[j]) this.min[j] = v;
      if (v > this.max[j]) this.max[j] = v;
    }));
    this.fitted = true;
    return this;
  }

  scaleOne(v, j) {
    const range = this.max[j] - this.min[j];
    return range === 0 ? 0 : (v - this.min[j]) / range;
  }

  describe(j) { return `(x − ${round(this.min[j], 3)}) ÷ ${round(this.max[j] - this.min[j], 3)}`; }
}

/** StandardScaler — 평균 0, 표준편차 1 로 맞춘다 */
export class StandardScaler extends BaseScaler {
  fit(X) {
    const d = X[0]?.length ?? 0;
    this.mean = new Array(d).fill(0);
    this.scale = new Array(d).fill(1);
    for (let j = 0; j < d; j += 1) {
      const col = X.map((r) => r[j]);
      this.mean[j] = mean(col);
      // 사이킷런은 모집단 표준편차(n으로 나눔)를 쓴다
      const m = this.mean[j];
      const varp = col.reduce((a, b) => a + (b - m) ** 2, 0) / (col.length || 1);
      this.scale[j] = Math.sqrt(varp) || 1;
    }
    this.fitted = true;
    return this;
  }

  scaleOne(v, j) { return (v - this.mean[j]) / this.scale[j]; }

  describe(j) { return `(x − ${round(this.mean[j], 3)}) ÷ ${round(this.scale[j], 3)}`; }
}

/** RobustScaler — 중앙값과 IQR 로 맞춘다. 이상치에 잘 흔들리지 않는다 */
export class RobustScaler extends BaseScaler {
  fit(X) {
    const d = X[0]?.length ?? 0;
    this.center = new Array(d).fill(0);
    this.scale = new Array(d).fill(1);
    for (let j = 0; j < d; j += 1) {
      const col = X.map((r) => r[j]).sort((a, b) => a - b);
      this.center[j] = quantile(col, 0.5);
      this.scale[j] = (quantile(col, 0.75) - quantile(col, 0.25)) || 1;
    }
    this.fitted = true;
    return this;
  }

  scaleOne(v, j) { return (v - this.center[j]) / this.scale[j]; }

  describe(j) { return `(x − ${round(this.center[j], 3)}) ÷ ${round(this.scale[j], 3)}`; }
}

export const SCALERS = {
  MinMaxScaler: { make: () => new MinMaxScaler(), note: '모든 값을 0 ~ 1 사이로' },
  StandardScaler: { make: () => new StandardScaler(), note: '평균 0, 표준편차 1 로' },
  RobustScaler: { make: () => new RobustScaler(), note: '중앙값 0, IQR 1 로 (이상치에 강함)' },
  none: { make: () => null, note: '스케일링하지 않음' },
};

export function round(v, n = 3) {
  if (typeof v !== 'number' || Number.isNaN(v)) return v;
  const p = 10 ** n;
  return Math.round(v * p) / p;
}

export { isMissing, std };
