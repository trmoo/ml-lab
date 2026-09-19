/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 파이프라인 — 데이터 한 벌을 「모델에 넣을 수 있는 숫자 표」로 바꾸는 전체 과정.
 *
 * 전처리 단계를 순서대로 이어 붙인 것이다.
 *   ① 쓰지 않을 컬럼 버리기
 *   ② 결측치 채우기
 *   ③ 파생변수 만들기
 *   ④ 범주형 → 숫자 (순서형은 직접 매핑, 나머지는 원-핫/레이블)
 *   ⑤ X 와 y 로 나누기
 *
 * 마지막에 encodeRow() 를 함께 돌려준다.
 * 새 데이터 하나를 넣어 예측할 때 「학습 때와 똑같은 변환」을 적용하기 위한 것으로,
 * 초보자가 가장 자주 놓치는 부분이다.
 */
import { fillValue, isMissing, round } from './preprocess.js';
import { mode } from './frame.js';

export const IMPUTE_PRESETS = {
  meanMode: { label: '수치형 평균 · 범주형 최빈값', num: 'mean', cat: 'mode' },
  medianMode: { label: '수치형 중앙값 · 범주형 최빈값', num: 'median', cat: 'mode' },
  zeroMode: { label: '수치형 0 · 범주형 unknown', num: 'zero', cat: 'unknown' },
  dropRows: { label: '결측치가 있는 행을 모두 삭제', num: 'dropRow', cat: 'dropRow' },
};

/**
 * @param {object} ds       datasets.js 의 데이터셋
 * @param {object} opts
 *   imputePreset  'meanMode' | 'medianMode' | 'zeroMode' | 'dropRows'
 *   imputePlan    컬럼별로 따로 정하고 싶을 때 { 컬럼: 방법 }
 *   encode        'onehot' | 'label'
 *   useOrdinal    순서형 컬럼을 직접 매핑할지 (기본 true)
 *   dropColumns   분석에서 뺄 컬럼
 *   useDerived    파생변수를 만들지
 */
export function buildMatrix(ds, opts = {}) {
  const {
    imputePreset = 'medianMode', imputePlan = null, encode = 'onehot',
    useOrdinal = true, dropColumns = [], useDerived = true,
  } = opts;

  const log = [];
  const drop = new Set([...(ds.idColumns || []), ...dropColumns]);
  let frame = ds.frame.drop([...drop].filter((c) => ds.frame.columns.includes(c)));
  if (drop.size) log.push(`컬럼 삭제: ${[...drop].join(', ')}`);

  const preset = IMPUTE_PRESETS[imputePreset] || IMPUTE_PRESETS.medianMode;

  /* ② 결측치 — 채운 값을 따로 적어 둔다 (새 데이터 예측 때 다시 쓴다) */
  const imputeValues = {};
  const dropRowCols = [];
  frame = frame.copy();
  frame.columns.forEach((c) => {
    const how = imputePlan?.[c] ?? (frame.isNumeric(c) ? preset.num : preset.cat);
    if (how === 'keep') return;
    if (how === 'dropRow') { dropRowCols.push(c); return; }
    const arr = frame.col(c);
    const v = how === 'unknown' && frame.isNumeric(c) ? 0 : fillValue(arr, how);
    if (v === null || (typeof v === 'number' && Number.isNaN(v))) return;
    imputeValues[c] = v;
    for (let i = 0; i < arr.length; i += 1) if (isMissing(arr[i])) arr[i] = v;
  });
  if (dropRowCols.length) {
    const before = frame.length;
    frame = frame.dropna(dropRowCols);
    log.push(`결측 행 삭제: ${before} → ${frame.length}행`);
  }
  const missingFilled = Object.keys(imputeValues).length;
  if (missingFilled) {
    log.push(`결측치 채움: ${missingFilled}개 컬럼 (${preset.label})`);
  }

  /* ③ 파생변수 — a ÷ b. b 가 0 이면 0 으로 둔다 (0으로 나누기 방지) */
  const derived = [];
  if (useDerived && ds.derived) {
    ds.derived.forEach((d) => {
      if (!frame.columns.includes(d.a) || !frame.columns.includes(d.b)) return;
      const A = frame.col(d.a);
      const B = frame.col(d.b);
      frame = frame.withColumn(d.name, A.map((v, i) => (B[i] ? v / B[i] : 0)));
      derived.push(d);
      log.push(`파생변수 ${d.name} 생성 (${d.label})`);
    });
  }

  /* ④ 범주형 → 숫자 */
  const ordinalMaps = {};
  if (useOrdinal && ds.ordinal) {
    Object.entries(ds.ordinal).forEach(([c, map]) => {
      if (!frame.columns.includes(c) || frame.isNumeric(c)) return;
      const arr = frame.col(c);
      ordinalMaps[c] = map;
      frame = frame.withColumn(c, arr.map((v) => {
        if (isMissing(v)) return NaN;
        const k = String(v);
        return Object.prototype.hasOwnProperty.call(map, k) ? map[k] : NaN;
      }));
      log.push(`순서형 인코딩 ${c}: ${Object.entries(map).map(([k, n]) => `${k}→${n}`).join(', ')}`);
    });
  }

  const target = ds.target;
  const catCols = frame.categoricalColumns().filter((c) => c !== target);
  const oneHotLevels = {};
  const labelMaps = {};

  if (encode === 'label') {
    catCols.forEach((c) => {
      const arr = frame.col(c);
      const levels = [...new Set(arr.filter((v) => !isMissing(v)).map(String))]
        .sort((a, b) => a.localeCompare(b, 'ko'));
      labelMaps[c] = levels;
      frame = frame.withColumn(c, arr.map((v) => {
        const k = isMissing(v) ? null : String(v);
        const at = k === null ? -1 : levels.indexOf(k);
        return at < 0 ? 0 : at;
      }));
    });
    if (catCols.length) log.push(`레이블 인코딩: ${catCols.join(', ')} (값을 0,1,2… 로)`);
  } else {
    catCols.forEach((c) => {
      const arr = frame.col(c).map((v) => (isMissing(v) ? 'unknown' : String(v)));
      const levels = [...new Set(arr)].sort((a, b) => a.localeCompare(b, 'ko'));
      oneHotLevels[c] = levels;
      frame = frame.drop([c]);
      levels.forEach((lv) => {
        frame = frame.withColumn(`${c}_${lv}`, arr.map((v) => (v === lv ? 1 : 0)));
      });
    });
    if (catCols.length) {
      const made = Object.values(oneHotLevels).reduce((a, l) => a + l.length, 0);
      log.push(`원-핫 인코딩: ${catCols.join(', ')} → 컬럼 ${made}개`);
    }
  }

  /* ⑤ X, y 로 나누기 */
  const featureNames = frame.columns.filter((c) => c !== target);
  const yRaw = frame.col(target);
  const X = [];
  const y = [];
  const keptRows = [];
  for (let i = 0; i < frame.length; i += 1) {
    if (isMissing(yRaw[i])) continue;
    const row = featureNames.map((c) => frame.col(c)[i]);
    if (row.some((v) => typeof v !== 'number' || Number.isNaN(v))) continue;
    X.push(row); y.push(yRaw[i]); keptRows.push(i);
  }
  const lost = frame.length - X.length;
  if (lost > 0) log.push(`남은 결측치 때문에 ${lost}행 제외 (사용 ${X.length}행)`);

  /* 새 데이터 하나를 학습 때와 똑같이 변환하는 함수 */
  const encodeRow = (raw) => {
    const work = { ...raw };
    Object.keys(imputeValues).forEach((c) => {
      if (isMissing(work[c])) work[c] = imputeValues[c];
    });
    derived.forEach((d) => {
      const b = Number(work[d.b]);
      work[d.name] = b ? Number(work[d.a]) / b : 0;
    });
    Object.entries(ordinalMaps).forEach(([c, map]) => {
      if (typeof work[c] === 'string') work[c] = map[work[c]] ?? 0;
    });
    return featureNames.map((name) => {
      if (Object.prototype.hasOwnProperty.call(work, name)) {
        const v = Number(work[name]);
        return Number.isFinite(v) ? v : 0;
      }
      // 원-핫으로 늘어난 컬럼 (원래컬럼_값)
      const hit = Object.entries(oneHotLevels)
        .map(([c, levels]) => {
          const lv = levels.find((l) => name === `${c}_${l}`);
          return lv === undefined ? null : { c, lv };
        }).find(Boolean);
      if (hit) return String(work[hit.c]) === hit.lv ? 1 : 0;
      if (Object.prototype.hasOwnProperty.call(labelMaps, name)) {
        const at = labelMaps[name].indexOf(String(work[name]));
        return at < 0 ? 0 : at;
      }
      return 0;
    });
  };

  return {
    X, y, featureNames, frame, keptRows, log,
    imputeValues, oneHotLevels, labelMaps, ordinalMaps, derived,
    encodeRow,
    /** 예측 체험에서 슬라이더 기본값으로 쓸 「대표적인 한 줄」 */
    typicalRow: typicalRowOf(ds, imputeValues),
    summary: {
      rows: X.length, features: featureNames.length,
      droppedRows: lost, encode, imputePreset,
    },
  };
}

/**
 * 예측 목표에 어떤 값이 몇 개씩 있는지 센다 (분류 문제용).
 *
 * 한쪽 정답이 아주 드물면(불균형 데이터) 정확도가 아무 의미가 없어진다.
 * 예를 들어 불량이 0.3% 뿐이면 「전부 정상」이라고만 찍어도 정확도가 99.7% 다.
 * 그래서 majorityRatio(다수 쪽으로만 찍었을 때의 정확도)를 함께 돌려준다.
 */
export function classBalance(ds) {
  const y = ds.frame.col(ds.target);
  const counts = new Map();
  y.forEach((v) => {
    if (v === null || Number.isNaN(v)) return;
    counts.set(v, (counts.get(v) || 0) + 1);
  });
  const total = [...counts.values()].reduce((a, b) => a + b, 0);
  const rows = [...counts].sort((a, b) => b[1] - a[1]).map(([v, n]) => ({
    value: v,
    name: ds.classNames?.[v] ?? String(v),
    n,
    ratio: total ? n / total : 0,
  }));
  return { rows, total, majorityRatio: rows.length ? rows[0].ratio : 1 };
}

/** 수치형은 중앙값, 범주형은 최빈값으로 채운 「평범한 한 줄」 */
function typicalRowOf(ds, imputeValues) {
  const out = {};
  ds.frame.columns.forEach((c) => {
    if (c === ds.target) return;
    if (ds.frame.isNumeric(c)) {
      const v = ds.frame.values(c).slice().sort((a, b) => a - b);
      out[c] = round(v.length ? v[Math.floor(v.length / 2)] : (imputeValues[c] ?? 0), 3);
    } else {
      out[c] = mode(ds.frame.col(c)) ?? 'unknown';
    }
  });
  return out;
}
