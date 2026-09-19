/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 탭 ② 전처리 실험실
 *
 * 머신러닝 파이프라인의 전처리 단계에 해당한다.
 *   ① 결측치 처리      — 방법을 바꿔 가며 분포가 어떻게 달라지는지 본다
 *   ② 인코딩          — 원-핫 / 레이블 / 순서형 직접 매핑을 표로 비교한다
 *   ③ 훈련·검증 분리   — test_size 와 random_state 를 만져 본다
 *   ④ 스케일링        — 세 스케일러의 결과를 나란히 보고,
 *                        검증 데이터에 fit_transform 을 쓰면 왜 안 되는지 실험한다
 */
import { getDataset, datasetOptions, DATASET_IDS } from '../data/datasets.js';
import {
  h, card, table, figure, selectRow, pills, num, codeBlock, foldout, badge, checkRow,
  sliderRow, redraw,
} from '../lib/ui.js';
import { histChart, barChart } from '../lib/chart.js';
import { histogram, mean, std, quantile } from '../lib/frame.js';
import {
  MinMaxScaler, StandardScaler, RobustScaler, fillValue, isMissing, trainTestSplit,
} from '../lib/preprocess.js';

const state = {
  dsId: DATASET_IDS[0],   // 첫 회차. 수업 순서가 바뀌어도 따라간다
  fillCol: null,
  fillHow: 'mean',
  encodeCol: null,
  encodeKind: 'onehot',
  testSize: 0.3,
  seed: 100,
  stratify: true,
  scaler: 'StandardScaler',
  scaleCols: null,
  leak: false,
};

/** 다른 탭(문제 풀기)에서 이 탭을 열 때 데이터셋을 미리 골라 둔다 */
export function useDataset(dsId) {
  if (!dsId || dsId === state.dsId) return;
  state.dsId = dsId;
  // 데이터 고르기 단추를 눌렀을 때와 똑같이 이전 데이터에 묶인 선택을 비운다
  state.fillCol = null; state.encodeCol = null; state.scaleCols = null;
}

export function renderPrep(root) {
  // redraw 로 감싸면 옵션을 눌러 다시 그려도 보던 자리에 그대로 머무른다
  redraw(root, () => {
    const ds = getDataset(state.dsId);
    root.append(
      picker(root),
      missingCard(ds, root),
      encodeCard(ds, root),
      splitCard(ds, root),
      scaleCard(ds, root),
    );
  });
}

function picker(root) {
  const grid = h('div', { class: 'dspick' }, datasetOptions().map((o) => h('button', {
    type: 'button',
    class: `dsbtn${o.id === state.dsId ? ' on' : ''}`,
    onclick: () => {
      state.dsId = o.id;
      state.fillCol = null; state.encodeCol = null; state.scaleCols = null;
      renderPrep(root);
    },
  }, h('b', {}, `${o.round}회차 · ${o.short}`), h('span', {}, o.taskLabel))));
  return card('실습 데이터 고르기', grid);
}

/* ── ① 결측치 처리 ───────────────────────────────────── */
const FILL_KO = {
  mean: '평균값(mean)', median: '중앙값(median)', mode: '최빈값(mode)',
  zero: '0 으로', dropRow: '그 행을 삭제', dropCol: '그 컬럼을 삭제',
};

function missingCard(ds, root) {
  const info = ds.frame.info().filter((r) => r.missing > 0);
  const rows = ds.frame.info().map((r) => ({
    cells: [r.name, r.dtype === 'float64' ? '수치형' : '범주형',
      r.missing ? h('span', { class: 'mi' }, `${r.missing}개`) : h('span', { class: 'na' }, '없음'),
      r.missing ? `${((r.missing / ds.frame.length) * 100).toFixed(1)}%` : '—',
      r.missing
        ? (r.dtype === 'float64' ? '평균 · 중앙값 · 0 · 행 삭제' : '최빈값 · unknown · 행 삭제')
        : '처리 불필요'],
    hi: r.missing > 0,
  }));

  if (!info.length) {
    return card('① 결측치 처리 — fillna()',
      table(['컬럼', '타입', '결측치', '비율', '쓸 수 있는 방법'], rows),
      h('div', { class: 'ok' }, '이 데이터에는 결측치가 없다. 다른 회차를 골라 보자.'));
  }

  if (!state.fillCol || !info.some((r) => r.name === state.fillCol)) {
    state.fillCol = info[0].name;
  }
  const col = state.fillCol;
  const numeric = ds.frame.isNumeric(col);
  const howOptions = numeric
    ? ['mean', 'median', 'zero', 'dropRow']
    : ['mode', 'dropRow', 'dropCol'];
  if (!howOptions.includes(state.fillHow)) [state.fillHow] = howOptions;

  const original = ds.frame.col(col);
  const cleanVals = original.filter((v) => !isMissing(v));
  const filler = fillValue(original, state.fillHow);

  // 처리 뒤 값 배열을 만든다
  let after = null;
  let afterNote = '';
  if (state.fillHow === 'dropRow') {
    after = cleanVals;
    afterNote = `결측 ${original.length - cleanVals.length}개 행을 지웠다. `
      + `${original.length} → ${cleanVals.length}행`;
  } else if (state.fillHow === 'dropCol') {
    afterNote = '컬럼 자체를 지우므로 남는 값이 없다. 정말 필요 없는 정보일 때만 쓴다.';
  } else {
    after = original.map((v) => (isMissing(v) ? filler : v));
    afterNote = `결측 ${original.length - cleanVals.length}개를 `
      + `${typeof filler === 'number' ? num(filler) : filler} 로 채웠다.`;
  }

  const body = [];
  if (numeric && after) {
    const before = { label: '처리 전(결측 제외)', ...histogram(cleanVals, 24) };
    const post = { label: '처리 후', ...histogram(after, 24) };
    body.push(figure(histChart([before, post], { xLabel: col, yLabel: '개수' })));
    const rowsStat = [
      ['개수', cleanVals.length, after.length],
      ['평균', mean(cleanVals), mean(after)],
      ['표준편차', std(cleanVals), std(after)],
      ['중앙값', quantile(cleanVals.slice().sort((a, b) => a - b), 0.5),
        quantile(after.slice().sort((a, b) => a - b), 0.5)],
    ].map(([k, a, b]) => ({
      cells: [k, num(a), num(b),
        typeof a === 'number' && typeof b === 'number' && a !== 0
          ? `${(((b - a) / Math.abs(a)) * 100).toFixed(2)}%` : '—'],
    }));
    body.push(table(['통계량', '처리 전', '처리 후', '변화율'], rowsStat));
    if (state.fillHow === 'mean' || state.fillHow === 'median') {
      body.push(h('div', { class: 'note' },
        '평균이나 중앙값으로 채우면 그 값 하나에 막대가 몰려 솟는다. '
        + '평균은 그대로지만 ', h('b', {}, '표준편차가 줄어든다'),
        ' — 실제보다 데이터가 고르게 보이는 부작용이다.'));
    }
    if (state.fillHow === 'zero') {
      body.push(h('div', { class: 'warn' },
        '0 으로 채우기는 「0 이 실제로 뜻이 있는 값」일 때만 써야 한다. '
        + '연봉이나 온도에 0 을 넣으면 없던 이상치를 만드는 셈이다.'));
    }
  } else if (!numeric && after) {
    const groupsBefore = countOf(cleanVals);
    const groupsAfter = countOf(after);
    body.push(figure(barChart(groupsAfter.map((g) => ({
      label: g.k, value: g.n, highlight: g.k === filler,
    })), { xLabel: col, yLabel: '개수' }),
    `색이 진한 막대가 최빈값이다. 결측치를 여기에 몰아 넣게 된다.`));
    body.push(table(['값', '처리 전', '처리 후'], groupsAfter.map((g) => ({
      cells: [g.k, num(groupsBefore.find((x) => x.k === g.k)?.n ?? 0), num(g.n)],
      hi: g.k === filler,
    })), { rowClass: (r) => (r.hi ? 'hi' : '') }));
  }

  return card('① 결측치 처리 — fillna() / dropna()',
    h('p', { class: 'lead' },
      '모델은 빈칸을 계산할 수 없다. 채우거나 지워야 하는데, 방법마다 데이터가 조금씩 달라진다.'),
    table(['컬럼', '타입', '결측치', '비율', '쓸 수 있는 방법'], rows,
      { rowClass: (r) => (r.hi ? 'hi' : '') }),
    h('div', { class: 'ctrls' },
      selectRow('처리할 컬럼', info.map((r) => ({ value: r.name, label: `${r.name} (${r.missing}개)` })),
        col, (v) => { state.fillCol = v; renderPrep(root); }),
      h('div', { class: 'ctrl' }, h('span', { class: 'ctrl-l' }, '처리 방법'),
        pills(howOptions.map((k) => ({ value: k, label: FILL_KO[k] })),
          state.fillHow, (v) => { state.fillHow = v; renderPrep(root); }, { small: true }))),
    h('div', { class: 'tip' }, afterNote),
    ...body,
    foldout('파이썬으로는 이렇게 씁니다', codeBlock(pyFill(ds, col, state.fillHow)).outerHTML));
}

function pyFill(ds, col, how) {
  const pre = `${ds.df.replace('_df', '_pre')}`;
  const head = `${pre} = ${ds.df}.copy()   # 원본은 남겨 둔다\n\n`;
  if (how === 'dropRow') return `${head}${pre} = ${pre}.dropna(subset=['${col}'])`;
  if (how === 'dropCol') return `${head}${pre} = ${pre}.drop('${col}', axis=1)`;
  if (how === 'mode') {
    return `${head}fill = ${pre}['${col}'].mode()[0]   # 최빈값\n`
      + `${pre}['${col}'] = ${pre}['${col}'].fillna(fill)`;
  }
  if (how === 'zero') return `${head}${pre}['${col}'] = ${pre}['${col}'].fillna(0)`;
  return `${head}fill = ${pre}['${col}'].${how}()\n`
    + `${pre}['${col}'] = ${pre}['${col}'].fillna(fill)`;
}

function countOf(arr) {
  const m = new Map();
  arr.forEach((v) => m.set(String(v), (m.get(String(v)) || 0) + 1));
  return [...m].map(([k, n]) => ({ k, n })).sort((a, b) => b.n - a.n);
}

/* ── ② 인코딩 ───────────────────────────────────────── */
function encodeCard(ds, root) {
  const catCols = ds.frame.categoricalColumns().filter((c) => c !== ds.target);
  if (!catCols.length) {
    return card('② 인코딩 — 글자를 숫자로',
      h('div', { class: 'ok' }, '이 데이터에는 범주형 컬럼이 없어 인코딩할 것이 없다.'));
  }
  if (!state.encodeCol || !catCols.includes(state.encodeCol)) [state.encodeCol] = catCols;
  const col = state.encodeCol;
  const ordinalMap = ds.ordinal?.[col] || null;
  const kinds = [
    { value: 'onehot', label: '원-핫 (get_dummies)' },
    { value: 'label', label: '레이블 (LabelEncoder)' },
  ];
  if (ordinalMap) kinds.push({ value: 'ordinal', label: '순서형 직접 매핑 (map)' });
  if (state.encodeKind === 'ordinal' && !ordinalMap) state.encodeKind = 'onehot';

  const raw = ds.frame.col(col).map((v) => (isMissing(v) ? 'unknown' : String(v)));
  const levels = [...new Set(raw)].sort((a, b) => a.localeCompare(b, 'ko'));
  const sample = [0, 1, 2, 3, 4, 5, 6].filter((i) => i < ds.frame.length);

  let headers = [];
  let rows = [];
  let explain = '';

  if (state.encodeKind === 'onehot') {
    headers = ['행', col, ...levels.map((l) => `${col}_${l}`)];
    rows = sample.map((i) => ({
      cells: [i, raw[i], ...levels.map((l) => h('span', {
        class: raw[i] === l ? 'mi' : 'na',
      }, raw[i] === l ? '1' : '0'))],
    }));
    explain = `값 ${levels.length}종류가 각각 컬럼 하나가 되어 열이 ${levels.length}개 늘어난다. `
      + '순서가 없는 값(브랜드·지역·색상)에 쓴다. 값 종류가 너무 많으면 열이 폭발하니 주의한다.';
  } else if (state.encodeKind === 'label') {
    const map = new Map(levels.map((l, k) => [l, k]));
    headers = ['행', col, `${col} (숫자)`];
    rows = sample.map((i) => ({ cells: [i, raw[i], h('b', {}, map.get(raw[i]))] }));
    explain = `사전순으로 0, 1, 2 … 를 붙인다 (${levels.map((l, k) => `${l}→${k}`).join(', ')}). `
      + '열이 늘지 않아 간편하지만, 모델은 「2가 1보다 크다」고 오해할 수 있다. '
      + '값이 두 개거나 진짜 순서가 있을 때만 안전하다.';
  } else {
    headers = ['행', col, `${col} (숫자)`];
    rows = sample.map((i) => ({
      cells: [i, raw[i], h('b', {}, ordinalMap[raw[i]] ?? '—')],
    }));
    explain = `순서를 사람이 정해 준다 (${Object.entries(ordinalMap).map(([k, v]) => `${k}→${v}`).join(', ')}). `
      + '학력·직급·계약기간처럼 크기 비교가 뜻이 있는 값에 가장 알맞다.';
  }

  const dist = countOf(raw);
  return card('② 인코딩 — 글자를 숫자로',
    h('p', { class: 'lead' }, '머신러닝 모델은 글자를 계산할 수 없다. 반드시 숫자로 바꿔야 한다.'),
    h('div', { class: 'ctrls' },
      selectRow('컬럼', catCols.map((c) => ({ value: c, label: `${c} (${new Set(ds.frame.col(c).map(String)).size}종)` })),
        col, (v) => { state.encodeCol = v; renderPrep(root); }),
      h('div', { class: 'ctrl' }, h('span', { class: 'ctrl-l' }, '방법'),
        pills(kinds, state.encodeKind, (v) => { state.encodeKind = v; renderPrep(root); }, { small: true })),
      ordinalMap ? h('span', { class: 'ctrl-hint' }, badge('순서가 있는 컬럼', 'v')) : null),
    figure(barChart(dist.map((d) => ({ label: d.k, value: d.n })),
      { xLabel: col, yLabel: '개수' }), `${col} 의 값 분포 — 종류 ${levels.length}개`),
    h('div', { class: 'ctrl-l' }, '변환 전 → 변환 후 (앞 7줄)'),
    table(headers, rows, { className: 'tbl-s' }),
    h('div', { class: 'tip' }, explain),
    foldout('파이썬으로는 이렇게 씁니다', codeBlock(pyEncode(ds, col, state.encodeKind, ordinalMap)).outerHTML));
}

function pyEncode(ds, col, kind, ordinalMap) {
  const pre = ds.df.replace('_df', '_pre');
  if (kind === 'onehot') {
    return `# 순서가 없는 범주형 → 원-핫 인코딩\n`
      + `${pre} = pd.get_dummies(${pre}, columns=['${col}'])`;
  }
  if (kind === 'label') {
    return `from sklearn.preprocessing import LabelEncoder\n\n`
      + `le = LabelEncoder()\n`
      + `${pre}['${col}'] = le.fit_transform(${pre}['${col}'])\n`
      + `print(le.classes_)   # 어떤 값이 몇 번이 되었는지 확인`;
  }
  const body = Object.entries(ordinalMap).map(([k, v]) => `'${k}': ${v}`).join(', ');
  return `# 순서가 있는 범주형 → 직접 매핑\n`
    + `${pre}['${col}'] = ${pre}['${col}'].map({${body}})`;
}

/* ── ③ 훈련·검증 분리 ───────────────────────────────── */
function splitCard(ds, root) {
  const isCls = ds.task === 'classification';
  const labels = isCls ? ds.frame.col(ds.target) : null;
  const n = ds.frame.length;
  const { train, valid } = trainTestSplit(n, {
    testSize: state.testSize,
    seed: state.seed,
    stratify: isCls && state.stratify ? labels : null,
  });

  // 분리가 잘 되었는지 — 정답 비율(분류) 또는 평균(회귀) 을 비교
  let compare = null;
  if (isCls) {
    const ratio = (idx) => idx.filter((i) => labels[i] === 1).length / (idx.length || 1);
    const rows = [
      ['전체', n, ratio(Array.from({ length: n }, (_, i) => i))],
      ['훈련(train)', train.length, ratio(train)],
      ['검증(valid)', valid.length, ratio(valid)],
    ].map(([k, cnt, r]) => ({ cells: [k, num(cnt), `${(r * 100).toFixed(2)}%`] }));
    compare = table(['구분', '행 수', `${ds.classNames?.[1] ?? '1'} 비율`], rows);
  } else {
    const y = ds.frame.col(ds.target);
    const avg = (idx) => mean(idx.map((i) => y[i]));
    const rows = [
      ['전체', n, avg(Array.from({ length: n }, (_, i) => i))],
      ['훈련(train)', train.length, avg(train)],
      ['검증(valid)', valid.length, avg(valid)],
    ].map(([k, cnt, m]) => ({ cells: [k, num(cnt), num(m)] }));
    compare = table(['구분', '행 수', `${ds.target} 평균`], rows);
  }

  const bar = h('div', { class: 'layerrow' },
    h('div', {
      class: 'layerbox',
      style: `flex:${train.length};background:#dbeafe;border-color:#93c5fd;text-align:center`,
    }, `훈련 ${train.length}행 (${((1 - state.testSize) * 100).toFixed(0)}%)`),
    h('div', {
      class: 'layerbox',
      style: `flex:${Math.max(valid.length, 1)};background:#fef3c7;border-color:#fcd34d;text-align:center`,
    }, `검증 ${valid.length}행 (${(state.testSize * 100).toFixed(0)}%)`));

  return card('③ 훈련·검증 데이터 분리 — train_test_split()',
    h('p', { class: 'lead' },
      '모델을 가르칠 데이터(훈련)와 실력을 재 볼 데이터(검증)를 반드시 나눠야 한다. '
      + '외운 문제로 시험을 보면 실력을 알 수 없기 때문이다.'),
    h('div', { class: 'ctrls' },
      sliderRow('검증 비율 test_size', {
        min: 0.1, max: 0.5, step: 0.05, value: state.testSize,
        format: (v) => `${(v * 100).toFixed(0)}%`,
        onChange: (v) => { state.testSize = v; renderPrep(root); },
      }),
      sliderRow('random_state', {
        min: 0, max: 300, step: 1, value: state.seed,
        onChange: (v) => { state.seed = v; renderPrep(root); },
        hint: '같은 값이면 언제나 같게 나뉜다',
      }),
      isCls
        ? checkRow('stratify=y 쓰기', state.stratify,
          (v) => { state.stratify = v; renderPrep(root); }, '정답 비율을 유지')
        : h('span', { class: 'ctrl-hint' }, '회귀 문제라 stratify 는 쓰지 않는다')),
    bar,
    compare,
    isCls
      ? h('div', { class: state.stratify ? 'ok' : 'warn' },
        state.stratify
          ? 'stratify=y 를 켰으므로 훈련·검증의 정답 비율이 전체와 거의 같다. 불균형 데이터에서 특히 중요하다.'
          : 'stratify 를 끄면 우연히 한쪽에 정답이 몰릴 수 있다. random_state 를 바꿔 보며 비율이 흔들리는 것을 확인해 보자.')
      : h('div', { class: 'tip' },
        'random_state 를 바꾸면 평균이 조금씩 달라진다. 그래서 모델 성능을 비교할 때는 '
        + 'random_state 를 고정해 두어야 「모델 차이」인지 「운 차이」인지 알 수 있다.'),
    foldout('파이썬으로는 이렇게 씁니다', codeBlock(
      `from sklearn.model_selection import train_test_split\n\n`
      + `X = ${ds.df.replace('_df', '_encoded')}.drop('${ds.target}', axis=1)\n`
      + `y = ${ds.df.replace('_df', '_encoded')}['${ds.target}']\n\n`
      + `X_train, X_valid, y_train, y_valid = train_test_split(\n`
      + `    X, y, test_size=${state.testSize}, random_state=${state.seed}`
      + `${isCls && state.stratify ? ', stratify=y' : ''})`).outerHTML));
}

/* ── ④ 스케일링 + 데이터 누수 실험 ──────────────────── */
function scaleCard(ds, root) {
  const numCols = ds.frame.numericColumns().filter((c) => c !== ds.target);
  if (numCols.length < 2) return h('div');
  if (!state.scaleCols) state.scaleCols = numCols.slice(0, 5);
  const cols = state.scaleCols.filter((c) => numCols.includes(c));

  // 결측치를 중앙값으로 채운 값으로 실험한다 (스케일링만 보려는 것)
  const filled = cols.map((c) => {
    const arr = ds.frame.col(c);
    const f = fillValue(arr, 'median');
    return arr.map((v) => (isMissing(v) ? f : v));
  });
  const n = ds.frame.length;
  const X = Array.from({ length: n }, (_, i) => filled.map((a) => a[i]));
  const { train, valid } = trainTestSplit(n, { testSize: state.testSize, seed: state.seed });
  const Xtr = train.map((i) => X[i]);
  const Xva = valid.map((i) => X[i]);

  const make = () => {
    if (state.scaler === 'MinMaxScaler') return new MinMaxScaler();
    if (state.scaler === 'RobustScaler') return new RobustScaler();
    return new StandardScaler();
  };
  const scaler = make().fit(Xtr);
  const trS = scaler.transform(Xtr);
  const vaS = scaler.transform(Xva);
  // 데이터 누수: 검증 데이터로 다시 fit 해 버린 경우
  const leaked = make().fit(Xva).transform(Xva);

  const statRows = cols.map((c, j) => {
    const rawTr = Xtr.map((r) => r[j]);
    const sTr = trS.map((r) => r[j]);
    const sVa = vaS.map((r) => r[j]);
    return {
      cells: [c, `${num(Math.min(...rawTr))} ~ ${num(Math.max(...rawTr))}`,
        scaler.describe(j),
        `${num(Math.min(...sTr))} ~ ${num(Math.max(...sTr))}`,
        num(mean(sTr)), num(std(sTr)),
        `${num(Math.min(...sVa))} ~ ${num(Math.max(...sVa))}`],
    };
  });

  const j0 = 0;
  const beforeHist = { label: '원래 값', ...histogram(Xtr.map((r) => r[j0]), 24) };
  const afterHist = { label: '스케일링 후', ...histogram(trS.map((r) => r[j0]), 24) };

  const leakRows = cols.map((c, j) => {
    const right = vaS.map((r) => r[j]);
    const wrong = leaked.map((r) => r[j]);
    const diff = mean(right.map((v, i) => Math.abs(v - wrong[i])));
    return { cells: [c, num(mean(right)), num(mean(wrong)), num(diff)] };
  });

  return card('④ 스케일링 — 단위를 맞추고, 데이터 누수를 피하기',
    h('p', { class: 'lead' },
      '주행거리(20만)와 연비(8.6)를 그대로 넣으면 거리 계산에서 주행거리가 모든 것을 정한다. '
      + 'KNN·SVM·신경망에는 스케일링이 사실상 필수다. (트리 계열은 영향을 거의 받지 않는다)'),
    h('div', { class: 'ctrls' },
      h('div', { class: 'ctrl' }, h('span', { class: 'ctrl-l' }, '스케일러'),
        pills([
          { value: 'MinMaxScaler', label: 'MinMaxScaler', title: '0~1 로' },
          { value: 'StandardScaler', label: 'StandardScaler', title: '평균 0, 표준편차 1' },
          { value: 'RobustScaler', label: 'RobustScaler', title: '중앙값과 IQR 기준' },
        ], state.scaler, (v) => { state.scaler = v; renderPrep(root); }, { small: true })),
      h('div', { class: 'ctrl' }, h('span', { class: 'ctrl-l' }, '볼 컬럼 (최대 6개)'),
        h('div', { class: 'pills pills-s' }, numCols.map((c) => h('button', {
          type: 'button', class: `pill${cols.includes(c) ? ' on' : ''}`,
          onclick: () => {
            const set = new Set(cols);
            if (set.has(c)) set.delete(c); else if (set.size < 6) set.add(c);
            state.scaleCols = numCols.filter((x) => set.has(x));
            if (!state.scaleCols.length) state.scaleCols = [numCols[0]];
            renderPrep(root);
          },
        }, c))))),
    table(['컬럼', '원래 범위(훈련)', '변환식', '변환 후(훈련)', '평균', '표준편차', '변환 후(검증)'],
      statRows, { className: 'tbl-s' }),
    figure(histChart([beforeHist], { xLabel: `${cols[j0]} — 원래 값`, yLabel: '개수' })),
    figure(histChart([afterHist], { xLabel: `${cols[j0]} — ${state.scaler} 적용 후`, yLabel: '개수' }),
      '모양은 그대로고 눈금만 바뀐다. 스케일링은 분포를 바꾸는 것이 아니라 단위를 맞추는 것이다.'),
    h('div', { class: 'tip' },
      state.scaler === 'MinMaxScaler'
        ? '검증 데이터 범위가 0~1 을 살짝 넘을 수 있다. 훈련 데이터에서 배운 최솟값·최댓값을 그대로 쓰기 때문인데, 정상이다.'
        : state.scaler === 'RobustScaler'
          ? '중앙값과 IQR 로 나누므로 이상치가 있어도 가운데 값들이 좁게 뭉치지 않는다.'
          : '훈련 데이터의 평균은 정확히 0, 표준편차는 1 이 된다. 검증 데이터는 0 근처지만 정확히 0 은 아니다.'),
    h('hr', { class: 'sep' }),
    h('div', { class: 'ctrl-l' }, '🔬 데이터 누수(Data Leakage) 실험'),
    h('p', { class: 'lead' },
      '검증 데이터에 실수로 fit_transform() 을 쓰면 어떻게 될까? '
      + '아래 표의 두 번째·세 번째 열을 비교해 보자.'),
    table(['컬럼', h('span', {}, '올바름: transform()'), h('span', { class: 'mi' }, '잘못: fit_transform()'),
      '평균 차이'], leakRows, { className: 'tbl-s' }),
    h('div', { class: 'warn' },
      h('b', {}, '왜 문제인가 — '),
      '검증 데이터로 fit 하면 「아직 보지 않아야 할 데이터의 평균·최댓값」을 모델이 미리 알게 된다. '
      + '그러면 검증 점수가 실제 실력보다 좋게 나오고, 진짜 새 데이터에서는 성능이 뚝 떨어진다. '
      + '규칙은 하나다 — ',
      h('b', {}, 'fit 은 훈련 데이터에만, 검증·시험 데이터에는 transform 만.')),
    foldout('파이썬으로는 이렇게 씁니다', codeBlock(
      `from sklearn.preprocessing import ${state.scaler}\n\n`
      + `scaler = ${state.scaler}()\n\n`
      + `X_train_scaled = scaler.fit_transform(X_train)   # 규칙을 배우고 + 적용\n`
      + `X_valid_scaled = scaler.transform(X_valid)       # 배운 규칙만 적용\n\n`
      + `# X_valid_scaled = scaler.fit_transform(X_valid)  ← 이렇게 하면 데이터 누수!`).outerHTML));
}
