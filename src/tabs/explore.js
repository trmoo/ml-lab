/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 탭 ① 데이터 탐험실
 *
 * 머신러닝 파이프라인의 데이터 탐색 단계에 해당한다.
 *   head() · info() · describe() · groupby() · crosstab() · corr()
 *   그리고 seaborn 으로 그리던 그림들을 브라우저에서 직접 본다.
 *
 * 모델을 만들기 전에 데이터를 먼저 들여다보는 습관을 들이는 것이 이 탭의 목적이다.
 */
import { getDataset, datasetOptions, DATASET_IDS } from '../data/datasets.js';
import {
  h, card, table, figure, selectRow, pills, num, codeBlock, foldout, badge, clear, redraw,
} from '../lib/ui.js';
import {
  barChart, histChart, scatterChart, boxChart, heatmapChart, hBarChart, emptyChart,
} from '../lib/chart.js';
import { histogram, boxStats, pearson } from '../lib/frame.js';
import { classBalance } from '../lib/pipeline.js';

const state = {
  dsId: DATASET_IDS[0],   // 첫 회차. 수업 순서가 바뀌어도 따라간다
  groupKey: null,
  groupValue: null,
  agg: 'mean',
  plotKind: 'scatter',
  xCol: null,
  yCol: null,
  hueCol: '',
  bins: 20,
};

/** 다른 탭(문제 풀기)에서 이 탭을 열 때 데이터셋을 미리 골라 둔다 */
export function useDataset(dsId) {
  if (!dsId || dsId === state.dsId) return;
  state.dsId = dsId;
  // 데이터 고르기 단추를 눌렀을 때와 똑같이 이전 데이터에 묶인 선택을 비운다
  state.groupKey = null; state.groupValue = null;
    state.xCol = null; state.yCol = null; state.hueCol = '';
}

export function renderExplore(root) {
  // redraw 로 감싸면 옵션을 눌러 다시 그려도 보던 자리에 그대로 머무른다
  redraw(root, () => {
    const ds = getDataset(state.dsId);
    root.append(
      datasetPicker(root),
      overviewCard(ds),
      targetCard(ds),
      infoCard(ds),
      describeCard(ds),
      groupCard(ds, root),
      plotCard(ds, root),
      corrCard(ds),
    );
  });
}

/* ── 데이터셋 고르기 ── */
function datasetPicker(root) {
  const opts = datasetOptions();
  const grid = h('div', { class: 'dspick' }, opts.map((o) => h('button', {
    type: 'button',
    class: `dsbtn${o.id === state.dsId ? ' on' : ''}`,
    onclick: () => {
      state.dsId = o.id;
      state.groupKey = null; state.groupValue = null;
      state.xCol = null; state.yCol = null; state.hueCol = '';
      renderExplore(root);
    },
  },
  h('b', {}, `${o.round}회차 · ${o.label}`),
  h('span', {}, `예측 목표는 ${o.taskLabel} 문제`))));
  return card('실습 데이터 고르기', grid);
}

/* ── 개요 ── */
function overviewCard(ds) {
  const cols = ds.frame.columns.map((c) => h('div', {
    class: `colcard${c === ds.target ? ' tgt' : ''}`,
  },
  h('b', {}, c),
  c === ds.target ? badge('예측 목표', 'g') : null,
  h('div', { class: 'cd' }, ds.colDesc[c] || (ds.frame.isNumeric(c) ? '수치형' : '범주형'))));

  return card(`${ds.round}회차 — ${ds.label}`,
    h('p', { class: 'lead' }, ds.background),
    h('div', { class: 'kv' },
      h('span', {}, h('b', {}, '파일 '), ds.file),
      h('span', {}, h('b', {}, '행 '), `${ds.frame.length.toLocaleString('ko-KR')}개`),
      h('span', {}, h('b', {}, '열 '), `${ds.frame.columns.length}개`),
      h('span', {}, h('b', {}, '문제 종류 '),
        ds.task === 'regression' ? '회귀 (숫자 맞히기)' : '분류 (둘 중 하나 맞히기)'),
      h('span', {}, h('b', {}, '예측 목표 '), `${ds.target}${ds.unit ? ` (${ds.unit})` : ''}`)),
    ds.note ? h('div', { class: 'note' }, ds.note) : null,
    h('div', { class: 'colgrid' }, cols),
    foldout('파이썬으로는 이렇게 씁니다', codeBlock(
      `import pandas as pd\n\n`
      + `${ds.df} = pd.read_csv('${ds.file.split(' ')[0]}')\n`
      + `${ds.df}.head()          # 위에서 5줄만 보기\n`
      + `${ds.df}.shape           # (행, 열) 개수`).outerHTML));
}

/* ── 예측 목표(Target) 살펴보기 ──
 * 모델을 만들기 전에 「맞혀야 하는 값」이 어떻게 생겼는지 먼저 봐야 한다.
 * 특히 분류 문제에서 한쪽 정답이 아주 드물면 정확도가 아무 의미가 없어진다.
 */
function targetCard(ds) {
  if (ds.task === 'classification') {
    const bal = classBalance(ds);
    const minority = bal.rows[bal.rows.length - 1];
    const imbalanced = minority && minority.ratio < 0.2;
    return card(`예측 목표 살펴보기 — ${ds.target}`,
      h('p', { class: 'lead' },
        '분류 문제에서는 정답이 몇 개씩 있는지를 꼭 먼저 세어 봐야 한다.'),
      figure(barChart(bal.rows.map((r) => ({
        label: `${r.name} (${r.value})`, value: r.n,
        color: r.value === 1 ? '#dc2626' : '#2563eb',
      })), { xLabel: ds.target, yLabel: '개수' })),
      table(['값', '뜻', '개수', '비율'], bal.rows.map((r) => ({
        cells: [String(r.value), r.name, num(r.n), `${(r.ratio * 100).toFixed(2)}%`],
        hi: r === minority && imbalanced,
      })), { rowClass: (r) => (r.hi ? 'hi' : '') }),
      imbalanced
        ? h('div', { class: 'warn' },
          h('b', {}, '불균형 데이터다 — 정확도에 속지 말자. '),
          `「${minority.name}」이 전체의 ${(minority.ratio * 100).toFixed(2)}% (${minority.n}개) 뿐이다. `,
          `아무것도 배우지 않고 전부 「${bal.rows[0].name}」이라고만 찍어도 `,
          h('b', {}, `정확도 ${(bal.majorityRatio * 100).toFixed(2)}%`),
          ' 가 나온다. ',
          h('br', {}),
          '이런 데이터에서는 정확도 대신 ',
          h('b', {}, '재현율(놓치지 않았나) · F1 · ROC AUC'),
          ' 를 봐야 한다. 「모델 학습실」에서 직접 확인해 보자.')
        : h('div', { class: 'ok' },
          `가장 많은 값이 ${(bal.majorityRatio * 100).toFixed(1)}% 로 크게 치우치지 않았다. `
          + '정확도를 지표로 써도 크게 어긋나지 않는다.'),
      foldout('파이썬으로는 이렇게 씁니다', codeBlock(
        `${ds.df}['${ds.target}'].value_counts()\n`
        + `${ds.df}['${ds.target}'].value_counts(normalize=True)   # 비율로 보기`).outerHTML));
  }

  // 회귀 — 목표값의 분포
  const vals = ds.frame.values(ds.target);
  const st = ds.frame.describe([ds.target])[0];
  const skew = st.mean - st.median;
  return card(`예측 목표 살펴보기 — ${ds.target}`,
    h('p', { class: 'lead' },
      '맞혀야 하는 값이 어떻게 퍼져 있는지 본다. 한쪽으로 몰려 있으면 모델이 그쪽으로 끌려간다.'),
    figure(histChart([{ label: '', ...histogram(vals, 30) }],
      { xLabel: `${ds.target}${ds.unit ? ` (${ds.unit})` : ''}`, yLabel: '개수' })),
    h('div', { class: 'kv' },
      h('span', {}, h('b', {}, '평균 '), num(st.mean)),
      h('span', {}, h('b', {}, '중앙값 '), num(st.median)),
      h('span', {}, h('b', {}, '최소~최대 '), `${num(st.min)} ~ ${num(st.max)}`),
      h('span', {}, h('b', {}, '표준편차 '), num(st.std))),
    Math.abs(skew) > st.std * 0.3
      ? h('div', { class: 'note' },
        `평균(${num(st.mean)})과 중앙값(${num(st.median)})이 꽤 다르다 — 분포가 `,
        h('b', {}, skew > 0 ? '오른쪽으로 긴 꼬리' : '왼쪽으로 긴 꼬리'),
        '를 가진 모양이다. 이런 데이터는 MSE 가 큰 값 몇 개에 크게 흔들리므로 '
        + 'MAE 도 함께 보는 편이 좋다.')
      : h('div', { class: 'ok' }, '평균과 중앙값이 비슷해 크게 치우치지 않은 분포다.'));
}

/* ── info() : 타입과 결측치 ── */
function infoCard(ds) {
  const info = ds.frame.info();
  const rows = info.map((r) => ({
    cells: [
      r.name,
      ds.colDesc[r.name] || '',
      r.dtype === 'float64' ? '수치형 float64' : '범주형 object',
      num(r.nonNull),
      r.missing
        ? h('span', { class: 'mi' }, `${r.missing}`)
        : h('span', { class: 'na' }, '0'),
      r.unique === null ? '—' : `${r.unique}종`,
      r.coerced ? h('span', { class: 'mi' }, `${r.coerced}`) : '—',
    ],
    hi: r.missing > 0,
    tgt: r.name === ds.target,
  }));
  const totalMissing = info.reduce((a, r) => a + r.missing, 0);
  const worst = info.filter((r) => r.missing > 0)
    .sort((a, b) => b.missing - a.missing);
  // 숫자 자리에 글자가 섞여 있어 결측치로 바꾼 컬럼
  const dirty = info.filter((r) => r.coerced > 0);

  const head = ds.frame.head(8);
  // CSV 에 적힌 값을 그대로 보여 준다 (단위를 만·억으로 줄이지 않는다)
  const raw = (v) => {
    if (v === null || (typeof v === 'number' && Number.isNaN(v))) {
      return h('span', { class: 'na' }, 'NaN');
    }
    return typeof v === 'number' ? String(Math.round(v * 1e6) / 1e6) : v;
  };
  const headTable = table(ds.frame.columns, head.map((r) => r.map(raw)),
    { className: 'tbl-s' });

  return card('① 데이터 살펴보기 — head() 와 info()',
    h('p', { class: 'lead' }, '먼저 위에서 몇 줄만 눈으로 확인하고, 그다음 컬럼마다 타입과 결측치를 센다.'),
    h('div', { class: 'ctrl-l' }, 'head(8) — 위에서 8줄'),
    headTable,
    h('hr', { class: 'sep' }),
    h('div', { class: 'ctrl-l' }, 'info() — 컬럼별 타입과 결측치'),
    table(['컬럼', { label: '설명', align: 'left' }, '타입', '결측 아닌 값', '결측치', '값 종류',
      '숫자 아니던 값'], rows, {
      rowClass: (r) => (r.tgt ? 'tgt' : (r.hi ? 'hi' : '')),
    }),
    dirty.length
      ? h('div', { class: 'warn' },
        h('b', {}, '숫자가 아닌 값이 섞여 있던 컬럼 — '),
        dirty.map((r) => `${r.name}(${r.coerced}개)`).join(', '),
        '. 숫자 자리에 ', h('code', {}, "'-'"), ' 나 ', h('code', {}, "'미확인'"),
        ' 같은 글자가 들어 있어 결측치로 바꿔 두었다.',
        h('br', {}),
        '파이썬에서는 이런 컬럼을 pandas 가 통째로 ', h('code', {}, 'object'),
        ' 타입으로 읽어 버리므로 ',
        h('code', {}, "pd.to_numeric(컬럼, errors='coerce')"),
        ' 로 직접 숫자형으로 바꿔 줘야 한다. 위 결측치 개수에는 이 값들도 들어 있다.')
      : null,
    totalMissing
      ? h('div', { class: 'note' },
        `결측치가 모두 ${totalMissing}개 있다. 가장 많은 컬럼은 `,
        h('b', {}, worst[0].name),
        `(${worst[0].missing}개). 결측치가 있으면 모델 학습이 되지 않으므로 `,
        h('b', {}, '「전처리 실험실」'),
        '에서 채우거나 지워야 한다.')
      : h('div', { class: 'ok' }, '결측치가 없는 깔끔한 데이터다.'),
    foldout('파이썬으로는 이렇게 씁니다', codeBlock(
      `${ds.df}.info()                    # 타입 · 결측 아닌 값 개수를 한눈에\n`
      + `${ds.df}.isnull().sum()           # 컬럼별 결측치 개수만 세기\n`
      + `${ds.df}['${ds.target}'].value_counts()   # 범주형 값이 몇 개씩 있나`).outerHTML));
}

/* ── describe() ── */
function describeCard(ds) {
  const st = ds.frame.describe();
  const rows = st.map((s) => ({
    cells: [s.name, num(s.count), num(s.mean), num(s.std), num(s.min),
      num(s.q1), num(s.median), num(s.q3), num(s.max)],
    tgt: s.name === ds.target,
  }));
  return card('② 기초 통계 — describe()',
    h('p', { class: 'lead' },
      '평균과 중앙값이 크게 다르면 분포가 한쪽으로 치우쳐 있다는 뜻이고, '
      + '최솟값·최댓값이 사분위수에서 너무 멀면 이상치를 의심한다.'),
    table(['컬럼', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], rows, {
      className: 'tbl-s', rowClass: (r) => (r.tgt ? 'tgt' : ''),
    }),
    foldout('파이썬으로는 이렇게 씁니다',
      codeBlock(`${ds.df}.describe()`).outerHTML));
}

/* ── groupby ── */
function groupCard(ds, root) {
  const catCols = ds.frame.categoricalColumns();
  const numCols = ds.frame.numericColumns();
  if (!catCols.length || !numCols.length) return h('div');
  if (!state.groupKey || !catCols.includes(state.groupKey)) [state.groupKey] = catCols;
  if (!state.groupValue || !numCols.includes(state.groupValue)) {
    state.groupValue = numCols.includes(ds.target) ? ds.target : numCols[0];
  }

  const g = ds.frame.groupby([state.groupKey], [state.groupValue], state.agg);
  const rows = g.rows.map((r, i) => ({
    cells: [r.label, num(r.n), num(r.cells[0])],
    top: i === 0,
  }));
  const chart = barChart(g.rows.map((r, i) => ({
    label: r.label, value: r.cells[0], highlight: i === 0,
  })), {
    xLabel: state.groupKey,
    yLabel: `${state.groupValue} 의 ${AGG_KO[state.agg]}`,
  });

  const body = h('div', {});
  const rebuild = () => { clear(body); body.append(inner()); };
  const inner = () => h('div', {},
    h('div', { class: 'ctrls' },
      selectRow('묶을 기준 (범주형)', catCols.map((c) => ({ value: c, label: c })),
        state.groupKey, (v) => { state.groupKey = v; renderExplore(root); }),
      selectRow('볼 값 (수치형)', numCols.map((c) => ({ value: c, label: c })),
        state.groupValue, (v) => { state.groupValue = v; renderExplore(root); }),
      h('div', { class: 'ctrl' }, h('span', { class: 'ctrl-l' }, '집계 방법'),
        pills(Object.entries(AGG_KO).map(([k, v]) => ({ value: k, label: v })),
          state.agg, (v) => { state.agg = v; renderExplore(root); }, { small: true }))),
    figure(chart),
    table([state.groupKey, '개수', `${state.groupValue} ${AGG_KO[state.agg]}`], rows, {
      rowClass: (r) => (r.top ? 'hi' : ''),
    }),
    h('div', { class: 'tip' },
      `${AGG_KO[state.agg]}이 가장 큰 값은 `,
      h('b', {}, g.rows[0]?.label ?? '—'),
      ` (${num(g.rows[0]?.cells[0])}). `,
      '「가장 ~한 것을 답안에 쓰세요」 라고 묻는 문제는 이렇게 확인한다.',
      g.skipped
        ? h('span', {}, h('br', {}),
          `기준 컬럼이 결측치인 ${g.skipped}행은 그룹에서 빠졌다 — pandas 의 groupby 도 똑같이 동작한다.`)
        : null),
    foldout('파이썬으로는 이렇게 씁니다', codeBlock(
      `${ds.df}.groupby('${state.groupKey}')['${state.groupValue}'].${state.agg}()\n\n`
      + `# 가장 큰 값을 가진 그룹 이름만 바로 얻고 싶다면\n`
      + `${ds.df}.groupby('${state.groupKey}')['${state.groupValue}'].${state.agg}().idxmax()`).outerHTML));
  body.append(inner());
  return card('③ 그룹별로 묶어 보기 — groupby()', body);
}

const AGG_KO = {
  mean: '평균', median: '중앙값', sum: '합계', count: '개수', max: '최대', min: '최소', std: '표준편차',
};

/* ── 시각화 실험실 ── */
const PLOTS = {
  scatter: { label: '산점도', py: 'scatterplot', need: ['num', 'num'] },
  hist: { label: '히스토그램', py: 'histplot', need: ['num'] },
  box: { label: '상자그림', py: 'boxplot', need: ['cat', 'num'] },
  bar: { label: '막대그래프', py: 'barplot', need: ['cat', 'num'] },
  count: { label: '개수 세기', py: 'countplot', need: ['cat'] },
};

function plotCard(ds, root) {
  const numCols = ds.frame.numericColumns();
  const catCols = ds.frame.categoricalColumns();
  const kind = PLOTS[state.plotKind] ? state.plotKind : 'scatter';
  state.plotKind = kind;
  const spec = PLOTS[kind];

  // 기본 축 정하기
  const wantCatX = spec.need[0] === 'cat';
  const xPool = wantCatX ? catCols : numCols;
  if (!xPool.length) return h('div');
  if (!state.xCol || !xPool.includes(state.xCol)) {
    state.xCol = wantCatX ? catCols[0] : (numCols.find((c) => c !== ds.target) || numCols[0]);
  }
  if (!state.yCol || !numCols.includes(state.yCol)) {
    state.yCol = numCols.includes(ds.target) ? ds.target : numCols[numCols.length - 1];
  }

  const hueOptions = [{ value: '', label: '(구분 없음)' },
    ...catCols.map((c) => ({ value: c, label: c })),
    ...(ds.task === 'classification' ? [{ value: ds.target, label: `${ds.target} (정답)` }] : [])];
  if (state.hueCol && !hueOptions.some((o) => o.value === state.hueCol)) state.hueCol = '';

  let svg = emptyChart('그래프를 고르세요');
  let caption = '';
  let py = '';

  if (kind === 'scatter') {
    const groups = splitByHue(ds, state.hueCol).map((g) => ({
      label: g.label,
      points: g.idx.map((i) => ({ x: ds.frame.col(state.xCol)[i], y: ds.frame.col(state.yCol)[i] }))
        .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y)),
    })).filter((g) => g.points.length);
    const r = pearson(ds.frame.col(state.xCol), ds.frame.col(state.yCol));
    svg = scatterChart(groups, {
      xLabel: state.xCol, yLabel: state.yCol, line: fitLine(ds, state.xCol, state.yCol),
    });
    caption = `상관계수 r = ${r.toFixed(3)} — ${corrWord(r)}`;
    py = `sns.scatterplot(data=${ds.df}, x='${state.xCol}', y='${state.yCol}'`
      + `${state.hueCol ? `, hue='${state.hueCol}'` : ''})\n`
      + `sns.regplot(data=${ds.df}, x='${state.xCol}', y='${state.yCol}')   # 회귀선까지`;
  } else if (kind === 'hist') {
    const series = splitByHue(ds, state.hueCol).map((g) => {
      const vals = g.idx.map((i) => ds.frame.col(state.yCol)[i]);
      return { label: g.label, ...histogram(vals, state.bins) };
    }).filter((s) => s.counts.length);
    svg = histChart(series, { xLabel: state.yCol, yLabel: '개수' });
    caption = `${state.yCol} 의 분포를 ${state.bins}칸으로 나눠 세었다.`;
    py = `sns.histplot(data=${ds.df}, x='${state.yCol}', bins=${state.bins}, kde=True`
      + `${state.hueCol ? `, hue='${state.hueCol}'` : ''})`;
  } else if (kind === 'box') {
    const groups = groupIndex(ds, state.xCol).map((g) => ({
      label: g.label, stats: boxStats(g.idx.map((i) => ds.frame.col(state.yCol)[i])),
    }));
    svg = boxChart(groups, { xLabel: state.xCol, yLabel: state.yCol });
    const withOut = groups.filter((g) => g.stats).sort((a, b) =>
      b.stats.outliers.length - a.stats.outliers.length)[0];
    caption = withOut
      ? `상자 안의 굵은 선이 중앙값, 상자 밖 점이 이상치다. 이상치가 가장 많은 그룹은 ${withOut.label} (${withOut.stats.outliers.length}개).`
      : '';
    py = `sns.boxplot(data=${ds.df}, x='${state.xCol}', y='${state.yCol}')`;
  } else if (kind === 'bar') {
    const g = ds.frame.groupby([state.xCol], [state.yCol], 'mean');
    svg = barChart(g.rows.map((r, i) => ({ label: r.label, value: r.cells[0], highlight: i === 0 })),
      { xLabel: state.xCol, yLabel: `${state.yCol} 평균` });
    caption = 'barplot 은 그룹별 「평균」을 막대로 보여 준다.';
    py = `sns.barplot(data=${ds.df}, x='${state.xCol}', y='${state.yCol}')`;
  } else {
    const g = groupIndex(ds, state.xCol);
    svg = barChart(g.map((x) => ({ label: x.label, value: x.idx.length })),
      { xLabel: state.xCol, yLabel: '개수' });
    caption = 'countplot 은 값이 몇 번 나오는지만 센다.';
    py = `sns.countplot(data=${ds.df}, x='${state.xCol}'`
      + `${state.hueCol ? `, hue='${state.hueCol}'` : ''})`;
  }

  const needsX = spec.need[0] === 'cat' || kind === 'scatter';
  const controls = h('div', { class: 'ctrls' },
    h('div', { class: 'ctrl' }, h('span', { class: 'ctrl-l' }, '그래프 종류'),
      pills(Object.entries(PLOTS).map(([k, v]) => ({ value: k, label: v.label, title: `sns.${v.py}()` })),
        kind, (v) => { state.plotKind = v; state.xCol = null; renderExplore(root); }, { small: true })),
    needsX
      ? selectRow(spec.need[0] === 'cat' ? 'x축 (범주형)' : 'x축 (수치형)',
        xPool.map((c) => ({ value: c, label: c })), state.xCol,
        (v) => { state.xCol = v; renderExplore(root); })
      : null,
    kind !== 'count'
      ? selectRow(kind === 'hist' ? '볼 컬럼 (수치형)' : 'y축 (수치형)',
        numCols.map((c) => ({ value: c, label: c })), state.yCol,
        (v) => { state.yCol = v; renderExplore(root); })
      : null,
    ['scatter', 'hist', 'count'].includes(kind)
      ? selectRow('색 구분 (hue)', hueOptions, state.hueCol,
        (v) => { state.hueCol = v; renderExplore(root); })
      : null,
    kind === 'hist'
      ? selectRow('구간 수 (bins)', [10, 15, 20, 30, 40, 60].map((b) => ({ value: b, label: `${b}칸` })),
        state.bins, (v) => { state.bins = Number(v); renderExplore(root); })
      : null);

  return card('④ 시각화 실험실 — seaborn 그림 그려 보기',
    controls, figure(svg, caption),
    foldout('파이썬으로는 이렇게 씁니다', codeBlock(
      `import seaborn as sns\nimport matplotlib.pyplot as plt\n\n${py}\nplt.show()`).outerHTML));
}

/** hue 에 따라 행 번호를 나눈다 */
function splitByHue(ds, hue) {
  if (!hue) return [{ label: '', idx: Array.from({ length: ds.frame.length }, (_, i) => i) }];
  return groupIndex(ds, hue);
}

/* 값별로 행 번호를 묶는다.
 * 결측치인 행은 뺀다 — seaborn 도 그림을 그릴 때 결측치를 그냥 건너뛴다. */
function groupIndex(ds, col) {
  const arr = ds.frame.col(col);
  const map = new Map();
  for (let i = 0; i < arr.length; i += 1) {
    let key = arr[i];
    if (key === null || (typeof key === 'number' && Number.isNaN(key))) continue;
    if (typeof key === 'number' && ds.classNames) key = ds.classNames[key] ?? key;
    const k = String(key);
    if (!map.has(k)) map.set(k, []);
    map.get(k).push(i);
  }
  return [...map].sort((a, b) => a[0].localeCompare(b[0], 'ko'))
    .map(([label, idx]) => ({ label, idx }));
}

/** 최소제곱 직선 — regplot 의 회귀선 */
function fitLine(ds, xc, yc) {
  const xs = ds.frame.col(xc);
  const ys = ds.frame.col(yc);
  let n = 0;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i += 1) {
    const x = xs[i];
    const y = ys[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    n += 1; sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  if (n < 2) return null;
  const denom = n * sxx - sx * sx;
  if (!denom) return null;
  const slope = (n * sxy - sx * sy) / denom;
  return { slope, intercept: (sy - slope * sx) / n };
}

function corrWord(r) {
  const a = Math.abs(r);
  const dir = r > 0 ? '양(같이 커짐)' : '음(반대로 움직임)';
  if (Number.isNaN(r)) return '계산할 수 없다';
  if (a >= 0.8) return `아주 강한 ${dir}의 상관. 다중공선성을 의심해 볼 수 있다`;
  if (a >= 0.5) return `뚜렷한 ${dir}의 상관`;
  if (a >= 0.3) return `약한 ${dir}의 상관`;
  return '거의 상관이 없다';
}

/* ── 상관계수 히트맵 ── */
function corrCard(ds) {
  const numCols = ds.frame.numericColumns();
  if (numCols.length < 2) return h('div');
  const c = ds.frame.corr(numCols);
  const ti = numCols.indexOf(ds.target);
  let ranking = null;
  if (ti >= 0) {
    ranking = numCols.map((name, j) => ({ name, r: c.m[ti][j] }))
      .filter((o) => o.name !== ds.target && Number.isFinite(o.r))
      .sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
  }
  // |r| >= 0.8 인 특성 짝 = 다중공선성 의심
  const multi = [];
  for (let i = 0; i < numCols.length; i += 1) {
    for (let j = i + 1; j < numCols.length; j += 1) {
      if (numCols[i] === ds.target || numCols[j] === ds.target) continue;
      if (Math.abs(c.m[i][j]) >= 0.8) multi.push(`${numCols[i]} ↔ ${numCols[j]} (${c.m[i][j].toFixed(2)})`);
    }
  }

  return card('⑤ 상관계수 히트맵 — corr()',
    h('p', { class: 'lead' },
      '두 변수가 함께 커지면 +1 쪽(빨강), 반대로 움직이면 −1 쪽(파랑)이다. '
      + '0 에 가까우면 선형 관계가 거의 없다.'),
    figure(heatmapChart(c)),
    ranking
      ? h('div', {},
        h('div', { class: 'ctrl-l' }, `예측 목표(${ds.target})와의 상관 순위`),
        figure(hBarChart(ranking.slice(0, 10).map((o) => ({
          label: `${o.name} (${o.r.toFixed(2)})`,
          value: Math.abs(o.r),
          color: o.r >= 0 ? '#dc2626' : '#2563eb',
        }))), '막대 길이는 |r|, 빨강은 양의 상관 · 파랑은 음의 상관'))
      : null,
    multi.length
      ? h('div', { class: 'warn' },
        h('b', {}, '다중공선성 주의 — '),
        `특성끼리 |r| ≥ 0.8 인 짝이 있다: ${multi.join(', ')}. `,
        '거의 같은 정보를 두 번 넣는 셈이라 선형 회귀의 계수 해석이 불안해진다.')
      : h('div', { class: 'ok' }, '특성끼리 |r| ≥ 0.8 인 짝은 없다. 다중공선성 걱정은 적다.'),
    foldout('파이썬으로는 이렇게 씁니다', codeBlock(
      `corr_cols = ${JSON.stringify(numCols.slice(0, 6)).replace(/"/g, "'")}\n`
      + `plt.figure(figsize=(10, 8))\n`
      + `sns.heatmap(${ds.df}[corr_cols].corr(), annot=True, cmap='viridis')`).outerHTML));
}
