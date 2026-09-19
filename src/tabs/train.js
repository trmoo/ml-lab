/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 탭 ③ 모델 학습실
 *
 * 머신러닝 파이프라인의 학습·평가·예측 단계에 해당한다.
 *   ① 전처리 설정을 고른다 (결측치·인코딩·스케일러·분리 비율)
 *   ② 모델과 하이퍼파라미터를 고르고 「학습 시작」을 누른다 — 진짜로 학습한다
 *   ③ 훈련 점수와 검증 점수를 나란히 보며 과적합을 확인한다
 *   ④ 특성 중요도 · 실제값 vs 예측값 · 혼동 행렬 · ROC 곡선을 본다
 *   ⑤ 슬라이더로 새 데이터를 만들어 예측해 본다 
 */
import { getDataset, datasetOptions, DATASET_IDS } from '../data/datasets.js';
import {
  h, card, table, figure, selectRow, pills, num, codeBlock, foldout, badge, clear, checkRow,
  sliderRow, progress, redraw,
} from '../lib/ui.js';
import { scatterChart, hBarChart, lineChart } from '../lib/chart.js';
import { buildMatrix, IMPUTE_PRESETS, classBalance } from '../lib/pipeline.js';
import { trainTestSplit, SCALERS } from '../lib/preprocess.js';
import { tick } from '../lib/tick.js';
import {
  LinearRegression, ElasticNetRegression, LogisticRegression, KNeighbors,
  DecisionTree, RandomForest, GradientBoosting,
} from '../lib/models.js';
import {
  regressionScores, classificationScores, rocCurve, rocAucScore,
} from '../lib/metrics.js';

/* ── 모델 목록 ─────────────────────────────────────────
 * 흔히 쓰는 모델은 20가지가 넘지만 계산 방식은 몇 갈래로 묶인다.
 * 갈래를 대표하는 모델만 골라 넣고, 같은 갈래의 다른 모델 이름을 함께 적어 둔다.
 */
const REG_MODELS = {
  linear: {
    label: '선형 회귀', py: 'LinearRegression', family: '선형',
    also: '가장 기본. 직선(평면) 하나로 예측한다',
    params: [], make: () => new LinearRegression({ alpha: 0 }),
  },
  ridge: {
    label: '릿지 회귀', py: 'Ridge', family: '선형 + L2 규제',
    also: '계수를 전체적으로 작게 눌러 과적합을 줄인다',
    params: ['alpha'], make: (p) => new LinearRegression({ alpha: p.alpha }),
  },
  lasso: {
    label: '라쏘 회귀', py: 'Lasso', family: '선형 + L1 규제',
    also: '쓸모없는 변수의 계수를 아예 0 으로 만든다',
    params: ['alpha'], make: (p) => new ElasticNetRegression({ alpha: p.alpha, l1Ratio: 1 }),
  },
  elastic: {
    label: '엘라스틱넷', py: 'ElasticNet', family: '선형 + L1·L2 규제',
    also: '라쏘와 릿지를 섞은 것. l1_ratio 로 비율을 정한다',
    params: ['alpha', 'l1Ratio'],
    make: (p) => new ElasticNetRegression({ alpha: p.alpha, l1Ratio: p.l1Ratio }),
  },
  knn: {
    label: 'KNN 회귀', py: 'KNeighborsRegressor', family: '거리 기반',
    also: '가장 비슷한 k개의 평균으로 답한다. 스케일링이 꼭 필요하다',
    params: ['k'], make: (p) => new KNeighbors({ k: p.k, task: 'regression' }),
  },
  tree: {
    label: '결정 트리 회귀', py: 'DecisionTreeRegressor', family: '트리',
    also: '예·아니오 질문을 이어 붙여 나눈다. 깊이를 키우면 금방 과적합된다',
    params: ['maxDepth', 'minSamplesLeaf'],
    make: (p) => new DecisionTree({
      task: 'regression', maxDepth: p.maxDepth, minSamplesLeaf: p.minSamplesLeaf,
    }),
  },
  forest: {
    label: '랜덤 포레스트 회귀', py: 'RandomForestRegressor', family: '트리 앙상블(배깅)',
    also: '트리 여러 개를 따로 키워 평균을 낸다',
    params: ['nEstimators', 'maxDepth'], async: true,
    make: (p) => new RandomForest({
      task: 'regression', nEstimators: p.nEstimators, maxDepth: p.maxDepth, seed: 100,
    }),
  },
  boost: {
    label: '그래디언트 부스팅 회귀', py: 'GradientBoostingRegressor', family: '트리 앙상블(부스팅)',
    also: 'XGBoost · LightGBM 도 같은 갈래다',
    params: ['nEstimators', 'learningRate', 'maxDepth'], async: true,
    make: (p) => new GradientBoosting({
      task: 'regression', nEstimators: p.nEstimators,
      learningRate: p.learningRate, maxDepth: p.maxDepth, seed: 100,
    }),
  },
};

const CLS_MODELS = {
  logistic: {
    label: '로지스틱 회귀', py: 'LogisticRegression', family: '선형 분류',
    also: '선형식을 시그모이드에 넣어 0~1 확률로 만든다',
    params: ['C'], make: (p) => new LogisticRegression({ C: p.C, maxIter: 400 }),
  },
  knn: {
    label: 'KNN 분류', py: 'KNeighborsClassifier', family: '거리 기반',
    also: '가까운 k개의 다수결로 정한다',
    params: ['k'], make: (p) => new KNeighbors({ k: p.k, task: 'classification' }),
  },
  tree: {
    label: '결정 트리 분류', py: 'DecisionTreeClassifier', family: '트리',
    also: '지니 불순도가 가장 많이 줄어드는 곳에서 자른다',
    params: ['maxDepth', 'minSamplesLeaf'],
    make: (p) => new DecisionTree({
      task: 'classification', maxDepth: p.maxDepth, minSamplesLeaf: p.minSamplesLeaf,
    }),
  },
  forest: {
    label: '랜덤 포레스트 분류', py: 'RandomForestClassifier', family: '트리 앙상블(배깅)',
    also: '트리 여러 개의 다수결로 정한다',
    params: ['nEstimators', 'maxDepth'], async: true,
    make: (p) => new RandomForest({
      task: 'classification', nEstimators: p.nEstimators, maxDepth: p.maxDepth, seed: 100,
    }),
  },
  boost: {
    label: '그래디언트 부스팅 분류', py: 'GradientBoostingClassifier', family: '트리 앙상블(부스팅)',
    also: 'AdaBoost · XGBoost 도 같은 갈래다',
    params: ['nEstimators', 'learningRate', 'maxDepth'], async: true,
    make: (p) => new GradientBoosting({
      task: 'classification', nEstimators: p.nEstimators,
      learningRate: p.learningRate, maxDepth: p.maxDepth, seed: 100,
    }),
  },
};

const PARAM_SPEC = {
  alpha: { label: 'alpha (규제 세기)', min: 0.01, max: 10, step: 0.01, def: 1, dec: 2, py: 'alpha' },
  l1Ratio: { label: 'l1_ratio', min: 0, max: 1, step: 0.05, def: 0.5, dec: 2, py: 'l1_ratio' },
  k: { label: 'n_neighbors (k)', min: 1, max: 40, step: 1, def: 5, py: 'n_neighbors' },
  maxDepth: { label: 'max_depth (깊이)', min: 1, max: 20, step: 1, def: 6, py: 'max_depth' },
  minSamplesLeaf: {
    label: 'min_samples_leaf', min: 1, max: 50, step: 1, def: 1, py: 'min_samples_leaf',
  },
  nEstimators: {
    label: 'n_estimators (트리 수)', min: 10, max: 300, step: 10, def: 100, py: 'n_estimators',
  },
  learningRate: {
    label: 'learning_rate', min: 0.01, max: 0.5, step: 0.01, def: 0.1, dec: 2, py: 'learning_rate',
  },
  C: { label: 'C (규제의 역수)', min: 0.01, max: 10, step: 0.01, def: 1, dec: 2, py: 'C' },
};

const state = {
  dsId: DATASET_IDS[0],   // 첫 회차. 수업 순서가 바뀌어도 따라간다
  model: 'forest',
  params: {},
  impute: 'medianMode',
  encode: 'onehot',
  scaler: 'StandardScaler',
  testSize: 0.3,
  seed: 100,
  useDerived: true,
  dropColumns: [],
  result: null,
  busy: false,
  predRow: null,
};

Object.entries(PARAM_SPEC).forEach(([k, s]) => { state.params[k] = s.def; });

/** 다른 탭(문제 풀기)에서 이 탭을 열 때 데이터셋을 미리 골라 둔다 */
export function useDataset(dsId) {
  if (!dsId || dsId === state.dsId) return;
  state.dsId = dsId;
  // 데이터 고르기 단추를 눌렀을 때와 똑같이 이전 데이터에 묶인 선택을 비운다
  state.result = null; state.dropColumns = []; state.predRow = null;
}

export function renderTrain(root) {
  // redraw 로 감싸면 옵션을 눌러 다시 그려도 보던 자리에 그대로 머무른다
  redraw(root, () => {
    const ds = getDataset(state.dsId);
    const models = ds.task === 'regression' ? REG_MODELS : CLS_MODELS;
    if (!models[state.model]) state.model = ds.task === 'regression' ? 'forest' : 'logistic';

    root.append(
      picker(root),
      prepCard(ds, root),
      modelCard(ds, models, root),
      resultArea(ds, root),
    );
  });
}

function picker(root) {
  const grid = h('div', { class: 'dspick' }, datasetOptions().map((o) => h('button', {
    type: 'button',
    class: `dsbtn${o.id === state.dsId ? ' on' : ''}`,
    onclick: () => {
      state.dsId = o.id; state.result = null; state.dropColumns = []; state.predRow = null;
      renderTrain(root);
    },
  }, h('b', {}, `${o.round}회차 · ${o.short}`), h('span', {}, o.taskLabel))));
  return card('실습 데이터 고르기', grid);
}

/* ── ① 전처리 설정 ── */
function prepCard(ds, root) {
  const cols = ds.frame.columns.filter((c) => c !== ds.target
    && !(ds.idColumns || []).includes(c));
  const prep = buildMatrix(ds, {
    imputePreset: state.impute, encode: state.encode,
    dropColumns: state.dropColumns, useDerived: state.useDerived,
  });

  return card('① 전처리 설정',
    h('div', { class: 'ctrls' },
      selectRow('결측치 처리',
        Object.entries(IMPUTE_PRESETS).map(([k, v]) => ({ value: k, label: v.label })),
        state.impute, (v) => { state.impute = v; state.result = null; renderTrain(root); }),
      h('div', { class: 'ctrl' }, h('span', { class: 'ctrl-l' }, '범주형 인코딩'),
        pills([{ value: 'onehot', label: '원-핫' }, { value: 'label', label: '레이블' }],
          state.encode, (v) => { state.encode = v; state.result = null; renderTrain(root); },
          { small: true })),
      h('div', { class: 'ctrl' }, h('span', { class: 'ctrl-l' }, '스케일러'),
        pills(Object.keys(SCALERS).map((k) => ({ value: k, label: k === 'none' ? '안 함' : k })),
          state.scaler, (v) => { state.scaler = v; state.result = null; renderTrain(root); },
          { small: true })),
      sliderRow('검증 비율 test_size', {
        min: 0.1, max: 0.5, step: 0.05, value: state.testSize,
        format: (v) => `${(v * 100).toFixed(0)}%`,
        onInput: (v) => { state.testSize = v; state.result = null; },
      }),
      sliderRow('random_state', {
        min: 0, max: 300, step: 1, value: state.seed,
        onInput: (v) => { state.seed = v; state.result = null; },
      }),
      ds.derived
        ? checkRow(`파생변수 ${ds.derived.map((d) => d.name).join(', ')} 만들기`, state.useDerived,
          (v) => { state.useDerived = v; state.result = null; renderTrain(root); })
        : null),
    h('div', { class: 'ctrl' },
      h('span', { class: 'ctrl-l' }, '분석에서 뺄 컬럼 (눌러서 켜고 끄기)'),
      h('div', { class: 'pills pills-s' }, cols.map((c) => h('button', {
        type: 'button',
        class: `pill${state.dropColumns.includes(c) ? ' on' : ''}`,
        onclick: () => {
          state.dropColumns = state.dropColumns.includes(c)
            ? state.dropColumns.filter((x) => x !== c)
            : [...state.dropColumns, c];
          state.result = null;
          renderTrain(root);
        },
      }, c)))),
    h('div', { class: 'tip' },
      '전처리 결과 — 사용 행 ', h('b', {}, num(prep.summary.rows)),
      '개 · 특성(열) ', h('b', {}, num(prep.summary.features)), '개',
      prep.summary.droppedRows ? ` (결측 때문에 ${prep.summary.droppedRows}행 제외)` : ''),
    foldout('전처리가 한 일 자세히 보기',
      `<ul>${prep.log.map((l) => `<li>${l}</li>`).join('')}</ul>`
      + `<p>만들어진 특성 이름 ${prep.featureNames.length}개</p>`
      + `<pre class="code">${prep.featureNames.join(', ')}</pre>`));
}

/* ── ② 모델 고르기 ── */
function modelCard(ds, models, root) {
  const spec = models[state.model];
  const paramCtrls = spec.params.map((p) => {
    const s = PARAM_SPEC[p];
    return sliderRow(s.label, {
      min: s.min, max: s.max, step: s.step, value: state.params[p],
      format: (v) => (s.dec ? v.toFixed(s.dec) : String(v)),
      onInput: (v) => { state.params[p] = v; },
    });
  });

  const pg = progress('학습 준비됨');
  const btn = h('button', {
    class: 'btn', disabled: state.busy,
    onclick: async () => {
      state.busy = true;
      btn.disabled = true;
      btn.textContent = '학습 중…';
      try {
        state.result = await runTraining(ds, models, (r, t) => pg.set(r, t));
      } catch (err) {
        state.result = { error: String(err?.message || err) };
        console.error(err);
      }
      state.busy = false;
      renderTrain(root);
    },
  }, '▶ 학습 시작');

  return card('② 모델과 하이퍼파라미터',
    h('div', { class: 'ctrl' }, h('span', { class: 'ctrl-l' }, '모델'),
      pills(Object.entries(models).map(([k, v]) => ({
        value: k, label: v.label, title: `사이킷런의 ${v.py}`,
      })), state.model, (v) => { state.model = v; state.result = null; renderTrain(root); })),
    h('div', { class: 'kv' },
      h('span', {}, h('b', {}, '사이킷런 이름 '), h('code', {}, spec.py)),
      h('span', {}, h('b', {}, '갈래 '), badge(spec.family, 'v')),
      h('span', {}, spec.also)),
    paramCtrls.length
      ? h('div', { class: 'ctrls' }, paramCtrls)
      : h('div', { class: 'tip' }, '이 모델에는 조절할 하이퍼파라미터가 없다.'),
    h('div', { class: 'btnrow' }, btn, pg.node),
    foldout('파이썬으로는 이렇게 씁니다', codeBlock(pyModel(ds, spec)).outerHTML));
}

function pyModel(ds, spec) {
  const args = spec.params.map((p) => {
    const s = PARAM_SPEC[p];
    const v = s.dec ? Number(state.params[p]).toFixed(s.dec) : state.params[p];
    return `${s.py}=${v}`;
  });
  if (!['linear', 'knn'].includes(state.model)) args.push('random_state=100');
  const varName = `model_${state.model}`;
  const scale = state.scaler === 'none' ? ''
    : `from sklearn.preprocessing import ${state.scaler}\n\n`
      + `scaler = ${state.scaler}()\n`
      + `X_train_scaled = scaler.fit_transform(X_train)\n`
      + `X_valid_scaled = scaler.transform(X_valid)\n\n`;
  const Xtr = state.scaler === 'none' ? 'X_train' : 'X_train_scaled';
  const Xva = state.scaler === 'none' ? 'X_valid' : 'X_valid_scaled';
  const metrics = ds.task === 'regression'
    ? 'from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n\n'
      + `pred = ${varName}.predict(${Xva})\n`
      + "print('MSE :', mean_squared_error(y_valid, pred))\n"
      + "print('MAE :', mean_absolute_error(y_valid, pred))\n"
      + "print('R2  :', r2_score(y_valid, pred))"
    : 'from sklearn.metrics import (accuracy_score, precision_score, recall_score,\n'
      + '                             f1_score, roc_auc_score, confusion_matrix)\n\n'
      + `pred = ${varName}.predict(${Xva})\n`
      + `proba = ${varName}.predict_proba(${Xva})[:, 1]\n`
      + "print('정확도 :', accuracy_score(y_valid, pred))\n"
      + "print('정밀도 :', precision_score(y_valid, pred))\n"
      + "print('재현율 :', recall_score(y_valid, pred))\n"
      + "print('F1     :', f1_score(y_valid, pred))\n"
      + "print('ROC AUC:', roc_auc_score(y_valid, proba))\n"
      + 'print(confusion_matrix(y_valid, pred))';
  return `${scale}${importOf(spec.py)}\n\n`
    + `${varName} = ${spec.py}(${args.join(', ')})\n`
    + `${varName}.fit(${Xtr}, y_train)\n\n${metrics}`;
}

function importOf(py) {
  const m = {
    LinearRegression: 'sklearn.linear_model', Ridge: 'sklearn.linear_model',
    Lasso: 'sklearn.linear_model', ElasticNet: 'sklearn.linear_model',
    LogisticRegression: 'sklearn.linear_model',
    KNeighborsRegressor: 'sklearn.neighbors', KNeighborsClassifier: 'sklearn.neighbors',
    DecisionTreeRegressor: 'sklearn.tree', DecisionTreeClassifier: 'sklearn.tree',
    RandomForestRegressor: 'sklearn.ensemble', RandomForestClassifier: 'sklearn.ensemble',
    GradientBoostingRegressor: 'sklearn.ensemble', GradientBoostingClassifier: 'sklearn.ensemble',
  }[py];
  return `from ${m} import ${py}`;
}

/* ── 실제 학습 ── */
async function runTraining(ds, models, onProgress) {
  const spec = models[state.model];
  onProgress(0.05, '전처리 중…');
  await tick();

  const prep = buildMatrix(ds, {
    imputePreset: state.impute, encode: state.encode,
    dropColumns: state.dropColumns, useDerived: state.useDerived,
  });
  if (prep.X.length < 20) {
    throw new Error('쓸 수 있는 행이 너무 적습니다. 결측치 처리 방법을 바꿔 보세요.');
  }
  if (!prep.featureNames.length) {
    throw new Error('특성이 하나도 없습니다. 뺀 컬럼을 다시 켜 주세요.');
  }

  const isCls = ds.task === 'classification';
  const { train, valid } = trainTestSplit(prep.X.length, {
    testSize: state.testSize, seed: state.seed,
    stratify: isCls ? prep.y : null,
  });
  let Xtr = train.map((i) => prep.X[i]);
  let Xva = valid.map((i) => prep.X[i]);
  const ytr = train.map((i) => prep.y[i]);
  const yva = valid.map((i) => prep.y[i]);

  const scaler = SCALERS[state.scaler].make();
  if (scaler) {
    scaler.fit(Xtr);                    // 규칙은 훈련 데이터에서만 배운다
    Xtr = scaler.transform(Xtr);
    Xva = scaler.transform(Xva);
  }

  onProgress(0.15, '학습 중…');
  const model = spec.make(state.params);
  const t0 = performance.now();
  if (spec.async) {
    await model.fit(Xtr, ytr, {
      onProgress: (r) => onProgress(0.15 + r * 0.75, `학습 중… ${Math.round(r * 100)}%`),
    });
  } else {
    await tick();
    model.fit(Xtr, ytr);
  }
  onProgress(0.95, '평가 중…');
  await tick();

  const predTr = model.predict(Xtr);
  const predVa = model.predict(Xva);
  const probTr = isCls && model.predictProba ? model.predictProba(Xtr) : null;
  const probVa = isCls && model.predictProba ? model.predictProba(Xva) : null;

  const scoreTr = isCls ? classificationScores(ytr, predTr, probTr) : regressionScores(ytr, predTr);
  const scoreVa = isCls ? classificationScores(yva, predVa, probVa) : regressionScores(yva, predVa);
  const seconds = (performance.now() - t0) / 1000;

  onProgress(1, `학습 완료 (${seconds.toFixed(1)}초)`);
  return {
    spec, prep, model, scaler, isCls,
    ytr, yva, predTr, predVa, probVa,
    scoreTr, scoreVa,
    importances: model.importances ? model.importances() : null,
    nTrain: Xtr.length, nValid: Xva.length, seconds,
    treeStats: model.stats ? model.stats() : null,
  };
}

/* ── 결과 화면 ── */
function resultArea(ds, root) {
  const r = state.result;
  if (!r) {
    return card('③ 학습 결과',
      h('div', { class: 'tip' }, '위에서 모델을 고르고 「학습 시작」을 누르면 여기에 결과가 나온다.'));
  }
  if (r.error) return card('③ 학습 결과', h('div', { class: 'warn' }, r.error));
  const wrap = h('div', {});
  wrap.append(scoreCard(ds, r), detailCard(ds, r), predictCard(ds, r, root));
  return wrap;
}

function scoreCard(ds, r) {
  const keys = r.isCls
    ? ['정확도', '정밀도', '재현율', 'F1', 'ROC AUC']
    : ['MSE', 'RMSE', 'MAE', 'R²'];
  const metric = (k) => {
    const tr = r.scoreTr[k];
    const va = r.scoreVa[k];
    const higherBetter = r.isCls || k === 'R²';
    // 훈련 점수와 검증 점수의 차이 = 과적합 정도
    const gapBad = higherBetter ? tr - va > 0.12 : va > tr * 1.6;
    return h('div', { class: `metric${gapBad ? ' bad' : ''}` },
      h('div', { class: 'metric-l' }, k),
      h('div', { class: 'metric-v' }, num(va, 4)),
      h('div', { class: 'metric-sub' }, `훈련 ${num(tr, 4)}`));
  };

  const mainKey = r.isCls ? '정확도' : 'R²';
  const gap = r.scoreTr[mainKey] - r.scoreVa[mainKey];
  let verdict;
  if (gap > 0.15) {
    verdict = h('div', { class: 'warn' },
      h('b', {}, '과적합 신호 — '),
      `훈련 ${mainKey} 가 ${num(r.scoreTr[mainKey], 3)} 인데 검증 ${mainKey} 는 ${num(r.scoreVa[mainKey], 3)} 다. `,
      '외운 문제는 잘 풀지만 새 문제는 못 푸는 상태다. max_depth 를 줄이거나 '
      + 'min_samples_leaf 를 늘려 모델을 단순하게 만들어 보자.',
      r.scoreVa[mainKey] < (r.isCls ? 0.6 : 0.2)
        ? h('span', {}, h('br', {}),
          h('b', {}, '다만 검증 점수 자체가 낮다는 점도 함께 보자. '),
          '단순하게 만들어도 크게 오르지 않는다면 과적합이 아니라 '
          + '데이터에 예측할 단서가 거의 없는 것이다. '
          + '「데이터 탐험실」의 상관 히트맵에서 예측 목표와 이어진 변수가 정말 있는지 확인해 보자.')
        : null);
  } else if (r.scoreVa[mainKey] < (r.isCls ? 0.6 : 0.2)) {
    /* 훈련 점수까지 낮으면 두 가지를 구분해야 한다.
     *   · 모델이 너무 단순해서 못 배운 것(과소적합) → 복잡하게 만들면 오른다
     *   · 데이터에 애초에 신호가 거의 없는 것        → 무엇을 해도 안 오른다
     * 앱이 함부로 하나로 단정하지 않고, 확인하는 방법을 알려 준다. */
    const trainAlsoLow = r.scoreTr[mainKey] < (r.isCls ? 0.7 : 0.35);
    verdict = h('div', { class: 'note' },
      h('b', {}, '점수가 낮다 — 이유는 두 가지일 수 있다. '),
      trainAlsoLow
        ? h('span', {},
          `훈련 ${mainKey} 도 ${num(r.scoreTr[mainKey], 3)} 로 낮다. `,
          h('b', {}, '① 모델이 너무 단순한 경우(과소적합)'),
          '거나, ',
          h('b', {}, '② 데이터에 애초에 예측할 만한 단서가 거의 없는 경우'),
          '다. ',
          h('br', {}),
          '구분하는 방법 — max_depth 를 크게 올리고 다시 학습해 보자. '
          + '훈련 점수가 쑥 오르면 ①이었던 것이고, 훈련 점수도 그대로면 ②다. '
          + '②라면 「데이터 탐험실」의 상관 히트맵에서 예측 목표와 이어진 변수가 '
          + '정말 있는지 확인해 보자. 세상의 모든 데이터가 예측 가능한 것은 아니다.')
        : h('span', {},
          `훈련 ${mainKey} 는 ${num(r.scoreTr[mainKey], 3)} 인데 검증이 낮다. `
          + '특성을 더 만들거나 다른 모델을 써 보자.'));
  } else {
    verdict = h('div', { class: 'ok' },
      h('b', {}, '괜찮은 상태 — '),
      `훈련과 검증의 ${mainKey} 차이가 ${num(Math.abs(gap), 3)} 로 크지 않다.`);
  }

  /* 불균형 데이터에서 정확도만 보면 크게 잘못 판단한다.
   * 「전부 다수 쪽으로 찍기」 점수를 함께 보여 줘 비교하게 한다. */
  let imbalanceNote = null;
  if (r.isCls) {
    const bal = classBalance(ds);
    const minority = bal.rows[bal.rows.length - 1];
    if (minority && minority.ratio < 0.2) {
      const baseline = bal.majorityRatio;
      const beatsBaseline = r.scoreVa.정확도 > baseline + 0.005;
      imbalanceNote = h('div', { class: 'warn' },
        h('b', {}, '이 데이터는 불균형이다 — 정확도를 그대로 믿으면 안 된다. '),
        `전부 「${bal.rows[0].name}」이라고만 찍어도 정확도 ${num(baseline, 4)} 가 나온다. `,
        `지금 모델의 정확도는 ${num(r.scoreVa.정확도, 4)} — `,
        beatsBaseline ? '그보다 조금 낫다. ' : h('b', {}, '즉 「찍기」와 다를 바가 없다. '),
        h('br', {}),
        r.scoreVa.F1 < 0.05
          ? h('span', {}, h('b', {}, `F1 이 ${num(r.scoreVa.F1, 3)} 다`),
            ` — 임계값 0.5 를 넘는 예측이 없어 「${minority.name}」을 하나도 찾아내지 못했다. `,
            '아래 「임계값을 바꾸면?」 표에서 임계값을 어디까지 내려야 찾기 시작하는지 확인해 보자. ',
            `ROC AUC ${num(r.scoreVa['ROC AUC'], 3)} 는 임계값과 무관한 지표라, `,
            '이 값이 0.5 보다 높다면 모델이 순위는 어느 정도 매기고 있다는 뜻이다.')
          : h('span', {}, `재현율 ${num(r.scoreVa.재현율, 3)} · F1 ${num(r.scoreVa.F1, 3)} `,
            `· ROC AUC ${num(r.scoreVa['ROC AUC'], 3)} 를 함께 봐야 한다.`));
    }
  }

  return card('③ 학습 결과 — 큰 숫자가 검증 점수, 작은 글씨가 훈련 점수',
    h('div', { class: 'kv' },
      h('span', {}, h('b', {}, '모델 '), r.spec.label),
      h('span', {}, h('b', {}, '훈련 행 '), num(r.nTrain)),
      h('span', {}, h('b', {}, '검증 행 '), num(r.nValid)),
      h('span', {}, h('b', {}, '특성 '), num(r.prep.featureNames.length)),
      h('span', {}, h('b', {}, '걸린 시간 '), `${r.seconds.toFixed(2)}초`),
      r.treeStats
        ? h('span', {}, h('b', {}, '트리 '), `깊이 ${r.treeStats.depth} · 잎 ${r.treeStats.leaves}개`)
        : null),
    h('div', { class: 'metrics' }, keys.map(metric)),
    imbalanceNote,
    verdict,
    r.isCls ? null : h('div', { class: 'tip' },
      'MSE·RMSE·MAE 는 작을수록 좋고, R² 는 1 에 가까울수록 좋다. '
      + 'R² 가 0 이면 「그냥 평균으로 찍기」와 같은 수준이라는 뜻이다.'));
}

function detailCard(ds, r) {
  const parts = [];

  /* 특성 중요도 */
  if (r.importances) {
    const items = r.prep.featureNames.map((name, j) => ({ label: name, value: r.importances[j] }))
      .filter((o) => Number.isFinite(o.value))
      .sort((a, b) => b.value - a.value).slice(0, 12);
    const total = items.reduce((a, b) => a + b.value, 0);
    const isLinear = ['linear', 'ridge', 'lasso', 'elastic', 'logistic'].includes(state.model);
    parts.push(h('div', {},
      h('div', { class: 'ctrl-l' },
        isLinear ? '계수 크기 |coef_| — 어느 변수가 세게 작용했나' : '특성 중요도 feature_importances_'),
      figure(hBarChart(items), total > 0
        ? `1등은 ${items[0].label}. 11번 문제에서 찾던 값이 바로 이것이다.`
        : '모두 0 이다 — 규제(alpha)가 너무 세서 모델이 아무 변수도 쓰지 않았다.')));
    if (!isLinear && total > 0) {
      parts.push(h('div', { class: 'tip' },
        '「데이터 탐험실」의 상관 히트맵에서는 별것 아니어 보이던 변수가 여기서 상위에 오를 수 있다. '
        + '상관계수는 ', h('b', {}, '곧은 관계만'), ' 잡아내지만, 트리 모델은 '
        + '「너무 높아도 낮아도 위험」 같은 ', h('b', {}, '굽은 관계'), ' 도 찾아내기 때문이다.'));
    }
    if (['lasso', 'elastic'].includes(state.model)) {
      const zeros = r.importances.filter((v) => v === 0).length;
      parts.push(h('div', { class: 'tip' },
        `계수가 정확히 0 이 된 특성이 ${zeros}개다. L1 규제(라쏘)의 특징으로, `
        + '모델이 스스로 「이 변수는 안 쓴다」고 결정한 것이다. alpha 를 키우면 더 늘어난다.'));
    }
  } else {
    parts.push(h('div', { class: 'note' },
      'KNN 은 「가까운 이웃을 본다」가 전부여서 특성 중요도를 내놓지 않는다. '
      + '중요도를 보고 싶으면 트리 계열이나 선형 모델을 골라 보자.'));
  }

  if (!r.isCls) {
    /* 회귀: 실제값 vs 예측값 */
    const points = r.yva.map((v, i) => ({ x: v, y: r.predVa[i] }));
    parts.push(h('div', {},
      h('div', { class: 'ctrl-l' }, '실제값 vs 예측값 (검증 데이터)'),
      figure(scatterChart([{ label: '', points }], {
        xLabel: `실제 ${ds.target}`, yLabel: `예측 ${ds.target}`, diagonal: true, r: 2.8,
      }), '점선(대각선) 위에 점이 모일수록 잘 맞힌 것이다.')));

    const errs = r.yva.map((v, i) => r.predVa[i] - v);
    const worst = errs.map((e, i) => ({ i, e })).sort((a, b) => Math.abs(b.e) - Math.abs(a.e))
      .slice(0, 5);
    parts.push(h('div', {},
      h('div', { class: 'ctrl-l' }, '가장 크게 틀린 5개'),
      table(['실제값', '예측값', '오차(예측−실제)'], worst.map((o) => ({
        cells: [num(r.yva[o.i]), num(r.predVa[o.i]), num(o.e)],
      })), { className: 'tbl-s' })));
  } else {
    /* 분류: 혼동 행렬 + ROC + 임계값 */
    const cm = r.scoreVa._cm;
    const cn = ds.classNames || { 0: '0', 1: '1' };
    parts.push(h('div', {},
      h('div', { class: 'ctrl-l' }, '혼동 행렬 (검증 데이터)'),
      h('table', { class: 'cm' },
        h('tr', {}, h('th', {}, ''), h('th', {}, `예측: ${cn[0]}`), h('th', {}, `예측: ${cn[1]}`)),
        h('tr', {}, h('th', {}, `실제: ${cn[0]}`),
          h('td', { class: 'tn' }, h('div', { class: 'cm-n' }, num(cm.tn)),
            h('div', { class: 'cm-l' }, 'TN 맞게 음성')),
          h('td', { class: 'fp' }, h('div', { class: 'cm-n' }, num(cm.fp)),
            h('div', { class: 'cm-l' }, 'FP 헛경보'))),
        h('tr', {}, h('th', {}, `실제: ${cn[1]}`),
          h('td', { class: 'fn' }, h('div', { class: 'cm-n' }, num(cm.fn)),
            h('div', { class: 'cm-l' }, 'FN 놓침')),
          h('td', { class: 'tp' }, h('div', { class: 'cm-n' }, num(cm.tp)),
            h('div', { class: 'cm-l' }, 'TP 맞게 양성')))),
      h('div', { class: 'tip' },
        `정밀도 = TP ÷ (TP+FP) = ${cm.tp} ÷ ${cm.tp + cm.fp} — 「${cn[1]}」이라 한 것 중 맞은 비율.`,
        h('br', {}),
        `재현율 = TP ÷ (TP+FN) = ${cm.tp} ÷ ${cm.tp + cm.fn} — 진짜 「${cn[1]}」 중 찾아낸 비율.`)));

    if (r.probVa) {
      const pts = rocCurve(r.yva, r.probVa);
      parts.push(h('div', {},
        h('div', { class: 'ctrl-l' }, 'ROC 곡선'),
        figure(lineChart([{ label: `AUC = ${num(rocAucScore(r.yva, r.probVa), 3)}`, points: pts }], {
          xLabel: '헛경보율 (FPR)', yLabel: '찾아낸율 (TPR)',
          xMin: 0, xMax: 1, yMin: 0, yMax: 1, diagonal: true,
        }), '점선(대각선)은 동전 던지기 수준. 곡선이 왼쪽 위로 붕 뜰수록 좋은 모델이다.')));

      /* 결정 트리·KNN 은 예측 확률이 몇 개 값으로만 나온다.
       * 그러면 임계값을 아무리 낮춰도 결과가 그대로라 수업이 성립하지 않는다.
       * 후보가 하나뿐이면 왜 그런지 알려 주고 다른 모델을 권한다. */
      const cand = thresholdCandidates(r.probVa);
      const distinctProb = new Set(r.probVa.map((v) => Math.round(v * 1e6))).size;
      const rows = cand.map((th) => {
        const pred = r.probVa.map((p) => (p >= th ? 1 : 0));
        const s = classificationScores(r.yva, pred, r.probVa);
        const nPos = pred.reduce((a, b) => a + b, 0);
        return {
          cells: [fmtThreshold(th), num(nPos), num(s.정확도, 3), num(s.정밀도, 3),
            num(s.재현율, 3), num(s.F1, 3), `${s._cm.fp} / ${s._cm.fn}`],
          on: Math.abs(th - 0.5) < 1e-9,
          f1: s.F1,
        };
      });
      const bestF1 = rows.reduce((a, b) => (b.f1 > a.f1 ? b : a), rows[0]);
      const tooFewSteps = cand.length < 2;
      parts.push(h('div', {},
        h('div', { class: 'ctrl-l' }, '임계값을 바꾸면? — 정밀도와 재현율은 서로 맞바뀐다'),
        tooFewSteps
          ? h('div', { class: 'warn' },
            h('b', {}, '이 모델로는 임계값 실험을 할 수 없다. '),
            `${r.spec.label}는 예측 확률이 ${distinctProb}가지 값으로만 나온다. `,
            '잎(leaf)에 모인 데이터의 비율이 그대로 확률이 되기 때문이다. '
            + '값이 띄엄띄엄해서 임계값을 낮춰도 예측이 그대로다. ',
            h('br', {}),
            h('b', {}, '랜덤 포레스트'), '나 ', h('b', {}, '로지스틱 회귀'),
            ' 로 바꿔 보자. 확률이 촘촘하게 나와 임계값을 내릴수록 '
            + '재현율이 오르는 모습을 볼 수 있다.')
          : null,
        table(['임계값', `「${cn[1]}」예측 수`, '정확도', '정밀도', '재현율', 'F1', '헛경보/놓침'],
          rows, { rowClass: (x) => (x.on ? 'hi' : ''), className: 'tbl-s' }),
        h('div', { class: 'note' },
          '임계값을 낮추면 「양성」이라 부르는 것이 많아져 재현율은 오르지만 정밀도는 떨어진다. '
          + '불량 검사처럼 놓치면 안 되는 문제라면 임계값을 낮추는 편이 낫다. '
          + '기본값 0.5 가 늘 정답은 아니다.',
          bestF1 && bestF1.f1 > 0 && !bestF1.on
            ? h('span', {}, h('br', {}),
              `이 모델에서는 임계값 ${bestF1.cells[0]} 일 때 F1 이 가장 높다(${num(bestF1.f1, 3)}). `,
              '0.5 를 그대로 쓰는 것보다 낫다는 뜻이다.')
            : null)));
    }
  }

  return card('④ 자세히 들여다보기', ...parts);
}

/**
 * 임계값 표에 쓸 값들을 고른다.
 *
 * 0.2 · 0.3 … 0.8 처럼 고정된 값만 쓰면, 예측 확률이 모두 0.1 아래에 몰려 있는
 * 불균형 데이터에서는 어느 줄이나 결과가 똑같아 아무것도 배울 수 없다.
 * 그래서 「예측 확률이 높은 순으로 상위 몇 %」 지점을 임계값으로 삼는다.
 */
function thresholdCandidates(prob) {
  const sorted = [...prob].filter(Number.isFinite).sort((a, b) => b - a);
  const n = sorted.length;
  const out = new Set([0.5]);
  [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35].forEach((frac) => {
    const k = Math.min(n - 1, Math.max(0, Math.round(n * frac) - 1));
    const v = sorted[k];
    // 불균형 데이터에서는 확률이 0.000001 보다도 작을 수 있으니 자르지 않는다
    if (Number.isFinite(v) && v > 0 && v < 1) out.add(v);
  });
  return [...out].sort((a, b) => a - b);
}

/** 임계값 표시 — 아주 작은 값은 지수 표기로 (0.0000 이 여러 줄 나오는 것을 막는다) */
function fmtThreshold(v) {
  if (v < 1e-4) return v.toExponential(1);
  return v.toFixed(4);
}

/* ── ⑤ 예측 체험 ── */
function predictCard(ds, r, root) {
  if (!state.predRow) state.predRow = { ...r.prep.typicalRow };
  const cols = ds.frame.columns.filter((c) => c !== ds.target
    && !(ds.idColumns || []).includes(c) && !state.dropColumns.includes(c));

  const out = h('div', { class: 'predout' });
  // 예측 결과 상자만 다시 그린다 (화면 전체가 아니라서 슬라이더를 끄는 도중에도 안전하다)
  const drawPrediction = () => {
    clear(out);
    const x = r.prep.encodeRow(state.predRow);
    const xs = r.scaler ? r.scaler.transform([x]) : [x];
    if (r.isCls) {
      const p = r.model.predictProba ? r.model.predictProba(xs)[0] : r.model.predict(xs)[0];
      const cn = ds.classNames || { 0: '0', 1: '1' };
      out.append(
        h('div', { class: 'pl' }, `「${cn[1]}」일 확률`),
        h('div', { class: 'pv' }, `${(p * 100).toFixed(1)}%`),
        h('div', { class: 'gauge' }, h('div', { style: `width:${Math.round(p * 100)}%` })),
        h('div', { class: 'pl' }, `임계값 0.5 기준 예측 → ${p >= 0.5 ? cn[1] : cn[0]}`));
    } else {
      const v = r.model.predict(xs)[0];
      out.append(
        h('div', { class: 'pl' }, `예측 ${ds.target}`),
        h('div', { class: 'pv' }, `${num(v, 2)} ${ds.unit || ''}`));
    }
  };

  const ctrls = cols.map((c) => {
    if (ds.frame.isNumeric(c)) {
      const vals = ds.frame.values(c);
      const lo = Math.min(...vals);
      const hi = Math.max(...vals);
      const step = (hi - lo) > 50 ? Math.max(1, Math.round((hi - lo) / 200)) : (hi - lo) / 100;
      return sliderRow(c, {
        min: lo, max: hi, step, value: Number(state.predRow[c]) || lo,
        format: (v) => num(v, 2),
        onInput: (v) => { state.predRow[c] = v; drawPrediction(); },
      });
    }
    const levels = [...new Set(ds.frame.col(c).filter((v) => v !== null).map(String))]
      .sort((a, b) => a.localeCompare(b, 'ko'));
    return selectRow(c, levels.map((l) => ({ value: l, label: l })),
      String(state.predRow[c]), (v) => { state.predRow[c] = v; drawPrediction(); });
  });

  drawPrediction();
  return card('⑤ 예측 체험 — 새 데이터를 넣어 보기',
    h('p', { class: 'lead' },
      '값을 움직이면 방금 학습한 모델이 곧바로 다시 예측한다. '
      + '학습할 때 쓴 결측치 채움값·인코딩·스케일러가 똑같이 적용된다 — 14번 문제의 핵심이다.'),
    out,
    h('div', { class: 'ctrls' }, ctrls),
    h('div', { class: 'btnrow' },
      h('button', {
        class: 'btn btn-3 btn-s',
        onclick: () => { state.predRow = { ...r.prep.typicalRow }; renderTrain(root); },
      }, '평범한 값으로 되돌리기'),
      h('button', {
        class: 'btn btn-3 btn-s',
        onclick: () => {
          const i = Math.floor(Math.random() * ds.frame.length);
          const row = {};
          cols.forEach((c) => {
            const v = ds.frame.col(c)[i];
            row[c] = (v === null || Number.isNaN(v)) ? r.prep.typicalRow[c] : v;
          });
          state.predRow = row;
          renderTrain(root);
        },
      }, '실제 데이터에서 한 줄 뽑아 오기')),
    foldout('파이썬으로는 이렇게 씁니다', codeBlock(
      '# 새 데이터도 반드시 학습 때 쓴 scaler 로 transform 만 한다\n'
      + `new_data = np.array([[ ... 특성 ${r.prep.featureNames.length}개 ... ]])\n`
      + 'new_scaled = scaler.transform(new_data)\n\n'
      + 'pred = model.predict(new_scaled)\n'
      + 'print(pred[0])').outerHTML));
}
