/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 탭 ④ 딥러닝 실험실
 *
 * keras 로 만드는 Dense 층 신경망을 흉내 낸다.
 * Sequential 모델에 Dense 층을 쌓고 학습시키는 과정을 그대로 브라우저에서 돌린다.
 *
 * 이 탭의 볼거리는 「손실 곡선」이다.
 * epoch 을 늘리면 훈련 손실은 계속 내려가는데 검증 손실은 어느 순간부터 올라간다.
 * 그 갈라지는 지점이 바로 과적합이 시작되는 곳이다.
 */
import { getDataset, datasetOptions, DATASET_IDS } from '../data/datasets.js';
import {
  h, card, table, figure, selectRow, pills, num, codeBlock, foldout, sliderRow, progress,
  badge, redraw,
} from '../lib/ui.js';
import { lineChart, scatterChart, emptyChart } from '../lib/chart.js';
import { buildMatrix } from '../lib/pipeline.js';
import { trainTestSplit, StandardScaler } from '../lib/preprocess.js';
import { NeuralNetwork } from '../lib/nn.js';
import { regressionScores, classificationScores, rocAucScore } from '../lib/metrics.js';

const state = {
  dsId: DATASET_IDS[0],   // 첫 회차. 수업 순서가 바뀌어도 따라간다
  layers: [64, 32],
  activation: 'relu',
  dropout: 0,
  epochs: 40,
  batchSize: 32,
  learningRate: 0.01,
  testSize: 0.3,
  seed: 100,
  result: null,
  busy: false,
};

/** 다른 탭(문제 풀기)에서 이 탭을 열 때 데이터셋을 미리 골라 둔다 */
export function useDataset(dsId) {
  if (!dsId || dsId === state.dsId) return;
  state.dsId = dsId;
  // 데이터 고르기 단추를 눌렀을 때와 똑같이 이전 데이터에 묶인 선택을 비운다
  state.result = null;
}

export function renderDeep(root) {
  // redraw 로 감싸면 옵션을 눌러 다시 그려도 보던 자리에 그대로 머무른다
  redraw(root, () => {
    const ds = getDataset(state.dsId);
    root.append(
      picker(root),
      designCard(ds, root),
      resultCard(ds),
    );
  });
}

function picker(root) {
  const grid = h('div', { class: 'dspick' }, datasetOptions().map((o) => h('button', {
    type: 'button',
    class: `dsbtn${o.id === state.dsId ? ' on' : ''}`,
    onclick: () => { state.dsId = o.id; state.result = null; renderDeep(root); },
  }, h('b', {}, `${o.round}회차 · ${o.short}`), h('span', {}, o.taskLabel))));
  return card('실습 데이터 고르기', grid);
}

/* ── 모델 설계 ── */
function designCard(ds, root) {
  const isCls = ds.task === 'classification';
  const prep = buildMatrix(ds, { imputePreset: 'medianMode', encode: 'onehot' });
  const nIn = prep.featureNames.length;

  // 층 구조 그림
  const boxes = [h('div', { class: 'layerbox', style: 'background:#e0f2fe;border-color:#7dd3fc' },
    `입력 ${nIn}개`)];
  state.layers.forEach((u) => {
    boxes.push(h('span', { class: 'arrow' }, '→'));
    boxes.push(h('div', { class: 'layerbox' },
      `Dense(${u}) ${state.activation}`,
      state.dropout > 0 ? h('span', {}, ` + Dropout(${state.dropout})`) : null));
  });
  boxes.push(h('span', { class: 'arrow' }, '→'));
  boxes.push(h('div', { class: 'layerbox', style: 'background:#dcfce7;border-color:#86efac' },
    `Dense(1) ${isCls ? 'sigmoid' : 'linear'}`));

  const params = totalParams(nIn, state.layers);

  const layerCtrls = state.layers.map((u, i) => sliderRow(`은닉층 ${i + 1} 노드 수`, {
    min: 4, max: 128, step: 4, value: u,
    onChange: (v) => { state.layers[i] = v; state.result = null; renderDeep(root); },
  }));

  const pg = progress('학습 준비됨');
  const btn = h('button', {
    class: 'btn', disabled: state.busy,
    onclick: async () => {
      state.busy = true;
      btn.disabled = true;
      btn.textContent = '학습 중…';
      const live = document.getElementById('nn-live');
      try {
        state.result = await runTraining(ds, prep, (r, t, hist) => {
          pg.set(r, t);
          if (live && hist) live.innerHTML = lossChart(hist, isCls);
        });
      } catch (err) {
        state.result = { error: String(err?.message || err) };
        console.error(err);
      }
      state.busy = false;
      renderDeep(root);
    },
  }, '▶ 학습 시작');

  return card('① 신경망 설계하기',
    h('p', { class: 'lead' },
      'Dense 층은 「앞 층의 모든 노드와 연결된 층」이다. '
      + '노드가 많고 층이 깊으면 복잡한 관계를 배울 수 있지만 그만큼 과적합도 쉬워진다.'),
    h('div', { class: 'layerrow' }, boxes),
    h('div', { class: 'kv' },
      h('span', {}, h('b', {}, '학습할 가중치 '), `${num(params)}개`),
      h('span', {}, h('b', {}, '손실 함수 '),
        h('code', {}, isCls ? 'binary_crossentropy' : 'mean_squared_error')),
      h('span', {}, h('b', {}, '옵티마이저 '), h('code', {}, 'adam')),
      h('span', {}, badge(isCls ? '이진 분류' : '회귀', 'v'))),
    h('div', { class: 'ctrls' },
      h('div', { class: 'ctrl' }, h('span', { class: 'ctrl-l' }, '은닉층 개수'),
        pills([1, 2, 3].map((n) => ({ value: n, label: `${n}개` })), state.layers.length,
          (v) => {
            const n = Number(v);
            const def = [64, 32, 16];
            state.layers = Array.from({ length: n }, (_, i) => state.layers[i] ?? def[i]);
            state.result = null;
            renderDeep(root);
          }, { small: true })),
      ...layerCtrls,
      h('div', { class: 'ctrl' }, h('span', { class: 'ctrl-l' }, '활성화 함수'),
        pills(['relu', 'tanh', 'sigmoid'].map((a) => ({ value: a, label: a })),
          state.activation, (v) => { state.activation = v; state.result = null; renderDeep(root); },
          { small: true })),
      selectRow('Dropout (과적합 방지)',
        [0, 0.1, 0.2, 0.3, 0.5].map((d) => ({ value: d, label: d === 0 ? '안 씀' : `${d}` })),
        state.dropout, (v) => { state.dropout = Number(v); state.result = null; renderDeep(root); },
        '학습할 때만 노드를 임시로 끈다')),
    h('div', { class: 'ctrls' },
      sliderRow('epochs (몇 바퀴 돌까)', {
        min: 5, max: 150, step: 5, value: state.epochs,
        onInput: (v) => { state.epochs = v; },
      }),
      selectRow('batch_size', [8, 16, 32, 64, 128].map((b) => ({ value: b, label: `${b}` })),
        state.batchSize, (v) => { state.batchSize = Number(v); },
        '한 번에 몇 개씩 보고 가중치를 고칠까'),
      sliderRow('learning_rate', {
        min: 0.0005, max: 0.05, step: 0.0005, value: state.learningRate,
        format: (v) => v.toFixed(4),
        onInput: (v) => { state.learningRate = v; },
        hint: '너무 크면 튀고 너무 작으면 안 배운다',
      }),
      sliderRow('검증 비율', {
        min: 0.1, max: 0.5, step: 0.05, value: state.testSize,
        format: (v) => `${(v * 100).toFixed(0)}%`,
        onInput: (v) => { state.testSize = v; },
      })),
    h('div', { class: 'btnrow' }, btn, pg.node),
    h('div', { id: 'nn-live', html: emptyChart('학습을 시작하면 손실 곡선이 여기 그려집니다') }),
    foldout('파이썬으로는 이렇게 씁니다', codeBlock(pyKeras(ds, nIn, isCls)).outerHTML));
}

function totalParams(nIn, layers) {
  let prev = nIn;
  let sum = 0;
  [...layers, 1].forEach((u) => { sum += prev * u + u; prev = u; });
  return sum;
}

function pyKeras(ds, nIn, isCls) {
  const lines = state.layers.map((u, i) => {
    const first = i === 0 ? `, input_shape=[${nIn}]` : '';
    const drop = state.dropout > 0 ? `\n    Dropout(${state.dropout}),` : '';
    return `    Dense(${u}, activation='${state.activation}'${first}),${drop}`;
  }).join('\n');
  return 'import tensorflow as tf\n'
    + 'from tensorflow.keras.models import Sequential\n'
    + `from tensorflow.keras.layers import Dense${state.dropout > 0 ? ', Dropout' : ''}\n\n`
    + 'model_dl = Sequential([\n'
    + `${lines}\n`
    + `    Dense(1, activation='${isCls ? 'sigmoid' : 'linear'}')\n`
    + '])\n\n'
    + `model_dl.compile(optimizer='adam',\n`
    + `                 loss='${isCls ? 'binary_crossentropy' : 'mean_squared_error'}'`
    + `${isCls ? ",\n                 metrics=['accuracy']" : ''})\n\n`
    + 'history = model_dl.fit(\n'
    + '    X_train_scaled, y_train,\n'
    + `    epochs=${state.epochs},\n`
    + `    batch_size=${state.batchSize},\n`
    + '    validation_data=(X_valid_scaled, y_valid))\n\n'
    + '# 손실 곡선 그리기\n'
    + "plt.plot(history.history['loss'], label='train')\n"
    + "plt.plot(history.history['val_loss'], label='valid')\n"
    + 'plt.legend()\nplt.show()';
}

/* ── 학습 ── */
async function runTraining(ds, prep, onProgress) {
  const isCls = ds.task === 'classification';
  onProgress(0.03, '전처리 중…');

  const { train, valid } = trainTestSplit(prep.X.length, {
    testSize: state.testSize, seed: state.seed, stratify: isCls ? prep.y : null,
  });
  const scaler = new StandardScaler().fit(train.map((i) => prep.X[i]));
  const Xtr = scaler.transform(train.map((i) => prep.X[i]));
  const Xva = scaler.transform(valid.map((i) => prep.X[i]));
  const ytrRaw = train.map((i) => prep.y[i]);
  const yvaRaw = valid.map((i) => prep.y[i]);

  /* 회귀는 목표값도 0~1 로 줄인다.
   * 가격이 8000 이면 손실(MSE)이 수백만이 되어 곡선이 아무것도 안 보이고
   * adam 의 기본 학습률로는 좀처럼 줄지 않는다. 예측할 때 다시 되돌린다. */
  let yLo = 0;
  let yHi = 1;
  if (!isCls) {
    yLo = Math.min(...ytrRaw);
    yHi = Math.max(...ytrRaw);
    if (yHi === yLo) yHi = yLo + 1;
  }
  const norm = (v) => (v - yLo) / (yHi - yLo);
  const back = (v) => v * (yHi - yLo) + yLo;
  const ytr = isCls ? ytrRaw : ytrRaw.map(norm);
  const yva = isCls ? yvaRaw : yvaRaw.map(norm);

  // Dropout 은 은닉층 뒤에만 붙인다. 출력층에 걸면 예측값이 흔들려 버린다.
  const layers = state.layers.map((u) => ({
    units: u, activation: state.activation, dropout: state.dropout,
  }));
  layers.push({ units: 1, activation: isCls ? 'sigmoid' : 'linear', dropout: 0 });

  const net = new NeuralNetwork(prep.featureNames.length, layers, {
    loss: isCls ? 'binary_crossentropy' : 'mse',
    learningRate: state.learningRate,
    seed: state.seed,
  });

  const t0 = performance.now();
  await net.fit(Xtr, ytr, {
    epochs: state.epochs,
    batchSize: state.batchSize,
    validationData: [Xva, yva],
    onEpoch: (ep, hist) => {
      onProgress(0.03 + (ep / state.epochs) * 0.94,
        `epoch ${ep}/${state.epochs} · 손실 ${hist.loss[ep - 1].toFixed(5)}`, hist);
    },
  });
  const seconds = (performance.now() - t0) / 1000;

  const rawTr = net.predict(Xtr);
  const rawVa = net.predict(Xva);
  const predTr = isCls ? rawTr : rawTr.map(back);
  const predVa = isCls ? rawVa : rawVa.map(back);

  let scoreTr;
  let scoreVa;
  if (isCls) {
    scoreTr = classificationScores(ytrRaw, predTr.map((p) => (p >= 0.5 ? 1 : 0)), predTr);
    scoreVa = classificationScores(yvaRaw, predVa.map((p) => (p >= 0.5 ? 1 : 0)), predVa);
  } else {
    scoreTr = regressionScores(ytrRaw, predTr);
    scoreVa = regressionScores(yvaRaw, predVa);
  }

  onProgress(1, `학습 완료 (${seconds.toFixed(1)}초)`);
  return {
    isCls, history: net.history, net, scoreTr, scoreVa,
    yvaRaw, predVa, seconds, summary: net.summary(),
    nTrain: Xtr.length, nValid: Xva.length,
    yScaled: !isCls, yLo, yHi,
  };
}

/* ── 손실 곡선 ── */
function lossChart(history, isCls) {
  const series = [{
    label: '훈련 손실', points: history.loss.map((v, i) => ({ x: i + 1, y: v })),
  }];
  if (history.valLoss.length) {
    series.push({
      label: '검증 손실', color: '#dc2626',
      points: history.valLoss.map((v, i) => ({ x: i + 1, y: v })),
    });
  }
  return lineChart(series, {
    xLabel: 'epoch', yLabel: isCls ? '손실 (binary crossentropy)' : '손실 (MSE)', yMin: 0,
  });
}

/* ── 결과 ── */
function resultCard(ds) {
  const r = state.result;
  if (!r) {
    return card('② 학습 결과',
      h('div', { class: 'tip' }, '「학습 시작」을 누르면 epoch 마다 손실 곡선이 그려진다.'));
  }
  if (r.error) return card('② 학습 결과', h('div', { class: 'warn' }, r.error));

  const keys = r.isCls
    ? ['정확도', '정밀도', '재현율', 'F1', 'ROC AUC']
    : ['MSE', 'RMSE', 'MAE', 'R²'];
  const metrics = keys.map((k) => h('div', { class: 'metric' },
    h('div', { class: 'metric-l' }, k),
    h('div', { class: 'metric-v' }, num(r.scoreVa[k], 4)),
    h('div', { class: 'metric-sub' }, `훈련 ${num(r.scoreTr[k], 4)}`)));

  /* 과적합을 두 가지로 살펴본다.
   *   ① 검증 손실이 어느 epoch 에서 가장 낮았나 (그 뒤로 올라가면 과적합)
   *   ② 훈련 점수와 검증 점수의 차이가 큰가 */
  const vl = r.history.valLoss;
  let bestEp = 1;
  vl.forEach((v, i) => { if (v < vl[bestEp - 1]) bestEp = i + 1; });
  const lastGap = vl[vl.length - 1] - vl[bestEp - 1];
  const curveTurned = bestEp < vl.length * 0.85 && lastGap > vl[bestEp - 1] * 0.05;
  const mainKey = r.isCls ? '정확도' : 'R²';
  const scoreGap = r.scoreTr[mainKey] - r.scoreVa[mainKey];
  const gapBig = scoreGap > 0.15;

  const parts = [
    h('div', { class: 'kv' },
      h('span', {}, h('b', {}, '훈련 행 '), num(r.nTrain)),
      h('span', {}, h('b', {}, '검증 행 '), num(r.nValid)),
      h('span', {}, h('b', {}, 'epoch '), num(state.epochs)),
      h('span', {}, h('b', {}, '걸린 시간 '), `${r.seconds.toFixed(1)}초`)),
    h('div', { class: 'metrics' }, metrics),
    figure(lossChart(r.history, r.isCls),
      '파란 선은 훈련 손실, 빨간 선은 검증 손실이다.'),
    curveTurned
      ? h('div', { class: 'warn' },
        h('b', {}, '과적합 — 검증 손실이 꺾였다. '),
        `검증 손실이 ${bestEp} epoch 에서 가장 낮았고(${num(vl[bestEp - 1], 5)}) `,
        `그 뒤로는 다시 올라가 마지막에는 ${num(vl[vl.length - 1], 5)} 가 되었다. `,
        `epoch 을 ${bestEp} 로 줄이거나, Dropout 을 켜거나, `,
        'keras 의 EarlyStopping 을 쓰면 가장 좋은 지점에서 멈출 수 있다.')
      : gapBig
        ? h('div', { class: 'warn' },
          h('b', {}, '과적합 — 훈련 점수만 좋다. '),
          `훈련 ${mainKey} ${num(r.scoreTr[mainKey], 3)} 인데 검증 ${mainKey} 는 `,
          `${num(r.scoreVa[mainKey], 3)} 다(차이 ${num(scoreGap, 3)}). `,
          '검증 손실 곡선은 아직 내려가는 중이지만 훈련 데이터를 외우는 쪽으로 기울었다. '
          + '노드 수를 줄이거나 Dropout 을 켜 보자.')
        : h('div', { class: 'ok' },
          `훈련과 검증의 ${mainKey} 차이가 ${num(Math.abs(scoreGap), 3)} 로 크지 않고, `
          + `검증 손실도 ${bestEp} epoch 까지 잘 내려갔다. `
          + 'epoch 을 더 늘려 보면 어디서 꺾이는지 볼 수 있다.'),
    h('div', { class: 'ctrl-l' }, '층 구조 요약 — keras 의 model.summary()'),
    table(['층', '노드 수', '활성화', 'Dropout', '가중치 개수'],
      r.summary.map((s, i) => ({
        cells: [i === r.summary.length - 1 ? `${s.name} (출력층)` : s.name,
          num(s.units), s.activation, s.dropout ? `${s.dropout}` : '—', num(s.params)],
      }))),
  ];

  if (!r.isCls) {
    parts.push(figure(scatterChart([{
      label: '',
      points: r.yvaRaw.map((v, i) => ({ x: v, y: r.predVa[i] })),
    }], {
      xLabel: `실제 ${ds.target}`, yLabel: `예측 ${ds.target}`, diagonal: true, r: 2.8,
    }), '점선 위에 점이 모일수록 잘 맞힌 것이다.'));
    parts.push(h('div', { class: 'note' },
      h('b', {}, '한 가지 알아 둘 것 — '),
      `여기서는 목표값 ${ds.target} 도 0~1 로 줄여서 학습시킨 뒤 예측할 때 되돌렸다 `,
      `(원래 범위 ${num(r.yLo)} ~ ${num(r.yHi)}). `,
      '원래 값을 그대로 넣으면 손실이 수백만이 되어 곡선이 안 보이고 '
      + '학습도 아주 느려진다. 위에 적힌 MSE·R² 는 원래 단위로 되돌린 값이다.'));
  } else {
    parts.push(h('div', { class: 'tip' },
      `AUC ${num(rocAucScore(r.yvaRaw, r.predVa), 3)} — `,
      '출력층이 sigmoid 이므로 결과가 0~1 확률로 나온다. '
      + '14번 문제에서 predict() 결과를 「이탈 확률」로 읽던 것이 이것이다.'));
  }

  parts.push(h('div', { class: 'tip' },
    h('b', {}, '실험해 볼 것 — '),
    '① 노드 수를 4로 줄이면? (과소적합) ② epoch 을 150으로 늘리면? (과적합) '
    + '③ learning_rate 를 0.05로 올리면? (손실이 튄다) ④ Dropout 0.3을 켜면? '
    + '(훈련 손실은 나빠지지만 검증 손실은 좋아질 수 있다)'));

  return card('② 학습 결과', ...parts);
}
