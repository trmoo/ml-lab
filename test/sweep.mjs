/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 전수 점검 — node test/sweep.mjs
 *
 * 데이터셋 10개 × 모델 8~5개를 모두 학습시켜 본다.
 * 「어느 조합에서 NaN 이 나오거나 터지지 않는지」와
 * 「점수가 상식적인 범위에 들어오는지」를 확인하는 것이 목적이다.
 * 시간이 좀 걸리므로 npm test 와 따로 두었다.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { META, DATASET_IDS } from '../src/data/meta.js';
import { buildDataset } from '../src/data/build.js';
import { buildMatrix } from '../src/lib/pipeline.js';
import { trainTestSplit, StandardScaler } from '../src/lib/preprocess.js';
import {
  LinearRegression, ElasticNetRegression, LogisticRegression, KNeighbors,
  DecisionTree, RandomForest, GradientBoosting,
} from '../src/lib/models.js';
import { regressionScores, classificationScores } from '../src/lib/metrics.js';
import { NeuralNetwork } from '../src/lib/nn.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const load = (key) => buildDataset(key,
  META[key].files.map((f) => readFileSync(join(ROOT, 'data', f), 'utf8')));

const REG = {
  '선형회귀': () => new LinearRegression({ alpha: 0 }),
  '릿지': () => new LinearRegression({ alpha: 1 }),
  '라쏘': () => new ElasticNetRegression({ alpha: 1, l1Ratio: 1 }),
  '엘라스틱넷': () => new ElasticNetRegression({ alpha: 1, l1Ratio: 0.5 }),
  'KNN': () => new KNeighbors({ k: 5, task: 'regression' }),
  '결정트리': () => new DecisionTree({ task: 'regression', maxDepth: 6 }),
  '랜덤포레스트': () => new RandomForest({ task: 'regression', nEstimators: 60, maxDepth: 10 }),
  '부스팅': () => new GradientBoosting({
    task: 'regression', nEstimators: 60, learningRate: 0.1, maxDepth: 3,
  }),
};
const CLS = {
  '로지스틱': () => new LogisticRegression({ C: 1, maxIter: 400 }),
  'KNN': () => new KNeighbors({ k: 5, task: 'classification' }),
  '결정트리': () => new DecisionTree({ task: 'classification', maxDepth: 6 }),
  '랜덤포레스트': () => new RandomForest({ task: 'classification', nEstimators: 60, maxDepth: 10 }),
  '부스팅': () => new GradientBoosting({
    task: 'classification', nEstimators: 60, learningRate: 0.1, maxDepth: 3,
  }),
};

const problems = [];
const pad = (s, n) => String(s).padEnd(n, ' ');
const padL = (s, n) => String(s).padStart(n, ' ');

for (const key of DATASET_IDS) {
  let ds;
  try {
    ds = load(key);
  } catch (err) {
    problems.push(`${key}: 데이터 준비 실패 — ${err.message}`);
    continue;
  }
  const isCls = ds.task === 'classification';
  console.log(`\n${'═'.repeat(78)}`);
  console.log(`${ds.round}회차 · ${ds.label}  [${isCls ? '분류' : '회귀'}] `
    + `${ds.frame.length}행 × ${ds.frame.columns.length}열  타깃=${ds.target}`);

  // 전처리 방법을 두 가지로 각각 확인한다
  for (const preset of ['medianMode', 'dropRows']) {
    const prep = buildMatrix(ds, { imputePreset: preset, encode: 'onehot' });
    if (!prep.X.length) { problems.push(`${key}/${preset}: 남는 행이 0`); continue; }
    if (!prep.X.every((r) => r.every(Number.isFinite))) {
      problems.push(`${key}/${preset}: X 에 NaN 이 남았다`);
    }
    console.log(`  [${preset}] ${prep.X.length}행 × ${prep.featureNames.length}열`);

    const { train, valid } = trainTestSplit(prep.X.length, {
      testSize: 0.3, seed: 100, stratify: isCls ? prep.y : null,
    });
    const sc = new StandardScaler().fit(train.map((i) => prep.X[i]));
    const Xtr = sc.transform(train.map((i) => prep.X[i]));
    const Xva = sc.transform(valid.map((i) => prep.X[i]));
    const ytr = train.map((i) => prep.y[i]);
    const yva = valid.map((i) => prep.y[i]);

    if (isCls) {
      const ones = ytr.filter((v) => v === 1).length;
      if (!ones || ones === ytr.length) problems.push(`${key}/${preset}: 훈련 데이터에 한쪽 정답만 있다`);
    }

    if (preset !== 'medianMode') continue;   // 모델 전수는 기본 전처리에서만

    const models = isCls ? CLS : REG;
    for (const [name, make] of Object.entries(models)) {
      const model = make();
      const t0 = process.hrtime.bigint();
      try {
        if (model.fit.constructor.name === 'AsyncFunction') await model.fit(Xtr, ytr);
        else model.fit(Xtr, ytr);
        const pred = model.predict(Xva);
        const prob = isCls && model.predictProba ? model.predictProba(Xva) : null;
        const ms = Number(process.hrtime.bigint() - t0) / 1e6;
        const s = isCls ? classificationScores(yva, pred, prob) : regressionScores(yva, pred);
        const main = isCls ? s.정확도 : s['R²'];
        const extra = isCls ? `F1 ${s.F1.toFixed(3)} AUC ${(s['ROC AUC'] || 0).toFixed(3)}`
          : `MAE ${s.MAE.toFixed(1)}`;
        const flag = !Number.isFinite(main) ? ' ❌NaN'
          : (isCls ? (main < 0.5 ? ' ⚠낮음' : '') : (main < 0 ? ' ⚠음수R²' : ''));
        console.log(`    ${pad(name, 12)} ${isCls ? '정확도' : 'R²'} `
          + `${padL(main.toFixed(4), 8)}  ${pad(extra, 22)} ${padL(ms.toFixed(0), 5)}ms${flag}`);
        if (!Number.isFinite(main)) problems.push(`${key}/${name}: 점수가 NaN`);
        if (pred.some((v) => !Number.isFinite(v))) problems.push(`${key}/${name}: 예측값에 NaN`);
        if (ms > 8000) problems.push(`${key}/${name}: ${ms.toFixed(0)}ms — 너무 느리다`);
        const imp = model.importances ? model.importances() : null;
        if (imp && imp.some((v) => !Number.isFinite(v))) {
          problems.push(`${key}/${name}: 특성 중요도에 NaN`);
        }
      } catch (err) {
        console.log(`    ${pad(name, 12)} ❌ ${err.message}`);
        problems.push(`${key}/${name}: 예외 — ${err.message}`);
      }
    }

    /* 신경망도 한 번씩 */
    const yLo = Math.min(...ytr);
    const yHi = Math.max(...ytr) || 1;
    const norm = (v) => (isCls ? v : (v - yLo) / ((yHi - yLo) || 1));
    const net = new NeuralNetwork(prep.featureNames.length, [
      { units: 32, activation: 'relu', dropout: 0 },
      { units: 16, activation: 'relu', dropout: 0 },
      { units: 1, activation: isCls ? 'sigmoid' : 'linear', dropout: 0 },
    ], { loss: isCls ? 'binary_crossentropy' : 'mse', learningRate: 0.01, seed: 100 });
    const t1 = process.hrtime.bigint();
    const hist = await net.fit(Xtr, ytr.map(norm), {
      epochs: 25, batchSize: 32, validationData: [Xva, yva.map(norm)],
    });
    const nnMs = Number(process.hrtime.bigint() - t1) / 1e6;
    const rawVa = net.predict(Xva);
    const predVa = isCls ? rawVa : rawVa.map((v) => v * ((yHi - yLo) || 1) + yLo);
    const s = isCls
      ? classificationScores(yva, predVa.map((p) => (p >= 0.5 ? 1 : 0)), predVa)
      : regressionScores(yva, predVa);
    const main = isCls ? s.정확도 : s['R²'];
    console.log(`    ${pad('신경망', 12)} ${isCls ? '정확도' : 'R²'} `
      + `${padL(main.toFixed(4), 8)}  손실 ${hist.loss[0].toFixed(4)}→`
      + `${hist.loss[hist.loss.length - 1].toFixed(4)}  ${padL(nnMs.toFixed(0), 5)}ms`);
    if (!Number.isFinite(main)) problems.push(`${key}/신경망: 점수가 NaN`);
    if (!Number.isFinite(hist.loss[hist.loss.length - 1])) {
      problems.push(`${key}/신경망: 손실이 NaN 으로 터졌다`);
    }
    if (nnMs > 12000) problems.push(`${key}/신경망: ${nnMs.toFixed(0)}ms — 너무 느리다`);

    /* encodeRow 도 확인 */
    const vec = prep.encodeRow(prep.typicalRow);
    if (vec.length !== prep.featureNames.length || !vec.every(Number.isFinite)) {
      problems.push(`${key}: encodeRow 결과가 이상하다`);
    }
  }
}

console.log(`\n${'═'.repeat(78)}`);
if (problems.length) {
  console.log(`문제 ${problems.length}건:`);
  problems.forEach((p) => console.log(`  ✗ ${p}`));
} else {
  console.log('모든 조합에서 문제 없음.');
}
process.exit(problems.length ? 1 : 0);
