/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 라이브러리 자체 점검 — node test/lib.test.mjs
 *
 * 브라우저를 열지 않고도 계산이 맞는지, 속도가 괜찮은지 확인한다.
 * 사이킷런으로 구한 값과 비교할 수 있는 것은 숫자를 적어 두었다.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { Frame, pearson, mode, quantile, histogram, boxStats } from '../src/lib/frame.js';
import {
  MinMaxScaler, StandardScaler, RobustScaler, trainTestSplit, fillValue,
} from '../src/lib/preprocess.js';
import {
  LinearRegression, ElasticNetRegression, LogisticRegression, KNeighbors,
  DecisionTree, RandomForest, GradientBoosting, makeBinner,
} from '../src/lib/models.js';
import {
  mse, r2Score, mae, accuracyScore, precisionScore, recallScore, f1Score,
  rocAucScore, confusionMatrix,
} from '../src/lib/metrics.js';
import { NeuralNetwork } from '../src/lib/nn.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
let pass = 0;
let fail = 0;

function ok(name, cond, extra = '') {
  if (cond) { pass += 1; console.log(`  ✓ ${name}${extra ? ` — ${extra}` : ''}`); } else {
    fail += 1; console.log(`  ✗ ${name}${extra ? ` — ${extra}` : ''}`);
  }
}
const near = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol;
const timed = (label, fn) => {
  const t = process.hrtime.bigint();
  const out = fn();
  const ms = Number(process.hrtime.bigint() - t) / 1e6;
  return { out, ms, label };
};

console.log('\n[1] Frame — CSV 읽기와 기초 통계');
const car = Frame.fromCsv(readFileSync(join(ROOT, 'data/car_resale.csv'), 'utf8'));
ok('행 개수 960', car.length === 960, `${car.length}`);
ok('열 개수 11', car.columns.length === 11);
ok('maker 결측 44개', car.missing('maker') === 44, `${car.missing('maker')}`);
ok('body_color 결측 36개', car.missing('body_color') === 36, `${car.missing('body_color')}`);
ok('resale_price_manwon 은 수치형', car.isNumeric('resale_price_manwon'));
ok('maker 는 범주형', !car.isNumeric('maker'));
ok('예측 목표에는 결측치가 없다', car.missing('resale_price_manwon') === 0);

const desc = car.describe(['used_years']);
ok('used_years 범위 0~15', desc[0].min === 0 && desc[0].max === 15,
  `${desc[0].min}~${desc[0].max}`);
ok('중앙값과 사분위수 순서가 맞다',
  desc[0].q1 <= desc[0].median && desc[0].median <= desc[0].q3);

/* 앱의 groupby 가 pandas 와 같은 답을 내는지 — 기대값은 pandas 로 미리 구해 적어 두었다.
 *   d.groupby('power_source')['resale_price_manwon'].mean().idxmax()          → '경유'
 *   d.groupby(['maker','trim_level'])['resale_price_manwon'].mean().idxmax()  → ('가온', '프리미엄') */
const grp = car.groupby(['power_source'], ['resale_price_manwon'], 'mean');
ok('power_source 평균 시세 1등 = 경유 (pandas 와 같다)', grp.rows[0].label === '경유', grp.rows[0].label);

const grp2 = car.groupby(['maker', 'trim_level'], ['resale_price_manwon'], 'mean');
ok('maker·trim_level 조합 1등 = 가온 · 프리미엄 (pandas 와 같다)',
  grp2.rows[0].label === '가온 · 프리미엄', grp2.rows[0].label);

console.log('\n[2] 상관계수 — 데이터를 만들 때 설계한 성질이 살아 있는지');
// tools/make_data.py 가 일부러 만들어 넣은 관계들이다. 이게 깨지면 수업이 성립하지 않는다.
const rUsed = pearson(car.col('used_years'), car.col('resale_price_manwon'));
ok('사용연수는 시세와 강한 음의 상관 (r ≤ −0.7)', rUsed <= -0.7, rUsed.toFixed(3));
const rUD = pearson(car.col('used_years'), car.col('total_distance_km'));
ok('사용연수 ↔ 주행거리 다중공선성 (r ≥ 0.8)', rUD >= 0.8, rUD.toFixed(3));
const rDP = pearson(car.col('displacement_cc'), car.col('max_power_ps'));
ok('배기량 ↔ 출력 다중공선성 (r ≥ 0.8)', rDP >= 0.8, rDP.toFixed(3));
const rFuel = pearson(car.col('fuel_economy_kmpl'), car.col('resale_price_manwon'));
ok('연비는 시세와 거의 무관 (|r| ≤ 0.2)', Math.abs(rFuel) <= 0.2, rFuel.toFixed(3));

console.log('\n[3] 보조 함수');
ok('mode', mode(['a', 'b', 'b', null, 'b', 'a']) === 'b');
ok('quantile 중앙값', quantile([1, 2, 3, 4], 0.5) === 2.5);
ok('quantile 선형보간', near(quantile([0, 10], 0.25), 2.5));
ok('histogram 합 = 개수', histogram([1, 2, 3, 4, 5], 5).counts.reduce((a, b) => a + b) === 5);
const bs = boxStats([1, 2, 3, 4, 5, 6, 7, 8, 9, 100]);
ok('boxStats 가 100 을 이상치로 잡는다', bs.outliers.includes(100));
ok('fillna mean', near(fillValue([1, 2, NaN, 3], 'mean'), 2));
ok('fillna median', near(fillValue([1, 2, NaN, 100], 'median'), 2));

console.log('\n[4] 스케일러');
const Xs = [[0, 10], [5, 20], [10, 30]];
const mm = new MinMaxScaler().fit(Xs);
ok('MinMax 최솟값 0 · 최댓값 1', JSON.stringify(mm.transform(Xs))
  === JSON.stringify([[0, 0], [0.5, 0.5], [1, 1]]));
const ss = new StandardScaler().fit(Xs);
const st = ss.transform(Xs);
ok('Standard 평균 0', near(st.reduce((a, r) => a + r[0], 0) / 3, 0, 1e-12));
ok('Standard 표준편차 1 (모집단 기준, 사이킷런과 같다)',
  near(Math.sqrt(st.reduce((a, r) => a + r[0] ** 2, 0) / 3), 1, 1e-12));
const rs = new RobustScaler().fit([[1], [2], [3], [4], [100]]);
ok('Robust 중앙값이 0', near(rs.transform([[3]])[0][0], 0));

console.log('\n[5] train_test_split');
const sp = trainTestSplit(1000, { testSize: 0.3, seed: 100 });
ok('검증 300 · 훈련 700', sp.valid.length === 300 && sp.train.length === 700);
ok('겹치는 행이 없다', new Set([...sp.train, ...sp.valid]).size === 1000);
const sp2 = trainTestSplit(1000, { testSize: 0.3, seed: 100 });
ok('같은 seed 면 같은 결과', JSON.stringify(sp.valid) === JSON.stringify(sp2.valid));
const sp3 = trainTestSplit(1000, { testSize: 0.3, seed: 7 });
ok('다른 seed 면 다른 결과', JSON.stringify(sp.valid) !== JSON.stringify(sp3.valid));
const labels = Array.from({ length: 1000 }, (_, i) => (i < 200 ? 1 : 0));
const sp4 = trainTestSplit(1000, { testSize: 0.3, seed: 1, stratify: labels });
const ratio = (idx) => idx.filter((i) => labels[i] === 1).length / idx.length;
ok('stratify 가 비율을 유지한다 (0.2)',
  near(ratio(sp4.train), 0.2, 0.01) && near(ratio(sp4.valid), 0.2, 0.01),
  `train ${ratio(sp4.train).toFixed(3)} · valid ${ratio(sp4.valid).toFixed(3)}`);

console.log('\n[6] 평가지표');
const yT = [3, -0.5, 2, 7];
const yP = [2.5, 0.0, 2, 8];
ok('mse 0.375 (사이킷런 예제와 같은 값)', near(mse(yT, yP), 0.375));
ok('mae 0.5', near(mae(yT, yP), 0.5));
ok('r2 0.9486', near(r2Score(yT, yP), 0.9486081370449679, 1e-9));
const cT = [0, 1, 1, 0, 1, 1, 0, 0];
const cP = [0, 1, 0, 0, 1, 1, 1, 0];
const cm = confusionMatrix(cT, cP);
ok('혼동 행렬 TN3 FP1 FN1 TP3',
  cm.tn === 3 && cm.fp === 1 && cm.fn === 1 && cm.tp === 3, JSON.stringify(cm));
ok('정확도 0.75', near(accuracyScore(cT, cP), 0.75));
ok('정밀도 0.75', near(precisionScore(cT, cP), 0.75));
ok('재현율 0.75', near(recallScore(cT, cP), 0.75));
ok('F1 0.75', near(f1Score(cT, cP), 0.75));
ok('roc_auc 완벽 분리는 1.0', near(rocAucScore([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1));
ok('roc_auc 무작위는 0.5', near(rocAucScore([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5));

console.log('\n[7] 선형 모델 — 정답을 아는 문제로');
// y = 3 x1 - 2 x2 + 5
const Xl = [];
const yl = [];
for (let i = 0; i < 60; i += 1) {
  const a = (i % 10) - 5;
  const b = Math.floor(i / 10) - 3;
  Xl.push([a, b]); yl.push(3 * a - 2 * b + 5);
}
const lr = new LinearRegression().fit(Xl, yl);
ok('LinearRegression 계수 [3, -2]',
  near(lr.coef[0], 3, 1e-8) && near(lr.coef[1], -2, 1e-8),
  `[${lr.coef.map((v) => v.toFixed(6)).join(', ')}]`);
ok('절편 5', near(lr.intercept, 5, 1e-8), lr.intercept.toFixed(6));
ok('R² = 1', near(r2Score(yl, lr.predict(Xl)), 1, 1e-12));

const ridge = new LinearRegression({ alpha: 100 }).fit(Xl, yl);
ok('Ridge 는 계수를 0 쪽으로 줄인다',
  Math.abs(ridge.coef[0]) < Math.abs(lr.coef[0]),
  `${ridge.coef[0].toFixed(3)} < ${lr.coef[0].toFixed(3)}`);

// x2 는 정답과 무관한 잡음 → 라쏘가 0 으로 만들어야 한다
const Xn = Xl.map(([a], i) => [a, ((i * 37) % 11) - 5]);
const yn = Xn.map(([a]) => 3 * a + 5);
const lasso = new ElasticNetRegression({ alpha: 1.5, l1Ratio: 1 }).fit(Xn, yn);
ok('Lasso 가 쓸모없는 변수 계수를 정확히 0 으로',
  lasso.coef[1] === 0 && Math.abs(lasso.coef[0]) > 2,
  `[${lasso.coef.map((v) => v.toFixed(4)).join(', ')}]`);

console.log('\n[8] 로지스틱 회귀');
const Xc = [];
const yc = [];
for (let i = 0; i < 200; i += 1) {
  const v = (i / 100) - 1;
  Xc.push([v]); yc.push(v > 0 ? 1 : 0);
}
const log = new LogisticRegression({ C: 100, maxIter: 600 }).fit(Xc, yc);
ok('선형으로 나뉘는 문제를 거의 완벽히 맞힌다',
  accuracyScore(yc, log.predict(Xc)) > 0.97,
  accuracyScore(yc, log.predict(Xc)).toFixed(3));
ok('손실이 줄어든다',
  log.history[log.history.length - 1] < log.history[0],
  `${log.history[0].toFixed(4)} → ${log.history[log.history.length - 1].toFixed(4)}`);
ok('확률이 0~1 안에 있다', log.predictProba(Xc).every((p) => p >= 0 && p <= 1));

console.log('\n[9] 트리 — 계단 모양 함수를 잘 맞히는지');
const Xt = Array.from({ length: 200 }, (_, i) => [i / 200]);
const yt = Xt.map(([v]) => (v < 0.3 ? 1 : v < 0.7 ? 5 : 2));
const dt = new DecisionTree({ task: "regression", maxDepth: 4 }).fit(Xt, yt);
/* 히스토그램 방식이라 분기점을 구간 경계에서만 고를 수 있다.
 * 실제 경계(0.3, 0.7)와 살짝 어긋나 몇 점을 놓치는 것이 정상이다. */
ok('결정 트리가 계단 함수를 거의 그대로 재현', r2Score(yt, dt.predict(Xt)) > 0.96,
  r2Score(yt, dt.predict(Xt)).toFixed(4));
const st1 = dt.stats();
ok('트리 통계(깊이·잎)가 나온다', st1.depth >= 2 && st1.leaves >= 3, JSON.stringify(st1));

const deep = new DecisionTree({ task: 'regression', maxDepth: 20 }).fit(Xt, yt);
const shallow = new DecisionTree({ task: 'regression', maxDepth: 1 }).fit(Xt, yt);
ok('깊게 만들면 훈련 성능이 더 좋다 (과적합의 씨앗)',
  r2Score(yt, deep.predict(Xt)) >= r2Score(yt, shallow.predict(Xt)));

ok('binner 의 구간이 오름차순', (() => {
  const b = makeBinner(Xt, 8);
  return b.edges[0].every((v, i) => i === 0 || v > b.edges[0][i - 1]);
})());

console.log('\n[10] KNN');
const knn = new KNeighbors({ k: 1, task: 'regression' }).fit(Xt, yt);
ok('k=1 이면 훈련 데이터를 그대로 맞힌다', near(r2Score(yt, knn.predict(Xt)), 1, 1e-9));
const knnC = new KNeighbors({ k: 5, task: 'classification' }).fit(Xc, yc);
ok('KNN 분류 정확도', accuracyScore(yc, knnC.predict(Xc)) > 0.95,
  accuracyScore(yc, knnC.predict(Xc)).toFixed(3));

console.log('\n[11] 앙상블 — 실제 데이터로 학습 + 속도');
const { buildMatrix } = await import('../src/lib/pipeline.js');
// datasets.js 는 ?raw 임포트를 쓰므로 노드에서 못 읽는다. 최소 정보만 흉내 낸다.
const dsCar = {
  target: 'resale_price_manwon', frame: car, task: 'regression', derived: null,
  ordinal: null, idColumns: [],
};
const prep = buildMatrix(dsCar, { imputePreset: 'medianMode', encode: 'onehot' });
ok('전처리 후 행 960 · 특성 21', prep.X.length === 960 && prep.featureNames.length === 21,
  `${prep.X.length}행 ${prep.featureNames.length}열`);
ok('전처리 후 결측치가 없다', prep.X.every((r) => r.every(Number.isFinite)));

const s = trainTestSplit(prep.X.length, { testSize: 0.3, seed: 100 });
const sc = new StandardScaler().fit(s.train.map((i) => prep.X[i]));
const XtrS = sc.transform(s.train.map((i) => prep.X[i]));
const XvaS = sc.transform(s.valid.map((i) => prep.X[i]));
const ytr = s.train.map((i) => prep.y[i]);
const yva = s.valid.map((i) => prep.y[i]);

const bin = timed('makeBinner', () => makeBinner(XtrS));
console.log(`  · makeBinner ${bin.ms.toFixed(1)}ms`);

const one = timed('트리 1개', () =>
  new DecisionTree({ task: 'regression', maxDepth: 10, maxFeatures: 5 })
    .fit(XtrS, ytr, { binner: bin.out }));
console.log(`  · 결정 트리 1개(깊이 10) ${one.ms.toFixed(1)}ms`);

const forest = new RandomForest({
  task: 'regression', nEstimators: 100, maxDepth: 10, seed: 100,
});
const t0 = process.hrtime.bigint();
await forest.fit(XtrS, ytr);
const forestMs = Number(process.hrtime.bigint() - t0) / 1e6;
const rfR2 = r2Score(yva, forest.predict(XvaS));
console.log(`  · 랜덤 포레스트 100개 학습 ${forestMs.toFixed(0)}ms`);
ok('랜덤 포레스트 검증 R² > 0.8', rfR2 > 0.8, rfR2.toFixed(4));
ok('랜덤 포레스트 학습이 3초 안에 끝난다', forestMs < 3000, `${forestMs.toFixed(0)}ms`);
const impSum = forest.importances().reduce((a, b) => a + b, 0);
ok('특성 중요도 합이 1', near(impSum, 1, 1e-6), impSum.toFixed(6));

const gb = new GradientBoosting({
  task: 'regression', nEstimators: 100, learningRate: 0.1, maxDepth: 3, seed: 100,
});
const t1 = process.hrtime.bigint();
await gb.fit(XtrS, ytr);
const gbMs = Number(process.hrtime.bigint() - t1) / 1e6;
const gbR2 = r2Score(yva, gb.predict(XvaS));
console.log(`  · 그래디언트 부스팅 100개 학습 ${gbMs.toFixed(0)}ms`);
ok('부스팅 검증 R² > 0.8', gbR2 > 0.8, gbR2.toFixed(4));
ok('부스팅 학습이 3초 안에 끝난다', gbMs < 3000, `${gbMs.toFixed(0)}ms`);
ok('부스팅 훈련 손실이 계속 줄어든다',
  gb.trainLoss[gb.trainLoss.length - 1] < gb.trainLoss[0]);

console.log('\n[12] 분류 데이터로 앙상블');
const churn = Frame.fromCsv(readFileSync(join(ROOT, 'data/subscriber_churn.csv'), 'utf8'));
const dsCh = {
  target: 'left_service', frame: churn, task: 'classification', derived: null,
  ordinal: { plan_term: { 무약정: 0, '12개월': 1, '24개월': 2 } }, idColumns: [],
};
const pc = buildMatrix(dsCh, { imputePreset: 'medianMode', encode: 'onehot' });
ok('고객 이탈 데이터 전처리 성공', pc.X.length > 2000, `${pc.X.length}행 ${pc.featureNames.length}열`);
const sc2 = trainTestSplit(pc.X.length, { testSize: 0.3, seed: 100, stratify: pc.y });
const scal2 = new StandardScaler().fit(sc2.train.map((i) => pc.X[i]));
const Xtr2 = scal2.transform(sc2.train.map((i) => pc.X[i]));
const Xva2 = scal2.transform(sc2.valid.map((i) => pc.X[i]));
const ytr2 = sc2.train.map((i) => pc.y[i]);
const yva2 = sc2.valid.map((i) => pc.y[i]);

const rfc = new RandomForest({
  task: 'classification', nEstimators: 100, maxDepth: 10, seed: 100,
});
const t2 = process.hrtime.bigint();
await rfc.fit(Xtr2, ytr2);
const rfcMs = Number(process.hrtime.bigint() - t2) / 1e6;
console.log(`  · 랜덤 포레스트 분류 (${Xtr2.length}행) ${rfcMs.toFixed(0)}ms`);
const acc = accuracyScore(yva2, rfc.predict(Xva2));
const auc = rocAucScore(yva2, rfc.predictProba(Xva2));
ok('이탈 예측 정확도 > 0.7', acc > 0.7, acc.toFixed(4));
ok('이탈 예측 ROC AUC > 0.6', auc > 0.6, auc.toFixed(4));
ok('분류 학습도 4초 안에', rfcMs < 4000, `${rfcMs.toFixed(0)}ms`);

const logc = new LogisticRegression({ C: 1, maxIter: 400 });
const t3 = process.hrtime.bigint();
logc.fit(Xtr2, ytr2);
const logMs = Number(process.hrtime.bigint() - t3) / 1e6;
console.log(`  · 로지스틱 회귀 (${Xtr2.length}행 × ${pc.featureNames.length}열) ${logMs.toFixed(0)}ms`);
ok('로지스틱 회귀가 2초 안에', logMs < 2000, `${logMs.toFixed(0)}ms`);

console.log('\n[13] 신경망');
const nn = new NeuralNetwork(prep.featureNames.length, [
  { units: 32, activation: 'relu' },
  { units: 16, activation: 'relu' },
  { units: 1, activation: 'linear' },
], { loss: 'mse', learningRate: 0.01, seed: 100 });
// y 를 0~1 로 줄여야 학습이 안정적이다 (실제 앱에서도 같게 처리한다)
const yMax = Math.max(...ytr);
const t4 = process.hrtime.bigint();
const hist = await nn.fit(XtrS, ytr.map((v) => v / yMax), { epochs: 20, batchSize: 32 });
const nnMs = Number(process.hrtime.bigint() - t4) / 1e6;
console.log(`  · 신경망 20 epoch (${XtrS.length}행) ${nnMs.toFixed(0)}ms`);
ok('손실이 줄어든다', hist.loss[hist.loss.length - 1] < hist.loss[0] * 0.6,
  `${hist.loss[0].toFixed(5)} → ${hist.loss[hist.loss.length - 1].toFixed(5)}`);
const nnR2 = r2Score(yva, nn.predict(XvaS).map((v) => v * yMax));
ok('신경망 검증 R² > 0.6', nnR2 > 0.6, nnR2.toFixed(4));
ok('20 epoch 이 6초 안에', nnMs < 6000, `${nnMs.toFixed(0)}ms`);
ok('summary() 가 층 정보를 준다', nn.summary().length === 3
  && nn.summary()[0].params === prep.featureNames.length * 32 + 32);

console.log('\n[14] encodeRow — 새 데이터 변환 (새 데이터 예측)');
const rowVec = prep.encodeRow(prep.typicalRow);
ok('encodeRow 길이가 특성 개수와 같다', rowVec.length === prep.featureNames.length);
ok('encodeRow 결과가 모두 숫자', rowVec.every(Number.isFinite));
const oneHotSum = prep.featureNames
  .map((n, j) => (n.startsWith('maker_') ? rowVec[j] : 0)).reduce((a, b) => a + b, 0);
ok('원-핫 부분은 정확히 하나만 1', oneHotSum === 1, `합 ${oneHotSum}`);

console.log('\n[15] 데이터셋 11개가 모두 준비되는지');
const { META, DATASET_IDS } = await import('../src/data/meta.js');
const { buildDataset } = await import('../src/data/build.js');
const built = {};
DATASET_IDS.forEach((key) => {
  try {
    built[key] = buildDataset(key, META[key].files.map((f) =>
      readFileSync(join(ROOT, 'data', f), 'utf8')));
  } catch (err) {
    ok(`${key} 준비`, false, err.message);
  }
});
ok('10개 모두 준비됨', Object.keys(built).length === DATASET_IDS.length,
  `${Object.keys(built).length}개`);
ok('컬럼마다 설명이 붙어 있다',
  DATASET_IDS.every((key) => Object.keys(built[key]?.colDesc || {}).length > 0));
ok('주택(7회차)은 두 파일이 합쳐져 열이 12개',
  built.home && built.home.frame.columns.length === 12,
  `${built.home?.frame.columns.length}열`);
ok('주택 합친 뒤에도 1440행', built.home?.frame.length === 1440, `${built.home?.frame.length}행`);

/* 순서형 매핑에 빠진 값이 있으면 그 행이 NaN 이 되어 조용히 사라진다.
 * 눈에 띄지 않는 종류의 실수라 여기서 반드시 막는다. */
let ordinalBad = [];
DATASET_IDS.forEach((key) => {
  const m = META[key];
  const ds = built[key];
  if (!m.ordinal || !ds) return;
  Object.entries(m.ordinal).forEach(([col, map]) => {
    if (!ds.frame.columns.includes(col)) { ordinalBad.push(`${key}/${col} 컬럼 없음`); return; }
    const missing = [...new Set(ds.frame.values(col).map(String))].filter((v) => !(v in map));
    if (missing.length) ordinalBad.push(`${key}/${col}: ${missing.join(', ')}`);
  });
});
ok('순서형 매핑이 데이터의 모든 값을 덮는다', ordinalBad.length === 0, ordinalBad.join(' | '));

/* 전처리에서 행이 통째로 사라지면 학생이 이유를 알 수 없다 */
const lost = [];
DATASET_IDS.forEach((key) => {
  const ds = built[key];
  if (!ds) return;
  const p = buildMatrix(ds, { imputePreset: 'medianMode', encode: 'onehot' });
  if (p.summary.droppedRows > 0) lost.push(`${key}: ${p.summary.droppedRows}행`);
  if (!p.X.length) lost.push(`${key}: 남는 행이 0`);
});
ok('기본 전처리에서 잃는 행이 없다', lost.length === 0, lost.join(' | '));

console.log(`\n결과: 통과 ${pass} · 실패 ${fail}\n`);
process.exit(fail ? 1 : 0);
