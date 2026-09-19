/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 머신러닝 모델들 — 브라우저에서 실제로 학습한다.
 *
 * 수업에서 다루는 모델들을 갈래별로 직접 구현했다.
 * 파이썬 사이킷런과 계산 방식을 최대한 같게 맞췄으므로,
 * scikit-learn 으로 돌린 점수와 비슷한 값이 나온다. (완전히 같지는 않다)
 *
 * 트리 계열은 「히스토그램 방식」으로 만들었다.
 * 값을 미리 32칸 구간으로 나눠 두고 그 구간 경계만 후보로 삼는 방법인데,
 * LightGBM 이 쓰는 방식과 같고 교실 PC 에서도 순식간에 학습이 끝난다.
 */
import { makeRng } from './rng.js';
import { mean, quantile } from './frame.js';
import { tick } from './tick.js';

/* ── 선형 계열 ──────────────────────────────────────────── */

/** 연립방정식 A w = b 를 가우스 소거법으로 푼다 (정규방정식용) */
function solveLinearSystem(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c += 1) {
    // 부분 피벗 — 절댓값이 가장 큰 행을 위로 올려 계산이 튀지 않게 한다
    let piv = c;
    for (let r = c + 1; r < n; r += 1) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-12) continue;
    [M[c], M[piv]] = [M[piv], M[c]];
    const d = M[c][c];
    for (let j = c; j <= n; j += 1) M[c][j] /= d;
    for (let r = 0; r < n; r += 1) {
      if (r === c) continue;
      const f = M[r][c];
      if (!f) continue;
      for (let j = c; j <= n; j += 1) M[r][j] -= f * M[c][j];
    }
  }
  return M.map((row) => row[n]);
}

/**
 * 선형 회귀 / 릿지 회귀.
 * alpha=0 이면 LinearRegression, alpha>0 이면 Ridge 와 같다.
 * 정규방정식 (XᵀX + αI)w = Xᵀy 를 직접 풀어 한 번에 답을 구한다.
 */
export class LinearRegression {
  constructor({ alpha = 0 } = {}) { this.alpha = alpha; }

  fit(X, y) {
    const n = X.length;
    const d = X[0].length;
    // 절편은 규제(alpha)에서 빼야 하므로, 평균을 뺀 뒤 따로 계산한다
    this.xMean = Array.from({ length: d }, (_, j) => mean(X.map((r) => r[j])));
    this.yMean = mean(y);
    const Xc = X.map((r) => r.map((v, j) => v - this.xMean[j]));
    const yc = y.map((v) => v - this.yMean);

    const A = Array.from({ length: d }, () => new Array(d).fill(0));
    const b = new Array(d).fill(0);
    for (let i = 0; i < n; i += 1) {
      const row = Xc[i];
      for (let j = 0; j < d; j += 1) {
        b[j] += row[j] * yc[i];
        for (let k = j; k < d; k += 1) A[j][k] += row[j] * row[k];
      }
    }
    for (let j = 0; j < d; j += 1) {
      for (let k = 0; k < j; k += 1) A[j][k] = A[k][j];
      A[j][j] += this.alpha;
    }
    this.coef = solveLinearSystem(A, b);
    this.intercept = this.yMean
      - this.coef.reduce((a, w, j) => a + w * this.xMean[j], 0);
    return this;
  }

  predict(X) {
    return X.map((r) => this.intercept + r.reduce((a, v, j) => a + v * this.coef[j], 0));
  }

  importances() { return this.coef.map(Math.abs); }
}

/**
 * 라쏘 / 엘라스틱넷 — 좌표 하강법(coordinate descent).
 * 가중치를 하나씩 돌아가며 최적값으로 바꾸는데, L1 규제 때문에
 * 쓸모없는 변수의 가중치가 정확히 0 이 된다. (변수 선택 효과)
 */
export class ElasticNetRegression {
  constructor({ alpha = 1, l1Ratio = 1, maxIter = 300, tol = 1e-5 } = {}) {
    Object.assign(this, { alpha, l1Ratio, maxIter, tol });
  }

  fit(X, y) {
    const n = X.length;
    const d = X[0].length;
    this.xMean = Array.from({ length: d }, (_, j) => mean(X.map((r) => r[j])));
    this.yMean = mean(y);
    const Xc = X.map((r) => r.map((v, j) => v - this.xMean[j]));
    const yc = y.map((v) => v - this.yMean);

    const w = new Array(d).fill(0);
    const z = Array.from({ length: d }, (_, j) => Xc.reduce((a, r) => a + r[j] * r[j], 0));
    const resid = yc.slice();
    const l1 = this.alpha * this.l1Ratio;
    const l2 = this.alpha * (1 - this.l1Ratio);

    for (let it = 0; it < this.maxIter; it += 1) {
      let maxMove = 0;
      for (let j = 0; j < d; j += 1) {
        if (z[j] === 0) continue;
        // j 번째 변수를 뺀 상태의 잔차와 x_j 의 내적
        let rho = 0;
        for (let i = 0; i < n; i += 1) rho += Xc[i][j] * (resid[i] + w[j] * Xc[i][j]);
        // 소프트 임계값 — L1 규제가 작은 가중치를 0 으로 눌러 버린다
        const num = Math.sign(rho) * Math.max(Math.abs(rho) / n - l1, 0);
        const newW = num / (z[j] / n + l2);
        const move = newW - w[j];
        if (move !== 0) {
          for (let i = 0; i < n; i += 1) resid[i] -= move * Xc[i][j];
          w[j] = newW;
          maxMove = Math.max(maxMove, Math.abs(move));
        }
      }
      if (maxMove < this.tol) break;
    }
    this.coef = w;
    this.intercept = this.yMean - w.reduce((a, v, j) => a + v * this.xMean[j], 0);
    return this;
  }

  predict(X) {
    return X.map((r) => this.intercept + r.reduce((a, v, j) => a + v * this.coef[j], 0));
  }

  importances() { return this.coef.map(Math.abs); }
}

/**
 * 로지스틱 회귀 — 경사 하강법.
 * 선형식의 결과를 시그모이드에 넣어 0~1 확률로 바꾼다.
 */
export class LogisticRegression {
  constructor({ C = 1, lr = 0.5, maxIter = 400 } = {}) {
    Object.assign(this, { C, lr, maxIter });
    this.history = [];
  }

  fit(X, y) {
    const n = X.length;
    const d = X[0].length;
    this.coef = new Array(d).fill(0);
    this.intercept = 0;
    const l2 = 1 / (this.C * n);       // 사이킷런의 C 는 규제의 역수다
    this.history = [];

    for (let it = 0; it < this.maxIter; it += 1) {
      const gW = new Array(d).fill(0);
      let gB = 0;
      let loss = 0;
      for (let i = 0; i < n; i += 1) {
        const p = sigmoid(this.intercept + dot(X[i], this.coef));
        const err = p - y[i];
        gB += err;
        for (let j = 0; j < d; j += 1) gW[j] += err * X[i][j];
        loss -= y[i] * Math.log(p + 1e-12) + (1 - y[i]) * Math.log(1 - p + 1e-12);
      }
      for (let j = 0; j < d; j += 1) {
        this.coef[j] -= this.lr * (gW[j] / n + l2 * this.coef[j]);
      }
      this.intercept -= this.lr * (gB / n);
      if (it % 20 === 0) this.history.push(loss / n);
    }
    return this;
  }

  predictProba(X) {
    return X.map((r) => sigmoid(this.intercept + dot(r, this.coef)));
  }

  predict(X, threshold = 0.5) {
    return this.predictProba(X).map((p) => (p >= threshold ? 1 : 0));
  }

  importances() { return this.coef.map(Math.abs); }
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

/* ── KNN ────────────────────────────────────────────────── */

/**
 * K-최근접 이웃. 학습이랄 게 없고, 예측할 때 가장 가까운 k개를 찾아 답을 낸다.
 * 그래서 스케일링을 안 하면 단위가 큰 변수가 거리를 독차지해 버린다.
 */
export class KNeighbors {
  constructor({ k = 5, task = 'regression' } = {}) { Object.assign(this, { k, task }); }

  fit(X, y) { this.X = X; this.y = y; return this; }

  neighborsOf(row) {
    const k = Math.min(this.k, this.X.length);
    // 가장 가까운 k개만 남기는 작은 목록을 유지한다
    const best = [];
    for (let i = 0; i < this.X.length; i += 1) {
      const p = this.X[i];
      let d2 = 0;
      for (let j = 0; j < p.length; j += 1) { const t = p[j] - row[j]; d2 += t * t; }
      if (best.length < k) {
        best.push({ i, d2 });
        if (best.length === k) best.sort((a, b) => a.d2 - b.d2);
      } else if (d2 < best[k - 1].d2) {
        best[k - 1] = { i, d2 };
        for (let m = k - 1; m > 0 && best[m].d2 < best[m - 1].d2; m -= 1) {
          [best[m], best[m - 1]] = [best[m - 1], best[m]];
        }
      }
    }
    return best;
  }

  predict(X) {
    return X.map((row) => {
      const nb = this.neighborsOf(row);
      if (this.task === 'regression') {
        return nb.reduce((a, o) => a + this.y[o.i], 0) / nb.length;
      }
      const ones = nb.reduce((a, o) => a + (this.y[o.i] === 1 ? 1 : 0), 0);
      return ones * 2 >= nb.length ? 1 : 0;
    });
  }

  predictProba(X) {
    return X.map((row) => {
      const nb = this.neighborsOf(row);
      return nb.reduce((a, o) => a + (this.y[o.i] === 1 ? 1 : 0), 0) / nb.length;
    });
  }

  importances() { return null; }
}

/* ── 트리 계열 ──────────────────────────────────────────── */

/**
 * 값을 미리 구간(bin)으로 나눠 둔다.
 * 이렇게 하면 분기 후보가 최대 31개로 줄어 트리 학습이 아주 빨라진다.
 */
export function makeBinner(X, nBins = 64) {
  const d = X[0].length;
  const n = X.length;
  const edges = [];
  for (let j = 0; j < d; j += 1) {
    const col = X.map((r) => r[j]).sort((a, b) => a - b);
    const lo = col[0];
    const hi = col[col.length - 1];
    const cuts = [];
    for (let q = 1; q < nBins; q += 1) {
      const v = quantile(col, q / nBins);
      /* 경계가 최솟값과 같으면 왼쪽 구간이 비어 버려 분기에 쓸 수 없다.
       * 0/1 로 된 원-핫 컬럼처럼 한쪽 값이 몰려 있으면 이런 경계가 잔뜩 나오는데,
       * 이것을 걸러 내지 않으면 그 컬럼으로는 아예 나눌 수 없게 된다. */
      if (v <= lo) continue;
      if (v > hi) break;
      if (!cuts.length || v > cuts[cuts.length - 1]) cuts.push(v);
    }
    edges.push(Float64Array.from(cuts));
  }
  const maxBins = Math.max(2, ...edges.map((e) => e.length + 1));
  const binIdx = new Uint8Array(n * d);
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < d; j += 1) binIdx[i * d + j] = binOf(edges[j], X[i][j]);
  }
  return { edges, binIdx, d, nBins: maxBins };
}

/** 값 하나가 몇 번째 구간인지 — 경계보다 작으면 그 구간 */
function binOf(cuts, x) {
  let lo = 0;
  let hi = cuts.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (x < cuts[mid]) hi = mid; else lo = mid + 1;
  }
  return lo;
}

/**
 * 결정 트리 (CART).
 * 회귀는 「나눈 뒤 오차제곱합이 가장 많이 줄어드는 곳」,
 * 분류는 「지니 불순도가 가장 많이 줄어드는 곳」에서 자른다.
 */
export class DecisionTree {
  constructor({
    task = 'regression', maxDepth = 5, minSamplesSplit = 2, minSamplesLeaf = 1,
    maxFeatures = null, seed = 0,
  } = {}) {
    Object.assign(this, { task, maxDepth, minSamplesSplit, minSamplesLeaf, maxFeatures, seed });
  }

  /** binner 를 넘기면 다시 만들지 않고 함께 쓴다 (랜덤 포레스트에서 중요) */
  fit(X, y, { binner = null, rows = null, sampleWeight = null } = {}) {
    this.binner = binner || makeBinner(X);
    this.d = this.binner.d;
    this.nFeatures = this.d;
    this.rng = makeRng(this.seed + 1);
    this.gainByFeature = new Array(this.d).fill(0);
    const idx = rows || Int32Array.from({ length: X.length }, (_, i) => i);
    this.y = y;
    this.w = sampleWeight;
    this.root = this.build(idx, 0);
    const total = this.gainByFeature.reduce((a, b) => a + b, 0);
    this.featureImportances = total > 0
      ? this.gainByFeature.map((g) => g / total)
      : this.gainByFeature.slice();
    return this;
  }

  leafValue(idx) {
    if (this.task === 'regression') {
      let s = 0;
      let w = 0;
      for (const i of idx) {
        const wi = this.w ? this.w[i] : 1;
        s += this.y[i] * wi; w += wi;
      }
      return w ? s / w : 0;
    }
    let ones = 0;
    let w = 0;
    for (const i of idx) {
      const wi = this.w ? this.w[i] : 1;
      ones += (this.y[i] === 1 ? wi : 0); w += wi;
    }
    return w ? ones / w : 0;                 // 1일 확률
  }

  build(idx, depth) {
    const value = this.leafValue(idx);
    if (depth >= this.maxDepth || idx.length < this.minSamplesSplit) {
      return { leaf: true, value, n: idx.length };
    }
    const split = this.bestSplit(idx);
    if (!split) return { leaf: true, value, n: idx.length };

    this.gainByFeature[split.j] += split.gain;
    const { binIdx, d } = this.binner;
    const left = [];
    const right = [];
    for (const i of idx) {
      (binIdx[i * d + split.j] <= split.bin ? left : right).push(i);
    }
    return {
      leaf: false,
      j: split.j,
      threshold: this.binner.edges[split.j][split.bin],
      n: idx.length,
      value,
      left: this.build(left, depth + 1),
      right: this.build(right, depth + 1),
    };
  }

  /** 후보 변수마다 구간별 합계를 모아 한 번에 최적 분기점을 찾는다 */
  bestSplit(idx) {
    const { binIdx, d, nBins } = this.binner;
    const feats = this.pickFeatures();
    const cnt = new Float64Array(nBins);
    const sum = new Float64Array(nBins);
    let best = null;

    for (const j of feats) {
      cnt.fill(0); sum.fill(0);
      let totalN = 0;
      let totalS = 0;
      for (const i of idx) {
        const b = binIdx[i * d + j];
        const wi = this.w ? this.w[i] : 1;
        const yi = this.task === 'regression' ? this.y[i] : (this.y[i] === 1 ? 1 : 0);
        cnt[b] += wi; sum[b] += yi * wi;
        totalN += wi; totalS += yi * wi;
      }
      if (totalN <= 0) continue;
      let nL = 0;
      let sL = 0;
      const lastBin = nBins - 1;
      for (let b = 0; b < lastBin; b += 1) {
        nL += cnt[b]; sL += sum[b];
        const nR = totalN - nL;
        if (nL < this.minSamplesLeaf || nR < this.minSamplesLeaf) continue;
        const sR = totalS - sL;
        // 회귀·분류 모두 (합²/개수) 를 크게 만드는 쪽이 좋은 분기다.
        //  · 회귀: 오차제곱합 감소량과 같은 식
        //  · 분류: 지니 불순도 감소량과 같은 식 (0/1 타깃일 때)
        const gain = (sL * sL) / nL + (sR * sR) / nR - (totalS * totalS) / totalN;
        if (gain > 1e-12 && (!best || gain > best.gain)) best = { j, bin: b, gain };
      }
    }
    return best;
  }

  pickFeatures() {
    if (!this.maxFeatures || this.maxFeatures >= this.d) {
      return Array.from({ length: this.d }, (_, j) => j);
    }
    const pool = Array.from({ length: this.d }, (_, j) => j);
    const out = [];
    for (let k = 0; k < this.maxFeatures; k += 1) {
      const p = Math.floor(this.rng() * pool.length);
      out.push(pool[p]);
      pool.splice(p, 1);
    }
    return out;
  }

  predictOne(row) {
    let node = this.root;
    while (!node.leaf) node = (row[node.j] < node.threshold) ? node.left : node.right;
    return node.value;
  }

  predict(X) {
    const raw = X.map((r) => this.predictOne(r));
    return this.task === 'regression' ? raw : raw.map((p) => (p >= 0.5 ? 1 : 0));
  }

  predictProba(X) { return X.map((r) => this.predictOne(r)); }

  importances() { return this.featureImportances; }

  /** 트리의 깊이와 잎 개수 — 과적합 이야기를 할 때 보여 주면 좋다 */
  stats() {
    let leaves = 0;
    let depth = 0;
    const walk = (node, dep) => {
      depth = Math.max(depth, dep);
      if (node.leaf) { leaves += 1; return; }
      walk(node.left, dep + 1); walk(node.right, dep + 1);
    };
    walk(this.root, 0);
    return { leaves, depth };
  }
}

/**
 * 랜덤 포레스트 — 결정 트리 여러 개의 평균.
 * ① 행을 무작위로 뽑아(부트스트랩) 트리마다 조금 다른 데이터를 준다
 * ② 분기할 때마다 변수도 일부만 후보로 준다
 * → 트리들이 서로 다른 실수를 하게 되고, 평균을 내면 실수가 상쇄된다.
 */
export class RandomForest {
  constructor({
    task = 'regression', nEstimators = 100, maxDepth = 10,
    minSamplesLeaf = 1, maxFeatures = 'sqrt', seed = 100,
  } = {}) {
    Object.assign(this, { task, nEstimators, maxDepth, minSamplesLeaf, maxFeatures, seed });
  }

  async fit(X, y, { onProgress = null } = {}) {
    const n = X.length;
    const d = X[0].length;
    const binner = makeBinner(X);
    const mf = resolveMaxFeatures(this.maxFeatures, d);
    this.trees = [];
    const rng = makeRng(this.seed);

    for (let t = 0; t < this.nEstimators; t += 1) {
      const rows = new Int32Array(n);
      for (let i = 0; i < n; i += 1) rows[i] = Math.floor(rng() * n);   // 부트스트랩
      const tree = new DecisionTree({
        task: this.task,
        maxDepth: this.maxDepth,
        minSamplesLeaf: this.minSamplesLeaf,
        maxFeatures: mf,
        seed: this.seed + t * 131,
      });
      tree.fit(X, y, { binner, rows });
      this.trees.push(tree);
      if (onProgress && (t % 5 === 4 || t === this.nEstimators - 1)) {
        onProgress((t + 1) / this.nEstimators);
        await tick();
      }
    }
    // 특성 중요도는 트리들의 평균
    this.featureImportances = new Array(d).fill(0);
    this.trees.forEach((tr) => tr.featureImportances.forEach((v, j) => {
      this.featureImportances[j] += v / this.trees.length;
    }));
    return this;
  }

  predictProba(X) {
    return X.map((row) => {
      let s = 0;
      for (const t of this.trees) s += t.predictOne(row);
      return s / this.trees.length;
    });
  }

  predict(X) {
    const raw = this.predictProba(X);
    return this.task === 'regression' ? raw : raw.map((p) => (p >= 0.5 ? 1 : 0));
  }

  importances() { return this.featureImportances; }
}

/**
 * 그래디언트 부스팅 — 앞 모델이 틀린 만큼(잔차)을 다음 트리가 이어서 맞춘다.
 * XGBoost · LightGBM · AdaBoost 도 모두 이 「부스팅」 갈래에 속한다.
 */
export class GradientBoosting {
  constructor({
    task = 'regression', nEstimators = 100, learningRate = 0.1,
    maxDepth = 3, seed = 100,
  } = {}) {
    Object.assign(this, { task, nEstimators, learningRate, maxDepth, seed });
  }

  async fit(X, y, { onProgress = null } = {}) {
    const n = X.length;
    const d = X[0].length;
    const binner = makeBinner(X);
    this.trees = [];
    this.trainLoss = [];

    if (this.task === 'regression') {
      this.init = mean(y);
      const pred = new Array(n).fill(this.init);
      for (let t = 0; t < this.nEstimators; t += 1) {
        const resid = y.map((v, i) => v - pred[i]);
        const tree = new DecisionTree({
          task: 'regression', maxDepth: this.maxDepth, seed: this.seed + t,
        });
        tree.fit(X, resid, { binner });
        const step = tree.predict(X);
        for (let i = 0; i < n; i += 1) pred[i] += this.learningRate * step[i];
        this.trees.push(tree);
        this.trainLoss.push(mean(y.map((v, i) => (v - pred[i]) ** 2)));
        if (onProgress && (t % 5 === 4 || t === this.nEstimators - 1)) {
          onProgress((t + 1) / this.nEstimators); await tick();
        }
      }
    } else {
      const p0 = Math.min(Math.max(mean(y), 1e-6), 1 - 1e-6);
      this.init = Math.log(p0 / (1 - p0));                 // 로그 오즈에서 출발
      const score = new Array(n).fill(this.init);
      for (let t = 0; t < this.nEstimators; t += 1) {
        const prob = score.map(sigmoid);
        const grad = y.map((v, i) => v - prob[i]);         // 음의 기울기
        const tree = new DecisionTree({
          task: 'regression', maxDepth: this.maxDepth, seed: this.seed + t,
        });
        tree.fit(X, grad, { binner });
        // 로지스틱 손실은 잎마다 뉴턴 한 걸음으로 값을 다시 정한다
        relabelLeaves(tree, X, grad, prob);
        const step = tree.predict(X);
        for (let i = 0; i < n; i += 1) score[i] += this.learningRate * step[i];
        this.trees.push(tree);
        this.trainLoss.push(mean(y.map((v, i) => {
          const p = sigmoid(score[i]);
          return -(v * Math.log(p + 1e-12) + (1 - v) * Math.log(1 - p + 1e-12));
        })));
        if (onProgress && (t % 5 === 4 || t === this.nEstimators - 1)) {
          onProgress((t + 1) / this.nEstimators); await tick();
        }
      }
    }

    this.featureImportances = new Array(d).fill(0);
    this.trees.forEach((tr) => tr.featureImportances.forEach((v, j) => {
      this.featureImportances[j] += v / this.trees.length;
    }));
    return this;
  }

  rawScore(X) {
    return X.map((row) => {
      let s = this.init;
      for (const t of this.trees) s += this.learningRate * t.predictOne(row);
      return s;
    });
  }

  predictProba(X) { return this.rawScore(X).map(sigmoid); }

  predict(X) {
    if (this.task === 'regression') return this.rawScore(X);
    return this.predictProba(X).map((p) => (p >= 0.5 ? 1 : 0));
  }

  importances() { return this.featureImportances; }
}

/** 잎마다 값을 Σ기울기 / Σp(1−p) 로 다시 정한다 (분류 부스팅의 뉴턴 갱신) */
function relabelLeaves(tree, X, grad, prob) {
  const buckets = new Map();
  for (let i = 0; i < X.length; i += 1) {
    let node = tree.root;
    while (!node.leaf) node = (X[i][node.j] < node.threshold) ? node.left : node.right;
    if (!buckets.has(node)) buckets.set(node, []);
    buckets.get(node).push(i);
  }
  for (const [node, idx] of buckets) {
    let num = 0;
    let den = 0;
    for (const i of idx) { num += grad[i]; den += prob[i] * (1 - prob[i]); }
    node.value = den > 1e-9 ? num / den : 0;
  }
}

function resolveMaxFeatures(spec, d) {
  if (spec === 'sqrt') return Math.max(1, Math.round(Math.sqrt(d)));
  if (spec === 'log2') return Math.max(1, Math.round(Math.log2(d)));
  if (spec === 'third') return Math.max(1, Math.round(d / 3));
  if (typeof spec === 'number') return Math.max(1, Math.min(d, Math.round(spec)));
  return d;
}

export { tick, sigmoid };
