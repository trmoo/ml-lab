/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 아주 작은 신경망 — keras 의 Dense 층 모델을 브라우저에서 돌린다.
 *
 * Sequential 모델에 Dense 층을 쌓는 것과 같은 구조이고,
 * 순전파 → 손실 계산 → 역전파 → 가중치 갱신(Adam) 을 그대로 구현했다.
 * 학습이 진행되는 동안 손실 곡선이 그려지므로,
 * 「epoch 을 늘리면 훈련 손실은 계속 내려가지만 검증 손실은 어느 순간부터 올라간다」
 * 는 과적합의 모습을 눈으로 볼 수 있다.
 */
import { makeNormalRng, shuffledIndex } from './rng.js';
import { tick } from './tick.js';

const ACT = {
  relu: { f: (z) => (z > 0 ? z : 0), df: (z, a) => (z > 0 ? 1 : 0) },
  sigmoid: { f: (z) => 1 / (1 + Math.exp(-z)), df: (z, a) => a * (1 - a) },
  tanh: { f: (z) => Math.tanh(z), df: (z, a) => 1 - a * a },
  linear: { f: (z) => z, df: () => 1 },
};

/** Dense 층 하나 — 입력 nIn 개를 받아 nOut 개를 낸다 */
class Dense {
  constructor(nIn, nOut, activation, normal) {
    this.nIn = nIn;
    this.nOut = nOut;
    this.act = ACT[activation] || ACT.linear;
    this.actName = activation;
    // He 초기화 — relu 를 쓸 때 신호가 죽거나 터지지 않게 분산을 맞춘다
    const scale = Math.sqrt(2 / nIn);
    this.W = Array.from({ length: nOut }, () =>
      Float64Array.from({ length: nIn }, () => normal() * scale));
    this.b = new Float64Array(nOut);
    // Adam 이 쓰는 두 개의 이동평균
    this.mW = this.W.map(() => new Float64Array(nIn));
    this.vW = this.W.map(() => new Float64Array(nIn));
    this.mb = new Float64Array(nOut);
    this.vb = new Float64Array(nOut);
    this.dropout = 0;
  }

  forward(x, training) {
    const z = new Float64Array(this.nOut);
    const a = new Float64Array(this.nOut);
    for (let o = 0; o < this.nOut; o += 1) {
      const w = this.W[o];
      let s = this.b[o];
      for (let i = 0; i < this.nIn; i += 1) s += w[i] * x[i];
      z[o] = s;
      a[o] = this.act.f(s);
    }
    // Dropout — 학습할 때만 일부 노드를 임시로 껐다 켠다 (과적합 방지)
    let mask = null;
    if (training && this.dropout > 0) {
      mask = new Float64Array(this.nOut);
      const keep = 1 - this.dropout;
      for (let o = 0; o < this.nOut; o += 1) {
        mask[o] = Math.random() < keep ? 1 / keep : 0;
        a[o] *= mask[o];
      }
    }
    return { x, z, a, mask };
  }
}

export class NeuralNetwork {
  /**
   * @param {number} nIn 입력 특성 개수
   * @param {Array<{units:number, activation:string, dropout?:number}>} layers
   * @param {'mse'|'binary_crossentropy'} loss
   */
  constructor(nIn, layers, {
    loss = 'mse', learningRate = 0.001, seed = 100,
  } = {}) {
    const normal = makeNormalRng(seed);
    this.layers = [];
    let prev = nIn;
    layers.forEach((spec) => {
      const layer = new Dense(prev, spec.units, spec.activation, normal);
      layer.dropout = spec.dropout || 0;
      this.layers.push(layer);
      prev = spec.units;
    });
    this.loss = loss;
    this.lr = learningRate;
    this.t = 0;
    this.history = { loss: [], valLoss: [] };
  }

  forward(x, training = false) {
    const caches = [];
    let cur = x;
    for (const layer of this.layers) {
      const c = layer.forward(cur, training);
      caches.push(c);
      cur = c.a;
    }
    return { out: cur, caches };
  }

  predict(X) {
    return X.map((row) => this.forward(row, false).out[0]);
  }

  /** 한 표본에 대한 기울기를 구해 층별 누적 기울기에 더한다 */
  backward(caches, yTrue, grads) {
    const last = this.layers.length - 1;
    const outA = caches[last].a;
    // 출력층 델타
    //   회귀(mse + linear)  : 2(ŷ − y)
    //   분류(BCE + sigmoid) : ŷ − y  ← 시그모이드 미분과 약분되어 아주 깔끔하다
    let delta = new Float64Array(outA.length);
    if (this.loss === 'binary_crossentropy') {
      for (let o = 0; o < outA.length; o += 1) delta[o] = outA[o] - yTrue;
    } else {
      for (let o = 0; o < outA.length; o += 1) {
        delta[o] = 2 * (outA[o] - yTrue) * this.layers[last].act.df(caches[last].z[o], outA[o]);
      }
    }

    for (let l = last; l >= 0; l -= 1) {
      const layer = this.layers[l];
      const cache = caches[l];
      if (cache.mask) for (let o = 0; o < delta.length; o += 1) delta[o] *= cache.mask[o];
      const g = grads[l];
      for (let o = 0; o < layer.nOut; o += 1) {
        const dz = delta[o];
        if (dz === 0) continue;
        const gw = g.W[o];
        for (let i = 0; i < layer.nIn; i += 1) gw[i] += dz * cache.x[i];
        g.b[o] += dz;
      }
      if (l === 0) break;
      // 앞 층으로 델타를 넘긴다
      const prevLayer = this.layers[l - 1];
      const prevCache = caches[l - 1];
      const next = new Float64Array(layer.nIn);
      for (let o = 0; o < layer.nOut; o += 1) {
        const dz = delta[o];
        if (dz === 0) continue;
        const w = layer.W[o];
        for (let i = 0; i < layer.nIn; i += 1) next[i] += dz * w[i];
      }
      for (let i = 0; i < layer.nIn; i += 1) {
        next[i] *= prevLayer.act.df(prevCache.z[i], prevCache.a[i]);
      }
      delta = next;
    }
  }

  makeGrads() {
    return this.layers.map((l) => ({
      W: l.W.map(() => new Float64Array(l.nIn)),
      b: new Float64Array(l.nOut),
    }));
  }

  /** Adam — 기울기의 이동평균을 써서 학습률을 알아서 조절한다 */
  applyGrads(grads, batchSize) {
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;
    this.t += 1;
    const c1 = 1 - b1 ** this.t;
    const c2 = 1 - b2 ** this.t;
    this.layers.forEach((layer, l) => {
      const g = grads[l];
      for (let o = 0; o < layer.nOut; o += 1) {
        const w = layer.W[o];
        const gw = g.W[o];
        const mw = layer.mW[o];
        const vw = layer.vW[o];
        for (let i = 0; i < layer.nIn; i += 1) {
          const grad = gw[i] / batchSize;
          mw[i] = b1 * mw[i] + (1 - b1) * grad;
          vw[i] = b2 * vw[i] + (1 - b2) * grad * grad;
          w[i] -= (this.lr * (mw[i] / c1)) / (Math.sqrt(vw[i] / c2) + eps);
        }
        const gb = g.b[o] / batchSize;
        layer.mb[o] = b1 * layer.mb[o] + (1 - b1) * gb;
        layer.vb[o] = b2 * layer.vb[o] + (1 - b2) * gb * gb;
        layer.b[o] -= (this.lr * (layer.mb[o] / c1)) / (Math.sqrt(layer.vb[o] / c2) + eps);
      }
    });
  }

  lossOf(yTrue, yHat) {
    if (this.loss === 'binary_crossentropy') {
      const p = Math.min(Math.max(yHat, 1e-12), 1 - 1e-12);
      return -(yTrue * Math.log(p) + (1 - yTrue) * Math.log(1 - p));
    }
    return (yTrue - yHat) ** 2;
  }

  evaluate(X, y) {
    let s = 0;
    for (let i = 0; i < X.length; i += 1) s += this.lossOf(y[i], this.forward(X[i]).out[0]);
    return s / X.length;
  }

  /** 학습 — keras 의 fit() 과 같은 자리 */
  async fit(X, y, {
    epochs = 50, batchSize = 32, validationData = null,
    onEpoch = null, seed = 7,
  } = {}) {
    const n = X.length;
    for (let ep = 0; ep < epochs; ep += 1) {
      const order = shuffledIndex(n, seed + ep);
      let epochLoss = 0;
      for (let start = 0; start < n; start += batchSize) {
        const batch = order.slice(start, start + batchSize);
        const grads = this.makeGrads();
        batch.forEach((i) => {
          const { out, caches } = this.forward(X[i], true);
          epochLoss += this.lossOf(y[i], out[0]);
          this.backward(caches, y[i], grads);
        });
        this.applyGrads(grads, batch.length);
      }
      this.history.loss.push(epochLoss / n);
      if (validationData) {
        this.history.valLoss.push(this.evaluate(validationData[0], validationData[1]));
      }
      if (onEpoch) {
        onEpoch(ep + 1, this.history);
        await tick();                                // 화면이 갱신될 틈을 준다
      }
    }
    return this.history;
  }

  /** 층 구조 요약 — keras 의 summary() 처럼 */
  summary() {
    return this.layers.map((l, i) => ({
      name: `Dense ${i + 1}`,
      units: l.nOut,
      activation: l.actName,
      dropout: l.dropout,
      params: l.nIn * l.nOut + l.nOut,
    }));
  }
}
