/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 평가지표 — 회귀·분류에서 흔히 쓰는 것들.
 *
 * 회귀는 「얼마나 빗나갔나」, 분류는 「무엇을 맞히고 무엇을 틀렸나」를 본다.
 */

/* ── 회귀 ───────────────────────────────────────────────── */

/** MSE — 오차를 제곱해서 평균. 크게 틀린 것에 더 큰 벌점을 준다. 작을수록 좋다 */
export function mse(yTrue, yPred) {
  let s = 0;
  for (let i = 0; i < yTrue.length; i += 1) s += (yTrue[i] - yPred[i]) ** 2;
  return s / yTrue.length;
}

/** RMSE — MSE 에 제곱근. 원래 단위로 돌아오므로 해석하기 쉽다 */
export function rmse(yTrue, yPred) { return Math.sqrt(mse(yTrue, yPred)); }

/** MAE — 오차의 절댓값 평균. 이상치에 덜 흔들린다 */
export function mae(yTrue, yPred) {
  let s = 0;
  for (let i = 0; i < yTrue.length; i += 1) s += Math.abs(yTrue[i] - yPred[i]);
  return s / yTrue.length;
}

/** R² (결정계수) — 1에 가까울수록 좋다. 0이면 「평균으로 찍는 것」과 같은 수준 */
export function r2Score(yTrue, yPred) {
  const m = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < yTrue.length; i += 1) {
    ssRes += (yTrue[i] - yPred[i]) ** 2;
    ssTot += (yTrue[i] - m) ** 2;
  }
  return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
}

export function regressionScores(yTrue, yPred) {
  return {
    MSE: mse(yTrue, yPred),
    RMSE: rmse(yTrue, yPred),
    MAE: mae(yTrue, yPred),
    'R²': r2Score(yTrue, yPred),
  };
}

/* ── 분류 ───────────────────────────────────────────────── */

/**
 * 혼동 행렬 — 분류 평가의 출발점.
 *   TN 정상을 정상이라 함 | FP 정상을 불량이라 함(헛경보)
 *   FN 불량을 정상이라 함(놓침) | TP 불량을 불량이라 함
 */
export function confusionMatrix(yTrue, yPred) {
  let tn = 0;
  let fp = 0;
  let fn = 0;
  let tp = 0;
  for (let i = 0; i < yTrue.length; i += 1) {
    const t = yTrue[i] === 1 ? 1 : 0;
    const p = yPred[i] === 1 ? 1 : 0;
    if (t === 0 && p === 0) tn += 1;
    else if (t === 0 && p === 1) fp += 1;
    else if (t === 1 && p === 0) fn += 1;
    else tp += 1;
  }
  return { tn, fp, fn, tp };
}

export function accuracyScore(yTrue, yPred) {
  const { tn, fp, fn, tp } = confusionMatrix(yTrue, yPred);
  const all = tn + fp + fn + tp;
  return all ? (tn + tp) / all : 0;
}

/** 정밀도 — 「1이라고 한 것」 중 진짜 1의 비율. 헛경보를 줄이고 싶을 때 본다 */
export function precisionScore(yTrue, yPred) {
  const { fp, tp } = confusionMatrix(yTrue, yPred);
  return tp + fp ? tp / (tp + fp) : 0;
}

/** 재현율 — 「진짜 1」 중 찾아낸 비율. 놓치면 큰일인 문제(불량·질병)에서 중요하다 */
export function recallScore(yTrue, yPred) {
  const { fn, tp } = confusionMatrix(yTrue, yPred);
  return tp + fn ? tp / (tp + fn) : 0;
}

/** F1 — 정밀도와 재현율의 조화평균. 둘 다 챙기고 싶을 때 */
export function f1Score(yTrue, yPred) {
  const p = precisionScore(yTrue, yPred);
  const r = recallScore(yTrue, yPred);
  return p + r ? (2 * p * r) / (p + r) : 0;
}

/**
 * ROC AUC — 임계값을 0부터 1까지 움직여도 순위를 잘 매기는지 보는 지표.
 * 0.5 는 동전 던지기, 1.0 은 완벽. 순위 통계(Mann–Whitney U)로 계산한다.
 */
export function rocAucScore(yTrue, yProb) {
  const pairs = yTrue.map((t, i) => ({ t: t === 1 ? 1 : 0, p: yProb[i] }))
    .sort((a, b) => a.p - b.p);
  // 같은 확률끼리는 평균 순위를 준다
  let i = 0;
  let sumRankPos = 0;
  let nPos = 0;
  let nNeg = 0;
  while (i < pairs.length) {
    let j = i;
    while (j + 1 < pairs.length && pairs[j + 1].p === pairs[i].p) j += 1;
    const avgRank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k += 1) {
      if (pairs[k].t === 1) { sumRankPos += avgRank; nPos += 1; } else nNeg += 1;
    }
    i = j + 1;
  }
  if (!nPos || !nNeg) return 0.5;
  return (sumRankPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

/** ROC 곡선 좌표 — 임계값을 낮춰 가며 (헛경보율, 찾아낸율) 을 이어 그린다 */
export function rocCurve(yTrue, yProb) {
  const pairs = yTrue.map((t, i) => ({ t: t === 1 ? 1 : 0, p: yProb[i] }))
    .sort((a, b) => b.p - a.p);
  const nPos = pairs.filter((o) => o.t === 1).length;
  const nNeg = pairs.length - nPos;
  const pts = [{ x: 0, y: 0 }];
  let tp = 0;
  let fp = 0;
  pairs.forEach((o) => {
    if (o.t === 1) tp += 1; else fp += 1;
    pts.push({ x: nNeg ? fp / nNeg : 0, y: nPos ? tp / nPos : 0 });
  });
  return pts;
}

export function classificationScores(yTrue, yPred, yProb) {
  const cm = confusionMatrix(yTrue, yPred);
  return {
    정확도: accuracyScore(yTrue, yPred),
    정밀도: precisionScore(yTrue, yPred),
    재현율: recallScore(yTrue, yPred),
    F1: f1Score(yTrue, yPred),
    'ROC AUC': yProb ? rocAucScore(yTrue, yProb) : NaN,
    _cm: cm,
  };
}
