/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 씨앗(seed)을 정해 두면 언제나 같은 순서로 나오는 난수 생성기.
 *
 * 사이킷런의 random_state 와 똑같은 역할이다.
 * random_state 를 고정하면 몇 번을 다시 실행해도 훈련/검증 데이터가 같은 자리에서
 * 갈라지기 때문에, 「모델을 바꿨더니 점수가 올랐다」를 제대로 비교할 수 있다.
 */

/** mulberry32 — 짧고 품질이 괜찮은 난수 알고리즘 */
export function makeRng(seed) {
  let a = (seed >>> 0) || 1;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 0 ~ n-1 을 뒤섞은 배열 (피셔–예이츠) */
export function shuffledIndex(n, seed) {
  const rng = makeRng(seed);
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx;
}

/** 정규분포(평균 0, 표준편차 1) 난수 — 신경망 가중치 초기화에 쓴다 */
export function makeNormalRng(seed) {
  const rng = makeRng(seed);
  let spare = null;
  return function normal() {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    let s = 0;
    do {
      u = rng() * 2 - 1;
      v = rng() * 2 - 1;
      s = u * u + v * v;
    } while (s === 0 || s >= 1);
    const mul = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * mul;
    return u * mul;
  };
}
