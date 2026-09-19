/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 탭 사이를 오가는 길잡이.
 *
 * 문제 풀기 탭에서 「실험실에서 직접 해 보기」를 누르면 다른 탭으로 넘어가야 하는데,
 * 탭 모듈이 main.js 를 불러오면 서로가 서로를 부르는 고리가 생긴다.
 * 그래서 main.js 가 여기에 이동 함수를 맡겨 두고, 탭은 여기만 부른다. */
let handler = null;

export function onNavigate(fn) { handler = fn; }

/** @param {string} tab 탭 id  @param {string} [dsId] 그 탭에서 고를 데이터셋 열쇠 */
export function navigate(tab, dsId) {
  if (handler) handler(tab, dsId);
}
