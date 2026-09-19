/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 긴 계산 중간에 화면이 갱신될 틈을 주는 함수.
 *
 * setTimeout(fn, 0) 은 브라우저가 최소 4ms 로 늘려 버리고,
 * 탭이 뒤에 가려져 있으면 1초까지 늦춘다.
 * 그래서 진행률을 스무 번만 갱신해도 학습보다 기다리는 시간이 더 길어진다.
 * MessageChannel 은 그런 지연이 없어 곧바로 한 바퀴를 돌려준다.
 */
const channel = typeof MessageChannel === 'function' ? new MessageChannel() : null;

export const tick = () => (channel
  ? new Promise((resolve) => {
    channel.port1.onmessage = () => resolve();
    channel.port2.postMessage(0);
  })
  : new Promise((resolve) => { setTimeout(resolve, 0); }));

export default tick;
