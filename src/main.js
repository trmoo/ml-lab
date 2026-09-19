/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 머신러닝 실험실 — 시작점.
 *
 * 탭 다섯 개가 머신러닝 파이프라인의 차례를 따른다.
 *   ① 데이터 탐험실   — 불러오기 · 살펴보기 · 시각화 · 상관
 *   ② 전처리 실험실   — 결측치 · 인코딩 · 분리 · 스케일링 · 데이터 누수
 *   ③ 모델 학습실     — 학습 · 평가 · 과적합 진단 · 새 데이터 예측
 *   ④ 딥러닝 실험실   — 은닉층 · 손실 곡선 · 조기 종료
 *   ⑤ 문제 풀기       — 회차마다 데이터 한 벌로 푸는 문제 (quiz/)
 */
import './style.css';
import { h, clear } from './lib/ui.js';
import { renderExplore } from './tabs/explore.js';
import { renderPrep } from './tabs/prep.js';
import { renderTrain } from './tabs/train.js';
import { renderDeep } from './tabs/deep.js';
import { renderQuiz } from './tabs/quiz.js';
import { onNavigate } from './lib/nav.js';
import { useDataset as exploreUse } from './tabs/explore.js';
import { useDataset as prepUse } from './tabs/prep.js';
import { useDataset as trainUse } from './tabs/train.js';
import { useDataset as deepUse } from './tabs/deep.js';

const TABS = [
  { id: 'explore', label: '데이터 탐험실', render: renderExplore, use: exploreUse },
  { id: 'prep', label: '전처리 실험실', render: renderPrep, use: prepUse },
  { id: 'train', label: '모델 학습실', render: renderTrain, use: trainUse },
  { id: 'deep', label: '딥러닝 실험실', render: renderDeep, use: deepUse },
  { id: 'quiz', label: '문제 풀기', render: renderQuiz },
];

let nav = null;
let view = null;
let current = TABS[0].id;

function show(id) {
  current = id;
  const tab = TABS.find((t) => t.id === id) || TABS[0];
  [...nav.children].forEach((b) => b.classList.toggle('on', b.dataset.id === id));
  clear(view);
  try {
    tab.render(view);
  } catch (err) {
    view.append(h('div', { class: 'warn' },
      h('b', {}, '화면을 그리는 중 문제가 생겼습니다. '), String(err?.message || err)));
    // 개발 중에 원인을 바로 찾을 수 있게 콘솔에도 남긴다
    console.error(err);
  }
  window.scrollTo({ top: 0 });
}

function start() {
  nav = document.getElementById('tabs');
  view = document.getElementById('view');
  if (!nav || !view) return;
  TABS.forEach((t, i) => {
    nav.append(h('button', {
      type: 'button', class: 'tab', role: 'tab', dataset: { id: t.id },
      onclick: () => show(t.id),
    }, h('span', { class: 'tab-n' }, i + 1), t.label));
  });
  // 문제 풀기 탭의 「실험실에서 직접 해 보기」 — 그 탭을 같은 데이터로 연다
  onNavigate((id, dsId) => {
    const tab = TABS.find((t) => t.id === id);
    if (dsId && tab?.use) tab.use(dsId);
    show(id);
  });
  show(current);
}

/* 빌드하면 이 스크립트가 <head> 안으로 들어간다.
 * 게다가 file:// 에서 열리게 하려고 type="module" 을 떼어 내므로
 * (모듈이 아닌 스크립트는 기다려 주지 않는다) 본문보다 먼저 실행된다.
 * 그래서 화면 요소가 준비될 때까지 기다린 뒤에 시작한다.
 * 개발 서버에서는 이미 준비돼 있어 곧바로 시작된다. */
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', start, { once: true });
} else {
  start();
}
