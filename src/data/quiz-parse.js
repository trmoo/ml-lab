/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 문제 은행 JSON → 화면에서 쓰기 좋은 모양.
 *
 * 브라우저(quiz.js)와 노드 점검(test/quiz.test.mjs)이 함께 쓴다. 그래서 DOM 을 건드리지 않는다.
 * 문제를 쓰는 규칙은 docs/문제-작성-규칙.md 에 있다.
 *
 * 빈칸은 코드 안에 ⟦정답⟧ 으로 적는다.
 * 파이썬의 df[['a', 'b']] 와 헷갈리지 않도록 대괄호가 아닌 ⟦ ⟧ (U+27E6 · U+27E7) 를 쓴다.
 */

export const TYPES = {
  blank: '빈칸 채우기',
  choice: '고르기',
  bug: '오류 찾기',
  order: '순서 맞추기',
};

export const STAGES = [
  '살펴보기', '시각화', '결측치', '인코딩', '파생변수',
  '분리·스케일링', '모델 학습', '평가', '딥러닝', '해석·예측',
];

const MARK = /⟦([^⟦⟧]*)⟧/g;

/** 빈칸 코드를 글 조각과 빈칸 조각으로 나눈다 */
export function splitBlanks(code) {
  const parts = [];
  const answers = [];
  let last = 0;
  for (const m of code.matchAll(MARK)) {
    if (m.index > last) parts.push({ t: 'text', v: code.slice(last, m.index) });
    parts.push({ t: 'blank', i: answers.length });
    answers.push(m[1]);
    last = m.index + m[0].length;
  }
  if (last < code.length) parts.push({ t: 'text', v: code.slice(last) });
  return { parts, answers };
}

/** 빈칸을 정답으로 채운 코드 (해설에서 보여 주고, 파이썬 점검에서 실행한다) */
export function fillBlanks(code) {
  return code.replace(MARK, (_, a) => a);
}

/**
 * @param {object} raw   quiz/<열쇠>.json 한 개
 * @param {number} round 회차 번호 (meta.js 의 순서)
 */
export function prepareRound(raw, round) {
  return {
    ...raw,
    round,
    problems: raw.problems.map((p, i) => {
      const q = { ...p, no: i + 1 };
      if (p.type === 'blank') {
        const { parts, answers } = splitBlanks(p.code);
        q.parts = parts;
        q.blanks = answers.map((answer, k) => ({
          answer,
          options: (p.blanks?.[k]?.options) || [],
        }));
        q.full = fillBlanks(p.code);
      }
      if (p.type === 'bug') {
        q.lines = p.code.split('\n');
        q.full = q.lines.map((l, k) => (k + 1 === p.answer ? p.fix : l)).join('\n');
      }
      return q;
    }),
  };
}

/** 정답 비교 — 따옴표 종류와 앞뒤 공백, 쉼표 뒤 띄어쓰기는 눈감아 준다 */
export function sameAnswer(given, answer) {
  const norm = (s) => String(s).trim()
    .replace(/^["']|["']$/g, '')
    .replace(/"/g, "'")
    .replace(/\s*,\s*/g, ', ')
    .replace(/\s+/g, ' ');
  if (norm(given) === norm(answer)) return true;
  const a = Number(norm(given));
  const b = Number(norm(answer));
  return norm(given) !== '' && Number.isFinite(a) && Number.isFinite(b) && a === b;
}
