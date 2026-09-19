/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 탭 ⑤ 문제 풀기
 *
 * 회차마다 데이터 한 벌을 두고, 그 데이터의 「가르칠 거리」를 중심으로 문제를 짰다.
 * 문제는 네 가지 모양이다.
 *   빈칸 채우기 — 코드의 핵심 낱말을 채운다 (보기를 누르거나 직접 입력)
 *   고르기     — 실행 결과를 예상하거나 해석을 고른다
 *   오류 찾기   — 잘못된 줄을 눌러 찾는다
 *   순서 맞추기 — 단계를 올바른 차례로 늘어놓는다
 *
 * 해설은 맞히거나 「정답 보기」를 누른 뒤에 열린다 — 먼저 스스로 생각해 보게 하려는 순서다.
 * 풀이 기록은 화면을 닫으면 사라진다. 학생 정보를 어디에도 저장하지 않는다.
 */
import { ROUNDS, TOTAL } from '../data/quiz.js';
import { TYPES, sameAnswer } from '../data/quiz-parse.js';
import { META } from '../data/meta.js';
import { h, card, codeBlock, foldout, md, badge, redraw } from '../lib/ui.js';
import { makeRng } from '../lib/rng.js';
import { navigate } from '../lib/nav.js';

const LAB_NAMES = {
  explore: '데이터 탐험실', prep: '전처리 실험실', train: '모델 학습실', deep: '딥러닝 실험실',
};

const state = {
  ri: 0, // 회차 순번 (ROUNDS 의 자리)
  qi: 0, // 문제 순번
  /** 문제 id → 푸는 중인 모습 */
  work: {},
};

/** 문제 하나의 풀이 상태 — 처음 볼 때 만든다 */
function workOf(q) {
  if (!state.work[q.id]) {
    state.work[q.id] = {
      answers: [], // 빈칸에 넣은 글
      checked: [], // 빈칸마다 'ok' | 'no'
      pick: null, // 고르기 · 오류 찾기에서 고른 것
      seq: [], // 순서 맞추기에서 늘어놓은 차례 (단계 번호)
      result: null, // 'ok' | 'no' | 'shown'
      hint: false,
    };
  }
  return state.work[q.id];
}

const solved = (q) => state.work[q.id]?.result === 'ok';
const opened = (q) => ['ok', 'shown'].includes(state.work[q.id]?.result);

export function renderQuiz(root) {
  redraw(root, () => {
    if (!ROUNDS.length) {
      root.append(card('문제 풀기', h('div', { class: 'note' }, '아직 문제가 없습니다.')));
      return;
    }
    const r = ROUNDS[state.ri] || ROUNDS[0];
    const q = r.problems[state.qi] || r.problems[0];
    root.append(pickerCard(root, r), problemCard(root, r, q));
  });
}

/* ── 회차·문제 고르기 ── */
function pickerCard(root, r) {
  const doneAll = ROUNDS.reduce((a, x) => a + x.problems.filter(solved).length, 0);

  const roundBtns = h('div', { class: 'rounds' }, ROUNDS.map((x, i) => h('button', {
    type: 'button',
    class: `rbtn${i === state.ri ? ' on' : ''}`,
    onclick: () => { state.ri = i; state.qi = 0; renderQuiz(root); },
  },
  `${x.round}회차 · ${META[x.key].short}`,
  h('small', {}, `${x.problems.filter(solved).length} / ${x.problems.length} 맞힘`))));

  const qBtns = h('div', { class: 'qnav' }, r.problems.map((p, i) => h('button', {
    type: 'button',
    class: `qbtn${i === state.qi ? ' on' : ''}${solved(p) ? ' done' : ''}`,
    title: `${TYPES[p.type]} · ${p.title}`,
    onclick: () => { state.qi = i; renderQuiz(root); },
  }, p.no)));

  return card('회차와 문제 고르기',
    h('p', { class: 'lead' },
      `10회차 · 모두 ${TOTAL}문제. 회차마다 데이터 한 벌을 가지고 탐색부터 평가까지 따라간다. `
      + '문제 옆 「실험실에서 직접 해 보기」를 누르면 같은 데이터로 그 탭이 열린다.'),
    roundBtns,
    h('div', { class: 'goal' },
      h('b', {}, `${r.round}회차 · ${META[r.key].label}`),
      h('span', {}, r.goal || '')),
    h('div', { class: 'kv' },
      h('span', {}, h('b', {}, '전체 '), `${doneAll} / ${TOTAL} 문제 맞힘`)),
    qBtns);
}

/* ── 문제 하나 ── */
function problemCard(root, r, q) {
  const w = workOf(q);
  const rerender = () => renderQuiz(root);

  const head = h('div', { class: 'qhead' },
    h('span', { class: 'qno' }, `${r.round}회차 문제 ${q.no}`),
    badge(q.stage, 'v'),
    badge(TYPES[q.type]),
    solved(q) ? badge('맞힘', 'g') : null);

  const body = [
    head,
    h('h4', { class: 'qtitle' }, q.title),
    h('div', { class: 'prompt', html: md(q.prompt) }),
  ];

  if (r.setup && q.no === 1) {
    body.push(foldout('이 회차에서 먼저 실행해 두는 코드', codeBlock(r.setup).outerHTML, true));
  } else if (r.setup) {
    body.push(foldout('이 회차에서 먼저 실행해 두는 코드', codeBlock(r.setup).outerHTML));
  }
  if (q.given) body.push(codeBlock(q.given, { title: '앞에서 실행해 둔 코드' }));

  const BUILD = { blank: blankBody, choice: choiceBody, bug: bugBody, order: orderBody };
  const grade = BUILD[q.type](body, q, w, rerender);

  /* 단추 줄 */
  const nextBtn = h('button', {
    class: 'btn btn-2',
    onclick: () => {
      if (state.qi + 1 < r.problems.length) state.qi += 1;
      else if (state.ri + 1 < ROUNDS.length) { state.ri += 1; state.qi = 0; }
      rerender();
    },
  }, '다음 문제 ▶');

  body.push(h('div', { class: 'btnrow' },
    h('button', { class: 'btn', onclick: () => { w.result = grade.check() ? 'ok' : 'no'; rerender(); } },
      '정답 확인'),
    h('button', { class: 'btn btn-2', onclick: () => { w.hint = true; rerender(); } }, '힌트 보기'),
    h('button', { class: 'btn btn-3', onclick: () => { grade.show(); w.result = 'shown'; rerender(); } },
      '정답 보기'),
    h('button', {
      class: 'btn btn-3',
      onclick: () => { delete state.work[q.id]; rerender(); },
    }, '다시 풀기'),
    nextBtn));

  if (w.hint) body.push(h('div', { class: 'tip', html: `<b>힌트</b> — ${md(grade.hint()).replace(/^<p>|<\/p>$/g, '')}` }));

  if (w.result === 'ok') {
    body.push(h('div', { class: 'ok' }, h('b', {}, '정답입니다. '), '아래 해설로 왜 그런지 확인해 보세요.'));
  } else if (w.result === 'no') {
    body.push(h('div', { class: 'warn' }, h('b', {}, '아직 아닙니다. '), grade.why()));
  } else if (w.result === 'shown') {
    body.push(h('div', { class: 'note' }, '정답을 펼쳤습니다. 해설을 읽고 다시 풀어 보세요.'));
  }

  if (opened(q)) {
    body.push(foldout('📘 해설', md(q.explain), true));
    if (q.full) body.push(foldout('완성된 코드', codeBlock(q.full).outerHTML));
  } else {
    body.push(h('div', { class: 'tip' },
      '해설은 맞히거나 「정답 보기」를 누르면 열린다. 먼저 스스로 풀어 보는 편이 훨씬 오래 남는다.'));
  }

  if (q.lab) {
    body.push(h('div', { class: 'labgo' },
      h('span', {}, `🔬 ${q.lab.say || '같은 데이터로 직접 확인해 보자.'}`),
      h('button', {
        class: 'btn btn-s',
        onclick: () => navigate(q.lab.tab, r.key),
      }, `${LAB_NAMES[q.lab.tab] || '실험실'}에서 직접 해 보기`)));
  }

  return card(null, ...body);
}

/* ── ① 빈칸 채우기 ─────────────────────────────────────── */
function blankBody(body, q, w, rerender) {
  const box = h('div', { class: 'blankcode' });
  const inputs = [];
  q.parts.forEach((part) => {
    if (part.t === 'text') { box.append(document.createTextNode(part.v)); return; }
    const i = part.i;
    const mark = w.checked[i] === 'ok' ? ' b-ok' : w.checked[i] === 'no' ? ' b-no' : '';
    const inp = h('input', {
      class: `blank${mark}${w.answers[i] ? ' filled' : ''}`,
      type: 'text',
      value: w.answers[i] ?? '',
      size: Math.max(5, q.blanks[i].answer.length + 2),
      'aria-label': `빈칸 ${i + 1}`,
      spellcheck: 'false',
      oninput: (e) => {
        w.answers[i] = e.target.value;
        w.checked[i] = undefined;
        e.target.classList.remove('b-ok', 'b-no');
        e.target.classList.toggle('filled', !!e.target.value);
      },
    });
    inputs.push(inp);
    box.append(inp);
  });
  body.push(h('div', { class: 'ctrl-l' }, '빈칸을 채우세요'), box);

  const bank = h('div', {});
  q.blanks.forEach((b, i) => {
    if (!b.options.length) return;
    const opts = shuffled([b.answer, ...b.options], seedOf(q.id) + i);
    bank.append(h('div', { class: 'blankrow' },
      h('span', { class: 'bkl' }, q.blanks.length > 1 ? `빈칸 ${i + 1}` : '보기'),
      ...opts.map((o) => h('button', {
        type: 'button', class: 'opt',
        onclick: () => {
          w.answers[i] = o;
          w.checked[i] = undefined;
          inputs[i].value = o;
          inputs[i].classList.remove('b-ok', 'b-no');
          inputs[i].classList.add('filled');
        },
      }, o))));
  });
  if (bank.children.length) body.push(bank);
  if (q.blanks.some((b) => !b.options.length)) {
    body.push(h('div', { class: 'tip' },
      '보기가 없는 빈칸은 직접 입력한다. 데이터를 봐야 알 수 있는 값이면 「데이터 탐험실」에서 확인해 보자.'));
  }

  return {
    check() {
      q.blanks.forEach((b, i) => { w.checked[i] = sameAnswer(w.answers[i] ?? '', b.answer) ? 'ok' : 'no'; });
      return w.checked.every((c) => c === 'ok') && w.checked.length === q.blanks.length;
    },
    show() {
      q.blanks.forEach((b, i) => { w.answers[i] = b.answer; w.checked[i] = undefined; });
    },
    hint() {
      return q.hint || q.blanks.map((b, i) =>
        `${q.blanks.length > 1 ? `빈칸 ${i + 1}: ` : ''}\`${firstHint(b.answer)}\``).join(' · ');
    },
    why() {
      const bad = w.checked.map((c, i) => (c === 'no' ? i + 1 : 0)).filter(Boolean);
      return q.blanks.length > 1 ? `빈칸 ${bad.join(', ')}번을 다시 보세요.` : '빈칸을 다시 보세요.';
    },
  };
}

/* ── ② 고르기 ─────────────────────────────────────────── */
function choiceBody(body, q, w, rerender) {
  if (q.code) body.push(codeBlock(q.code));
  body.push(h('div', { class: 'choices' }, q.choices.map((c, i) => {
    let cls = 'choice';
    if (w.pick === i) cls += ' on';
    if (w.result && w.result !== 'no' && i === q.answer) cls += ' right';
    if (w.result === 'no' && w.pick === i) cls += ' wrong';
    return h('button', {
      type: 'button', class: cls,
      onclick: () => { w.pick = i; if (w.result === 'no') w.result = null; rerender(); },
    }, h('span', { class: 'cn' }, '①②③④⑤⑥'[i]), h('span', { html: md(c).replace(/^<p>|<\/p>$/g, '') }));
  })));
  return {
    check: () => w.pick === q.answer,
    show: () => { w.pick = q.answer; },
    hint: () => q.hint || '보기마다 「왜 아닌지」를 하나씩 따져 보자.',
    why: () => (w.pick === null ? '먼저 하나를 고르세요.' : '다른 보기를 골라 보세요.'),
  };
}

/* ── ③ 오류 찾기 ──────────────────────────────────────── */
function bugBody(body, q, w, rerender) {
  body.push(h('div', { class: 'ctrl-l' }, '잘못된 줄을 눌러 고르세요'));
  body.push(h('div', { class: 'buglines' }, q.lines.map((line, k) => {
    const n = k + 1;
    let cls = 'bugline';
    if (w.pick === n) cls += ' on';
    if (w.result === 'no' && w.pick === n) cls += ' wrong';
    if (opened(q) && n === q.answer) cls += ' right';
    return h('button', {
      type: 'button', class: cls,
      onclick: () => { w.pick = n; if (w.result === 'no') w.result = null; rerender(); },
    }, h('span', { class: 'ln' }, n), h('code', {}, line || ' '));
  })));
  if (opened(q)) {
    body.push(h('div', { class: 'fixline' }, h('b', {}, `${q.answer}번 줄 바로잡기`),
      h('code', {}, q.fix)));
  }
  return {
    check: () => w.pick === q.answer,
    show: () => { w.pick = q.answer; },
    hint: () => q.hint || '한 줄씩 「이 줄이 끝나면 무엇이 바뀌었나」를 따라가 보자.',
    why: () => (w.pick === null ? '먼저 한 줄을 고르세요.' : '그 줄은 문제가 없습니다. 다른 줄을 보세요.'),
  };
}

/* ── ④ 순서 맞추기 ────────────────────────────────────── */
function orderBody(body, q, w, rerender) {
  const order = shuffled(q.steps.map((_, i) => i), seedOf(q.id));
  const left = order.filter((i) => !w.seq.includes(i));

  body.push(h('div', { class: 'ctrl-l' }, '아래 단계를 차례대로 누르세요 (늘어놓은 것을 누르면 빠집니다)'));
  body.push(h('div', { class: 'orderpool' }, left.length
    ? left.map((i) => h('button', {
      type: 'button', class: 'opt',
      onclick: () => { w.seq.push(i); if (w.result === 'no') w.result = null; rerender(); },
    }, q.steps[i]))
    : h('span', { class: 'ctrl-hint' }, '모두 늘어놓았습니다.')));

  body.push(h('ol', { class: 'orderlist' }, w.seq.map((i, k) => {
    let cls = 'orderitem';
    if (w.result === 'no') cls += i === k ? ' right' : ' wrong';
    return h('li', {},
      h('button', {
        type: 'button', class: cls,
        onclick: () => { w.seq.splice(k, 1); if (w.result === 'no') w.result = null; rerender(); },
      }, q.steps[i]));
  })));

  return {
    check: () => w.seq.length === q.steps.length && w.seq.every((i, k) => i === k),
    show: () => { w.seq = q.steps.map((_, i) => i); },
    hint: () => q.hint || `첫 단계는 「${q.steps[0]}」 입니다.`,
    why: () => (w.seq.length < q.steps.length
      ? '아직 다 늘어놓지 않았습니다.'
      : '빨간 칸부터 차례가 어긋났습니다. 눌러서 빼고 다시 늘어놓아 보세요.'),
  };
}

/* ── 도우미 ── */

/** 첫 글자 힌트 — 앞쪽 3분의 1 만 보여 준다. 한글은 초성으로 */
function firstHint(answer) {
  const s = String(answer);
  const CHO = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ';
  return [...s].slice(0, Math.max(1, Math.ceil(s.length / 3))).map((ch) => {
    const c = ch.charCodeAt(0);
    return c >= 0xac00 && c <= 0xd7a3 ? CHO[Math.floor((c - 0xac00) / 588)] : ch;
  }).join('') + '…';
}

/** 문제 id 로 씨앗을 만든다 — 다시 그려도 보기 순서가 바뀌지 않게 */
function seedOf(id) {
  let s = 7;
  for (const ch of String(id)) s = (s * 31 + ch.charCodeAt(0)) % 1000003;
  return s;
}

function shuffled(arr, seed) {
  const rng = makeRng(seed + 1);
  const out = [...new Set(arr)];
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  // 순서 맞추기에서 섞은 결과가 우연히 정답 순서와 같으면 문제가 안 되므로 한 칸 돌린다
  if (out.length > 1 && out.every((v, i) => v === arr[i])) out.push(out.shift());
  return out;
}
