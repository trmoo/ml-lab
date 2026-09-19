/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 문제 은행 모양 점검 — node test/quiz.test.mjs
 *
 * 문제 JSON 이 docs/문제-작성-규칙.md 를 지키는지 본다.
 * 코드를 실제로 돌려 정답이 데이터와 맞는지는 python tools/verify_quiz.py 가 따로 본다.
 */
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { TYPES, STAGES, splitBlanks, prepareRound, sameAnswer } from '../src/data/quiz-parse.js';
import { META, DATASET_IDS } from '../src/data/meta.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
let pass = 0;
let fail = 0;
function ok(name, cond, extra = '') {
  if (cond) { pass += 1; console.log(`  ✓ ${name}${extra ? ` — ${extra}` : ''}`); } else {
    fail += 1; console.log(`  ✗ ${name}${extra ? ` — ${extra}` : ''}`);
  }
}

console.log('\n[1] 빈칸 표시 풀기');
{
  const { parts, answers } = splitBlanks("df[['a', 'b']].⟦mean⟧() + ⟦1⟧");
  ok('⟦ ⟧ 만 빈칸으로 보고 대괄호 두 겹은 코드로 남긴다',
    answers.join('|') === 'mean|1' && parts[0].v === "df[['a', 'b']].");
  ok('정답 비교는 따옴표·쉼표 띄어쓰기를 눈감아 준다',
    sameAnswer('"a",\'b\'', "'a', 'b'") && sameAnswer('0.30', '0.3') && !sameAnswer('', '0'));
}

const LABS = ['explore', 'prep', 'train', 'deep'];
const LEAK = /inplace\s*=\s*True/;
const allIds = new Set();

DATASET_IDS.forEach((key) => {
  const path = join(ROOT, 'quiz', `${key}.json`);
  console.log(`\n[${META[key].round}회차] ${key}`);
  if (!existsSync(path)) { ok(`${key}.json 이 있다`, false); return; }
  let raw;
  try { raw = JSON.parse(readFileSync(path, 'utf8')); } catch (err) {
    ok(`${key}.json 이 올바른 JSON 이다`, false, err.message); return;
  }
  const ps = raw.problems || [];
  const bad = [];
  const note = (p, msg) => bad.push(`${p.id || '(id 없음)'}: ${msg}`);

  ok('goal 이 있다', typeof raw.goal === 'string' && raw.goal.length >= 10);
  ok('문제가 8~10개', ps.length >= 8 && ps.length <= 10, `${ps.length}개`);

  ps.forEach((p, i) => {
    const want = `${key}-${String(i + 1).padStart(2, '0')}`;
    if (p.id !== want) note(p, `id 는 ${want} 이어야 한다`);
    if (allIds.has(p.id)) note(p, 'id 가 겹친다');
    allIds.add(p.id);
    if (!TYPES[p.type]) note(p, `모르는 type '${p.type}'`);
    if (!STAGES.includes(p.stage)) note(p, `모르는 stage '${p.stage}'`);
    if (!p.title || p.title.length > 30) note(p, 'title 이 없거나 30자를 넘는다');
    if (!p.prompt || p.prompt.length < 10) note(p, 'prompt 가 너무 짧다');
    if (!p.explain || p.explain.length < 40) note(p, 'explain 이 40자보다 짧다');
    if (p.lab && !LABS.includes(p.lab.tab)) note(p, `lab.tab '${p.lab.tab}' 은 없는 탭`);
    if (p.check && !Array.isArray(p.check)) note(p, 'check 는 목록이어야 한다');
    const code = [p.code, p.fix, raw.setup].filter(Boolean).join('\n');
    if (LEAK.test(code)) note(p, 'inplace=True 를 쓰지 않는다 (pandas 3 에서 결과가 달라진다)');
    if (/read_csv\(\s*['"][^'"]*\//.test(code)) note(p, 'read_csv 에는 파일 이름만 적는다');
    for (const m of code.matchAll(/read_csv\(\s*['"]([^'"]+)['"]/g)) {
      if (!existsSync(join(ROOT, 'data', m[1]))) note(p, `없는 파일 ${m[1]}`);
    }
    if (p.run === false && !/keras|tensorflow/.test(p.code || '')) {
      note(p, 'run:false 는 keras 코드에만 쓴다');
    }

    if (p.type === 'blank') {
      const { answers } = splitBlanks(p.code || '');
      if (answers.length < 1 || answers.length > 3) note(p, `빈칸이 ${answers.length}개 (1~3개)`);
      if ((p.blanks || []).length !== answers.length) note(p, 'blanks 길이가 빈칸 개수와 다르다');
      answers.forEach((a, k) => {
        const opts = p.blanks?.[k]?.options || [];
        if (!a.trim()) note(p, `빈칸 ${k + 1} 의 정답이 비었다`);
        if (opts.length === 1 || opts.length > 5) note(p, `빈칸 ${k + 1} 의 오답이 ${opts.length}개 (0 또는 2~5개)`);
        if (opts.some((o) => sameAnswer(o, a))) note(p, `빈칸 ${k + 1} 의 오답에 정답이 섞였다`);
        if (new Set(opts).size !== opts.length) note(p, `빈칸 ${k + 1} 의 오답이 겹친다`);
      });
    }
    if (p.type === 'choice') {
      const cs = p.choices || [];
      if (cs.length < 3 || cs.length > 5) note(p, `보기가 ${cs.length}개 (3~5개)`);
      if (new Set(cs).size !== cs.length) note(p, '보기가 겹친다');
      if (!Number.isInteger(p.answer) || p.answer < 0 || p.answer >= cs.length) note(p, 'answer 자리가 보기 밖');
    }
    if (p.type === 'bug') {
      const lines = (p.code || '').split('\n');
      if (!Number.isInteger(p.answer) || p.answer < 1 || p.answer > lines.length) note(p, 'answer 줄 번호가 코드 밖');
      else if (!p.fix || p.fix.trim() === lines[p.answer - 1].trim()) note(p, 'fix 가 없거나 원래 줄과 같다');
      if (lines.length < 3) note(p, '오류 찾기 코드는 3줄 이상');
    }
    if (p.type === 'order') {
      const st = p.steps || [];
      if (st.length < 3 || st.length > 6) note(p, `단계가 ${st.length}개 (3~6개)`);
      if (new Set(st).size !== st.length) note(p, '단계가 겹친다');
    }
  });
  ok('문제마다 규칙을 지킨다', bad.length === 0, bad.join('\n      '));

  const types = new Set(ps.map((p) => p.type));
  const blanks = ps.filter((p) => p.type === 'blank').length;
  ok('문제 모양을 세 가지 이상 섞었다', types.size >= 3, [...types].join(', '));
  ok('빈칸 채우기가 절반을 넘지 않는다', blanks <= ps.length / 2, `${blanks} / ${ps.length}`);
  ok('오류 찾기가 하나 이상', ps.some((p) => p.type === 'bug'));
  ok('실험실로 이어지는 문제가 하나 이상', ps.some((p) => p.lab));
  const kerasN = ps.filter((p) => p.run === false).length;
  ok('keras 문제는 두 개까지', kerasN <= 2, `${kerasN}개`);

  const answersAt = ps.filter((p) => p.type === 'choice').map((p) => p.answer);
  const most = Math.max(0, ...[0, 1, 2, 3, 4].map((a) => answersAt.filter((x) => x === a).length));
  ok('고르기 정답 자리가 한쪽에 몰리지 않는다',
    answersAt.length < 3 || most <= Math.ceil(answersAt.length * 0.6), `정답 자리 ${answersAt.join(',')}`);

  const prepared = prepareRound({ key, ...raw }, META[key].round);
  ok('화면용으로 바꾸어도 문제 수가 같다', prepared.problems.length === ps.length);
});

console.log(`\n결과: 통과 ${pass} · 실패 ${fail}\n`);
process.exit(fail ? 1 : 0);
