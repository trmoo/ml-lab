/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 문제 은행 — quiz/<열쇠>.json 열 개를 수업 순서대로 묶는다.
 * JSON 으로 불러오므로 빌드하면 dist/index.html 안에 함께 들어간다. */
import exam from '../../quiz/exam.json';
import crop from '../../quiz/crop.json';
import car from '../../quiz/car.json';
import ad from '../../quiz/ad.json';
import pay from '../../quiz/pay.json';
import yieldRound from '../../quiz/yield.json';
import home from '../../quiz/home.json';
import churn from '../../quiz/churn.json';
import loan from '../../quiz/loan.json';
import quality from '../../quiz/quality.json';
import { META, DATASET_IDS } from './meta.js';
import { prepareRound } from './quiz-parse.js';

const RAW = { exam, crop, car, ad, pay, yield: yieldRound, home, churn, loan, quality };

/** 회차 순서대로 — 문제가 아직 없는 회차는 건너뛴다 */
export const ROUNDS = DATASET_IDS
  .filter((key) => RAW[key]?.problems?.length)
  .map((key) => prepareRound({ key, ...RAW[key] }, META[key].round));

export const TOTAL = ROUNDS.reduce((a, r) => a + r.problems.length, 0);
