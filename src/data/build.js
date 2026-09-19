/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* CSV 글자 → 데이터셋 객체.
 *
 * 브라우저(datasets.js)와 노드 점검 스크립트(test/)가 함께 쓴다.
 * 어느 쪽에서 불러왔든 완전히 같은 데이터가 나오게 하려고 한 곳에 모아 두었다.
 */
import { Frame } from '../lib/frame.js';
import { META, describeColumns, backgroundOf, fileLabel } from './meta.js';

/**
 * @param {string} key     META 의 열쇠 ('exam', 'crop', 'car', …)
 * @param {string[]} texts CSV 글자들 (주택은 두 개)
 */
export function buildDataset(key, texts) {
  const meta = META[key];
  if (!meta) throw new Error(`알 수 없는 데이터셋: ${key}`);
  const parts = texts.map((text) => Frame.fromCsv(text));
  let frame = parts[0];
  if (parts.length > 1 && meta.mergeKey) {
    frame = Frame.mergeOn(parts[0], parts[1], meta.mergeKey);
  }

  if (!frame.columns.includes(meta.target)) {
    throw new Error(`${key}: 예측 목표 컬럼 '${meta.target}' 이 데이터에 없습니다.`);
  }

  return {
    key,
    id: key,
    ...meta,
    file: fileLabel(key),
    frame,
    colDesc: describeColumns(key),
    background: backgroundOf(key),
    features: frame.columns.filter((c) =>
      c !== meta.target && !(meta.idColumns || []).includes(c)),
  };
}
