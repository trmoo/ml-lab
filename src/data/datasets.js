/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 실습 데이터 11개 파일을 앱 안에 넣어 둔다.
 *
 * CSV 를 ?raw 로 불러오므로 빌드하면 dist/index.html 한 파일 안에 데이터까지 들어간다.
 * → 교실에서 인터넷이 끊겨도 그냥 열린다.
 *
 * 파일 이름에는 회차 번호를 붙이지 않는다. 수업 순서(meta.js 의 DATASET_IDS)를
 * 바꿔도 파일 이름을 고칠 필요가 없게 하려는 것이다.
 * 데이터의 「설명」은 meta.js 에 따로 두었다.
 */
import csvExam from '../../data/exam_result.csv?raw';
import csvCrop from '../../data/crop_output.csv?raw';
import csvCar from '../../data/car_resale.csv?raw';
import csvAd from '../../data/ad_return.csv?raw';
import csvPay from '../../data/staff_pay.csv?raw';
import csvYield from '../../data/process_yield.csv?raw';
import csvHomeA from '../../data/home_value_a.csv?raw';
import csvHomeB from '../../data/home_value_b.csv?raw';
import csvChurn from '../../data/subscriber_churn.csv?raw';
import csvLoan from '../../data/loan_result.csv?raw';
import csvQuality from '../../data/product_quality.csv?raw';
import {
  META, DATASET_IDS, describeColumns, backgroundOf, fileLabel,
} from './meta.js';
import { buildDataset } from './build.js';

const RAW = {
  exam: [csvExam],
  crop: [csvCrop],
  car: [csvCar],
  ad: [csvAd],
  pay: [csvPay],
  yield: [csvYield],
  home: [csvHomeA, csvHomeB],
  churn: [csvChurn],
  loan: [csvLoan],
  quality: [csvQuality],
};

const cache = new Map();

/** 데이터셋 하나를 준비한다. 한 번 읽은 것은 다시 읽지 않는다 */
export function getDataset(key) {
  if (cache.has(key)) return cache.get(key);
  if (!RAW[key]) throw new Error(`알 수 없는 데이터셋: ${key}`);
  const ds = buildDataset(key, RAW[key]);
  cache.set(key, ds);
  return ds;
}

/** 화면의 데이터셋 고르기 단추에 쓸 목록 — 수업 순서대로 */
export function datasetOptions() {
  return DATASET_IDS.map((key) => {
    const m = META[key];
    return {
      id: key,
      round: m.round,
      label: m.label,
      short: m.short,
      task: m.task,
      taskLabel: m.task === 'regression' ? '회귀' : '분류',
    };
  });
}

export { META, DATASET_IDS, describeColumns, backgroundOf, fileLabel };
