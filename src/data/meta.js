/* 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
 * 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요. */
/* 실습 데이터 10종의 기본 정보.
 *
 * CSV 를 실제로 불러오는 일은 datasets.js 가 하고, 여기에는 「설명」만 둔다.
 * 이렇게 나눠 두면 노드에서 돌리는 점검 스크립트(test/)도 이 파일을 그대로 쓸 수 있다.
 *
 * 데이터는 모두 tools/make_data.py 가 만든 가상 자료다. 배경 이야기의 학교·회사·지명도 지어낸 것이다.
 *
 * **수업 순서는 DATASET_IDS 가 정한다.**
 *   1순위 — 회귀를 먼저, 분류를 나중에
 *   2순위 — 데이터 가공·분석 난이도 오름차순
 * 회차 번호(round)는 이 순서에서 나온다. 순서를 바꾸면 문제 은행의 회차도 함께 따라간다.
 */

/** 수업 순서 */
export const DATASET_IDS = ['exam', 'crop', 'car', 'ad', 'pay', 'yield', 'home', 'churn', 'loan', 'quality'];

/** 열쇠별 기본 정보 */
const BASE = {
  exam: {
    label: '기말고사 점수 예측', short: '기말점수',
    task: 'regression', target: 'final_point', unit: '점',
    files: ['exam_result.csv'], df: 'exam_df',
    ordinal: { extra_class: { 있음: 1, 없음: 0 }, club_join: { 있음: 1, 없음: 0 } },
    background: '한 고등학교가 학기 중간에 모은 학습 기록으로 기말고사 점수를 미리 짐작해 보려 한다. '
      + '공부 시간과 중간고사처럼 점수와 바로 이어진 변수가 있어서, 머신러닝을 처음 해 보기에 알맞은 데이터다.',
    columns: {
      study_hours_week: '한 주에 공부한 시간 (시간)',
      attend_percent: '출석률 (%)',
      extra_class: '방과 후 보충수업을 듣는지 (있음/없음)',
      track: '계열 (어문·공학·사회·자연·예술)',
      midterm_point: '중간고사 점수 (0~100)',
      homework_avg: '과제 점수 평균',
      online_lecture_count: '들은 온라인 강의 수',
      club_join: '동아리 활동 여부 (있음/없음)',
      final_point: '예측 목표 — 기말고사 점수 (0~100)',
    },
  },
  crop: {
    label: '농작물 수확량 예측', short: '수확량',
    task: 'regression', target: 'output_kg_per_ha', unit: 'kg/ha',
    files: ['crop_output.csv'], df: 'farm_df',
    derived: [{ name: 'nk_ratio', label: '질소 ÷ 칼리 비율', a: 'n_level', b: 'k_level' }],
    background: '여러 농장의 한 해 기상·토양·비료 기록으로 1헥타르당 수확량을 예측한다. '
      + '비가 얼마나 왔는지가 가장 크게 작용하고, 흙의 종류에 따라 수확량이 층을 이룬다.',
    columns: {
      rain_mm_year: '한 해 강수량 (mm)',
      mean_temp: '평균 기온 (℃)',
      sun_hours: '한 해 일조 시간 (시간)',
      soil_kind: '흙의 종류 (사질토·양토·점토)',
      soil_ph: '흙의 산도 (pH)',
      fertilizer: '주로 쓴 비료 (질소·복합·칼리·인산)',
      n_level: '흙 속 질소 수치',
      p_level: '흙 속 인 수치',
      k_level: '흙 속 칼리 수치',
      output_kg_per_ha: '예측 목표 — 1헥타르당 수확량 (kg)',
    },
  },
  car: {
    label: '중고차 시세 예측', short: '중고차',
    task: 'regression', target: 'resale_price_manwon', unit: '만원',
    files: ['car_resale.csv'], df: 'resale_df',
    background: '중고차 거래 기록으로 차의 적정 거래가를 예측한다. '
      + '오래 탄 차일수록 값이 크게 떨어지고, 서로 붙어 다니는 변수 짝이 두 개 들어 있다.',
    columns: {
      maker: '제조사 (가상의 네 회사)',
      used_years: '사용 연수 (년)',
      total_distance_km: '총 주행 거리 (km)',
      power_source: '동력원 (휘발유·경유·전기·하이브리드)',
      displacement_cc: '배기량 (cc)',
      max_power_ps: '최고 출력 (마력)',
      fuel_economy_kmpl: '연비 (km/L)',
      repair_count: '사고 수리 횟수',
      body_color: '차체 색',
      trim_level: '트림 등급 (스탠다드·플러스·프리미엄)',
      resale_price_manwon: '예측 목표 — 거래가 (만원)',
    },
  },
  ad: {
    label: '광고 성과 예측', short: '광고 성과',
    task: 'regression', target: 'return_rate_percent', unit: '%',
    files: ['ad_return.csv'], df: 'promo_df',
    derived: [{ name: 'click_rate', label: '클릭률 = 클릭 수 ÷ 노출 수', a: 'click_count', b: 'views' }],
    background: '광고 캠페인 기록으로 들인 돈 대비 얼마를 벌었는지(성과율)를 예측한다. '
      + '성과율은 손해(음수)와 이익(양수)이 섞여 있고, 노출 → 클릭 → 가입으로 이어지는 숫자들이 서로 얽혀 있다.',
    columns: {
      media: '광고 매체 (검색·방송·라디오·SNS)',
      budget_manwon: '광고비 (만원)',
      run_days: '광고 기간 (일)',
      age_group: '겨냥한 나이대',
      views: '노출 수',
      click_count: '클릭 수',
      signup_count: '가입 수',
      return_rate_percent: '예측 목표 — 성과율 (%, 음수면 손해)',
    },
  },
  pay: {
    label: '직원 연봉 예측', short: '연봉',
    task: 'regression', target: 'annual_pay_manwon', unit: '만원',
    files: ['staff_pay.csv'], df: 'pay_df',
    ordinal: { degree: { 고졸: 0, 학사: 1, 석사: 2, 박사: 3 }, job_rank: { 사원: 0, 선임: 1, 책임: 2, 수석: 3 }, foreign_language: { 초급: 0, 중급: 1, 고급: 2 } },
    background: '한 회사의 인사 기록으로 직원의 연봉을 예측한다. '
      + '학력·직급·외국어처럼 「순서가 있는」 범주가 세 개 있어서, 인코딩 방식을 고르는 연습을 하기에 좋다.',
    columns: {
      team: '부서 (경영지원·영업·연구개발·마케팅)',
      career_years: '경력 (년)',
      degree: '최종 학력 (고졸 < 학사 < 석사 < 박사)',
      job_rank: '직급 (사원 < 선임 < 책임 < 수석)',
      review_score: '인사 평가 점수',
      project_count: '참여한 프로젝트 수',
      team_headcount: '부서 인원',
      edu_hours: '한 해 교육 이수 시간',
      foreign_language: '외국어 수준 (초급 < 중급 < 고급)',
      annual_pay_manwon: '예측 목표 — 연봉 (만원)',
    },
  },
  yield: {
    label: '공정 양품률 예측', short: '양품률',
    task: 'regression', target: 'good_rate_percent', unit: '%',
    files: ['process_yield.csv'], df: 'proc_df',
    note: '이 데이터는 일부러 신호가 약하다. 어떤 모델을 써도 검증 점수가 낮은데, 「예측이 안 되는 데이터도 있다」를 보여 주려고 그렇게 만들었다.',
    background: '반도체 공정 기록으로 한 묶음의 양품률을 예측한다. '
      + '센서값이 많지만 대부분 양품률과 거의 관계가 없다. 좋은 모델을 써도 잘 맞히지 못하는 경우를 겪어 보는 데이터다.',
    columns: {
      line_code: '생산 라인 (L1·L2)',
      material_grade: '원료 등급 (M1·M2·M3)',
      chamber_temp: '챔버 온도',
      chamber_pressure: '챔버 압력',
      chamber_humidity: '챔버 습도',
      vibration_level: '진동 수준',
      cycle_time_sec: '한 번 가공하는 데 걸린 시간 (초)',
      power_kw: '전력 사용량 (kW)',
      worker_years: '작업자 경력 (년)',
      good_rate_percent: '예측 목표 — 양품률 (%)',
    },
  },
  home: {
    label: '주택 거래가 예측', short: '주택',
    task: 'regression', target: 'deal_price_eok', unit: '억 원',
    files: ['home_value_a.csv', 'home_value_b.csv'], df: 'home_df',
    mergeKey: 'home_id',
    idColumns: ['home_id'],
    note: '파일이 두 개다. home_id 를 열쇠(key)로 두 표를 나란히 붙여서 쓴다.',
    background: '가상 도시의 주택 거래 기록으로 거래가를 예측한다. '
      + '집의 기본 정보와 주변 환경 정보가 두 파일로 나뉘어 있어 먼저 하나로 합쳐야 하고, 결측치가 있는 컬럼도 많다.',
    columns: {
      home_id: '주택 번호 (두 파일을 잇는 열쇠)',
      district: '구 (가상의 지명)',
      floor_area_sqm: '전용 면적 (㎡)',
      room_count: '방 개수',
      bath_count: '욕실 개수',
      built_years: '지은 지 몇 년 (년)',
      frame_material: '구조 (목구조·벽돌조·철근콘크리트)',
      parking_slots: '주차 가능 대수',
      subway_within_1km: '1km 안 지하철역 수',
      shops_within_500m: '500m 안 상점 수',
      school_rating: '주변 학교 평가 (1~10)',
      deal_price_eok: '예측 목표 — 거래가 (억 원)',
    },
  },
  churn: {
    label: '가입자 이탈 예측', short: '가입자 이탈',
    task: 'classification', target: 'left_service',
    files: ['subscriber_churn.csv'], df: 'sub_df',
    classNames: { '0': '유지', '1': '이탈' },
    ordinal: { plan_term: { 무약정: 0, '12개월': 1, '24개월': 2 } },
    background: '통신 서비스 가입자 기록으로 누가 서비스를 떠날지(이탈) 예측한다. '
      + '이탈과 유지가 거의 반반이라 분류를 처음 해 보기에 알맞다. 숫자 컬럼 하나에 글자가 섞여 있다.',
    columns: {
      months_joined: '가입 기간 (개월)',
      plan_term: '약정 (무약정·12개월·24개월)',
      pay_method: '납부 방법',
      net_product: '함께 쓰는 인터넷 상품',
      monthly_fee_won: '월 요금 (원)',
      total_paid_won: '지금까지 낸 돈 (원) — 숫자 아닌 값이 섞여 있다',
      data_gb: '한 달 데이터 사용량 (GB)',
      cs_calls: '고객센터 상담 횟수',
      addon_count: '부가 서비스 개수',
      left_service: '예측 목표 — 이탈 여부 (1 이탈 · 0 유지)',
    },
  },
  loan: {
    label: '대출 심사 결과 예측', short: '대출 심사',
    task: 'classification', target: 'approved',
    files: ['loan_result.csv'], df: 'credit_df',
    classNames: { '0': '거절', '1': '승인' },
    ordinal: { house_type: { 자가: 0, 월세: 1, 전세: 2 }, past_overdue: { 없음: 0, 있음: 1 } },
    background: '대출 신청 기록으로 심사 결과(승인·거절)를 예측한다. '
      + '승인이 열에 아홉이라 살짝 불균형하다. 정확도 말고 정밀도와 재현율이 왜 필요한지 느껴 볼 수 있다.',
    columns: {
      job_type: '직업 형태 (미취업·정규직·계약직·자영업)',
      year_income_manwon: '연 소득 (만원)',
      credit_point: '신용 점수 (300~1000)',
      loan_reason: '대출 목적',
      loan_manwon: '신청 금액 (만원)',
      debt_ratio: '부채 비율 (%)',
      job_years: '근속 연수 (년)',
      house_type: '주거 형태 (자가·월세·전세)',
      past_overdue: '예전 연체 기록 (없음/있음)',
      approved: '예측 목표 — 심사 결과 (1 승인 · 0 거절)',
    },
  },
  quality: {
    label: '제품 불량 예측', short: '제품 불량',
    task: 'classification', target: 'is_defect',
    files: ['product_quality.csv'], df: 'qc_df',
    classNames: { '0': '정상', '1': '불량' },
    ordinal: { work_shift: { 낮: 0, 밤: 1 } },
    derived: [{ name: 'gauge_ratio', label: '계측값 A ÷ B', a: 'gauge_a', b: 'gauge_b' }],
    note: '불량이 1% 뿐이다. 정확도만 보면 안 되는 까닭을 배우는 데이터다.',
    background: '공장의 생산 기록으로 제품이 불량인지 예측한다. '
      + '불량은 100개 중 1개꼴이다. 「전부 정상」이라고만 찍어도 정확도가 99% 가 나오는 함정을 직접 겪어 본다.',
    columns: {
      line_no: '생산 설비 (1호기·2호기·3호기)',
      work_shift: '근무조 (낮·밤)',
      worker_years: '작업자 경력 (년)',
      work_min: '작업 시간 (분)',
      temp_c: '작업장 온도 (℃)',
      humid_percent: '작업장 습도 (%)',
      gauge_a: '계측값 A',
      gauge_b: '계측값 B',
      gauge_c: '계측값 C',
      is_defect: '예측 목표 — 불량 여부 (1 불량 · 0 정상)',
    },
  },
};

/** 회차 번호를 붙인 최종 정보 — 회차는 DATASET_IDS 순서에서 나온다 */
export const META = Object.fromEntries(
  DATASET_IDS.map((key, i) => [key, { ...BASE[key], round: i + 1 }]));

/** 컬럼 설명 사전 */
export function describeColumns(key) {
  return { ...(META[key]?.columns || {}) };
}

/** 배경 문단 */
export function backgroundOf(key) {
  return META[key]?.background || '';
}

/** 표시용 파일 이름 */
export function fileLabel(key) {
  const m = META[key];
  return m.files.length > 1 ? m.files.join(' + ') : m.files[0];
}
