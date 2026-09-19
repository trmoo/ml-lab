# -*- coding: utf-8 -*-
# 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유
# 학교 수업 목적으로만 이용해 주세요. 자세한 내용은 LICENSE 파일을 보세요.
"""실습 데이터를 만든다 — python tools/make_data.py

numpy 난수로 **처음부터 생성**하는 가상 데이터다. 실제 인물·기관·제품과 관계가 없다.

데이터마다 「가르칠 거리」가 하나씩 들어가도록 설계했다.
  · 회귀 7종 · 분류 3종
  · 예측 목표와 강하게 이어진 변수 하나, 거의 무관한 변수 하나
  · 서로 |r| ≥ 0.8 인 변수 짝 (다중공선성 이야기용)
  · 컬럼마다 다른 개수의 결측치
  · 제품 불량(quality)은 불량이 1% 뿐인 극단적 불균형 (정확도의 함정 이야기용)

씨앗을 고정했으므로 다시 돌려도 같은 데이터가 나온다.
"""
import csv
import sys
from pathlib import Path

import numpy as np

# 열쇠별 파일 이름 — 수업 순서(회차 번호)는 파일 이름에 넣지 않는다
FILES = {
    'exam': ['exam_result.csv'],
    'crop': ['crop_output.csv'],
    'car': ['car_resale.csv'],
    'ad': ['ad_return.csv'],
    'pay': ['staff_pay.csv'],
    'yield': ['process_yield.csv'],
    'home': ['home_value_a.csv', 'home_value_b.csv'],
    'churn': ['subscriber_churn.csv'],
    'loan': ['loan_result.csv'],
    'quality': ['product_quality.csv'],
}

# 열쇠별 행 개수
ROWS = {'exam': 1150, 'crop': 870, 'car': 960, 'ad': 480, 'pay': 770,
        'yield': 1920, 'home': 1440, 'churn': 2400, 'loan': 1720, 'quality': 2880}

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data'


# ── 도우미 ────────────────────────────────────────────────
def pick(rng, levels, weights, n):
    """범주형 값 뽑기"""
    p = np.array(weights, dtype=float)
    return rng.choice(levels, size=n, p=p / p.sum())


def holes(rng, values, n_missing):
    """결측치를 n_missing 개 뚫는다 (예측 목표에는 쓰지 않는다)"""
    out = list(values)
    if n_missing <= 0:
        return out
    idx = rng.choice(len(out), size=n_missing, replace=False)
    for i in idx:
        out[i] = None
    return out


def sprinkle_junk(rng, values, n_junk, tokens=('-', '미확인')):
    """숫자 컬럼에 「숫자가 아닌 값」을 몇 개 섞는다.

    실제 업무 데이터에는 숫자 자리에 '-' 나 '미확인' 같은 글자가 섞여 들어와
    pandas 가 그 컬럼을 통째로 object 타입으로 읽는 일이 흔하다.
    pd.to_numeric(errors='coerce') 를 가르치려면 그런 데이터가 실제로 있어야 한다.
    """
    out = list(values)
    spots = [i for i, v in enumerate(out) if v is not None]
    pick_at = rng.choice(len(spots), size=min(n_junk, len(spots)), replace=False)
    for k, at in enumerate(pick_at):
        out[spots[at]] = tokens[k % len(tokens)]
    return out


def rnd(a, digits=0):
    """소수점 자리를 맞춘다. 0이면 정수로."""
    a = np.asarray(a, dtype=float)
    if digits == 0:
        return np.round(a).astype(int)
    return np.round(a, digits)


def fmt_col(values, digits):
    """CSV 에 쓸 글자로 바꾼다. 결측치는 빈칸."""
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append('')
        elif isinstance(v, (int, np.integer)):
            out.append(str(int(v)))
        elif isinstance(v, (float, np.floating)):
            out.append(f'{v:.{digits}f}' if digits else str(int(round(v))))
        else:
            out.append(str(v))
    return out


def write_csv(name, columns):
    """columns = [(이름, 값들, 소수점자리), ...]"""
    path = OUT / name
    cols = [(c, fmt_col(v, d)) for c, v, d in columns]
    n = len(cols[0][1])
    with path.open('w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow([c for c, _ in cols])
        for i in range(n):
            w.writerow([v[i] for _, v in cols])
    print(f'  {name:28s} {n}행 × {len(cols)}열')
    return path


def noise(rng, n, sd):
    return rng.normal(0, sd, n)


# ── car · 중고차 시세 ───────────────────────────────────
def make_car(seed=20260703, n=None):
    """사용연수가 값을 크게 끌어내리고, 사용연수↔주행거리 / 배기량↔출력이 각각 붙어 있다."""
    n = n or ROWS['car']
    rng = np.random.default_rng(seed)

    used_years = rng.integers(0, 16, n)                    # 0~15년
    # 주행거리는 사용연수에 붙는다 → 다중공선성 짝 ① (|r| ≈ 0.86 을 노린다)
    total_km = np.clip(used_years * 14500 + noise(rng, n, 39000) + 12000, 900, None)
    # 배기량과 출력도 서로 붙는다 → 다중공선성 짝 ② (|r| ≈ 0.92)
    displacement = np.clip(rng.normal(2450, 880, n), 980, 4200)
    max_power = np.clip(displacement * 0.096 + noise(rng, n, 36) + 12, 70, 430)
    # 연비는 배기량이 클수록 낮아진다. 값과는 약한 음의 상관만 남는다
    fuel_economy = np.clip(24 - displacement * 0.0042 + noise(rng, n, 3.4), 3.0, 24.0)
    repair = rng.choice([0, 1, 2, 3, 4], n, p=[.56, .18, .13, .08, .05])

    maker = pick(rng, ['한빛', '가온', '다온', '라온'], [.30, .28, .19, .23], n)
    power_source = pick(rng, ['휘발유', '경유', '전기', '하이브리드'], [.27, .24, .25, .24], n)
    body_color = pick(rng, ['블랙', '실버', '화이트', '그외'], [.29, .24, .22, .25], n)
    trim = pick(rng, ['스탠다드', '플러스', '프리미엄'], [.35, .31, .34], n)

    # 범주형이 값에 주는 차이는 넉넉히 벌려 둔다.
    # 차이가 작으면 막대그래프에서 「어느 쪽이 가장 높은가」를 눈으로 읽을 수 없다.
    trim_mult = np.select([trim == '스탠다드', trim == '플러스'], [0.80, 1.00], 1.28)
    maker_mult = np.select(
        [maker == '한빛', maker == '가온', maker == '다온'], [1.04, 1.14, 0.90], 0.98)
    src_mult = np.select(
        [power_source == '경유', power_source == '전기', power_source == '하이브리드'],
        [1.32, 1.04, 0.96], 0.84)

    # 사용연수가 오를수록 값이 빠진다 → 강한 음의 상관.
    # 감쇠를 너무 급하게 잡으면 값 분포가 극단으로 치우쳐, 관계가 없는 컬럼에서도
    # 그룹 평균이 크게 흔들려 보인다. 완만하게 잡았다.
    base = 6900 * np.exp(-0.155 * used_years)
    price = base * trim_mult * maker_mult * src_mult
    # 배기량이 큰 차는 조금 더 비싸다 → 연비와 값 사이에 약한 음의 상관이 생긴다
    price = price * (1 + (displacement - 2450) / 2450 * 0.13)
    price = price * (1 - 0.045 * repair) + noise(rng, n, 210)
    price = np.clip(price, 100, None)

    cols = [
        ('maker', holes(rng, maker, 44), 0),
        ('used_years', rnd(used_years), 0),
        ('total_distance_km', rnd(total_km), 0),
        ('power_source', holes(rng, power_source, 25), 0),
        ('displacement_cc', rnd(displacement), 0),
        ('max_power_ps', rnd(max_power), 0),
        ('fuel_economy_kmpl', rnd(fuel_economy, 1), 1),
        ('repair_count', repair, 0),
        ('body_color', holes(rng, body_color, 36), 0),
        ('trim_level', holes(rng, trim, 30), 0),
        ('resale_price_manwon', rnd(price, 1), 1),
    ]
    write_csv(FILES['car'][0], cols)


# ── home · 주택 거래가 (파일 두 개) ──────────────────────
def make_home(seed=20260731):
    """면적이 값을 끌어올리고, 구(區)에 따라 같은 면적에서도 값대가 층을 이룬다."""
    n = ROWS['home']
    rng = np.random.default_rng(seed)

    home_id = np.arange(1, n + 1)
    district = pick(rng, ['별빛구', '한들구', '미르구', '솔빛구', '그외'],
                    [.27, .21, .20, .22, .10], n)
    floor_area = np.clip(rng.normal(124, 42, n), 45, 205)
    # 면적과 방 개수를 바짝 붙여 둔다 → 다중공선성(|r| ≥ 0.8) 이야기용 짝
    room_count = np.clip(np.round(floor_area / 34 + rng.normal(0, 0.32, n)), 1, 6).astype(int)
    bath_count = np.clip(np.round(room_count / 2 + rng.normal(0, 0.5, n)), 1, 3).astype(int)
    built_years = rng.integers(0, 41, n)
    frame = pick(rng, ['목구조', '벽돌조', '철근콘크리트'], [.33, .33, .34], n)
    parking = np.clip(np.round(rng.normal(1.8, 0.62, n)), 1, 3).astype(int)
    subway = rng.choice([0, 1, 2, 3], n, p=[.22, .28, .28, .22])
    shops = rng.integers(0, 16, n)
    school = rng.integers(1, 11, n)

    # 구마다 값대가 다르게 — 별빛구·한들구가 비싸다 (산점도로 읽어 낼 거리)
    dist_add = np.select(
        [district == '별빛구', district == '한들구', district == '미르구', district == '솔빛구'],
        [11.5, 8.2, 2.0, 0.5], -1.5)
    frame_add = np.select([frame == '철근콘크리트', frame == '벽돌조'], [1.4, 0.4], 3.1)

    price = (7.0 + floor_area * 0.105 + dist_add + frame_add
             - built_years * 0.135 + parking * 2.2 + subway * 1.15
             + shops * 0.05 + school * 0.28 + noise(rng, n, 2.3))
    price = np.clip(price, 5.0, None)

    write_csv(FILES['home'][0], [
        ('home_id', home_id, 0),
        ('district', holes(rng, district, 38), 0),
        ('floor_area_sqm', rnd(floor_area), 0),
        ('room_count', holes(rng, room_count, 47), 0),
        ('bath_count', holes(rng, bath_count, 29), 0),
        ('built_years', holes(rng, built_years, 36), 0),
    ])
    write_csv(FILES['home'][1], [
        ('home_id', home_id, 0),
        ('frame_material', holes(rng, frame, 52), 0),
        ('parking_slots', holes(rng, parking, 41), 0),
        ('subway_within_1km', holes(rng, subway, 44), 0),
        ('shops_within_500m', shops, 0),
        ('school_rating', school, 0),
        ('deal_price_eok', rnd(price, 1), 1),
    ])


# ── pay · 직원 연봉 ─────────────────────────────────────
def make_pay(seed=20260732):
    """경력이 연봉을 거의 결정한다. 성과점수·프로젝트 수는 거의 무관하게 둔다."""
    n = ROWS['pay']
    rng = np.random.default_rng(seed)

    career = rng.integers(0, 26, n)
    degree = pick(rng, ['고졸', '학사', '석사', '박사'], [.09, .52, .28, .11], n)
    rank_from_career = np.select(
        [career < 5, career < 11, career < 18], ['사원', '선임', '책임'], '수석')
    # 직급은 경력을 따라가되 가끔 어긋난다
    shuffle = rng.random(n) < 0.18
    job_rank = np.where(shuffle,
                        pick(rng, ['사원', '선임', '책임', '수석'], [.25, .25, .25, .25], n),
                        rank_from_career)
    team = pick(rng, ['경영지원', '영업', '연구개발', '마케팅'], [.25, .24, .26, .25], n)
    review = np.clip(rng.normal(80, 12, n), 58, 100)
    projects = rng.integers(1, 22, n)
    headcount = rng.integers(0, 16, n)
    edu_hours = rng.integers(8, 101, n)
    lang = pick(rng, ['초급', '중급', '고급'], [.33, .30, .37], n)

    degree_add = np.select([degree == '고졸', degree == '학사', degree == '석사'],
                           [-520, 0, 430], 880)
    rank_add = np.select([job_rank == '사원', job_rank == '선임', job_rank == '책임'],
                         [0, 380, 780], 1250)
    team_add = np.select([team == '연구개발', team == '영업', team == '마케팅'],
                         [260, 120, 40], 0)
    lang_add = np.select([lang == '고급', lang == '중급'], [190, 70], 0)

    # 경력이 거의 결정하지만 완전히 결정하지는 않게 잡음을 넉넉히 준다 (r ≈ 0.92)
    pay = (4050 + career * 165 + degree_add + rank_add + team_add + lang_add
           + noise(rng, n, 480))
    pay = np.clip(pay, 3600, None)

    write_csv(FILES['pay'][0], [
        ('team', holes(rng, team, 26), 0),
        ('career_years', career, 0),
        ('degree', holes(rng, degree, 28), 0),
        ('job_rank', holes(rng, job_rank, 34), 0),
        ('review_score', holes(rng, rnd(review, 1), 25), 1),
        ('project_count', holes(rng, projects, 30), 0),
        ('team_headcount', holes(rng, headcount, 24), 0),
        ('edu_hours', edu_hours, 0),
        ('foreign_language', lang, 0),
        ('annual_pay_manwon', rnd(pay, 0), 0),
    ])


# ── yield · 공정 양품률 ───────────────────────────────────
def make_yield(seed=20260733):
    """센서값 대부분은 양품률과 거의 무관하고, 작업자 경력만 조금 이어진다.
    「상관이 약할 때 모델이 얼마나 못 맞히는가」를 보여 주는 데이터다."""
    n = ROWS['yield']
    rng = np.random.default_rng(seed)

    line = pick(rng, ['L1', 'L2'], [.48, .52], n)
    material = pick(rng, ['M1', 'M2', 'M3'], [.36, .33, .31], n)
    temp = rng.normal(100, 5.2, n)
    pressure = rng.normal(50, 2.1, n)
    humid = np.clip(rng.normal(50, 11.8, n), 28, 72)
    vibration = np.clip(rng.normal(1.0, 0.20, n), 0.30, 1.70)
    cycle = np.clip(rng.normal(120, 9.9, n), 84, 160)
    power = np.clip(rng.normal(198, 22, n), 118, 268)
    worker = rng.integers(1, 12, n)

    line_add = np.where(line == 'L2', 0.22, -0.10)
    mat_add = np.select([material == 'M1', material == 'M2'], [0.18, -0.05], -0.12)
    good = (93.6 + worker * 0.105 + line_add + mat_add
            - np.abs(temp - 100) * 0.012 - np.abs(pressure - 50) * 0.010
            + noise(rng, n, 1.02))
    good = np.clip(good, 90.0, 98.5)

    write_csv(FILES['yield'][0], [
        ('line_code', line, 0),
        ('material_grade', material, 0),
        ('chamber_temp', holes(rng, rnd(temp, 2), 52), 2),
        ('chamber_pressure', holes(rng, rnd(pressure, 2), 66), 2),
        ('chamber_humidity', holes(rng, rnd(humid, 1), 74), 1),
        ('vibration_level', rnd(vibration, 4), 4),
        ('cycle_time_sec', holes(rng, rnd(cycle, 1), 58), 1),
        ('power_kw', rnd(power, 2), 2),
        ('worker_years', worker, 0),
        ('good_rate_percent', rnd(good, 2), 2),
    ])


# ── ad · 광고 성과 ─────────────────────────────────────
def make_ad(seed=20260734):
    """가입 전환 수가 성과와 가장 많이 이어진다. 성과는 음수와 양수가 섞이게 둔다."""
    n = ROWS['ad']
    rng = np.random.default_rng(seed)

    media = pick(rng, ['검색', '방송', '라디오', 'SNS'], [.29, .22, .24, .25], n)
    budget = rng.integers(200, 20100, n)                  # 만원
    run_days = rng.integers(7, 91, n)
    age_group = pick(rng, ['10대', '20·30대', '40·50대', '전연령'],
                     [.24, .30, .23, .23], n)

    # 매체마다 노출 효율과 클릭률이 다르다.
    # 매체 사이 격차를 너무 벌리면 성과율이 하한에 몰려 「중앙값이 가장 높은 매체」를
    # 물을 수 없게 된다. 검색이 가장 좋고 라디오가 가장 나쁘도록 완만하게 잡는다.
    view_rate = np.select([media == '검색', media == '방송', media == '라디오'],
                          [92.0, 74.0, 68.0], 85.0)
    views = np.clip(budget * view_rate * (1 + noise(rng, n, 0.20)), 8000, None)
    ctr = np.select([media == '검색', media == '방송', media == '라디오'],
                    [0.026, 0.019, 0.017], 0.023) * (1 + noise(rng, n, 0.22))
    clicks = np.clip(views * np.clip(ctr, 0.0008, None), 40, None)
    cvr = np.select([age_group == '20·30대', age_group == '40·50대'],
                    [0.085, 0.075], 0.062) * (1 + noise(rng, n, 0.25))
    signups = np.clip(clicks * np.clip(cvr, 0.006, None), 3, None)

    # 성과율 — 가입 1건이 벌어 주는 값(만원)으로 계산한다. 음수와 양수가 섞이게 잡았다
    value_per_signup = 9.2 * (1 + noise(rng, n, 0.12))
    revenue = signups * value_per_signup
    ret = (revenue - budget) / budget * 100 + noise(rng, n, 8.0)
    ret = np.clip(ret, -95, 320)

    write_csv(FILES['ad'][0], [
        ('media', media, 0),
        ('budget_manwon', holes(rng, budget, 17), 0),
        ('run_days', holes(rng, run_days, 24), 0),
        ('age_group', age_group, 0),
        ('views', rnd(views), 0),
        ('click_count', rnd(clicks), 0),
        ('signup_count', rnd(signups), 0),
        ('return_rate_percent', rnd(ret, 2), 2),
    ])


# ── exam · 기말고사 점수 ─────────────────────────────────
def make_exam(seed=20260735):
    """중간고사 점수와 주당 공부시간이 기말 점수를 크게 설명한다."""
    n = ROWS['exam']
    rng = np.random.default_rng(seed)

    study = rng.integers(1, 42, n)
    attend = np.clip(rng.normal(76, 14.5, n), 48, 100)
    extra = pick(rng, ['있음', '없음'], [.30, .70], n)
    track = pick(rng, ['어문', '공학', '사회', '자연', '예술'],
                 [.24, .21, .20, .18, .17], n)
    homework = np.clip(rng.normal(79, 12, n), 58, 100)
    online = rng.integers(0, 52, n)

    ability = (study * 1.02 + attend * 0.18 + homework * 0.16
               + np.where(extra == '있음', 3.4, 0) + noise(rng, n, 6.0))
    # 중간고사 점수답게 0~100 에 고르게 퍼지도록 잡는다.
    # (예전 척도는 평균 20점·최대 66점이라 시험 점수로 보이지 않았다)
    midterm = np.clip(ability * 1.50 - 20 + noise(rng, n, 8.0), 0, 100)
    club = pick(rng, ['있음', '없음'], [.50, .50], n)

    final = np.clip(18 + ability * 0.40 + midterm * 0.30 + noise(rng, n, 7.5), 0, 100)

    write_csv(FILES['exam'][0], [
        ('study_hours_week', study, 0),
        ('attend_percent', rnd(attend, 1), 1),
        ('extra_class', extra, 0),
        ('track', track, 0),
        ('midterm_point', holes(rng, rnd(midterm, 1), 45), 1),
        ('homework_avg', rnd(homework, 1), 1),
        ('online_lecture_count', online, 0),
        ('club_join', holes(rng, club, 51), 0),
        ('final_point', rnd(final, 1), 1),
    ])


# ── crop · 농작물 수확량 ─────────────────────────────────
def make_crop(seed=20260736):
    """연간 강수량이 수확량을 가장 크게 좌우한다. 토양 종류마다 수확량 층이 다르다."""
    n = ROWS['crop']
    rng = np.random.default_rng(seed)

    rain = rng.integers(480, 2050, n)
    temp = np.clip(rng.normal(17.5, 4.3, n), 9.5, 26.0)
    sun = rng.integers(1450, 3050, n)
    soil = pick(rng, ['사질토', '양토', '점토'], [.36, .33, .31], n)
    ph = np.clip(rng.normal(6.5, 0.85, n), 4.8, 8.2)
    fert = pick(rng, ['질소', '복합', '칼리', '인산'], [.24, .28, .24, .24], n)
    n_level = rng.uniform(10, 100, n)
    p_level = rng.uniform(5, 50, n)
    k_level = rng.uniform(5, 50, n)

    soil_add = np.select([soil == '양토', soil == '사질토'], [190, 60], -140)
    fert_add = np.select([fert == '복합', fert == '질소', fert == '칼리'],
                         [130, 60, 20], -30)
    out = (1180 + rain * 0.44 + sun * 0.16 + soil_add + fert_add
           - np.abs(ph - 6.5) * 38 + noise(rng, n, 118))
    out = np.clip(out, 1200, None)

    write_csv(FILES['crop'][0], [
        ('rain_mm_year', rain, 0),
        ('mean_temp', holes(rng, rnd(temp, 1), 25), 1),
        ('sun_hours', holes(rng, sun, 34), 0),
        ('soil_kind', soil, 0),
        ('soil_ph', rnd(ph, 1), 1),
        ('fertilizer', fert, 0),
        ('n_level', rnd(n_level, 2), 2),
        ('p_level', rnd(p_level, 2), 2),
        ('k_level', rnd(k_level, 2), 2),
        ('output_kg_per_ha', rnd(out), 0),
    ])


# ── churn · 가입자 이탈 (분류) ────────────────────────────
def make_churn(seed=20260737):
    """가입 기간이 짧고 요금이 비싸고 상담 전화가 많으면 떠난다."""
    n = ROWS['churn']
    rng = np.random.default_rng(seed)

    months = rng.integers(1, 75, n)
    plan = pick(rng, ['무약정', '12개월', '24개월'], [.30, .29, .41], n)
    pay_method = pick(rng, ['카드', '지류고지서', '계좌이체'], [.31, .34, .35], n)
    net = pick(rng, ['기가', '광랜', '없음'], [.33, .34, .33], n)
    fee = np.clip(rng.normal(86000, 37000, n), 19000, 152000)
    total = np.clip(fee * months * (1 + noise(rng, n, 0.10)), 20000, None)
    data_gb = rng.integers(8, 505, n)
    cs = rng.choice(np.arange(0, 11), n,
                    p=np.array([12, 11, 11, 10, 10, 9, 9, 8, 7, 7, 6]) / 100)
    addon = rng.choice([0, 1, 2, 3, 4, 5], n, p=[.16, .17, .18, .18, .16, .15])

    plan_z = np.select([plan == '무약정', plan == '12개월'], [0.9, 0.1], -0.8)
    logit = (1.95 - months * 0.058 + (fee - 86000) / 33000 * 0.80
             + cs * 0.19 - addon * 0.16 + plan_z * 0.60
             + np.where(pay_method == '지류고지서', 0.30, -0.05)
             + noise(rng, n, 0.40))
    p = 1 / (1 + np.exp(-logit))
    left = (rng.random(n) < p).astype(int)

    write_csv(FILES['churn'][0], [
        ('months_joined', holes(rng, months, 74), 0),
        ('plan_term', holes(rng, plan, 92), 0),
        ('pay_method', holes(rng, pay_method, 65), 0),
        ('net_product', holes(rng, net, 57), 0),
        ('monthly_fee_won', holes(rng, rnd(fee), 83), 0),
        # 숫자 자리에 '-'·'미확인' 이 섞여 있어 pandas 가 object 로 읽는다.
        # pd.to_numeric(errors='coerce') 가 실제로 필요해진다.
        ('total_paid_won', sprinkle_junk(rng, holes(rng, rnd(total), 69), 46), 0),
        ('data_gb', holes(rng, data_gb, 71), 0),
        ('cs_calls', holes(rng, cs, 66), 0),
        ('addon_count', holes(rng, addon, 78), 0),
        ('left_service', left, 0),
    ])
    print(f'      이탈 비율 {left.mean() * 100:.1f}%')


# ── loan · 대출 심사 (분류) ──────────────────────────────
def make_loan(seed=20260738):
    """신용점수가 높고 부채비율이 낮으면 승인된다. 승인률이 95% 가까워 조금 불균형하다."""
    n = ROWS['loan']
    rng = np.random.default_rng(seed)

    job = pick(rng, ['미취업', '정규직', '계약직', '자영업'], [.10, .47, .20, .23], n)
    income = rng.integers(1900, 20500, n)
    credit = np.clip(rng.normal(650, 200, n), 300, 1000)
    reason = pick(rng, ['주택', '생활', '사업', '그외'], [.23, .26, .24, .27], n)
    amount = rng.integers(100, 10200, n)
    debt = np.clip(rng.normal(245, 145, n), 1, 520)
    job_years = rng.integers(0, 32, n)
    house = pick(rng, ['자가', '월세', '전세'], [.34, .33, .33], n)
    overdue = pick(rng, ['없음', '있음'], [.80, .20], n)

    logit = (3.35 + (credit - 650) / 200 * 1.05 - (debt - 245) / 145 * 0.95
             + np.where(overdue == '있음', -0.85, 0.15)
             + np.where(job == '미취업', -0.9, 0.1)
             + noise(rng, n, 0.5))
    p = 1 / (1 + np.exp(-logit))
    approved = (rng.random(n) < p).astype(int)

    write_csv(FILES['loan'][0], [
        ('job_type', holes(rng, job, 66), 0),
        ('year_income_manwon', holes(rng, income, 61), 0),
        ('credit_point', holes(rng, rnd(credit), 58), 0),
        ('loan_reason', holes(rng, reason, 52), 0),
        ('loan_manwon', holes(rng, amount, 47), 0),
        ('debt_ratio', holes(rng, rnd(debt, 1), 53), 1),
        ('job_years', holes(rng, job_years, 60), 0),
        ('house_type', holes(rng, house, 68), 0),
        ('past_overdue', holes(rng, overdue, 57), 0),
        ('approved', approved, 0),
    ])
    print(f'      승인 비율 {approved.mean() * 100:.1f}%')


# ── quality · 제품 불량 (극단적 불균형 분류) ───────────────
def make_quality(seed=20260739):
    """불량이 1% 뿐이다. 정확도의 함정을 보여 주는 데이터.

    불량에 **실제 신호를 넣었다.** 온도가 정상 범위를 벗어나고
    작업자 경력이 짧고 계측값 A 가 튀면 불량이 될 확률이 오른다.
    이렇게 해야 「임계값을 낮추면 불량을 찾아낸다」는 수업이 성립한다.
    """
    n = ROWS['quality']
    rng = np.random.default_rng(seed)

    line = pick(rng, ['1호기', '2호기', '3호기'], [.37, .33, .30], n)
    shift = pick(rng, ['낮', '밤'], [.49, .51], n)
    worker = rng.integers(0, 17, n)
    work_min = np.clip(rng.normal(30, 5.0, n), 13, 49)
    temp = rng.normal(25, 3.1, n)
    humid = np.clip(rng.normal(50, 10.0, n), 15, 87)
    gauge_a = rng.normal(100, 9.9, n)
    gauge_b = rng.normal(199, 19.8, n)
    gauge_c = rng.normal(50, 5.1, n)

    # 위험 점수 — 높은 온도 · 짧은 경력 · 계측값 A 이탈 · 야간 근무 · 3호기
    #
    # 온도는 「높을수록 위험」(단조)을 크게, 「정상에서 벗어날수록 위험」(U자)을 작게 섞었다.
    # 단조 성분이 있어야 선형 모델과 상관계수도 신호를 잡을 수 있고,
    # U자 성분이 있어야 트리가 선형 모델보다 잘 맞히는 모습을 보여 줄 수 있다.
    risk = ((temp - 25) / 3.1 * 1.15
            + np.abs(temp - 25) / 3.1 * 0.45
            + np.maximum(0, 5 - worker) * 0.40
            + np.abs(gauge_a - 100) / 9.9 * 0.95
            + np.where(shift == '밤', 0.50, 0)
            + np.where(line == '3호기', 0.40, 0))
    logit = -9.4 + risk * 1.30 + noise(rng, n, 0.30)
    p = 1 / (1 + np.exp(-logit))
    defect = (rng.random(n) < p).astype(int)

    write_csv(FILES['quality'][0], [
        ('line_no', line, 0),
        ('work_shift', shift, 0),
        ('worker_years', holes(rng, worker, 93), 0),
        ('work_min', holes(rng, rnd(work_min, 1), 81), 1),
        ('temp_c', rnd(temp, 1), 1),
        ('humid_percent', rnd(humid, 1), 1),
        ('gauge_a', rnd(gauge_a, 2), 2),
        ('gauge_b', rnd(gauge_b, 2), 2),
        ('gauge_c', rnd(gauge_c, 2), 2),
        ('is_defect', defect, 0),
    ])
    print(f'      불량 {defect.sum()}건 / {n}건 = {defect.mean() * 100:.2f}%')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(parents=True, exist_ok=True)
    print('실습 데이터를 새로 만듭니다.\n')
    # 수업 순서와 무관하게 데이터는 열쇠 이름으로 만든다.
    make_exam()
    make_crop()
    make_car()
    make_ad()
    make_pay()
    make_yield()
    make_home()
    make_churn()
    make_loan()
    make_quality()
    print('\n끝. data/ 아래 파일들을 확인하세요.')


if __name__ == '__main__':
    main()
