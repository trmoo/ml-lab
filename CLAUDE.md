<!-- 머신러닝 실험실 — © 2026 티쳐무 · 모든 권리 보유 -->
# 머신러닝 실험실 (ml-lab)

## 앱의 목적과 대상 단원
- 고등학교 정보 · 인공지능 기초 · 데이터 과학의 머신러닝 단원.
- **파이썬 설치 없이 브라우저에서 머신러닝이 실제로 학습된다.** 모델 10종을 자바스크립트로 직접 구현했다.
- **공개 배포용 앱이다.** 데이터·코드·문제 은행이 모두 독자적으로 만든 것이라 공개 저장소 + Pages 로 배포한다.

## ⚠ 가장 중요한 규칙 — 문제 은행은 독자적으로
- `quiz/` 의 문제를 쓰거나 고칠 때 **다른 앱·문제집·강의 자료의 문제를 참고하지 않는다.**
  출발점은 ① 이 앱의 데이터와 설계 의도 ② 일반적인 머신러닝 지식 ③ pandas·scikit-learn 공식 사용법뿐이다.
- 규칙 전문은 `docs/문제-작성-규칙.md`. 문제를 쓰는 서브에이전트에게도 이 문서를 먼저 읽힌다.
- **`tools/local/originality.py`**(로컬 전용, `.gitignore`)가 이 PC 에만 있는 옛 자료와 새 문제를 견준다.
  코드 유사도 0.80 · 글(5글자 조각 포함률) 0.35 를 넘으면 멈춘다. 옛 문제를 미끼로 넣어
  **미끼가 안 걸리면 검사기 고장으로 실패**한다. 문제를 고친 뒤에는 꼭 돌릴 것.
  - 글 비교에서는 영문·숫자 낱말(컬럼 이름)을 뺀다 — 같은 데이터를 쓰면 이름은 당연히 겹친다.
  - 2026-09-19 첫 검사: 코드 최대 0.68, 글 최대 0.27(한 문제 0.39 → 다시 써서 0.16). 뼈대(API 호출 차례) 0.12~0.37.

## 기능 (탭 5개)
1. 데이터 탐험실 2. 전처리 실험실 3. 모델 학습실 4. 딥러닝 실험실 — 옛 수업용 앱에서 가져온 것(직접 만든 코드).
5. **문제 풀기** — 10회차 · **96문제**(고르기 43 · 빈칸 37 · 오류 찾기 11 · 순서 맞추기 5, 빈칸 65개).
   회차마다 그 데이터에만 있는 「가르칠 거리」가 중심이다(규칙 문서 §5 표).
   문제의 `lab` 단추가 같은 데이터로 실험실 탭을 연다 (`src/lib/nav.js` → 각 탭의 `useDataset()`).

## 구조
```
data/                  가상 CSV 11개 (tools/make_data.py 가 만든다. 다시 돌려도 같은 파일)
quiz/<열쇠>.json        문제 은행. 회차 하나 = 파일 하나
docs/문제-작성-규칙.md    문제 JSON 모양 · 코드 규칙 · 회차별 목표
src/data/meta.js       데이터 설명(컬럼·배경)과 수업 순서 DATASET_IDS
src/data/quiz-parse.js 문제 JSON → 화면용 (노드 시험도 쓴다. DOM 금지)
src/data/quiz.js       quiz/*.json 을 불러 묶는다
src/tabs/quiz.js       문제 풀기 탭 (네 가지 모양)
tools/verify_quiz.py   문제 코드를 진짜 파이썬으로 실행 + check 식 확인
tools/local/           로컬 전용 (저장소에 없음) — originality.py
```

## 구현에서 짚어 둘 점
- **빈칸 표시는 `⟦정답⟧`** (U+27E6/27E7). `[[ ]]` 를 쓰면 파이썬 `df[['a','b']]` 와 부딪힌다.
- **문제 코드는 pandas 2.2(코랩)와 3 에서 똑같이 돌아야 한다** — `inplace=True` 금지(시험이 막는다).
  pandas 3 에서 글자 컬럼은 `str` 로 찍혀 `object` 가 아니다. 자료형 이름을 정답으로 묻지 않는다.
- `verify_quiz.py` 는 seaborn·matplotlib·tensorflow 를 가짜 모듈로 바꿔 끼운다. keras 문제는 `"run": false` 로
  문법만 본다. FutureWarning 은 오류로 다룬다. check 식은 파이썬 `bool`/numpy `bool_` 만 받는다.
- `bug` 문제는 `fix` 로 바꾼 코드가 실행된다. 잘못된 코드가 다음 문제에 새지 않게 `"sandbox": true` 를 쓴다.
- 탭을 다시 그릴 때는 `redraw(root, …)`, 슬라이더의 화면 전체 다시 그리기는 `onChange` 에 (옛 앱과 같은 규칙).
- 회차 번호를 코드에 박지 않는다. 순서는 `meta.js` 의 `DATASET_IDS` 하나가 정한다.

## 명령
```bash
npm install
npm start              # 개발 서버 http://localhost:5187
npm run build          # dist/index.html 한 파일 (약 856KB, 데이터·문제 포함)
npm test               # 계산 점검 81가지 + 문제 은행 모양 점검 102가지
npm run verify         # 문제 96개를 파이썬으로 실행해 정답 확인
node test/sweep.mjs    # 데이터 10종 × 모델 전수 학습
PYTHONUTF8=1 python tools/local/originality.py   # 원본 대조 (로컬 전용)
```
- `grep -c 티쳐무 dist/index.html` 이 3 이상이어야 한다(vite.config.js · 워크플로가 검사).

## 현재 개발 상태 (2026-09-19)
- 탭 5개와 문제 은행 96문제 완성. `npm test`·`npm run verify`·원본 대조 모두 통과.
- 문제는 exam 회차를 먼저 직접 써서 본보기로 삼고, 나머지 9회차를 서브에이전트 5개(Sonnet 5)가 규칙 문서만 보고 썼다.
- 배포된 화면에서 네 가지 문제 모양의 채점(맞음·틀림), 96문제 전부 그리기·해설 열기,
  「실험실에서 직접 해 보기」가 같은 데이터로 탭을 여는 것까지 확인했다. 콘솔 오류 0건.

## 저장소 · 배포 (2026-09-19)
- 저장소: https://github.com/trmoo/ml-lab (공개) · Pages: https://trmoo.github.io/ml-lab/
- 포털 `comedu_portal` 의 「[인공지능] 학습 자료」에 🧪 머신러닝 실험실로 올렸다.
- Pages 는 처음부터 「GitHub Actions」(`build_type: workflow`)라 Jekyll 잡이 생기지 않았다.
- 워크플로: `npm ci → npm test → 파이썬 정답 점검(pip pandas scikit-learn) → sweep → build → 점검 → 배포`.
- ⚠⚠ **첫 푸시에서 파이썬 정답 점검이 깨졌다** — 로지스틱 회귀에 **표준화 없이** 원(₩) 단위 컬럼을 넣었더니
  최적화가 불안정해 깃허브 서버와 이 PC 의 숫자가 달랐다(churn-09 · quality-08/09).
  학생의 코랩에서도 똑같이 어긋날 수 있다. `make_pipeline(StandardScaler(), LogisticRegression(…))` 로 바꾸고
  solver 넷(lbfgs·newton-cg·newton-cholesky·saga)에서 같은 값이 나오는 것을 확인했다.
  **새 문제에 로지스틱 회귀·신경망처럼 반복 최적화하는 모델을 쓰면 반드시 표준화할 것.**
- ⚠ **임계값 문제는 확률이 임계값에 붙어 있지 않은지 볼 것** — 0.3 은 확률 하나가 0.0045 차이라 0.2(여유 0.034)로 바꿨고,
  `check` 에 `float(abs(proba - 0.2).min()) > 0.01` 을 넣어 지킨다. AUC 처럼 셋째 자리에서 갈리는 값은 둘째 자리로 비교한다.
- 배포본이 로컬 `dist/index.html` 보다 약 32KB 작다 — 로컬 CSV 가 CRLF 라서다(깃이 LF 로 올림). 내용은 같다.
- 확인: `curl -s https://trmoo.github.io/ml-lab/ | wc -c` 가 약 1,051,000 이면 정상(1KB 미만이면 브랜치 배포가 이긴 것).
