⚡ LSBS_Main_Project3_Electricity_Fare_Prediction

전기요금 예측 및 전력 모니터링 대시보드 (4인)

공장 단위 15분 실측 전력 데이터를 활용해
전기요금을 예측하고 주요 전력 지표를 모니터링하는 AI 기반 대시보드 프로젝트

📘 프로젝트 개요

목적

전력 사용 패턴(전력사용량, 역률, 작업유형 등)에 기반한 전기요금 예측 모델 구축

예측 결과와 주요 지표를 Streamlit 대시보드로 제공해 피크 수요·요금 관리 지원

데이터

공장(산업체) 단위 15분 시계열 전력 사용 데이터

사용 기간: 2024년 1–11월 학습, 12월 전기요금 예측

성과

MAE 기준 상위권 성능의 회귀 모델 개발

실시간 지표 요약, 피크 진단, 자동 보고서 생성 기능을 포함한 웹 대시보드 구현

🧩 데이터 구성
파일명	설명
train.csv	학습용 데이터 (전력사용량, 전기요금, 작업유형, 역률 등)
test.csv	예측 대상 기간 데이터
sample_submission.csv	제출 형식 예시

주요 컬럼 예시

컬럼명	설명
측정일시	전력 측정 시간 (15분 간격)
전력사용량_kWh	구간별 전력 사용량
전력요금	해당 구간의 전기요금 (타깃 변수)
역률	전력 효율 지표
작업유형	생산 / 휴식 / 점검 등 공정 상태 구분

※ 기업 보안상 실제 데이터는 repo에 포함되어 있지 않을 수 있으며, 재현 시 별도 경로 설정이 필요합니다.

🛠 사용 기술 (Tech Stack)

언어 & 라이브러리

Python, NumPy, Pandas, Scikit-learn

시각화: Matplotlib, Plotly

모델링

LightGBM, XGBoost, CatBoost, Histogram-GBR

NeuralForecast 기반 시계열 딥러닝 모델(N-HiTS/TFT, 실험용)

서비스 & 협업

Streamlit (웹 대시보드)

Git / GitHub (버전 관리 및 협업)

🔄 워크플로우

데이터 수집 & 이해

공장 15분 전력 사용·요금·역률·작업유형 데이터 구조 분석

EDA & 전처리

이상치/결측치 처리, 공정별/시간대별 사용 패턴 탐색

피처 엔지니어링

시간 피처(hour, dayofweek, month, holiday)

사인/코사인 변환을 통한 주기성 인코딩(Fourier Features)

전력단가(원/kWh), 피크 수요, 집계 통계(일·주·월) 생성

모델 학습 & 검증

TimeSeriesSplit 기반 검증 전략 설정

구간별(예: 주간/야간/휴일 등) 모델 분리 학습

LightGBM/XGBoost/CatBoost 성능 비교 후 가중 앙상블(Hybrid Ensemble) 구성

대시보드 & 보고서

예측 결과 및 주요 지표를 Streamlit 대시보드로 서비스

월별 요약 리포트를 Word 문서로 자동 생성

🤖 모델링 / 예측

목표 지표

타깃: 전력요금(원)

평가: MAE(Median Absolute Error) 최소화

검증 전략

11월 데이터를 검증 구간으로 고정하고
TimeSeriesSplit으로 시계열 누적 검증 수행

모델 조합

개별 모델: LightGBM, XGBoost, CatBoost, HGBR

최종: 모델별 성능(검증 MAE)에 비례한 가중 앙상블

파이프라인 예시

# 1. 전처리
X_train, y_train = preprocess(train)
X_valid, y_valid = preprocess(valid)
X_test = preprocess(test)

# 2. 모델 학습
lgbm = LGBMRegressor(**lgbm_params)
xgb  = XGBRegressor(**xgb_params)
cat  = CatBoostRegressor(**cat_params, verbose=0)

for model in [lgbm, xgb, cat]:
    model.fit(X_train, y_train)

# 3. 예측 및 앙상블
pred_lgbm = lgbm.predict(X_valid)
pred_xgb  = xgb.predict(X_valid)
pred_cat  = cat.predict(X_valid)

# 예시: 성능 기반 가중 평균
pred_valid = 0.4 * pred_lgbm + 0.35 * pred_xgb + 0.25 * pred_cat

📊 Streamlit 대시보드
탭 구성
탭	설명
Tab1. 주요 지표 요약	기간별 전력 사용량·요금·단가·피크 수요를 KPI 카드와 그래프로 표시
Tab2. 상세 분석 / 보고서	요일·시간대·작업유형별 사용 패턴, 역률 분석, 피크 구간 시각화, Word 보고서 자동 생성
Tab3. 요금 예측 결과	예측 vs 실제 비교, 모델별 예측값 비교, CSV 다운로드 기능 제공
주요 기능

실시간 모니터링

선택한 기간/공정에 대해 전력 사용량, 요금, 단가, 역률, 피크 수요를 한 화면에서 조회

피크 진단

시간대별·요일별 피크 발생 구간 하이라이트

기준선을 초과한 구간을 색상 경고로 표시

자동 보고서 생성

한 달 단위 주요 지표 요약 + 그래프를 Word 문서(.docx)로 자동 생성

관리자가 바로 보고용으로 활용 가능

👤 나의 역할 (Team of 4)

팀장

문제 정의, 분석 방향 설정, 전체 일정·협업 관리

데이터 분석 & 피처 엔지니어링

이상치/결측치 처리, 시간·공정·역률 기반 피처 설계

전력단가 및 피크 관련 파생 변수 설계

모델링

LightGBM / XGBoost / CatBoost 튜닝 및 가중 앙상블 구성

TimeSeriesSplit 기반 검증 전략 설계 및 MAE 기준 최적 모델 선정

대시보드 기획·구현

대시보드 화면 구조 설계(Tab 구성, KPI 카드, 그래프 레이아웃)

Streamlit 앱 주요 기능(필터, 그래프, 보고서 생성) 구현 및 사용성 테스트

🚀 실행 방법
# 1. 가상환경 생성 및 라이브러리 설치
conda create -n elec python=3.10
conda activate elec
pip install -r requirements.txt

# 2. 대시보드 실행
cd dashboard
streamlit run app.py

🧠 향후 개선 계획

전력요금제(시간대별/계약전력별)를 세분화한 요금 구조 반영

LSTM, TFT 등 시계열 특화 딥러닝 모델 추가 검증

피크 알람 및 이상치 탐지 기능 추가

MLOps 관점에서 주기적 재학습·배포 파이프라인 설계

이대로 그대로 복붙하니까 예쁘게 안 나와 사진처럼 나오게 해줘

"# ⚡ LSBS_Main_Project3_Electricity_Fare_Prediction
> 전력 사용 패턴 기반의 전기요금 예측 및 시각화 대시보드 프로젝트  
> *Electricity Fare Prediction Based on Power Consumption Data*

---

## 📘 프로젝트 개요

본 프로젝트는 **전력 사용량 데이터를 활용하여 전기요금을 예측**하고,  
이를 **시계열 분석 및 시각화 대시보드**로 제공하는 것을 목표로 합니다.

- **주제:** 전력 사용량 및 역률, 작업유형 등의 피처를 기반으로 전기요금 예측  
- **목표:** MAE(Median Absolute Error) 최소화  
- **데이터:** 공장/산업체 단위의 15분 단위 전력 사용량 시계열 데이터  
- **결과물:** 전력 사용 추세 분석, 피크 수요 진단, 자동 보고서 생성 기능이 포함된 Streamlit 대시보드

---



## 🧩 데이터 구성

| 파일명 | 설명 |
|--------|------|
| `train.csv` | 학습용 데이터 (전력 사용량, 전력요금, 작업유형, 역률 등) |
| `test.csv` | 예측용 데이터 |
| `sample_submission.csv` | 제출 형식 예시 |

**주요 컬럼 예시**
| 컬럼명 | 설명 |
|--------|------|
| 측정일시 | 전력 측정 시간 (15분 단위) |
| 전력사용량(kWh) | 구간별 전력 사용량 |
| 전력요금(원) | 해당 구간의 전력요금 (타깃 변수) |
| 역률 | 전력 효율 지표 |
| 작업유형 | 생산공정 / 휴식 / 점검 등 공정 상태 구분 |

---

## ⚙️ 데이터 전처리 및 피처 엔지니어링

1. **결측치 처리:** 이상값, 누락 데이터 보정  
2. **시간 피처 생성:**  
   - `hour`, `dayofweek`, `month`, `holiday`  
   - `sin/cos` 변환으로 주기적 패턴 반영 (Fourier Features)
3. **전력단가(원/kWh)** 추가 피처 생성  
4. **집계 피처:**  
   - 일별, 주별, 월별 평균 사용량  
   - 피크 수요 (상위 5%) 및 최소 사용량 등  
5. **라벨 인코딩:** 범주형 작업유형 처리

---

## 🤖 모델링

| 모델 | 설명 |
|------|------|
| LightGBM | 기본형 회귀 모델, 빠른 학습 속도 |
| CatBoost | 범주형 피처 처리에 강점 |
| Hybrid Ensemble | LightGBM + CatBoost + HGBR 가중 앙상블 |
| Neural Forecast (실험) | 시계열 기반 딥러닝(NHiTS/TFT) 테스트 |

**학습 파이프라인 요약**
```python
# 1. 전처리
X_train, y_train = preprocess(train)
X_test = preprocess(test)

# 2. 모델 학습
model = LGBMRegressor(**best_params)
model.fit(X_train, y_train)

# 3. 예측 및 제출
pred = model.predict(X_test)
submission = pd.DataFrame({'id': test['id'], 'target': pred})
submission.to_csv('./submissions/submission_final.csv', index=False)
```

---

## 📊 Streamlit 대시보드

| 탭 | 설명 |
|----|------|
| **Tab1. 주요지표 요약** | 기간별 전력 사용량, 요금, 단가, 피크 수요 등 주요 지표 |
| **Tab2. 상세 분석 및 보고서 생성** | 요일/시간대별 패턴, 역률 분석, 자동 Word 보고서 생성 |
| **Tab3. 요금 예측 결과** | 모델별 예측값 비교 및 다운로드 기능 |

---

## 🧾 자동 보고서 생성 기능

- **모듈:** `report_generator.py`
- **형식:** `.docx` (Word 문서)
- **포함 내용:**
  - 주요 지표 요약
  - 요일/시간대별 전력 사용 패턴
  - 피크 수요 지표
  - 전력 사용량 추이 그래프
  - 개선 제안 자동 생성

---

## 🎨 시각화 예시

- 기간별 전력 사용 추이  
- 요일·시간대별 평균 사용량 Heatmap  
- 피크 수요 표시 마커  
- 단가 변화 추이 Dual Line Chart  
- 역률 비교 Gauge / Scatter

---

## 🚀 실행 방법

```bash
conda create -n elec python=3.10
conda activate elec
pip install -r requirements.txt

cd dashboard
streamlit run app.py
```

---

## 🧠 향후 개선 계획

- LSTM 기반 시계열 모델 추가  
- 전력요금제별 세분화 반영  
- 이상치 탐지 및 알림 기능 추가  
- AutoML 기반 파라미터 최적화  

  "

이게 원래 사용하던 탬플릿인데 이거 참고해
