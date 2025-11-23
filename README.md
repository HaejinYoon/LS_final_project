# ⚡ LSBS_Main_Project3_Electricity_Fare_Prediction
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

---

## 🏷️ 저자 및 라이선스

- **Project Owner:** LSBS Main Project 3 Team 4
- **Team Members:** 도호현, 안형엽, 오윤서, 윤해진  
- **License:** MIT License  
