"""
전력사용량 예측 파이프라인 (3분할 버전 + 1~2월 제외 + Validation 분석)
- 1~2월 데이터 제외하고 학습 (3~11월 데이터만 사용)
- 휴무일 / 가동일-야간 / 가동일-주간 3개로 분리
- 각 그룹별로 XGBoost, LightGBM, CatBoost 튜닝 (75 trials)
- 총 9개 모델 앙상블
- Validation 일치도 상세 분석 추가
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
from lightgbm import LGBMRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 100)
print("🚀 전력사용량 예측 파이프라인 (3분할 + 1~2월 제외 + Validation 분석)")
print("=" * 100)

# ============================================================================
# STEP 1: 데이터 로드
# ============================================================================
print("\n[STEP 1] 데이터 로드")
print("-" * 100)

train = pd.read_csv("./data/raw/train.csv")
test = pd.read_csv("./data/raw/test.csv")

print(f"✓ Train shape: {train.shape}")
print(f"✓ Test shape: {test.shape}")

# ============================================================================
# STEP 2: 데이터 전처리
# ============================================================================
print("\n[STEP 2] 데이터 전처리")
print("-" * 100)

# 결측치 처리
train['단가'] = train['단가'].fillna(0)

# 측정일시를 datetime으로 변환
train['측정일시'] = pd.to_datetime(train['측정일시'])
test['측정일시'] = pd.to_datetime(test['측정일시'])

# 1~2월 데이터 확인 및 제외
jan_feb_count = len(train[train['month'].isin([1, 2])])
print(f"\n✓ 1~2월 데이터: {jan_feb_count}건 ({jan_feb_count/len(train)*100:.1f}%)")

# 1~2월 제외
train = train[~train['month'].isin([1, 2])].copy()
print(f"✓ 1~2월 제외 후 Train shape: {train.shape}")
print(f"✓ 사용 기간: 3~11월")

# 휴무일 이상치 제거
holiday_data = train[train['작업휴무'] == '휴무'].copy()
Q1 = holiday_data['전력사용량(kWh)'].quantile(0.25)
Q3 = holiday_data['전력사용량(kWh)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = ((holiday_data['전력사용량(kWh)'] < lower_bound) | 
            (holiday_data['전력사용량(kWh)'] > upper_bound))
holiday_clean = holiday_data[~outliers]
working_data = train[train['작업휴무'] == '가동'].copy()

train = pd.concat([working_data, holiday_clean], axis=0).sort_values('id').reset_index(drop=True)
print(f"✓ 이상치 제거: {outliers.sum()}건")

# ============================================================================
# STEP 3: 강화된 파생변수 생성
# ============================================================================

def create_enhanced_features(df, is_train=True, train_stats=None):
    """강화된 파생변수 생성"""
    df = df.copy()
    
    # === 기본 인코딩 ===
    df['시간대_인코딩'] = (df['시간대'] == '주간').astype(int)
    df['역률곱_역수'] = 1 / (df['지상역률(%)'] * df['진상역률(%)'] + 1e-10)
    
    시간대2_mapping = {
        '심야': 0, '심야전환': 1, '점심': 2,
        '저녁': 3, '오후근무': 4, '오전근무': 5
    }
    df['시간대2_인코딩'] = df['시간대2'].map(시간대2_mapping)
    
    작업유형_mapping = {
        'Light_Load': 0, 'Medium_Load': 1, 'Maximum_Load': 2
    }
    df['작업유형_인코딩'] = df['작업유형'].map(작업유형_mapping)
    
    # === 주기성 변수 ===
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_year'] = df['측정일시'].dt.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # === 기타 변수 ===
    df['heating_need'] = df['기온'].apply(lambda x: max(0, 15 - x))
    df['기온_hour_interaction'] = df['기온'] * df['hour']
    df['기온_구간'] = pd.cut(df['기온'], bins=[-20, 0, 10, 20, 40], labels=[0, 1, 2, 3])
    df['기온_구간'] = df['기온_구간'].astype(int)
    df['작업유형_hour'] = df['작업유형_인코딩'] * df['hour']
    df['역률곱'] = df['지상역률(%)'] * df['진상역률(%)']
    
    # === 통계 변수 (리키지 방지 - 10월까지만 사용) ===
    if is_train:
        stats = {}
        train_for_stats = df[df['month'] <= 10]
        stats['시간대2_평균전력'] = train_for_stats.groupby('시간대2')['전력사용량(kWh)'].mean().to_dict()
        stats['작업유형_평균전력'] = train_for_stats.groupby('작업유형')['전력사용량(kWh)'].mean().to_dict()
        stats['hour_평균전력'] = train_for_stats.groupby('hour')['전력사용량(kWh)'].mean().to_dict()
    else:
        stats = train_stats
    
    df['시간대2_평균전력'] = df['시간대2'].map(stats['시간대2_평균전력'])
    df['작업유형_평균전력'] = df['작업유형'].map(stats['작업유형_평균전력'])
    df['hour_평균전력'] = df['hour'].map(stats['hour_평균전력'])
    
    if is_train:
        return df, stats
    else:
        return df

train_featured, train_stats = create_enhanced_features(train, is_train=True)
test_featured = create_enhanced_features(test, is_train=False, train_stats=train_stats)

# ============================================================================
# STEP 4: Feature 목록 정의
# ============================================================================

feature_cols = [
    # 기본 변수
    'month', 'day', 'hour', 'minute', '기온',
    '지상역률(%)', '진상역률(%)',
    
    # 기존 파생변수
    '시간대_인코딩', '역률곱_역수', 
    '시간대2_인코딩', '작업유형_인코딩',
    'hour_sin', 'hour_cos', 'heating_need',
    
    # 주기성 변수
    'month_sin', 'month_cos',
    'day_of_year_sin', 'day_of_year_cos',
    
    # 강화 변수
    '기온_hour_interaction', '기온_구간',
    '작업유형_hour', '역률곱',
    
    # 통계 변수
    '시간대2_평균전력', '작업유형_평균전력', 'hour_평균전력'
]


# ============================================================================
# STEP 5: 데이터 분할 (3분할)
# ============================================================================
print("\n[STEP 5] 데이터 분할 (시간순 + 3분할)")
print("-" * 100)

train_data = train_featured[train_featured['month'] <= 10].copy()
val_data = train_featured[train_featured['month'] == 11].copy()


# 3분할: 휴무일 / 가동일-야간 / 가동일-주간
train_holiday = train_data[train_data['작업휴무'] == '휴무'].copy()
train_night = train_data[(train_data['작업휴무'] == '가동') & (train_data['시간대'] == '야간')].copy()
train_day = train_data[(train_data['작업휴무'] == '가동') & (train_data['시간대'] == '주간')].copy()

val_holiday = val_data[val_data['작업휴무'] == '휴무'].copy()
val_night = val_data[(val_data['작업휴무'] == '가동') & (val_data['시간대'] == '야간')].copy()
val_day = val_data[(val_data['작업휴무'] == '가동') & (val_data['시간대'] == '주간')].copy()


# Feature와 Target 분리
X_train_holiday = train_holiday[feature_cols]
y_train_holiday = train_holiday['전력사용량(kWh)']
X_val_holiday = val_holiday[feature_cols]
y_val_holiday = val_holiday['전력사용량(kWh)']

X_train_night = train_night[feature_cols]
y_train_night = train_night['전력사용량(kWh)']
X_val_night = val_night[feature_cols]
y_val_night = val_night['전력사용량(kWh)']

X_train_day = train_day[feature_cols]
y_train_day = train_day['전력사용량(kWh)']
X_val_day = val_day[feature_cols]
y_val_day = val_day['전력사용량(kWh)']

# ============================================================================
# STEP 6: 하이퍼파라미터 튜닝 함수 정의
# ============================================================================
def objective_xgb(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'reg:squarederror',
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict(X_val)
    return mean_absolute_error(y_val, pred)

def objective_lgb(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = model.predict(X_val)
    return mean_absolute_error(y_val, pred)

def objective_cat(trial, X_train, y_train, X_val, y_val):
    params = {
        'loss_function': 'MAE',
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_seed': 42,
        'verbose': False
    }
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    pred = model.predict(X_val)
    return mean_absolute_error(y_val, pred)

# ============================================================================
# STEP 7: 하이퍼파라미터 튜닝 (총 9개 모델)
# ============================================================================
print("\n[STEP 6] 하이퍼파라미터 튜닝 (XGBoost, LightGBM, CatBoost × 3그룹)")

# ============== 휴무일 ==============
print("\n[1/3] 휴무일 튜닝")
print("  XGBoost (75 trials)...")
study_xgb_holiday = optuna.create_study(direction='minimize')
study_xgb_holiday.optimize(
    lambda trial: objective_xgb(trial, X_train_holiday, y_train_holiday, X_val_holiday, y_val_holiday),
    n_trials=75, show_progress_bar=False
)
best_xgb_holiday_params = study_xgb_holiday.best_params
print(f"    ✓ Best MAE: {study_xgb_holiday.best_value:.4f}")

print("  LightGBM (75 trials)...")
study_lgb_holiday = optuna.create_study(direction='minimize')
study_lgb_holiday.optimize(
    lambda trial: objective_lgb(trial, X_train_holiday, y_train_holiday, X_val_holiday, y_val_holiday),
    n_trials=75, show_progress_bar=False
)
best_lgb_holiday_params = study_lgb_holiday.best_params
print(f"    ✓ Best MAE: {study_lgb_holiday.best_value:.4f}")

print("  CatBoost (75 trials)...")
study_cat_holiday = optuna.create_study(direction='minimize')
study_cat_holiday.optimize(
    lambda trial: objective_cat(trial, X_train_holiday, y_train_holiday, X_val_holiday, y_val_holiday),
    n_trials=75, show_progress_bar=False
)
best_cat_holiday_params = study_cat_holiday.best_params
print(f"    ✓ Best MAE: {study_cat_holiday.best_value:.4f}")

# ============== 가동일-야간 ==============
print("\n[2/3] 가동일-야간 튜닝")
print("  XGBoost (75 trials)...")
study_xgb_night = optuna.create_study(direction='minimize')
study_xgb_night.optimize(
    lambda trial: objective_xgb(trial, X_train_night, y_train_night, X_val_night, y_val_night),
    n_trials=75, show_progress_bar=False
)
best_xgb_night_params = study_xgb_night.best_params
print(f"    ✓ Best MAE: {study_xgb_night.best_value:.4f}")

print("  LightGBM (75 trials)...")
study_lgb_night = optuna.create_study(direction='minimize')
study_lgb_night.optimize(
    lambda trial: objective_lgb(trial, X_train_night, y_train_night, X_val_night, y_val_night),
    n_trials=75, show_progress_bar=False
)
best_lgb_night_params = study_lgb_night.best_params
print(f"    ✓ Best MAE: {study_lgb_night.best_value:.4f}")

print("  CatBoost (75 trials)...")
study_cat_night = optuna.create_study(direction='minimize')
study_cat_night.optimize(
    lambda trial: objective_cat(trial, X_train_night, y_train_night, X_val_night, y_val_night),
    n_trials=75, show_progress_bar=False
)
best_cat_night_params = study_cat_night.best_params
print(f"    ✓ Best MAE: {study_cat_night.best_value:.4f}")

# ============== 가동일-주간 ==============
print("\n[3/3] 가동일-주간 튜닝")
print("  XGBoost (75 trials)...")
study_xgb_day = optuna.create_study(direction='minimize')
study_xgb_day.optimize(
    lambda trial: objective_xgb(trial, X_train_day, y_train_day, X_val_day, y_val_day),
    n_trials=75, show_progress_bar=False
)
best_xgb_day_params = study_xgb_day.best_params
print(f"    ✓ Best MAE: {study_xgb_day.best_value:.4f}")

print("  LightGBM (75 trials)...")
study_lgb_day = optuna.create_study(direction='minimize')
study_lgb_day.optimize(
    lambda trial: objective_lgb(trial, X_train_day, y_train_day, X_val_day, y_val_day),
    n_trials=75, show_progress_bar=False
)
best_lgb_day_params = study_lgb_day.best_params
print(f"    ✓ Best MAE: {study_lgb_day.best_value:.4f}")

print("  CatBoost (75 trials)...")
study_cat_day = optuna.create_study(direction='minimize')
study_cat_day.optimize(
    lambda trial: objective_cat(trial, X_train_day, y_train_day, X_val_day, y_val_day),
    n_trials=75, show_progress_bar=False
)
best_cat_day_params = study_cat_day.best_params
print(f"    ✓ Best MAE: {study_cat_day.best_value:.4f}")

print("\n✓ 총 675 trials 튜닝 완료!")

# ============================================================================
# STEP 8: 최종 모델 학습
# ============================================================================
print("\n[STEP 7] 최종 모델 학습 (튜닝된 파라미터)")
print("-" * 100)

# 파라미터 준비
best_xgb_holiday_params.update({'objective': 'reg:squarederror', 'random_state': 42, 'n_jobs': -1})
best_lgb_holiday_params.update({'objective': 'regression', 'metric': 'mae', 'random_state': 42, 'n_jobs': -1, 'verbose': -1})
best_cat_holiday_params.update({'loss_function': 'MAE', 'random_seed': 42, 'verbose': False})

best_xgb_night_params.update({'objective': 'reg:squarederror', 'random_state': 42, 'n_jobs': -1})
best_lgb_night_params.update({'objective': 'regression', 'metric': 'mae', 'random_state': 42, 'n_jobs': -1, 'verbose': -1})
best_cat_night_params.update({'loss_function': 'MAE', 'random_seed': 42, 'verbose': False})

best_xgb_day_params.update({'objective': 'reg:squarederror', 'random_state': 42, 'n_jobs': -1})
best_lgb_day_params.update({'objective': 'regression', 'metric': 'mae', 'random_state': 42, 'n_jobs': -1, 'verbose': -1})
best_cat_day_params.update({'loss_function': 'MAE', 'random_seed': 42, 'verbose': False})

# 휴무일
print("  휴무일 모델 학습 중...")
xgb_holiday = xgb.XGBRegressor(**best_xgb_holiday_params)
xgb_holiday.fit(X_train_holiday, y_train_holiday, eval_set=[(X_val_holiday, y_val_holiday)], verbose=False)

lgb_holiday = LGBMRegressor(**best_lgb_holiday_params)
lgb_holiday.fit(X_train_holiday, y_train_holiday, eval_set=[(X_val_holiday, y_val_holiday)],
                callbacks=[lgb.early_stopping(50, verbose=False)])

cat_holiday = CatBoostRegressor(**best_cat_holiday_params)
cat_holiday.fit(X_train_holiday, y_train_holiday, eval_set=(X_val_holiday, y_val_holiday), verbose=False)

# 가동일-야간
print("  가동일-야간 모델 학습 중...")
xgb_night = xgb.XGBRegressor(**best_xgb_night_params)
xgb_night.fit(X_train_night, y_train_night, eval_set=[(X_val_night, y_val_night)], verbose=False)

lgb_night = LGBMRegressor(**best_lgb_night_params)
lgb_night.fit(X_train_night, y_train_night, eval_set=[(X_val_night, y_val_night)],
                callbacks=[lgb.early_stopping(50, verbose=False)])

cat_night = CatBoostRegressor(**best_cat_night_params)
cat_night.fit(X_train_night, y_train_night, eval_set=(X_val_night, y_val_night), verbose=False)

# 가동일-주간
print("  가동일-주간 모델 학습 중...")
xgb_day = xgb.XGBRegressor(**best_xgb_day_params)
xgb_day.fit(X_train_day, y_train_day, eval_set=[(X_val_day, y_val_day)], verbose=False)

lgb_day = LGBMRegressor(**best_lgb_day_params)
lgb_day.fit(X_train_day, y_train_day, eval_set=[(X_val_day, y_val_day)],
                callbacks=[lgb.early_stopping(50, verbose=False)])

cat_day = CatBoostRegressor(**best_cat_day_params)
cat_day.fit(X_train_day, y_train_day, eval_set=(X_val_day, y_val_day), verbose=False)

print("✓ 9개 모델 학습 완료")

# ============================================================================
# STEP 9: 앙상블 가중치 최적화
# ============================================================================
print("\n[STEP 8] 앙상블 가중치 최적화 (3그룹)")
print("-" * 100)

# 개별 예측
pred_xgb_holiday = xgb_holiday.predict(X_val_holiday)
pred_lgb_holiday = lgb_holiday.predict(X_val_holiday)
pred_cat_holiday = cat_holiday.predict(X_val_holiday)

pred_xgb_night = xgb_night.predict(X_val_night)
pred_lgb_night = lgb_night.predict(X_val_night)
pred_cat_night = cat_night.predict(X_val_night)

pred_xgb_day = xgb_day.predict(X_val_day)
pred_lgb_day = lgb_day.predict(X_val_day)
pred_cat_day = cat_day.predict(X_val_day)

# 휴무일 가중치 최적화
def objective_weights_holiday(weights):
    pred = weights[0]*pred_xgb_holiday + weights[1]*pred_lgb_holiday + weights[2]*pred_cat_holiday
    return mean_absolute_error(y_val_holiday, pred)

result_holiday = minimize(
    objective_weights_holiday,
    [0.33, 0.33, 0.34],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1}
)
optimal_weights_holiday = result_holiday.x

# 가동일-야간 가중치 최적화
def objective_weights_night(weights):
    pred = weights[0]*pred_xgb_night + weights[1]*pred_lgb_night + weights[2]*pred_cat_night
    return mean_absolute_error(y_val_night, pred)

result_night = minimize(
    objective_weights_night,
    [0.33, 0.33, 0.34],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1}
)
optimal_weights_night = result_night.x

# 가동일-주간 가중치 최적화
def objective_weights_day(weights):
    pred = weights[0]*pred_xgb_day + weights[1]*pred_lgb_day + weights[2]*pred_cat_day
    return mean_absolute_error(y_val_day, pred)

result_day = minimize(
    objective_weights_day,
    [0.33, 0.33, 0.34],
    bounds=[(0, 1), (0, 1), (0, 1)],
    constraints={'type': 'eq', 'fun': lambda w: sum(w) - 1}
)
optimal_weights_day = result_day.x

print(f"✓ 휴무일 최적 가중치: XGB={optimal_weights_holiday[0]:.3f}, LGB={optimal_weights_holiday[1]:.3f}, CAT={optimal_weights_holiday[2]:.3f}")
print(f"✓ 가동일-야간 최적 가중치: XGB={optimal_weights_night[0]:.3f}, LGB={optimal_weights_night[1]:.3f}, CAT={optimal_weights_night[2]:.3f}")
print(f"✓ 가동일-주간 최적 가중치: XGB={optimal_weights_day[0]:.3f}, LGB={optimal_weights_day[1]:.3f}, CAT={optimal_weights_day[2]:.3f}")

# 최적 가중치로 앙상블
pred_ensemble_holiday = (optimal_weights_holiday[0]*pred_xgb_holiday + 
                          optimal_weights_holiday[1]*pred_lgb_holiday + 
                          optimal_weights_holiday[2]*pred_cat_holiday)

pred_ensemble_night = (optimal_weights_night[0]*pred_xgb_night + 
                        optimal_weights_night[1]*pred_lgb_night + 
                        optimal_weights_night[2]*pred_cat_night)

pred_ensemble_day = (optimal_weights_day[0]*pred_xgb_day + 
                      optimal_weights_day[1]*pred_lgb_day + 
                      optimal_weights_day[2]*pred_cat_day)

# ============================================================================
# STEP 10: 전체 데이터로 최종 모델 학습
# ============================================================================
print("\n[STEP 11] 전체 Train 데이터로 최종 모델 학습")
print("-" * 100)

train_full_holiday = train_featured[train_featured['작업휴무'] == '휴무'].copy()
train_full_night = train_featured[(train_featured['작업휴무'] == '가동') & (train_featured['시간대'] == '야간')].copy()
train_full_day = train_featured[(train_featured['작업휴무'] == '가동') & (train_featured['시간대'] == '주간')].copy()

X_full_holiday = train_full_holiday[feature_cols]
y_full_holiday = train_full_holiday['전력사용량(kWh)']
X_full_night = train_full_night[feature_cols]
y_full_night = train_full_night['전력사용량(kWh)']
X_full_day = train_full_day[feature_cols]
y_full_day = train_full_day['전력사용량(kWh)']

# 휴무일
print("  휴무일 최종 모델 학습 중...")
final_xgb_holiday = xgb.XGBRegressor(**best_xgb_holiday_params)
final_xgb_holiday.fit(X_full_holiday, y_full_holiday, verbose=False)

final_lgb_holiday = LGBMRegressor(**best_lgb_holiday_params)
final_lgb_holiday.fit(X_full_holiday, y_full_holiday)

final_cat_holiday = CatBoostRegressor(**best_cat_holiday_params)
final_cat_holiday.fit(X_full_holiday, y_full_holiday, verbose=False)

# 가동일-야간
print("  가동일-야간 최종 모델 학습 중...")
final_xgb_night = xgb.XGBRegressor(**best_xgb_night_params)
final_xgb_night.fit(X_full_night, y_full_night, verbose=False)

final_lgb_night = LGBMRegressor(**best_lgb_night_params)
final_lgb_night.fit(X_full_night, y_full_night)

final_cat_night = CatBoostRegressor(**best_cat_night_params)
final_cat_night.fit(X_full_night, y_full_night, verbose=False)

# 가동일-주간
print("  가동일-주간 최종 모델 학습 중...")
final_xgb_day = xgb.XGBRegressor(**best_xgb_day_params)
final_xgb_day.fit(X_full_day, y_full_day, verbose=False)

final_lgb_day = LGBMRegressor(**best_lgb_day_params)
final_lgb_day.fit(X_full_day, y_full_day)

final_cat_day = CatBoostRegressor(**best_cat_day_params)
final_cat_day.fit(X_full_day, y_full_day, verbose=False)

print("✓ 최종 9개 모델 학습 완료")

# ============================================================================
# STEP 11: Test 데이터 예측
# ============================================================================
print("\n[STEP 12] Test 데이터 예측")
print("-" * 100)

test_holiday = test_featured[test_featured['작업휴무'] == '휴무'].copy()
test_night = test_featured[(test_featured['작업휴무'] == '가동') & (test_featured['시간대'] == '야간')].copy()
test_day = test_featured[(test_featured['작업휴무'] == '가동') & (test_featured['시간대'] == '주간')].copy()

X_test_holiday = test_holiday[feature_cols]
X_test_night = test_night[feature_cols]
X_test_day = test_day[feature_cols]

# 개별 모델 예측
pred_test_xgb_holiday = final_xgb_holiday.predict(X_test_holiday)
pred_test_lgb_holiday = final_lgb_holiday.predict(X_test_holiday)
pred_test_cat_holiday = final_cat_holiday.predict(X_test_holiday)

pred_test_xgb_night = final_xgb_night.predict(X_test_night)
pred_test_lgb_night = final_lgb_night.predict(X_test_night)
pred_test_cat_night = final_cat_night.predict(X_test_night)

pred_test_xgb_day = final_xgb_day.predict(X_test_day)
pred_test_lgb_day = final_lgb_day.predict(X_test_day)
pred_test_cat_day = final_cat_day.predict(X_test_day)

# 최적 가중치로 앙상블
pred_test_holiday = (optimal_weights_holiday[0]*pred_test_xgb_holiday + 
                      optimal_weights_holiday[1]*pred_test_lgb_holiday + 
                      optimal_weights_holiday[2]*pred_test_cat_holiday)

pred_test_night = (optimal_weights_night[0]*pred_test_xgb_night + 
                    optimal_weights_night[1]*pred_test_lgb_night + 
                    optimal_weights_night[2]*pred_test_cat_night)

pred_test_day = (optimal_weights_day[0]*pred_test_xgb_day + 
                  optimal_weights_day[1]*pred_test_lgb_day + 
                  optimal_weights_day[2]*pred_test_cat_day)

print(f"✓ 휴무일 예측 완료: {len(pred_test_holiday)}개")
print(f"✓ 가동일-야간 예측 완료: {len(pred_test_night)}개")
print(f"✓ 가동일-주간 예측 완료: {len(pred_test_day)}개")

# ============================================================================
# STEP 12: 후처리
# ============================================================================

# 실제 CSV 기반 회귀 계수 (1·2·11월 fitting 결과)
worktype_params = {
    "Light_Load":   {"a": 15.088489, "b": 1_072_302.0},
    "Medium_Load":  {"a": 2.404026,  "b": 1_096_892.0},
    "Maximum_Load": {"a": 2.732438,  "b": 1_107_024.0},
}

def predict_unit_price_by_work(work_type, lag_pf, lead_pf):
    """
    작업유형별 기본식 회귀계수에 따라 단가를 예측합니다.
    
    입력값:
        work_type : str ('Light_Load' / 'Medium_Load' / 'Maximum_Load')
        lag_pf    : float (지상역률 %, 예: 90.5)
        lead_pf   : float (진상역률 %, 예: 95.0)
    
    반환값:
        예측 단가 (float)
    """
    if work_type not in worktype_params:
        raise ValueError(f"'{work_type}'은(는) 유효한 작업유형이 아닙니다. "
                         f"허용값: {list(worktype_params.keys())}")
    
    a = worktype_params[work_type]["a"]
    b = worktype_params[work_type]["b"]
    # 역률곱 역수 계산 (0방지)
    inv_pf = 1.0 / max(lag_pf * lead_pf, 1e-6)
    
    # 단가 계산
    price = a + b * inv_pf
    return round(price, 3)


# 예측값 후처리
pred_test_holiday = np.maximum(pred_test_holiday, 0)
pred_test_night = np.maximum(pred_test_night, 0)
pred_test_day = np.maximum(pred_test_day, 0)

pred_test_holiday = np.minimum(pred_test_holiday, 5.0)
pred_test_night = np.minimum(pred_test_night, 15.0)

print("✓ 후처리 완료")
print(f"  - 휴무일: [0, 5] kWh")
print(f"  - 야간: [0, 15] kWh")
print(f"  - 주간: [0, ~] kWh")

# 예측값을 각 데이터프레임에 할당
test_holiday['전력사용량'] = pred_test_holiday
test_night['전력사용량'] = pred_test_night
test_day['전력사용량'] = pred_test_day

# concat 먼저 수행
test_result = pd.concat([test_holiday, test_night, test_day]).sort_values('id').reset_index(drop=True)


# ============================================================================
# STEP 13: 전기요금 계산 및 Submission
# ============================================================================
print("\n[STEP 15] 전기요금 계산 및 Submission 생성")
print("-" * 100)

# 단가 계산 (한 번만)

# 작업유형별 단가 계산
test_result['단가'] = test_result.apply(
    lambda row: predict_unit_price_by_work(
        row['작업유형'], 
        row['지상역률(%)'], 
        row['진상역률(%)']
    ), 
    axis=1
)

print("✓ 작업유형별 단가 계산 완료")
print(f"  - Light_Load 평균: {test_result[test_result['작업유형']=='Light_Load']['단가'].mean():.2f} 원/kWh")
print(f"  - Medium_Load 평균: {test_result[test_result['작업유형']=='Medium_Load']['단가'].mean():.2f} 원/kWh")
print(f"  - Maximum_Load 평균: {test_result[test_result['작업유형']=='Maximum_Load']['단가'].mean():.2f} 원/kWh")

# 전기요금 계산
test_result['전기요금'] = test_result['전력사용량'] * test_result['단가']

print("✓ 전기요금 계산 완료")


submission = pd.DataFrame({
    'id': test_result['id'],
    'target': test_result['전기요금']
})

submission.to_csv('sub보정완료1.csv', index=False)
