import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# ===== 사용자 설정 =====
csv_file_path = './data/raw/train.csv'
datetime_col = '측정일시'
value_col = '전력사용량(kWh)'
target_year = None                 # 예: 2024. None이면 데이터의 최댓값 연도 자동 선택
normalize_by_month_mean = False    # True면 월평균=1로 정규화(패턴 비교용)
palette_name = 'Set3'              # 'Set3' 또는 'tab20'

# ===== 폰트 =====
try:
    plt.rc('font', family='Malgun Gothic')   # Windows
except:
    try:
        plt.rc('font', family='AppleGothic') # macOS
    except:
        plt.rc('font', family='DejaVu Sans') # Fallback
plt.rcParams['axes.unicode_minus'] = False

# ===== 로드 & 전처리 =====
try:
    try:
        df = pd.read_csv(csv_file_path, encoding='cp949')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file_path, encoding='utf-8')

    # 기본 정리
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    df = df.dropna(subset=[datetime_col])
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

    # 연도 선택
    df['year'] = df[datetime_col].dt.year
    if target_year is None:
        target_year = int(df['year'].max())
    d = df[df['year'] == target_year].copy()

    # 월/요일 파생
    d['월'] = d[datetime_col].dt.month
    d['요일_idx'] = d[datetime_col].dt.weekday   # 월(0) ~ 일(6)
    weekday_labels = ['월','화','수','목','금','토','일']

    # === 1~11월만 사용 ===
    d = d[d['월'].between(1, 11)]

    # 요일 x 월 평균
    grp = d.groupby(['요일_idx','월'], as_index=False)[value_col].mean()

    # (옵션) 월평균=1 정규화 (패턴 비교)
    if normalize_by_month_mean:
        month_mean = grp.groupby('월')[value_col].transform('mean')
        grp[value_col] = grp[value_col] / month_mean

    # 피벗 (행=요일, 열=월)
    pv = grp.pivot_table(index='요일_idx', columns='월', values=value_col, aggfunc='mean').sort_index()
    pv.index = [weekday_labels[i] for i in pv.index]

    # ===== 색/마커 세팅: 1~11월 구분 확실히 =====
    cols = sorted(pv.columns.tolist())             # 실제 존재하는 월만
    cmap = cm.get_cmap(palette_name, 12)           # 충분히 구분되는 팔레트
    colors = [cmap(i) for i in range(len(cols))]
    markers = ['o','s','^','D','v','P','X','*','<','>','h'][:len(cols)]

    # ===== 플롯 =====
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, m in enumerate(cols):
        ax.plot(
            pv.index, pv[m],
            label=f'{m}월',
            color=colors[i],
            marker=markers[i],
            linewidth=2.3,
            markersize=6,
            alpha=0.95
        )

    title_core = f'{target_year}년 요일별 전력사용량 평균 — 월별 오버레이 (1~11월)'
    ax.set_title(
        title_core + (' / 월평균=1 정규화' if normalize_by_month_mean else ''),
        pad=12
    )
    ax.set_ylabel('상대 전력사용량 (월평균=1)' if normalize_by_month_mean else '전력사용량 (kWh)')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(title='월', ncol=4, frameon=True, framealpha=0.9,
              bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    print(f"[INFO] 사용 연도: {target_year}, 포함 월: {cols}")

except FileNotFoundError:
    print(f"오류: '{csv_file_path}' 파일을 찾을 수 없습니다. 경로를 확인하세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")
