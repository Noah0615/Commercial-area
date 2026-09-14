# -*- coding: utf-8 -*-
"""
===============================================================================
  상권 조기경보 AI 모델 (Commercial District Early-Warning AI)
  - Google Colab 실행용 전체 파이프라인
  - 2021~2024 학습 → 2025 GT 검증
===============================================================================

📌 Google Colab 사용법:
  1. 이 파일(.py)을 Colab에 업로드하거나, .ipynb로 변환하여 사용
  2. CSV 파일들을 Google Drive에 업로드 후 마운트
  3. 셀 단위로 나누어 실행 (### CELL 주석 기준)

📌 필요 라이브러리 (Colab에서 자동 설치):
  pip install torch torch-geometric scikit-learn pandas numpy matplotlib seaborn
"""

### CELL 1: 환경 설정 및 라이브러리 설치 =========================================
# Google Colab에서 실행 시 아래 주석을 해제하세요

# !pip install torch-geometric
# from google.colab import drive
# drive.mount('/content/drive')

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict

# 시각화 설정
plt.rcParams['font.family'] = 'DejaVu Sans'  # Colab에서는 'NanumGothic' 사용 가능
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

print("✅ 기본 라이브러리 로드 완료")

### CELL 2: 데이터 경로 설정 =====================================================
# ⚠️ Google Drive 마운트 후 경로를 수정하세요
# Colab 사용 시: BASE_DIR = '/content/drive/MyDrive/가명정보활용경진대회'
# 로컬 사용 시: 아래 경로 유지

BASE_DIR = '/Users/jeonjaewon/Desktop/project/학부연구실/가명정보활용경진대회'

# 파일 경로 정의
PATHS = {
    'area': os.path.join(BASE_DIR, '서울시 상권분석서비스(영역-상권).csv'),
    'sales_2021': os.path.join(BASE_DIR, '추정매출-상권', '서울시_상권분석서비스(추정매출-상권)_2021년.csv'),
    'sales_2022': os.path.join(BASE_DIR, '추정매출-상권', '서울시_상권분석서비스(추정매출-상권)_2022년.csv'),
    'sales_2023': os.path.join(BASE_DIR, '추정매출-상권', '서울시_상권분석서비스(추정매출-상권)_2023년.csv'),
    'sales_2024': os.path.join(BASE_DIR, '추정매출-상권', '서울시 상권분석서비스(추정매출-상권)_2024년.csv'),
    'sales_2025': os.path.join(BASE_DIR, '추정매출-상권', '서울시 상권분석서비스(추정매출-상권)_2025년.csv'),
    'restaurant': os.path.join(BASE_DIR, '식품_일반음식점.csv'),
    'cafe': os.path.join(BASE_DIR, '식품_휴게음식점.csv'),
}

# 파일 존재 확인
for name, path in PATHS.items():
    exists = "✅" if os.path.exists(path) else "❌"
    print(f"{exists} {name}: {path}")

### CELL 3: 데이터 로딩 ==========================================================
print("=" * 60)
print("📊 데이터 로딩 시작...")
print("=" * 60)

# ── 3-1. 상권 영역 데이터 (도화지) ──
df_area = pd.read_csv(PATHS['area'], encoding='cp949')
print(f"\n🗺️  상권 영역 데이터: {df_area.shape}")
print(f"   상권 구분: {df_area['상권_구분_코드_명'].value_counts().to_dict()}")

# ── 3-2. 추정매출 데이터 (2021~2025 통합) ──
sales_dfs = []
for year_key in ['sales_2021', 'sales_2022', 'sales_2023', 'sales_2024', 'sales_2025']:
    try:
        df = pd.read_csv(PATHS[year_key], encoding='cp949')
        sales_dfs.append(df)
        print(f"💰 {year_key}: {df.shape[0]:,} rows, 분기: {df['기준_년분기_코드'].unique()}")
    except Exception as e:
        print(f"⚠️  {year_key} 로딩 실패: {e}")

df_sales_all = pd.concat(sales_dfs, ignore_index=True)
print(f"\n💰 전체 매출 데이터: {df_sales_all.shape[0]:,} rows")

# ── 3-3. 식품 업소 데이터 (환자 상태) ──
df_restaurant = pd.read_csv(PATHS['restaurant'], encoding='cp949', low_memory=False)
df_cafe = pd.read_csv(PATHS['cafe'], encoding='cp949', low_memory=False)

# 서울 데이터만 필터링
df_restaurant = df_restaurant[df_restaurant['지번주소'].str.contains('서울', na=False)].copy()
df_cafe = df_cafe[df_cafe['지번주소'].str.contains('서울', na=False)].copy()

# 합치기
df_restaurant['업소구분'] = '일반음식점'
df_cafe['업소구분'] = '휴게음식점'
df_shops = pd.concat([df_restaurant, df_cafe], ignore_index=True)

print(f"\n🏪 서울 음식점 데이터: {df_shops.shape[0]:,} rows")
print(f"   영업상태: {df_shops['영업상태명'].value_counts().to_dict()}")

### CELL 4: 전처리 - 업소 데이터 ===================================================
print("\n" + "=" * 60)
print("🔧 전처리 시작...")
print("=" * 60)

# ── 4-1. 날짜 파싱 ──
df_shops['인허가일자'] = pd.to_datetime(df_shops['인허가일자'], errors='coerce')
df_shops['폐업일자'] = pd.to_datetime(df_shops['폐업일자'], errors='coerce')

# ── 4-2. 좌표 정제 (TM좌표 → float) ──
df_shops['X'] = pd.to_numeric(df_shops['좌표정보(X)'].astype(str).str.strip(), errors='coerce')
df_shops['Y'] = pd.to_numeric(df_shops['좌표정보(Y)'].astype(str).str.strip(), errors='coerce')

# 좌표가 있는 데이터만 유지
df_shops = df_shops.dropna(subset=['X', 'Y'])
print(f"   좌표 유효 업소 수: {df_shops.shape[0]:,}")

# ── 4-3. 폐업 여부 라벨링 ──
# 2025년 이내 폐업 여부
df_shops['is_closed'] = (df_shops['영업상태명'] == '폐업').astype(int)
# 최근 2년 내 폐업 (2023~2024 폐업 = 학습용, 2025 폐업 = 검증용)
df_shops['closed_2023_2024'] = (
    (df_shops['is_closed'] == 1) &
    (df_shops['폐업일자'] >= '2023-01-01') &
    (df_shops['폐업일자'] < '2025-01-01')
).astype(int)

df_shops['closed_2025'] = (
    (df_shops['is_closed'] == 1) &
    (df_shops['폐업일자'] >= '2025-01-01')
).astype(int)

print(f"   2023-2024 폐업: {df_shops['closed_2023_2024'].sum():,}")
print(f"   2025 폐업: {df_shops['closed_2025'].sum():,}")
print(f"   현재 영업중: {(df_shops['is_closed'] == 0).sum():,}")

### CELL 5: 업소 → 상권 매핑 (공간 결합) ==========================================
print("\n" + "=" * 60)
print("🔗 업소-상권 공간 결합 (Nearest Centroid)...")
print("=" * 60)

from scipy.spatial import cKDTree

# 상권 중심 좌표
area_coords = df_area[['엑스좌표_값', '와이좌표_값']].values.astype(float)
area_tree = cKDTree(area_coords)

# 업소 좌표로 가장 가까운 상권 찾기
shop_coords = df_shops[['X', 'Y']].values.astype(float)
distances, indices = area_tree.query(shop_coords, k=1)

# 매핑
df_shops['상권_코드'] = df_area.iloc[indices]['상권_코드'].values
df_shops['상권_코드_명'] = df_area.iloc[indices]['상권_코드_명'].values
df_shops['상권_구분'] = df_area.iloc[indices]['상권_구분_코드_명'].values
df_shops['거리_to_상권'] = distances

# 너무 먼 매핑은 제외 (2km 이상)
df_shops = df_shops[df_shops['거리_to_상권'] < 2000].copy()
print(f"   매핑 완료: {df_shops.shape[0]:,} 업소 → {df_shops['상권_코드'].nunique()} 상권")

### CELL 6: 상권별 폐업 통계 집계 =================================================
print("\n" + "=" * 60)
print("📈 상권별 폐업 통계 집계...")
print("=" * 60)

# 상권별 업소 현황 집계
district_stats = df_shops.groupby('상권_코드').agg(
    총_업소수=('is_closed', 'count'),
    폐업_업소수=('is_closed', 'sum'),
    폐업_23_24=('closed_2023_2024', 'sum'),
    폐업_2025=('closed_2025', 'sum'),
    영업중=('is_closed', lambda x: (x == 0).sum()),
    평균_영업기간_일=('인허가일자', lambda x: (datetime(2024, 12, 31) - x).dt.days.mean()),
    상권_구분=('상권_구분', 'first'),
).reset_index()

district_stats['폐업률'] = district_stats['폐업_업소수'] / district_stats['총_업소수']
district_stats['최근_폐업률'] = district_stats['폐업_23_24'] / district_stats['총_업소수']

print(f"   상권 수: {district_stats.shape[0]}")
print(f"   평균 폐업률: {district_stats['폐업률'].mean():.2%}")
print(f"   최근(23-24) 평균 폐업률: {district_stats['최근_폐업률'].mean():.2%}")

### CELL 7: 매출 피처 엔지니어링 ==================================================
print("\n" + "=" * 60)
print("⚙️  매출 피처 엔지니어링...")
print("=" * 60)

# 음식 관련 업종만 필터 (CS1xxxxx)
food_codes = [f'"CS10000{i}"' for i in range(1, 11)]
# 실제로는 따옴표 없이 저장된 경우도 있으므로
df_sales_all['서비스_업종_코드'] = df_sales_all['서비스_업종_코드'].astype(str).str.strip('"')
df_sales_food = df_sales_all[df_sales_all['서비스_업종_코드'].str.startswith('CS1')].copy()

# 분기 코드 정리
df_sales_food['기준_년분기_코드'] = df_sales_food['기준_년분기_코드'].astype(str).str.strip('"')
df_sales_food['상권_코드'] = df_sales_food['상권_코드'].astype(str).str.strip('"').astype(int)

# 숫자 컬럼 정리
numeric_cols = ['당월_매출_금액', '당월_매출_건수', '주중_매출_금액', '주말_매출_금액',
                '시간대_00~06_매출_금액', '시간대_06~11_매출_금액', '시간대_11~14_매출_금액',
                '시간대_14~17_매출_금액', '시간대_17~21_매출_금액', '시간대_21~24_매출_금액',
                '남성_매출_금액', '여성_매출_금액',
                '연령대_10_매출_금액', '연령대_20_매출_금액', '연령대_30_매출_금액',
                '연령대_40_매출_금액', '연령대_50_매출_금액', '연령대_60_이상_매출_금액',
                '주중_매출_건수', '주말_매출_건수']

for col in numeric_cols:
    df_sales_food[col] = pd.to_numeric(
        df_sales_food[col].astype(str).str.strip('"'), errors='coerce'
    ).fillna(0)

# ── 상권별-분기별 매출 집계 ──
sales_agg = df_sales_food.groupby(['상권_코드', '기준_년분기_코드']).agg(
    총매출=('당월_매출_금액', 'sum'),
    총건수=('당월_매출_건수', 'sum'),
    주중매출=('주중_매출_금액', 'sum'),
    주말매출=('주말_매출_금액', 'sum'),
    야간매출=('시간대_21~24_매출_금액', 'sum'),
    점심매출=('시간대_11~14_매출_금액', 'sum'),
    저녁매출=('시간대_17~21_매출_금액', 'sum'),
    남성매출=('남성_매출_금액', 'sum'),
    여성매출=('여성_매출_금액', 'sum'),
    청년매출=('연령대_20_매출_금액', 'sum'),      # 20대
    중장년매출=('연령대_40_매출_금액', 'sum'),     # 40대
    시니어매출=('연령대_60_이상_매출_금액', 'sum'), # 60+
    업종수=('서비스_업종_코드', 'nunique'),
).reset_index()

# ── 파생 피처 생성 ──
sales_agg['객단가'] = (sales_agg['총매출'] / sales_agg['총건수'].replace(0, np.nan)).fillna(0)
sales_agg['주말비중'] = (sales_agg['주말매출'] / sales_agg['총매출'].replace(0, np.nan)).fillna(0)
sales_agg['야간비중'] = (sales_agg['야간매출'] / sales_agg['총매출'].replace(0, np.nan)).fillna(0)
sales_agg['저녁비중'] = (sales_agg['저녁매출'] / sales_agg['총매출'].replace(0, np.nan)).fillna(0)
sales_agg['여성비중'] = (sales_agg['여성매출'] / sales_agg['총매출'].replace(0, np.nan)).fillna(0)
sales_agg['시니어비중'] = (sales_agg['시니어매출'] / sales_agg['총매출'].replace(0, np.nan)).fillna(0)

print(f"   매출 집계: {sales_agg.shape[0]:,} rows ({sales_agg['상권_코드'].nunique()} 상권)")
print(f"   분기: {sorted(sales_agg['기준_년분기_코드'].unique())}")

### CELL 8: 시계열 피처 생성 (변화율, 이동평균) ====================================
print("\n" + "=" * 60)
print("📉 시계열 피처 생성 (매출 변화율, 추세)...")
print("=" * 60)

# 분기를 시간 순서로 정렬
quarter_order = sorted(sales_agg['기준_년분기_코드'].unique())
quarter_map = {q: i for i, q in enumerate(quarter_order)}
sales_agg['분기순서'] = sales_agg['기준_년분기_코드'].map(quarter_map)
sales_agg = sales_agg.sort_values(['상권_코드', '분기순서'])

# ── 상권별 시계열 피처 ──
def compute_temporal_features(group):
    """상권별 시계열 피처 계산"""
    features = {}

    # 최근 4분기(1년) 매출 변화율
    if len(group) >= 2:
        recent = group['총매출'].iloc[-1]
        prev = group['총매출'].iloc[-2]
        features['매출_전분기_변화율'] = (recent - prev) / max(prev, 1)
    else:
        features['매출_전분기_변화율'] = 0

    # 최근 4분기 매출 추세 (선형 기울기)
    if len(group) >= 4:
        last4 = group['총매출'].iloc[-4:].values
        x = np.arange(4)
        if np.std(last4) > 0:
            slope = np.polyfit(x, last4, 1)[0]
            features['매출_4Q_추세'] = slope / max(np.mean(last4), 1)
        else:
            features['매출_4Q_추세'] = 0
    else:
        features['매출_4Q_추세'] = 0

    # 매출 변동성 (CV = std/mean)
    if len(group) >= 3:
        cv = group['총매출'].std() / max(group['총매출'].mean(), 1)
        features['매출_변동성'] = cv
    else:
        features['매출_변동성'] = 0

    # 객단가 변화율
    if len(group) >= 2:
        features['객단가_변화율'] = (
            (group['객단가'].iloc[-1] - group['객단가'].iloc[-2]) /
            max(group['객단가'].iloc[-2], 1)
        )
    else:
        features['객단가_변화율'] = 0

    # 건수 변화율
    if len(group) >= 2:
        features['건수_변화율'] = (
            (group['총건수'].iloc[-1] - group['총건수'].iloc[-2]) /
            max(group['총건수'].iloc[-2], 1)
        )
    else:
        features['건수_변화율'] = 0

    # 최근 분기 절대값
    features['최근_매출'] = group['총매출'].iloc[-1]
    features['최근_건수'] = group['총건수'].iloc[-1]
    features['최근_객단가'] = group['객단가'].iloc[-1]
    features['최근_주말비중'] = group['주말비중'].iloc[-1]
    features['최근_야간비중'] = group['야간비중'].iloc[-1]
    features['최근_저녁비중'] = group['저녁비중'].iloc[-1]
    features['최근_여성비중'] = group['여성비중'].iloc[-1]
    features['최근_시니어비중'] = group['시니어비중'].iloc[-1]
    features['최근_업종수'] = group['업종수'].iloc[-1]

    return pd.Series(features)

# 2024년 4분기(20244)까지의 데이터로 학습용 피처 생성
sales_train = sales_agg[sales_agg['기준_년분기_코드'] <= '20244']
temporal_features = sales_train.groupby('상권_코드').apply(compute_temporal_features).reset_index()

# 2025년 데이터로 검증용 피처 생성
sales_test = sales_agg.copy()  # 2025 포함 전체
temporal_features_test = sales_test.groupby('상권_코드').apply(compute_temporal_features).reset_index()

print(f"   학습용 시계열 피처: {temporal_features.shape}")
print(f"   검증용 시계열 피처: {temporal_features_test.shape}")

### CELL 9: 최종 학습 데이터 병합 =================================================
print("\n" + "=" * 60)
print("🔀 최종 학습 데이터 병합...")
print("=" * 60)

# 상권 영역 정보 (좌표, 면적, 구분)
area_info = df_area[['상권_코드', '상권_코드_명', '엑스좌표_값', '와이좌표_값',
                      '영역_면적', '상권_구분_코드_명', '자치구_코드_명']].copy()

# 상권 구분 원핫 인코딩
area_info = pd.get_dummies(area_info, columns=['상권_구분_코드_명'], prefix='상권')

# 면적 로그 변환
area_info['log_면적'] = np.log1p(area_info['영역_면적'])

# 최종 병합 - 학습 데이터
df_train = (
    district_stats
    .merge(area_info, on='상권_코드', how='inner')
    .merge(temporal_features, on='상권_코드', how='inner')
)

# 타겟 변수: 2025년 폐업 여부 (향후 6개월 위험도)
# 2023-2024 폐업률이 상위 25%인 상권을 '위험 상권'으로 라벨링
threshold = df_train['최근_폐업률'].quantile(0.75)
df_train['target'] = (df_train['최근_폐업률'] >= threshold).astype(int)

# 2025년 실제 폐업 데이터로 GT 라벨
df_train['gt_2025_closed'] = df_train['폐업_2025']
df_train['gt_2025_위험'] = (df_train['폐업_2025'] > 0).astype(int)

print(f"   최종 학습 데이터: {df_train.shape}")
print(f"   위험 상권 (학습 라벨): {df_train['target'].sum()} / {len(df_train)}")
print(f"   2025 실제 폐업 상권: {df_train['gt_2025_위험'].sum()} / {len(df_train)}")

### CELL 10: 시계열 입력 텐서 생성 (LSTM용) ========================================
print("\n" + "=" * 60)
print("🔢 시계열 입력 텐서 생성 (LSTM 입력용)...")
print("=" * 60)

# LSTM에 넣을 분기별 시계열 데이터
SEQUENCE_LENGTH = 8  # 최근 8분기 (2년)

# 시계열 피처 목록
ts_features = ['총매출', '총건수', '객단가', '주말비중', '야간비중',
               '저녁비중', '여성비중', '시니어비중', '업종수']

def create_sequences(sales_agg, district_codes, seq_len=8, cutoff_quarter='20244'):
    """상권별 시계열 시퀀스 생성"""
    sequences = []
    valid_codes = []

    filtered = sales_agg[sales_agg['기준_년분기_코드'] <= cutoff_quarter]

    for code in district_codes:
        district_data = filtered[filtered['상권_코드'] == code].sort_values('분기순서')

        if len(district_data) < 4:  # 최소 4분기 필요
            continue

        # 시퀀스 패딩
        seq = district_data[ts_features].values[-seq_len:]
        if len(seq) < seq_len:
            pad = np.zeros((seq_len - len(seq), len(ts_features)))
            seq = np.vstack([pad, seq])

        sequences.append(seq)
        valid_codes.append(code)

    return np.array(sequences), valid_codes

sequences, valid_codes = create_sequences(
    sales_agg, df_train['상권_코드'].values, SEQUENCE_LENGTH, '20244'
)

print(f"   시퀀스 shape: {sequences.shape}")
print(f"   유효 상권 수: {len(valid_codes)}")

# 유효 상권만 필터
df_train = df_train[df_train['상권_코드'].isin(valid_codes)].copy()
# 순서 맞추기
code_to_idx = {c: i for i, c in enumerate(valid_codes)}
df_train['seq_idx'] = df_train['상권_코드'].map(code_to_idx)
df_train = df_train.sort_values('seq_idx').reset_index(drop=True)

print(f"   최종 학습 데이터: {df_train.shape}")

### CELL 11: 공간 그래프 구성 =====================================================
print("\n" + "=" * 60)
print("🕸️  공간 그래프 구성 (인접 상권 연결)...")
print("=" * 60)

# 상권 좌표 추출
graph_coords = df_train[['엑스좌표_값', '와이좌표_값']].values.astype(float)

# KNN 기반 그래프 (k=5 인접 상권)
K_NEIGHBORS = 5
tree = cKDTree(graph_coords)
distances, neighbors = tree.query(graph_coords, k=K_NEIGHBORS + 1)  # 자기 자신 포함

# Edge list 생성
edge_src = []
edge_dst = []
edge_weight = []

for i in range(len(graph_coords)):
    for j in range(1, K_NEIGHBORS + 1):  # 자기 자신 제외
        neighbor_idx = neighbors[i][j]
        dist = distances[i][j]
        if dist < 3000:  # 3km 이내만 연결
            edge_src.append(i)
            edge_dst.append(neighbor_idx)
            # 거리 기반 가중치 (가까울수록 영향 큼)
            edge_weight.append(np.exp(-dist / 1000))

# 양방향 엣지
edge_src_bi = edge_src + edge_dst
edge_dst_bi = edge_dst + edge_src
edge_weight_bi = edge_weight + edge_weight

print(f"   노드 수: {len(graph_coords)}")
print(f"   엣지 수: {len(edge_src_bi)}")
print(f"   평균 연결 수: {len(edge_src_bi) / len(graph_coords):.1f}")

### CELL 12: PyTorch 모델 정의 =====================================================
print("\n" + "=" * 60)
print("🧠 모델 정의: SpatioTemporal GNN + LSTM")
print("=" * 60)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# GPU 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")

try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data
    HAS_PYG = True
    print("   ✅ PyTorch Geometric 사용 가능")
except ImportError:
    HAS_PYG = False
    print("   ⚠️  PyTorch Geometric 없음 → MLP+LSTM 대체 모델 사용")


class TemporalEncoder(nn.Module):
    """LSTM 기반 시계열 매출 변화 인코더"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.output_dim = hidden_dim * 2

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Self-Attention
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)

        return context


class SpatialGNN(nn.Module):
    """GCN/GAT 기반 상권 간 공간적 상호작용 인코더"""
    def __init__(self, input_dim, hidden_dim=64, heads=4):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=0.3)
        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr=None):
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.elu(h)
        h = F.dropout(h, p=0.3, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        h = F.elu(h)
        return h


class SpatialMLPFallback(nn.Module):
    """PyG 없을 때 사용하는 MLP 기반 공간 인코더 (이웃 평균 집계)"""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        self.output_dim = hidden_dim

    def forward(self, x, neighbor_features):
        # x: (N, input_dim), neighbor_features: (N, input_dim) - 이웃 평균
        combined = torch.cat([x, neighbor_features], dim=-1)
        return self.mlp(combined)


class EarlyWarningModel(nn.Module):
    """
    상권 조기경보 통합 모델
    = TemporalEncoder(LSTM) + SpatialGNN(GAT) + Static Features
    → Risk Score (0~1)
    """
    def __init__(self, ts_dim, static_dim, hidden_dim=64, use_gnn=True):
        super().__init__()
        self.use_gnn = use_gnn

        # 1. 시계열 인코더
        self.temporal = TemporalEncoder(ts_dim, hidden_dim)

        # 2. 공간 인코더
        if use_gnn and HAS_PYG:
            self.spatial = SpatialGNN(hidden_dim + static_dim, hidden_dim)
            spatial_out = hidden_dim
        else:
            self.spatial = SpatialMLPFallback(hidden_dim + static_dim, hidden_dim)
            spatial_out = hidden_dim

        # 3. 정적 피처 인코더
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.2),
        )

        # 4. 융합 & 예측 헤드
        fusion_dim = self.temporal.output_dim + spatial_out + hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 5. 매출 예측 헤드 (보조 태스크)
        self.sales_predictor = nn.Sequential(
            nn.Linear(self.temporal.output_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, ts_input, static_input, edge_index=None, edge_attr=None,
                neighbor_features=None):
        # 시계열 인코딩
        temporal_emb = self.temporal(ts_input)

        # 정적 피처 인코딩
        static_emb = self.static_encoder(static_input)

        # 공간 인코딩 (GNN 또는 MLP)
        node_feat = torch.cat([temporal_emb, static_input], dim=-1)

        if self.use_gnn and HAS_PYG and edge_index is not None:
            spatial_emb = self.spatial(node_feat, edge_index, edge_attr)
        elif neighbor_features is not None:
            spatial_emb = self.spatial(node_feat, neighbor_features)
        else:
            spatial_emb = self.spatial.mlp[:4](node_feat)  # 자기 자신만

        # 융합
        fused = torch.cat([temporal_emb, spatial_emb, static_emb], dim=-1)

        # 위험도 예측
        risk_score = self.classifier(fused)

        # 매출 예측 (보조)
        sales_pred = self.sales_predictor(temporal_emb)

        return risk_score, sales_pred

print("   ✅ 모델 정의 완료")

### CELL 13: 학습 데이터 준비 =====================================================
print("\n" + "=" * 60)
print("📦 학습 데이터 텐서 준비...")
print("=" * 60)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# ── 정적 피처 선택 ──
static_feature_cols = [
    '총_업소수', '폐업_업소수', '영업중', '평균_영업기간_일', 'log_면적',
    '매출_전분기_변화율', '매출_4Q_추세', '매출_변동성',
    '객단가_변화율', '건수_변화율',
    '최근_매출', '최근_건수', '최근_객단가',
    '최근_주말비중', '최근_야간비중', '최근_저녁비중',
    '최근_여성비중', '최근_시니어비중', '최근_업종수',
]

# 원핫 인코딩 컬럼 추가
onehot_cols = [c for c in df_train.columns if c.startswith('상권_')]
static_feature_cols += [c for c in onehot_cols if c in df_train.columns and
                        df_train[c].dtype in ['int64', 'float64', 'bool', 'uint8']]

# NaN 처리
df_train[static_feature_cols] = df_train[static_feature_cols].fillna(0)

# 스케일링
scaler_static = StandardScaler()
static_scaled = scaler_static.fit_transform(df_train[static_feature_cols].values)

# 시퀀스 스케일링
scaler_ts = StandardScaler()
seq_reshaped = sequences.reshape(-1, sequences.shape[-1])
seq_scaled = scaler_ts.fit_transform(seq_reshaped).reshape(sequences.shape)

# 타겟
y_target = df_train['target'].values
y_sales = df_train['최근_매출'].values

# 스케일링
scaler_sales = StandardScaler()
y_sales_scaled = scaler_sales.fit_transform(y_sales.reshape(-1, 1)).flatten()

# 텐서 변환
X_ts = torch.FloatTensor(seq_scaled)
X_static = torch.FloatTensor(static_scaled)
Y_target = torch.FloatTensor(y_target)
Y_sales = torch.FloatTensor(y_sales_scaled)

# 엣지 텐서
edge_index = torch.LongTensor([edge_src_bi, edge_dst_bi])
edge_attr = torch.FloatTensor(edge_weight_bi).unsqueeze(1)

# 이웃 피처 (MLP fallback 용)
neighbor_static = torch.zeros_like(X_static)
for i in range(len(graph_coords)):
    nbr_indices = [edge_dst_bi[j] for j in range(len(edge_src_bi)) if edge_src_bi[j] == i]
    if nbr_indices:
        # 시계열 인코딩 전이므로 정적 피처의 이웃 평균으로 대체
        neighbor_static[i] = X_static[nbr_indices].mean(dim=0)

print(f"   시계열 입력: {X_ts.shape}")
print(f"   정적 피처: {X_static.shape} ({len(static_feature_cols)} features)")
print(f"   타겟 분포: 위험={y_target.sum():.0f}, 정상={len(y_target) - y_target.sum():.0f}")
print(f"   엣지: {edge_index.shape}")

### CELL 14: 학습 루프 =============================================================
print("\n" + "=" * 60)
print("🚀 모델 학습 시작!")
print("=" * 60)

# ── 하이퍼파라미터 ──
HIDDEN_DIM = 64
LEARNING_RATE = 1e-3
EPOCHS = 150
WEIGHT_DECAY = 1e-4
PATIENCE = 20  # Early stopping

# ── 클래스 불균형 처리 (Focal Loss) ──
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal = weight * (1 - pt) ** self.gamma * bce
        return focal.mean()

# ── K-Fold 교차 검증 ──
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

fold_results = []
best_models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_static, y_target)):
    print(f"\n{'─' * 40}")
    print(f"📂 Fold {fold + 1}/{N_FOLDS}")
    print(f"   Train: {len(train_idx)}, Val: {len(val_idx)}")

    # 모델 초기화
    model = EarlyWarningModel(
        ts_dim=len(ts_features),
        static_dim=X_static.shape[1],
        hidden_dim=HIDDEN_DIM,
        use_gnn=HAS_PYG
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
    mse_loss = nn.MSELoss()

    # 데이터를 device로 이동
    X_ts_d = X_ts.to(device)
    X_static_d = X_static.to(device)
    Y_target_d = Y_target.to(device)
    Y_sales_d = Y_sales.to(device)
    edge_index_d = edge_index.to(device) if HAS_PYG else None
    edge_attr_d = edge_attr.to(device) if HAS_PYG else None

    # 이웃 피처 계산 (temporal encoding 후 업데이트 필요 → 여기서는 정적 피처 사용)
    nbr_concat = torch.cat([X_ts_d.mean(dim=1), X_static_d], dim=-1)
    neighbor_feat = torch.zeros_like(nbr_concat)
    for i in range(len(graph_coords)):
        nbr_indices = [edge_dst_bi[j] for j in range(len(edge_src_bi)) if edge_src_bi[j] == i]
        if nbr_indices:
            neighbor_feat[i] = nbr_concat[nbr_indices].mean(dim=0)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        # ── 학습 ──
        model.train()
        optimizer.zero_grad()

        risk_pred, sales_pred = model(
            X_ts_d[train_idx],
            X_static_d[train_idx],
            edge_index=edge_index_d,
            edge_attr=edge_attr_d,
            neighbor_features=neighbor_feat[train_idx] if not HAS_PYG else None
        )

        # 멀티태스크 손실
        loss_risk = focal_loss(risk_pred.squeeze(), Y_target_d[train_idx])
        loss_sales = mse_loss(sales_pred.squeeze(), Y_sales_d[train_idx]) * 0.1
        loss = loss_risk + loss_sales

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())

        # ── 검증 ──
        model.eval()
        with torch.no_grad():
            val_risk, val_sales = model(
                X_ts_d[val_idx],
                X_static_d[val_idx],
                edge_index=edge_index_d,
                edge_attr=edge_attr_d,
                neighbor_features=neighbor_feat[val_idx] if not HAS_PYG else None
            )
            val_loss = focal_loss(val_risk.squeeze(), Y_target_d[val_idx])
            val_losses.append(val_loss.item())

        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"   ⏹️  Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 30 == 0:
            print(f"   Epoch {epoch+1}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")

    # 최적 모델 로드
    model.load_state_dict(best_state)
    best_models.append(model)

    # ── Fold 평가 ──
    model.eval()
    with torch.no_grad():
        val_risk, _ = model(
            X_ts_d[val_idx], X_static_d[val_idx],
            edge_index=edge_index_d, edge_attr=edge_attr_d,
            neighbor_features=neighbor_feat[val_idx] if not HAS_PYG else None
        )
        val_probs = torch.sigmoid(val_risk.squeeze()).cpu().numpy()

    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

    val_preds = (val_probs >= 0.5).astype(int)
    val_true = y_target[val_idx]

    try:
        auc = roc_auc_score(val_true, val_probs)
    except:
        auc = 0

    f1 = f1_score(val_true, val_preds, zero_division=0)
    prec = precision_score(val_true, val_preds, zero_division=0)
    rec = recall_score(val_true, val_preds, zero_division=0)

    fold_results.append({
        'fold': fold + 1,
        'auc': auc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'best_val_loss': best_val_loss,
    })

    print(f"   📊 Fold {fold+1} 결과: AUC={auc:.4f}, F1={f1:.4f}, "
          f"Precision={prec:.4f}, Recall={rec:.4f}")

# ── 전체 결과 요약 ──
print("\n" + "=" * 60)
print("📊 K-Fold 교차 검증 결과 요약")
print("=" * 60)

results_df = pd.DataFrame(fold_results)
for metric in ['auc', 'f1', 'precision', 'recall']:
    print(f"   {metric.upper():>10}: {results_df[metric].mean():.4f} ± {results_df[metric].std():.4f}")

### CELL 15: GT 검증 (2025년 실제 데이터) ==========================================
print("\n" + "=" * 60)
print("✅ GT 검증: 2025년 실제 폐업 데이터 vs 예측")
print("=" * 60)

# 앙상블 예측 (5-Fold 모델 평균)
all_probs = np.zeros(len(df_train))

for model in best_models:
    model.eval()
    with torch.no_grad():
        risk_pred, _ = model(
            X_ts_d, X_static_d,
            edge_index=edge_index_d, edge_attr=edge_attr_d,
            neighbor_features=neighbor_feat if not HAS_PYG else None
        )
        probs = torch.sigmoid(risk_pred.squeeze()).cpu().numpy()
        all_probs += probs

all_probs /= len(best_models)

df_train['risk_score'] = all_probs
df_train['pred_위험'] = (all_probs >= 0.5).astype(int)

# ── GT 검증 1: 2025 폐업 상권 예측 정확도 ──
gt_labels = df_train['gt_2025_위험'].values
pred_labels = df_train['pred_위험'].values

try:
    gt_auc = roc_auc_score(gt_labels, all_probs)
except:
    gt_auc = 0

gt_f1 = f1_score(gt_labels, pred_labels, zero_division=0)
gt_prec = precision_score(gt_labels, pred_labels, zero_division=0)
gt_rec = recall_score(gt_labels, pred_labels, zero_division=0)

print(f"\n   🎯 2025년 GT 검증 결과:")
print(f"      AUC-ROC : {gt_auc:.4f}")
print(f"      F1 Score: {gt_f1:.4f}")
print(f"      Precision: {gt_prec:.4f} (경보 정밀도)")
print(f"      Recall  : {gt_rec:.4f} (실제 위기 포착율)")

# ── GT 검증 2: 생존 순서 채점 (C-index) ──
from sklearn.metrics import ndcg_score

# 2025 실제 폐업 수 vs 위험도 점수 상관관계
print(f"\n   📈 위험도-폐업수 Spearman 상관계수:")
from scipy.stats import spearmanr
corr, pval = spearmanr(df_train['risk_score'], df_train['폐업_2025'])
print(f"      ρ = {corr:.4f} (p = {pval:.4e})")

# 상위 10% 위험 상권에서의 실제 폐업 집중도
top10_threshold = np.percentile(all_probs, 90)
top10_mask = all_probs >= top10_threshold
total_closures_2025 = df_train['폐업_2025'].sum()
top10_closures = df_train.loc[top10_mask, '폐업_2025'].sum()

print(f"\n   🔝 상위 10% 위험 상권 분석:")
print(f"      상위 10% 상권 수: {top10_mask.sum()}")
print(f"      해당 상권 2025 폐업 수: {top10_closures}")
print(f"      전체 폐업 대비 집중도: {top10_closures/max(total_closures_2025,1):.1%}")

### CELL 16: 결과 시각화 ===========================================================
print("\n" + "=" * 60)
print("📊 결과 시각화...")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 위험도 분포
ax = axes[0, 0]
ax.hist(all_probs[gt_labels == 0], bins=30, alpha=0.6, label='Safe (2025)', color='#2196F3')
ax.hist(all_probs[gt_labels == 1], bins=30, alpha=0.6, label='Closed (2025)', color='#F44336')
ax.set_xlabel('Risk Score')
ax.set_ylabel('Count')
ax.set_title('Risk Score Distribution by GT Label')
ax.legend()

# 2. ROC Curve
from sklearn.metrics import roc_curve, auc as auc_fn
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(gt_labels, all_probs)
roc_auc = auc_fn(fpr, tpr)
ax.plot(fpr, tpr, color='#4CAF50', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve (2025 GT)')
ax.legend()

# 3. 상권 구분별 위험도
ax = axes[0, 2]
if '상권_구분' in df_train.columns:
    risk_by_type = df_train.groupby('상권_구분')['risk_score'].mean().sort_values(ascending=False)
    risk_by_type.plot(kind='bar', ax=ax, color=['#FF9800', '#F44336', '#2196F3', '#4CAF50'])
    ax.set_title('Avg Risk Score by District Type')
    ax.set_ylabel('Risk Score')
    ax.tick_params(axis='x', rotation=45)

# 4. Confusion Matrix
from sklearn.metrics import confusion_matrix
ax = axes[1, 0]
cm = confusion_matrix(gt_labels, pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Safe', 'Risk'], yticklabels=['Safe', 'Risk'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual (2025)')
ax.set_title('Confusion Matrix')

# 5. 위험 상권 Top 20
ax = axes[1, 1]
top20 = df_train.nlargest(20, 'risk_score')[['상권_코드_명', 'risk_score', '폐업_2025']]
colors = ['#F44336' if x > 0 else '#2196F3' for x in top20['폐업_2025'].values]
bars = ax.barh(range(len(top20)), top20['risk_score'].values, color=colors, alpha=0.8)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20['상권_코드_명'].values, fontsize=7)
ax.set_xlabel('Risk Score')
ax.set_title('Top 20 High-Risk Districts\n(Red=Actually Closed in 2025)')
ax.invert_yaxis()

# 6. 피처 중요도 (Gradient-based)
ax = axes[1, 2]
# 간단히 정적 피처와 risk의 상관관계로 피처 중요도 근사
importances = []
for col_idx, col_name in enumerate(static_feature_cols):
    corr_val = np.corrcoef(X_static[:, col_idx].numpy(), all_probs)[0, 1]
    importances.append((col_name, abs(corr_val)))

importances.sort(key=lambda x: x[1], reverse=True)
top_features = importances[:15]
ax.barh(range(len(top_features)),
        [x[1] for x in top_features],
        color='#9C27B0', alpha=0.7)
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels([x[0] for x in top_features], fontsize=7)
ax.set_xlabel('|Correlation with Risk|')
ax.set_title('Feature Importance (Top 15)')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'early_warning_results.png'), dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ 시각화 저장: early_warning_results.png")

### CELL 17: 예측 결과 저장 ========================================================
print("\n" + "=" * 60)
print("💾 예측 결과 저장...")
print("=" * 60)

# 결과 CSV 저장
output_cols = ['상권_코드', '상권_코드_명', '상권_구분', '자치구_코드_명',
               'risk_score', 'pred_위험', 'gt_2025_위험', '폐업_2025',
               '총_업소수', '영업중', '최근_매출', '매출_4Q_추세']

df_result = df_train[[c for c in output_cols if c in df_train.columns]].copy()
df_result = df_result.sort_values('risk_score', ascending=False)

output_path = os.path.join(BASE_DIR, 'early_warning_predictions.csv')
df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"   ✅ 예측 결과 저장: {output_path}")

# 모델 저장
model_path = os.path.join(BASE_DIR, 'early_warning_model.pt')
torch.save({
    'model_state_dicts': [m.state_dict() for m in best_models],
    'scaler_static': scaler_static,
    'scaler_ts': scaler_ts,
    'static_feature_cols': static_feature_cols,
    'ts_features': ts_features,
    'fold_results': fold_results,
}, model_path)
print(f"   ✅ 모델 저장: {model_path}")

### CELL 18: 이상 탐지 보조 검증 (K-Means) =========================================
print("\n" + "=" * 60)
print("🔍 이상 탐지 보조 검증 (K-Means Anomaly Detection)...")
print("=" * 60)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler as SS

# 매출 추세 기반 군집화
anomaly_features = df_train[['매출_전분기_변화율', '매출_4Q_추세', '매출_변동성',
                              '건수_변화율', '객단가_변화율']].fillna(0).values

scaler_anomaly = SS()
anomaly_scaled = scaler_anomaly.fit_transform(anomaly_features)

# K-Means 군집화 (3 클러스터: 정상/주의/위험)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(anomaly_scaled)
df_train['anomaly_cluster'] = clusters

# 각 클러스터의 평균 매출 추세로 위험 클러스터 식별
cluster_means = df_train.groupby('anomaly_cluster')['매출_4Q_추세'].mean()
risk_cluster = cluster_means.idxmin()  # 매출 추세가 가장 나쁜 클러스터
print(f"   클러스터별 매출추세: {cluster_means.to_dict()}")
print(f"   위험 클러스터: {risk_cluster}")

# 이상 탐지 결과와 모델 예측 일치율
df_train['anomaly_risk'] = (df_train['anomaly_cluster'] == risk_cluster).astype(int)
agreement = (df_train['pred_위험'] == df_train['anomaly_risk']).mean()
print(f"   모델-이상탐지 일치율: {agreement:.1%}")

# 이상 탐지로 포착한 위기 상권 중 실제 폐업 비율
anomaly_risk_mask = df_train['anomaly_risk'] == 1
if anomaly_risk_mask.sum() > 0:
    anomaly_precision = df_train.loc[anomaly_risk_mask, 'gt_2025_위험'].mean()
    print(f"   이상 탐지 정밀도: {anomaly_precision:.1%}")

print("\n" + "=" * 60)
print("🎉 전체 파이프라인 완료!")
print("=" * 60)
print(f"""
{'='*60}
📋 최종 요약
{'='*60}
• 학습 데이터: 2021~2024 (매출 + 업소 + 상권 영역)
• 검증 데이터: 2025 실제 폐업 (GT)
• 모델: SpatioTemporal GNN + LSTM (Attention)
• 학습 방법: 5-Fold CV + Focal Loss + Multi-task
• GT 검증 AUC: {gt_auc:.4f}
• 위험도-폐업 상관: ρ = {corr:.4f}
• 상위 10% 위험 상권의 폐업 집중도: {top10_closures/max(total_closures_2025,1):.1%}

📂 출력 파일:
  - early_warning_predictions.csv (예측 결과)
  - early_warning_results.png (시각화)
  - early_warning_model.pt (모델 가중치)
{'='*60}
""")
