import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import matplotlib.font_manager as fm
import os

# 페이지 레이아웃 설정
st.set_page_config(layout="wide", page_title="Bank Customer Churn Dashboard")

# 💡 한글 깨짐 방지: 시스템에 설치된 나눔고딕 폰트 지정 및 적용
font_path = '/fonts/NanumGothic.ttf'
if os.path.exists(font_path):
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
else:
    plt.rc('font', family='sans-serif')

# 1. 데이터 로드 및 전처리 캐싱
@st.cache_data
def load_data():
    url = "https://github.com/hsw8555/bank/raw/main/bank_customer.xlsx"
    df = pd.read_excel(url, sheet_name="bank_customer")
    # 결측치 처리
    df['Credit_Limit'] = df['Credit_Limit'].fillna(df['Credit_Limit'].median())
    df['Total_Trans_Amt'] = df['Total_Trans_Amt'].fillna(df['Total_Trans_Amt'].median())
    return df

df = load_data()

# 2. 모델 학습 캐싱 (요청된 5개 변수 활용)
@st.cache_resource
def train_model(data):
    features = ['Income_Category', 'Months_on_book', 'Credit_Limit', 'Total_Revolving_Bal', 'Total_Trans_Amt']
    X = data[features]
    y = data['Attrition_Flag']
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X, y)
    return rf

rf_model = train_model(df)

st.title("🏦 은행 고객 이탈 분석 및 예측 대시보드")

# 사이드바: 설정 및 고객 예측 입력
st.sidebar.header("⚙️ 대시보드 설정 및 예측")

# Card_Category 선택 ('전체' 옵션 포함)
card_list = ['전체'] + sorted(list(df['Card_Category'].unique()))
selected_card = st.sidebar.selectbox("Card Category 선택", card_list)

# 7. 고객 정보 입력에 의한 이탈 여부 예측
st.sidebar.subheader("🔮 실시간 고객 이탈 예측")
input_income = st.sidebar.number_input("Income Category (1~4)", min_value=1, max_value=4, value=2, step=1)
input_months = st.sidebar.number_input("Months on book", min_value=1, max_value=120, value=36, step=1)
input_credit = st.sidebar.number_input("Credit Limit", min_value=100.0, max_value=100000.0, value=5000.0, step=100.0)
input_revolving = st.sidebar.number_input("Total Revolving Bal", min_value=0.0, max_value=50000.0, value=1500.0, step=10.0)
input_trans_amt = st.sidebar.number_input("Total Trans Amt", min_value=10.0, max_value=50000.0, value=4000.0, step=50.0)

# 예측하기 버튼
if st.sidebar.button("이탈 여부 예측하기"):
    input_data = pd.DataFrame([[
        input_income, input_months, input_credit, input_revolving, input_trans_amt
    ]], columns=['Income_Category', 'Months_on_book', 'Credit_Limit', 'Total_Revolving_Bal', 'Total_Trans_Amt'])

    prediction = rf_model.predict(input_data)[0]
    probability = rf_model.predict_proba(input_data)[0]
    
    # 클래스 순서(classes_) 확인 후 레이블 매칭
    classes = list(rf_model.classes_)
    churn_idx = classes.index('이탈')
    keep_idx = classes.index('유지')

    st.sidebar.write(f"### 예측 결과: **{prediction}**")
    st.sidebar.write(f"(확률: 유지 {probability[keep_idx]:.2f} / 이탈 {probability[churn_idx]:.2f})")

# 메인 화면: 선택된 Card_Category별 필터링
if selected_card == '전체':
    filtered_df = df
else:
    filtered_df = df[df['Card_Category'] == selected_card]

# 5. KPI 영역
st.subheader(f"📊 Card Category: {selected_card} KPI")
col1, col2, col3 = st.columns(3)

total_cust = len(filtered_df)
active_cust = len(filtered_df[filtered_df['Attrition_Flag'] == '유지'])
churned_cust = len(filtered_df[filtered_df['Attrition_Flag'] == '이탈'])

with col1:
    st.metric("총 고객 수", f"{total_cust:,} 명")
with col2:
    st.metric("유지 고객 수", f"{active_cust:,} 명")
with col3:
    st.metric("이탈 고객 수", f"{churned_cust:,} 명")

# Attrition_Flag별 시각화 영역
st.markdown("---")
st.subheader("📈 이탈 여부(Attrition Flag)별 주요 지표 비교")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Total Revolving Balance 평균 비교
sns.barplot(ax=ax[0], x='Attrition_Flag', y='Total_Revolving_Bal', data=filtered_df, estimator=np.mean, errorbar=None, palette='pastel')
ax[0].set_title(f'{selected_card} - Attrition_Flag별 평균 회전 결제 잔액')
ax[0].set_xlabel('이탈 여부')
ax[0].set_ylabel('평균 잔액')

# Total Transaction Amount 평균 비교
sns.barplot(ax=ax[1], x='Attrition_Flag', y='Total_Trans_Amt', data=filtered_df, estimator=np.mean, errorbar=None, palette='pastel')
ax[1].set_title(f'{selected_card} - Attrition_Flag별 평균 거래 금액')
ax[1].set_xlabel('이탈 여부')
ax[1].set_ylabel('평균 거래 금액')

st.pyplot(fig)
