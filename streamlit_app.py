import streamlit as st
import pandas as pd

# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="마케팅 예산 & 영업이익 시뮬레이터",
    layout="wide"
)
st.title("📊 마케팅 예산 & 영업이익 시뮬레이터")

# =========================
# Google Sheets (원본 시트 → gviz CSV)
# =========================
CSV_URL = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vRt3yFjt4OwY6Ym1-xCIJy75-6ccqAzpmGWfa7j7BscDWYL9bl2AmWEJtxo7SDvmQbysb5UEM-jOM2A/pub?output=csv"
)

@st.cache_data
def load_data():
    return pd.read_csv(CSV_URL)

df = load_data()

# =========================
# 백데이터 검증
# =========================
# =========================
# 시나리오 컬럼 자동 인식
# =========================
scenario_col = df.columns[0]  # 첫 번째 컬럼을 시나리오명으로 사용
df = df.set_index(scenario_col)

st.info(f"ℹ️ 시나리오 컬럼으로 '{scenario_col}' 사용 중")

df = df.set_index("시나리오명")

# =========================
# 채널 그룹 정의
# =========================
CHANNEL_GROUP = {
    "퍼포먼스": ["네이버_광고", "쿠팡_광고", "그외_퍼포먼스"],
    "바이럴": ["네이버_바이럴", "인스타_바이럴", "커뮤니티_바이럴", "그외_바이럴"],
    "브랜드": ["기타_브랜드"],
}

# =========================
# Sidebar 입력
# =========================
st.sidebar.header("기본 정보")
product_name = st.sidebar.text_input("제품명", "테스트 제품")
category = st.sidebar.selectbox("카테고리", ["뷰티", "건강", "푸드", "리빙"])

st.sidebar.divider()
st.sidebar.header("제품 / 운영 지표")

price = st.sidebar.number_input("판매가 (원)", value=50_000, step=1_000)
cost_rate = st.sidebar.number_input("원가율 (%)", value=30.0) / 100
logistics_cost = st.sidebar.number_input("물류비 / 건 (원)", value=3_000, step=500)

marketing_budget = st.sidebar.number_input(
    "월 마케팅 총 예산 (원)", value=50_000_000, step=1_000_000
)
cpc = st.sidebar.number_input("예상 CPC (원)", value=300, step=10)
cvr = st.sidebar.number_input("예상 CVR (%)", value=2.0) / 100
headcount = st.sidebar.number_input("운영 인력 수 (명)", value=2)
salary = st.sidebar.number_input("인당 고정비 (원)", value=3_000_000, step=500_000)

# =========================
# 계산 함수
# =========================
def simulate_pl(ratio_row):
    ad_cost_detail = ratio_row * marketing_budget
    total_ad_cost = ad_cost_detail.sum()

    clicks = total_ad_cost / cpc
    orders = clicks * cvr
    revenue = orders * price

    cost_of_goods = revenue * cost_rate
    total_logistics = orders * logistics_cost
    labor_cost = headcount * salary

    total_cost = total_ad_cost + cost_of_goods + total_logistics + labor_cost
    operating_profit = revenue - total_cost
    operating_margin = (operating_profit / revenue * 100) if revenue > 0 else 0
    roas = reven
