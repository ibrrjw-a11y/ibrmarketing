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
    "https://docs.google.com/spreadsheets/d/"
    "1MueXw_UsT5EfVraCeWWMqC8_JrdFl0xMsiaNweA9Za8"
    "/gviz/tq?tqx=out:csv&gid=1704119896"
)

@st.cache_data
def load_data():
    return pd.read_csv(CSV_URL)

df_raw = load_data()

# =========================
# 시나리오 컬럼 자동 인식
# =========================
scenario_col = df_raw.columns[0]  # 첫 번째 컬럼 = 시나리오명
df = df_raw.set_index(scenario_col)

st.caption(f"ℹ️ 시나리오 컬럼으로 '{scenario_col}' 사용 중")

# =========================
# 비율 컬럼 자동 정규화
# =========================
def normalize_ratio(x):
    try:
        x = float(x)
        if x > 1:
            return x / 100
        return x
    except:
        return 0

df = df.applymap(normalize_ratio)

# =========================
# 채널 그룹 정의 (컬럼명 기반)
# =========================
CHANNEL_GROUP = {
    "퍼포먼스": [c for c in df.columns if "광고" in c or "퍼포먼스" in c],
    "바이럴": [c for c in df.columns if "바이럴" in c or "인스타" in c or "커뮤니티" in c],
    "브랜드": [c for c in df.columns if "브랜드" in c],
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
    roas = revenue / total_ad_cost if total_ad_cost > 0 else 0

    return {
        "매출": revenue,
        "광고비": total_ad_cost,
        "영업이익": operating_profit,
        "영업이익률": operating_margin,
        "ROAS": roas,
        "광고비_상세": ad_cost_detail,
    }

# =========================
# ① 단일 시나리오 분석
# =========================
st.header("① 단일 시나리오 분석")

scenario = st.selectbox("시나리오 선택", df.index.tolist())
result = simulate_pl(df.loc[scenario])

c1, c2, c3, c4 = st.columns(4)
c1.metric("예상 매출", f"{result['매출']:,.0f} 원")
c2.metric("총 광고비", f"{result['광고비']:,.0f} 원")
c3.metric(
    "영업이익",
    f"{result['영업이익']:,.0f} 원",
    f"{result['영업이익률']:.1f}%"
)
c4.metric("ROAS", f"{result['ROAS']:.2f}")

st.divider()

# =========================
# 광고비 그룹 구조
# =========================
st.subheader("📌 광고비 구조 (퍼포먼스 / 바이럴 / 브랜드)")

group_rows = []
for group, channels in CHANNEL_GROUP.items():
    if channels:
        group_rows.append({
            "구분": group,
            "광고비(원)": result["광고비_상세"][channels].sum()
        })

group_df = pd.DataFrame(group_rows)

st.dataframe(
    group_df.style.format({"광고비(원)": "{:,.0f}"}),
    use_container_width=True
)
st.bar_chart(group_df.set_index("구분"))

st.divider()

# =========================
# ② 시나리오 A/B/C 비교 (임원용)
# =========================
st.header("② 시나리오 A / B / C 비교")

compare_scenarios = st.multiselect(
    "비교할 시나리오 선택 (최대 3개 권장)",
    df.index.tolist(),
    default=df.index.tolist()[:3]
)

rows = []
for scn in compare_scenarios:
    r = simulate_pl(df.loc[scn])
    rows.append({
        "시나리오": scn,
        "매출(원)": r["매출"],
        "광고비(원)": r["광고비"],
        "영업이익(원)": r["영업이익"],
        "영업이익률(%)": r["영업이익률"],
        "ROAS": r["ROAS"],
    })

compare_df = pd.DataFrame(rows)

st.dataframe(
    compare_df.style.format({
        "매출(원)": "{:,.0f}",
        "광고비(원)": "{:,.0f}",
        "영업이익(원)": "{:,.0f}",
        "영업이익률(%)": "{:.1f}",
        "ROAS": "{:.2f}",
    }),
    use_container_width=True
)

st.subheader("📊 시나리오별 영업이익 비교")
st.bar_chart(compare_df.set_index("시나리오")[["영업이익(원)"]])
