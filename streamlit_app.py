import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =========================
# 페이지 설정
# =========================
st.set_page_config(page_title="마케팅 시뮬레이터", layout="wide")

# =========================
# 정돈된 디자인 CSS
# =========================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 14px;
    color: #212529;
}
h2 { font-size: 20px; font-weight: 600; }
h3 { font-size: 16px; font-weight: 600; }

div[data-testid="metric-container"] {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 14px;
}
div[data-testid="metric-container"] > div {
    font-size: 18px;
}
div[data-testid="metric-container"] label {
    font-size: 12px;
    color: #868e96;
}
section.main > div {
    gap: 2.2rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 사이드바 – 모드 선택
# =========================
st.sidebar.header("화면 모드")
view_mode = st.sidebar.radio("보기 모드 선택", ["내부용", "대행용"])

# =========================
# 내부용 입력
# =========================
if view_mode == "내부용":
    st.markdown("### 1. 백데이터 업로드")
    uploaded_file = st.file_uploader("시나리오 비율 엑셀 (.xlsx)", type=["xlsx"])

    st.markdown("### 2. 제품 / 운영 지표")
    price = st.number_input("판매가 (원)", value=50000, step=1000)
    cost_rate = st.number_input("원가율 (%)", value=30.0) / 100
    logistics_cost = st.number_input("물류비 (건당)", value=3000, step=500)

    marketing_budget = st.number_input("월 마케팅 총 예산", value=50000000, step=1000000)
    cpc = st.number_input("예상 CPC", value=300)
    cvr = st.number_input("예상 CVR (%)", value=2.0) / 100

    headcount = st.number_input("운영 인력 수", value=2)
    salary = st.number_input("인당 고정비", value=3000000)

    st.session_state["uploaded_file"] = uploaded_file
    st.session_state["inputs"] = {
        "price": price, "cost_rate": cost_rate,
        "logistics_cost": logistics_cost,
        "marketing_budget": marketing_budget,
        "cpc": cpc, "cvr": cvr,
        "headcount": headcount, "salary": salary
    }

else:
    uploaded_file = st.session_state.get("uploaded_file")
    inputs = st.session_state.get("inputs")
    if uploaded_file is None or inputs is None:
        st.warning("내부용에서 먼저 설정해주세요.")
        st.stop()

    price = inputs["price"]
    cost_rate = inputs["cost_rate"]
    logistics_cost = inputs["logistics_cost"]
    marketing_budget = inputs["marketing_budget"]
    cpc = inputs["cpc"]
    cvr = inputs["cvr"]
    headcount = inputs["headcount"]
    salary = inputs["salary"]

# =========================
# 엑셀 시트 자동 인식
# =========================
xls = pd.ExcelFile(uploaded_file)
sheet_to_use = "backdata" if "backdata" in xls.sheet_names else xls.sheet_names[0]
df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_to_use)
st.caption(f"📄 사용 중인 시트: {sheet_to_use}")

# =========================
# 시나리오 컬럼 자동 인식
# =========================
scenario_candidates = [
    c for c in df_raw.columns
    if any(k in str(c).lower() for k in ["시나리오", "scenario", "전략"])
]
if not scenario_candidates:
    st.error("❌ 시나리오 컬럼을 찾을 수 없습니다.")
    st.stop()

scenario_col = scenario_candidates[0]
df = df_raw.set_index(scenario_col)

# =========================
# 비율 정규화
# =========================
def normalize(x):
    try:
        v = float(str(x).replace("%", ""))
        return v / 100 if v > 1 else v
    except:
        return 0

df_ratio = df.applymap(normalize)

# =========================
# 손익 계산 함수
# =========================
def simulate_pl(ratio_row):

    # ✅ DataFrame → Series 강제 변환 (핵심)
    if isinstance(ratio_row, pd.DataFrame):
        ratio_row = ratio_row.iloc[0]

    ratio = pd.to_numeric(ratio_row, errors="coerce").fillna(0)

    ad_detail = ratio * marketing_budget
    total_ad = ad_detail.sum()

    clicks = total_ad / cpc
    orders = clicks * cvr
    revenue = orders * price

    cost_goods = revenue * cost_rate
    logistics = orders * logistics_cost
    labor = headcount * salary

    profit = revenue - (total_ad + cost_goods + logistics + labor)
    margin = (profit / revenue * 100) if revenue > 0 else 0
    roas = (revenue / total_ad) if total_ad > 0 else 0

    return revenue, total_ad, profit, margin, roas, ad_detail


# =========================
# 시나리오 선택 (단일)
# =========================
scenario = st.selectbox("기준 시나리오 선택", df_ratio.index)
rev, ad, prof, marg, roas, detail = simulate_pl(df_ratio.loc[scenario])

# =========================
# KPI 요약
# =========================
st.markdown("### 캠페인 핵심 지표")
k1, k2, k3, k4 = st.columns(4)
k1.metric("예상 매출", f"{rev:,.0f} 원")
k2.metric("총 광고비", f"{ad:,.0f} 원")
k3.metric("영업이익", f"{prof:,.0f} 원")
k4.metric("ROAS", f"{roas:.2f}")

# =========================
# 내부용 전략 비교 (막대 + 꺾은선)
# =========================
if view_mode == "내부용":

    st.divider()
    st.markdown("### 전략 비교 분석")

    compare_strategies = st.multiselect(
        "비교할 전략 선택",
        options=df_ratio.index.tolist(),
        default=df_ratio.index[:3].tolist()
    )

    metric_view = st.radio(
        "표시 지표 선택",
        ["예상 매출", "예상 광고비", "ROAS", "전체"],
        horizontal=True
    )

    rows = []
    for s in compare_strategies:
        r, a, _, _, ro, _ = simulate_pl(df_ratio.loc[s])
        rows.append({"전략": s, "예상 매출": r, "예상 광고비": a, "ROAS": ro})
    cmp_df = pd.DataFrame(rows)

    fig = go.Figure()

    if metric_view in ["예상 매출", "전체"]:
        fig.add_bar(x=cmp_df["전략"], y=cmp_df["예상 매출"], name="예상 매출")

    if metric_view in ["예상 광고비", "전체"]:
        fig.add_bar(x=cmp_df["전략"], y=cmp_df["예상 광고비"], name="예상 광고비")

    if metric_view in ["ROAS", "전체"]:
        fig.add_trace(go.Scatter(
            x=cmp_df["전략"], y=cmp_df["ROAS"],
            mode="lines+markers", name="ROAS", yaxis="y2"
        ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(title="금액 (원)", tickformat=","),
        yaxis2=dict(title="ROAS", overlaying="y", side="right"),
        font=dict(size=13),
        margin=dict(t=20)
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# 대행용 화면
# =========================
if view_mode == "대행용":
    st.divider()
    st.markdown("### 미디어 믹스 제안")
    st.info("※ 대행용 미디어믹스 템플릿 연동 영역")

    st.divider()
    st.markdown("### 광고비 구조")

    pie_df = detail.reset_index()
    pie_df.columns = ["매체", "광고비"]

    fig_pie = px.pie(pie_df, values="광고비", names="매체", hole=0.45)
    fig_pie.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)
