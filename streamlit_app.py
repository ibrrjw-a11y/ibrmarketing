import streamlit as st
import pandas as pd
import plotly.express as px

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

h1, h2, h3 {
    font-weight: 600;
}
h2 { font-size: 20px; }
h3 { font-size: 16px; }

div[data-testid="metric-container"] {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 14px;
    box-shadow: none;
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
view_mode = st.sidebar.radio(
    "보기 모드 선택",
    ["내부용", "대행용"],
    index=0
)

# =========================
# 내부용 입력
# =========================
if view_mode == "내부용":
    st.markdown("### 1. 백데이터 업로드")
    uploaded_file = st.file_uploader(
        "시나리오 비율 엑셀 (.xlsx)",
        type=["xlsx"]
    )

    st.markdown("### 2. 제품 / 운영 지표")

    price = st.number_input("판매가 (원)", value=50000, step=1000)
    cost_rate = st.number_input("원가율 (%)", value=30.0) / 100
    logistics_cost = st.number_input("물류비 (건당)", value=3000, step=500)

    marketing_budget = st.number_input(
        "월 마케팅 총 예산",
        value=50000000,
        step=1000000
    )
    cpc = st.number_input("예상 CPC", value=300)
    cvr = st.number_input("예상 CVR (%)", value=2.0) / 100

    headcount = st.number_input("운영 인력 수", value=2)
    salary = st.number_input("인당 고정비", value=3000000)

    st.session_state["uploaded_file"] = uploaded_file
    st.session_state["inputs"] = {
        "price": price,
        "cost_rate": cost_rate,
        "logistics_cost": logistics_cost,
        "marketing_budget": marketing_budget,
        "cpc": cpc,
        "cvr": cvr,
        "headcount": headcount,
        "salary": salary,
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
# 엑셀 로딩
# =========================
if uploaded_file is None:
    st.stop()

df_raw = pd.read_excel(uploaded_file, sheet_name="backdata")

# 시나리오 컬럼 자동 인식 (의미 기반)
scenario_candidates = [
    c for c in df_raw.columns
    if any(k in str(c).lower() for k in ["시나리오", "scenario", "전략"])
]

if not scenario_candidates:
    st.error("❌ 시나리오 컬럼을 찾을 수 없습니다.")
    st.write(df_raw.columns.tolist())
    st.stop()

scenario_col = scenario_candidates[0]
df = df_raw.set_index(scenario_col)

# =========================
# 비율 정규화 (파생)
# =========================
def normalize(x):
    try:
        v = float(str(x).replace("%", ""))
        return v / 100 if v > 1 else v
    except:
        return 0

df_ratio = df.applymap(normalize)

# =========================
# 시나리오 선택
# =========================
scenario = st.selectbox("시나리오 선택", df_ratio.index)

# =========================
# 손익 계산 함수
# =========================
def simulate_pl(ratio_row):
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

rev, ad, prof, marg, roas, detail = simulate_pl(df_ratio.loc[scenario])

# =========================
# KPI 요약 (공통)
# =========================
st.markdown("### 캠페인 핵심 지표")

k1, k2, k3, k4 = st.columns(4)
k1.metric("예상 매출", f"{rev:,.0f} 원")
k2.metric("총 광고비", f"{ad:,.0f} 원")
k3.metric("영업이익", f"{prof:,.0f} 원")
k4.metric("ROAS", f"{roas:.2f}")

# =========================
# 대행용 화면
# =========================
if view_mode == "대행용":

    st.divider()

    # 미디어믹스 자리
    st.markdown("### 미디어 믹스 제안")
    st.info("※ 본 영역은 대행용 미디어믹스 양식을 연결할 자리입니다.")

    st.divider()

    # 광고비 구조 (원형)
    st.markdown("### 광고비 구조")

    CHANNEL_GROUP = {
        "퍼포먼스": [c for c in detail.index if "퍼포먼스" in c],
        "바이럴": [c for c in detail.index if "바이럴" in c],
        "브랜드": [c for c in detail.index if "브랜드" in c or "기타" in c],
    }

    rows = []
    for g, cols in CHANNEL_GROUP.items():
        rows.append({
            "구분": g,
            "광고비": detail[cols].sum() if cols else 0
        })

    pie_df = pd.DataFrame(rows)

    fig_pie = px.pie(
        pie_df,
        values="광고비",
        names="구분",
        hole=0.45
    )
    fig_pie.update_traces(textinfo="percent+label")
    fig_pie.update_layout(font=dict(size=13))

    st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # 시나리오 비교
    st.markdown("### 시나리오 비교")

    compare_rows = []
    for s in df_ratio.index[:5]:
        r, a, p, m, ro, _ = simulate_pl(df_ratio.loc[s])
        compare_rows.append({
            "시나리오": s,
            "매출": r,
            "광고비": a,
            "영업이익": p,
            "영업이익률": m,
        })

    cmp_df = pd.DataFrame(compare_rows)

    metric = st.radio(
        "비교 지표 선택",
        ["매출", "광고비", "영업이익", "영업이익률"],
        horizontal=True
    )

    fig_bar = px.bar(
        cmp_df,
        x="시나리오",
        y=metric,
        text=metric
    )
    fig_bar.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )
    fig_bar.update_layout(
        font=dict(size=13),
        xaxis_tickangle=0,
        yaxis_title=None,
        xaxis_title=None
    )

    st.plotly_chart(fig_bar, use_container_width=True)
    st.stop()

# =========================
# 내부용 상세
# =========================
st.divider()
st.markdown("### 내부용 상세 데이터")

detail_df = detail.reset_index()
detail_df.columns = ["매체", "광고비(원)"]

st.dataframe(
    detail_df.style.format({"광고비(원)": "{:,.0f}"}),
    use_container_width=True
)
