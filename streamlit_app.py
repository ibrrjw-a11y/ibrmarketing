import streamlit as st
import pandas as pd

# =========================
# 페이지 설정
# =========================
st.set_page_config(
    page_title="마케팅 예산 & 영업이익 시뮬레이터",
    layout="wide"
)
st.title("📊 마케팅 예산 & 영업이익 시뮬레이터 (엑셀 업로드)")

# =========================
# 엑셀 업로드
# =========================
st.header("① 백데이터 엑셀 업로드")

uploaded_file = st.file_uploader(
    "시나리오 비율 엑셀 파일 업로드 (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("⬆️ 먼저 엑셀 파일을 업로드해주세요.")
    st.stop()

df_raw = pd.read_excel(uploaded_file)

# =========================
# 시나리오 컬럼 자동 인식
# =========================
scenario_col = df_raw.columns[0]
df = df_raw.set_index(scenario_col)

st.success(f"✅ '{scenario_col}' 컬럼을 시나리오명으로 인식했습니다.")

# =========================
# 비율 컬럼 자동 정규화
# =========================
def normalize(x):
    try:
        x = float(str(x).replace("%", ""))
        if x > 1:
            return x / 100
        return x
    except:
        return 0

df = df.applymap(normalize)

# =========================
# 채널 그룹 자동 분류
# =========================
CHANNEL_GROUP = {
    "퍼포먼스": [c for c in df.columns if "광고" in c or "퍼포먼스" in c],
    "바이럴": [c for c in df.columns if "바이럴" in c or "인스타" in c or "커뮤니티" in c],
    "브랜드": [c for c in df.columns if "브랜드" in c],
}

# =========================
# Sidebar 입력
# =========================
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
    ad_detail = ratio_row * marketing_budget
    ad_cost = ad_detail.sum()

    clicks = ad_cost / cpc
    orders = clicks * cvr
    revenue = orders * price

    cost_goods = revenue * cost_rate
    logistics = orders * logistics_cost
    labor = headcount * salary

    profit = revenue - (ad_cost + cost_goods + logistics + labor)
    margin = profit / revenue * 100 if revenue else 0
    roas = revenue / ad_cost if ad_cost else 0

    return revenue, ad_cost, profit, margin, roas, ad_detail

# =========================
# ② 단일 시나리오 분석
# =========================
st.header("② 단일 시나리오 분석")

scenario = st.selectbox("시나리오 선택", df.index.tolist())
rev, ad, prof, marg, roas, detail = simulate_pl(df.loc[scenario])

c1, c2, c3, c4 = st.columns(4)
c1.metric("예상 매출", f"{rev:,.0f} 원")
c2.metric("총 광고비", f"{ad:,.0f} 원")
c3.metric("영업이익", f"{prof:,.0f} 원", f"{marg:.1f}%")
c4.metric("ROAS", f"{roas:.2f}")

# =========================
# 광고비 구조
# =========================
st.subheader("📌 광고비 구조 (퍼포먼스 / 바이럴 / 브랜드)")

rows = []
for g, cols in CHANNEL_GROUP.items():
    rows.append({
        "구분": g,
        "광고비(원)": detail[cols].sum() if cols else 0
    })

group_df = pd.DataFrame(rows)

st.dataframe(
    group_df.style.format({"광고비(원)": "{:,.0f}"}),
    use_container_width=True
)
st.bar_chart(group_df.set_index("구분"))

# =========================
# ③ 시나리오 A/B/C 비교
# =========================
st.header("③ 시나리오 A / B / C 비교")

compare = st.multiselect(
    "비교할 시나리오 선택",
    df.index.tolist(),
    default=df.index.tolist()[:3]
)

rows = []
for s in compare:
    r, a, p, m, ro, _ = simulate_pl(df.loc[s])
    rows.append({
        "시나리오": s,
        "매출(원)": r,
        "광고비(원)": a,
        "영업이익(원)": p,
        "영업이익률(%)": m,
        "ROAS": ro,
    })

cmp_df = pd.DataFrame(rows)

st.dataframe(
    cmp_df.style.format({
        "매출(원)": "{:,.0f}",
        "광고비(원)": "{:,.0f}",
        "영업이익(원)": "{:,.0f}",
        "영업이익률(%)": "{:.1f}",
        "ROAS": "{:.2f}",
    }),
    use_container_width=True
)

st.bar_chart(cmp_df.set_index("시나리오")[["영업이익(원)"]])
