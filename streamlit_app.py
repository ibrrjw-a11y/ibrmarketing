import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# =========================
# Page / Theme
# =========================
st.set_page_config(page_title="마케팅/유통 시뮬레이터", layout="wide")

ACCENT = "#2F6FED"   # 제안서 스타일: 컬러 조금만
MUTED = "#6c757d"
BG = "#f8f9fa"

st.markdown(f"""
<style>
html, body, [class*="css"] {{
  font-size: 14px;
  color: #212529;
}}
h1, h2, h3 {{
  font-weight: 650;
}}
/* KPI 카드 톤 */
div[data-testid="metric-container"] {{
  background: {BG};
  border-radius: 12px;
  padding: 14px;
  border: 1px solid rgba(0,0,0,0.06);
}}
div[data-testid="metric-container"] label {{
  color: {MUTED};
  font-size: 12px;
}}
/* 섹션 간격 */
section.main > div {{
  gap: 2rem;
}}
/* 작은 캡션 */
.smallcap {{
  color: {MUTED};
  font-size: 12px;
}}
/* 신호등 뱃지 */
.badge {{
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 12px;
}}
.badge-green {{ background: rgba(25,135,84,0.12); color: rgb(25,135,84); }}
.badge-red {{ background: rgba(220,53,69,0.12); color: rgb(220,53,69); }}
</style>
""", unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def fmt_won(x):
    try:
        return f"{float(x):,.0f} 원"
    except:
        return "-"

def to_float(x, default=0.0):
    try:
        if pd.isna(x): 
            return default
        s = str(x).strip().replace(",", "")
        s = s.replace("%", "")
        if s == "":
            return default
        return float(s)
    except:
        return default

def normalize_ratio(x):
    """ratio value: supports 0.32, 32, '32%', etc."""
    v = to_float(x, default=np.nan)
    if np.isnan(v):
        return np.nan
    return v / 100.0 if v > 1 else v

# =========================
# preprocess_data (필수 구현)
# - 스택형 CSV: 섹션마다 '시나리오명' 헤더 row로 구분
# =========================
def preprocess_data(df_raw: pd.DataFrame):
    """
    df_raw: header=None로 읽은 전체 CSV
    returns: dict[str, pd.DataFrame] sections in order
      - 각 섹션 DF는 header row를 컬럼으로 사용하고,
        이후 row들은 데이터로 구성
    """
    # 첫 컬럼에서 '시나리오명' 헤더를 찾는다.
    col0 = df_raw.iloc[:, 0].astype(str).str.strip()
    header_idx = df_raw.index[col0.eq("시나리오명")].tolist()
    if not header_idx:
        raise ValueError("스택형 CSV에서 '시나리오명' 헤더 행을 찾지 못했습니다.")

    sections = []
    for i, h in enumerate(header_idx):
        start = h
        end = header_idx[i+1] - 1 if i + 1 < len(header_idx) else len(df_raw) - 1
        sec = df_raw.iloc[start:end+1].copy()

        # 섹션 내 완전 빈 컬럼 제거(섹션 기준), 데이터 손실 없음
        non_empty_cols = [c for c in sec.columns if not sec[c].isna().all()]
        sec = sec[non_empty_cols]

        # header row -> columns
        header = sec.iloc[0].tolist()
        header = [str(x).strip() if pd.notna(x) else "" for x in header]

        # 빈/중복 헤더를 유니크화
        seen = {}
        clean_header = []
        for j, name in enumerate(header):
            if name == "" or name.lower().startswith("unnamed"):
                name = f"_COL_{j+1}"
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
            else:
                seen[name] = 1
            clean_header.append(name)

        body = sec.iloc[1:].copy()
        body.columns = clean_header
        body = body.dropna(how="all")

        # '시나리오명' 컬럼 공백 제거
        if "시나리오명" in body.columns:
            body["시나리오명"] = body["시나리오명"].astype(str).str.strip()

        sections.append(body)

    # 섹션을 "내용 기반"으로 자동 분류(원하면 시나리오명/컬럼명 너가 바꿔도 웬만하면 동작)
    out = {"_sections": sections}

    def has_any(cols, keywords):
        s = " ".join(cols)
        return any(k in s for k in keywords)

    for sec in sections:
        cols = sec.columns.tolist()
        # 광고비 배분(Section 1): '광고비' 키워드가 여러개
        if "ad_alloc" not in out and has_any(cols, ["광고비"]):
            out["ad_alloc"] = sec
            continue
        # 판매 채널 믹스(Section 2): 올리브영/백화점/스마트스토어/쿠팡 등
        if "channel_mix" not in out and has_any(cols, ["스마트스토어", "올리브영", "백화점", "쿠팡", "자사몰", "오픈마켓", "마켓컬리"]):
            out["channel_mix"] = sec
            continue
        # 상세 미디어 믹스(Section 3): 퍼포먼스마케팅_ 구글/메타/틱톡 등
        if "media_mix" not in out and has_any(cols, ["퍼포먼스마케팅_", "구글", "메타", "틱톡", "크리테오"]):
            out["media_mix"] = sec
            continue
        # KPI 목표치(Section 4): CPC/CTR/CVR
        if "kpi" not in out and has_any(cols, ["CPC", "CTR", "CVR", "재구매율"]):
            out["kpi"] = sec
            continue

    return out

def scenario_list_from_sections(sections_dict):
    # 가능한 모든 섹션의 시나리오명을 유니온
    names = set()
    for k, v in sections_dict.items():
        if isinstance(v, pd.DataFrame) and "시나리오명" in v.columns:
            for x in v["시나리오명"].dropna().astype(str):
                x = x.strip()
                if x and x != "시나리오명":
                    names.add(x)
    return sorted(names)

def get_row_by_scenario(df, scenario):
    if df is None or not isinstance(df, pd.DataFrame) or "시나리오명" not in df.columns:
        return None
    sub = df[df["시나리오명"].astype(str).str.strip() == str(scenario).strip()]
    if sub.empty:
        return None
    # 중복 시나리오가 있어도 첫 행 사용 (필요시 추후 옵션화 가능)
    return sub.iloc[0]

# =========================
# Sidebar: Mode + Data Upload
# =========================
st.sidebar.title("마케팅/유통 시뮬레이터")
mode = st.sidebar.radio("모드 선택", ["내부 실무용", "브랜드사(임원용)", "대행사(제안용)"])

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("backdata_reframed.csv 업로드", type=["csv"])
if uploaded is None:
    st.info("좌측에서 스택형 CSV(backdata) 파일을 업로드하세요. (예: backdata_clean_stacked.csv)")
    st.stop()

# Read CSV raw (header=None)
raw_bytes = uploaded.getvalue()
raw_text = raw_bytes.decode("utf-8-sig", errors="replace")
df_raw = pd.read_csv(StringIO(raw_text), header=None)

# preprocess
try:
    data = preprocess_data(df_raw)
except Exception as e:
    st.error(f"❌ CSV 파싱 실패: {e}")
    st.stop()

scenarios = scenario_list_from_sections(data)
if not scenarios:
    st.error("❌ 시나리오 목록을 찾지 못했습니다. (각 섹션에 '시나리오명' 컬럼이 있어야 합니다.)")
    st.stop()

scenario = st.sidebar.selectbox("시나리오 선택", scenarios)

# =========================
# Pull section rows
# =========================
ad_alloc_df = data.get("ad_alloc")
channel_mix_df = data.get("channel_mix")
media_mix_df = data.get("media_mix")
kpi_df = data.get("kpi")

ad_row = get_row_by_scenario(ad_alloc_df, scenario)
ch_row = get_row_by_scenario(channel_mix_df, scenario)
mm_row = get_row_by_scenario(media_mix_df, scenario)
kpi_row = get_row_by_scenario(kpi_df, scenario)

# =========================
# Common: KPI defaults (for funnel & manager/executive)
# =========================
def get_kpi_value(name, default):
    if kpi_row is None:
        return default
    # KPI 섹션은 컬럼명이 다양할 수 있어 keyword 매칭
    cols = list(kpi_row.index)
    # direct
    for c in cols:
        if str(c).strip() == name:
            return to_float(kpi_row[c], default)
    # fuzzy
    for c in cols:
        if name in str(c):
            return to_float(kpi_row[c], default)
    return default

base_cpc = get_kpi_value("목표 평균 CPC", 300.0)
base_ctr = get_kpi_value("목표 평균 CTR", 0.012)  # 1.2% default
base_cvr = get_kpi_value("목표 평균 CVR", 0.02)   # 2% default

# KPI값이 퍼센트로 들어오면 보정
if base_ctr > 1: base_ctr = base_ctr / 100.0
if base_cvr > 1: base_cvr = base_cvr / 100.0

# =========================
# Mode A: 내부 실무용 (Manager)
# =========================
if mode == "내부 실무용":
    st.markdown("## 내부 실무용 대시보드")
    st.markdown('<div class="smallcap">정교한 손익 분석 및 전략 비교</div>', unsafe_allow_html=True)

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown("### 입력")
        calc_mode = st.radio("계산 방식", ["광고비 입력 → 매출 산출", "매출 입력 → 필요 광고비 산출"], horizontal=True)

        aov = st.number_input("객단가(판매가) (원)", value=50000, step=1000)
        cost_rate = st.number_input("원가율 (%)", value=30.0) / 100.0
        logistics_per_order = st.number_input("물류비(건당) (원)", value=3000, step=500)
        fixed_cost = st.number_input("고정비(인건비 등) (원)", value=6000000, step=500000)

        cpc = st.number_input("CPC (원)", value=float(base_cpc), step=10.0)
        cvr = st.number_input("CVR (%)", value=float(base_cvr*100.0), step=0.1) / 100.0

        if calc_mode.startswith("광고비 입력"):
            marketing_budget = st.number_input("총 광고비 (원)", value=50000000, step=1000000)
            target_revenue = None
        else:
            target_revenue = st.number_input("목표 매출 (원)", value=300000000, step=10000000)
            marketing_budget = None

    # --- Simulation core (Manager) ---
    def simulate_manager(ad_spend=None, revenue=None):
        # If revenue provided: compute required ad spend from CPC/CVR/AOV
        if revenue is not None:
            orders = revenue / aov if aov > 0 else 0
            clicks = orders / cvr if cvr > 0 else 0
            ad_spend = clicks * cpc
        else:
            # ad_spend provided -> compute revenue
            clicks = ad_spend / cpc if cpc > 0 else 0
            orders = clicks * cvr
            revenue = orders * aov

        cogs = revenue * cost_rate
        logistics = orders * logistics_per_order
        profit = revenue - (ad_spend + cogs + logistics + fixed_cost)
        contrib_margin = (revenue - ad_spend - logistics - cogs) / revenue * 100 if revenue > 0 else 0

        return {
            "revenue": float(revenue),
            "ad_spend": float(ad_spend),
            "orders": float(orders),
            "clicks": float(clicks),
            "cogs": float(cogs),
            "logistics": float(logistics),
            "fixed": float(fixed_cost),
            "profit": float(profit),
            "contrib_margin": float(contrib_margin),
            "roas": float(revenue / ad_spend) if ad_spend and ad_spend > 0 else 0.0
        }

    res = simulate_manager(ad_spend=marketing_budget, revenue=target_revenue)

    with right:
        st.markdown("### 결과")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("예상 매출", fmt_won(res["revenue"]))
        m2.metric("예상 광고비", fmt_won(res["ad_spend"]))
        m3.metric("영업이익", fmt_won(res["profit"]))
        m4.metric("공헌이익률", f"{res['contrib_margin']:.1f}%")

        st.markdown("### 비용 구조")
        cost_df = pd.DataFrame({
            "항목": ["광고비", "원가(매출원가)", "물류비", "고정비", "영업이익"],
            "금액": [res["ad_spend"], res["cogs"], res["logistics"], res["fixed"], res["profit"]]
        })
        fig_cost = px.bar(cost_df, x="항목", y="금액", text="금액")
        fig_cost.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_cost.update_layout(
            height=320, yaxis_title=None, xaxis_title=None,
            margin=dict(t=10, b=10), font=dict(size=13)
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    st.divider()
    st.markdown("### 시나리오 비교")
    compare_scenarios = st.multiselect("비교할 시나리오 선택", options=scenarios, default=scenarios[:3])

    metric_choice = st.radio("비교 지표", ["매출", "광고비", "영업이익", "공헌이익률", "ROAS"], horizontal=True)

    rows = []
    for s in compare_scenarios:
        # 동일한 입력 조건으로 각 시나리오 비교 (여기서는 시나리오별 CPC/CVR이 있다면 적용 가능)
        # 현재는 KPI 섹션이 시나리오별이라면 kpi_row를 바꿔야 하므로, 간단히 공통 KPI로 비교.
        r = simulate_manager(ad_spend=res["ad_spend"] if calc_mode.startswith("광고비") else None,
                             revenue=res["revenue"] if calc_mode.startswith("매출") else None)
        rows.append({
            "시나리오": s,
            "매출": r["revenue"],
            "광고비": r["ad_spend"],
            "영업이익": r["profit"],
            "공헌이익률": r["contrib_margin"],
            "ROAS": r["roas"]
        })

    cmp_df = pd.DataFrame(rows)
    fig_cmp = px.bar(cmp_df, x="시나리오", y=metric_choice, text=metric_choice)
    if metric_choice in ["공헌이익률", "ROAS"]:
        fig_cmp.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    else:
        fig_cmp.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_cmp.update_layout(height=380, xaxis_tickangle=0, yaxis_title=None, xaxis_title=None, margin=dict(t=10))
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.stop()

# =========================
# Mode B: 브랜드사(임원용) Executive
# =========================
if mode == "브랜드사(임원용)":
    st.markdown("## 브랜드사(임원용) 대시보드")
    st.markdown('<div class="smallcap">수입 유통사 관점의 의사결정: 판매량 예측 & 입점 전략</div>', unsafe_allow_html=True)

    st.markdown("### 입력")
    c1, c2, c3 = st.columns(3)
    with c1:
        total_budget = st.number_input("총 가용 예산 (원) (수입+마케팅 포함)", value=200000000, step=10000000)
    with c2:
        target_units = st.number_input("목표 수입 물량 (Total Unit)", value=10000, step=100)
    with c3:
        landed_cost = st.number_input("개당 수입 원가 (Landed Cost, 원)", value=12000, step=500)

    # Optional, but needed for net profit realism
    with st.expander("고급 옵션 (선택)"):
        price_mult = st.slider("예상 판매가 배수(판매가 = Landed Cost × 배수)", min_value=1.2, max_value=4.0, value=2.0, step=0.1)
        selling_price = landed_cost * price_mult
        st.caption(f"예상 판매가(추정): {selling_price:,.0f} 원")

        cpc = st.number_input("CPC (원)", value=float(base_cpc), step=10.0)
        cvr = st.number_input("CVR (%)", value=float(base_cvr*100.0), step=0.1) / 100.0

    # Budget split
    import_cost = target_units * landed_cost
    affordable_units = target_units
    if import_cost > total_budget and landed_cost > 0:
        affordable_units = int(total_budget // landed_cost)
        import_cost = affordable_units * landed_cost
    marketing_budget = max(total_budget - import_cost, 0.0)

    # Demand estimate via marketing funnel
    clicks = marketing_budget / cpc if cpc > 0 else 0
    orders = clicks * cvr
    units_sold = min(float(orders), float(affordable_units))
    sell_through = (units_sold / target_units * 100) if target_units > 0 else 0

    # Net profit: unit margin * units_sold - marketing
    unit_margin = max(selling_price - landed_cost, 0)
    net_profit = units_sold * unit_margin - marketing_budget

    # KPI Cards
    st.markdown("### 예상 판매 성과")
    k1, k2, k3 = st.columns([1, 1, 1])
    with k1:
        st.metric("총 예상 판매량 (Units Sold)", f"{units_sold:,.0f} 개")
    with k2:
        badge = "badge-green" if sell_through >= 100 else "badge-red"
        st.markdown(f"완판 예상율: <span class='badge {badge}'>{sell_through:.1f}%</span>", unsafe_allow_html=True)
        st.caption("목표 물량 대비 예상 판매량")
    with k3:
        st.metric("예상 순수익 (Net Profit)", fmt_won(net_profit))

    st.divider()

    # Channel recommendation (Section 2)
    st.markdown("### 유통 채널 추천 (상위 5)")
    if ch_row is None:
        st.info("채널 믹스(Section 2) 데이터를 찾지 못했습니다. (backdata에 판매 채널 믹스 섹션이 필요)")
    else:
        tmp = ch_row.drop(labels=["시나리오명"], errors="ignore")
        # make numeric ratios
        series = tmp.apply(normalize_ratio).dropna()
        series = series[series > 0]
        top = series.sort_values(ascending=False).head(5)

        top_df = pd.DataFrame({"채널": top.index.astype(str), "비중": top.values})
        top_df["비중(%)"] = top_df["비중"] * 100

        colA, colB = st.columns([1, 1])
        with colA:
            fig_bar = px.bar(top_df, x="채널", y="비중(%)", text="비중(%)")
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar.update_layout(height=340, yaxis_title=None, xaxis_title=None, margin=dict(t=10))
            st.plotly_chart(fig_bar, use_container_width=True)
        with colB:
            fig_pie = px.pie(top_df, values="비중(%)", names="채널", hole=0.45)
            fig_pie.update_traces(textinfo="percent+label")
            fig_pie.update_layout(height=340, margin=dict(t=10))
            st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # Budget burn donut: import / marketing / profit(or loss)
    st.markdown("### 예산 소진 현황")
    donut_labels = ["제품 수입비용", "마케팅 집행비"]
    donut_vals = [import_cost, marketing_budget]

    if net_profit >= 0:
        donut_labels.append("예상 수익")
        donut_vals.append(net_profit)
    else:
        donut_labels.append("예상 손실")
        donut_vals.append(abs(net_profit))

    donut_df = pd.DataFrame({"구성": donut_labels, "금액": donut_vals})
    fig_donut = px.pie(donut_df, values="금액", names="구성", hole=0.5)
    fig_donut.update_layout(height=360, margin=dict(t=10))
    st.plotly_chart(fig_donut, use_container_width=True)

    st.stop()

# =========================
# Mode C: 대행사(제안용) Agency
# =========================
st.markdown("## 대행사(제안용) 대시보드")
st.markdown('<div class="smallcap">클라이언트 설득을 위한 미디어 믹스 제안</div>', unsafe_allow_html=True)

# Section 3: treemap / sunburst
st.markdown("### 상세 미디어 믹스 (Section 3)")
if mm_row is None:
    st.info("상세 미디어 믹스(Section 3) 데이터를 찾지 못했습니다.")
else:
    tmp = mm_row.drop(labels=["시나리오명"], errors="ignore")
    vals = tmp.apply(normalize_ratio).dropna()
    vals = vals[vals > 0]

    if vals.empty:
        st.info("해당 시나리오의 미디어 믹스 값이 비어있습니다.")
    else:
        mm_long = pd.DataFrame({
            "채널": vals.index.astype(str),
            "비중": vals.values
        })
        # 상위/기타 묶기(너무 많으면 보기 불편)
        mm_long = mm_long.sort_values("비중", ascending=False)
        if len(mm_long) > 18:
            top18 = mm_long.head(18)
            other = mm_long.iloc[18:]["비중"].sum()
            mm_long = pd.concat([top18, pd.DataFrame([{"채널": "기타", "비중": other}])], ignore_index=True)

        fig_tm = px.treemap(mm_long, path=["채널"], values="비중")
        fig_tm.update_layout(height=420, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_tm, use_container_width=True)

st.divider()

# Funnel analysis (긍정적/평범/보수적)
st.markdown("### 퍼널 시뮬레이션 (노출 → 유입 → 전환)")
left, right = st.columns([1, 1])

with left:
    budget = st.number_input("투입 예산 (원)", value=50000000, step=1000000)
    cpc = st.number_input("CPC (원)", value=float(base_cpc), step=10.0)
    ctr = st.number_input("CTR (%)", value=float(base_ctr * 100.0), step=0.1) / 100.0
    cvr = st.number_input("CVR (%)", value=float(base_cvr * 100.0), step=0.1) / 100.0

with right:
    scenario_type = st.radio("가정 선택", ["보수적", "평범", "긍정적"], horizontal=True)

# multipliers
if scenario_type == "보수적":
    m_ctr, m_cvr, m_cpc = 0.85, 0.85, 1.10
elif scenario_type == "긍정적":
    m_ctr, m_cvr, m_cpc = 1.15, 1.15, 0.90
else:
    m_ctr, m_cvr, m_cpc = 1.00, 1.00, 1.00

ctr2 = max(ctr * m_ctr, 1e-6)
cvr2 = max(cvr * m_cvr, 1e-6)
cpc2 = max(cpc * m_cpc, 1e-6)

clicks = budget / cpc2
impressions = clicks / ctr2
conversions = clicks * cvr2

funnel_df = pd.DataFrame({
    "단계": ["노출(Impressions)", "유입(Clicks)", "전환(Conversions)"],
    "값": [impressions, clicks, conversions]
})

fig_funnel = go.Figure(go.Funnel(
    y=funnel_df["단계"],
    x=funnel_df["값"],
    textinfo="value+percent initial"
))
fig_funnel.update_layout(height=420, margin=dict(t=10, b=10), font=dict(size=13))
st.plotly_chart(fig_funnel, use_container_width=True)
