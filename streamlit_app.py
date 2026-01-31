import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from io import StringIO
import re
from datetime import datetime

# =========================================================
# Page
# =========================================================
st.set_page_config(page_title="마케팅/유통 시뮬레이터", layout="wide")

ACCENT = "#2F6FED"
MUTED = "#6c757d"
BG = "#f8f9fa"
CARD = "#ffffff"

st.markdown(f"""
<style>
html, body, [class*="css"] {{
  font-size: 14px;
  color: #212529;
}}
h1, h2, h3 {{
  font-weight: 700;
}}
section.main > div {{
  gap: 1.6rem;
}}
.smallcap {{
  color: {MUTED};
  font-size: 12px;
}}
.badge {{
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 800;
  font-size: 12px;
  border: 1px solid rgba(0,0,0,0.08);
  background: rgba(47,111,237,0.08);
  color: #1f4fd6;
}}
.card {{
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 14px;
  padding: 14px 14px;
  background: {CARD};
}}
hr.soft {{
  border: 0;
  border-top: 1px solid rgba(0,0,0,0.06);
  margin: 12px 0;
}}
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
</style>
""", unsafe_allow_html=True)

# =========================================================
# Utils
# =========================================================
def fmt_won(x):
    try:
        if x is None:
            return "-"
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return "-"
        return f"{float(x):,.0f} 원"
    except:
        return "-"

def fmt_int(x):
    try:
        if x is None:
            return "-"
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return "-"
        return f"{float(x):,.0f}"
    except:
        return "-"

def fmt_pct(x, digits=1):
    try:
        if x is None:
            return "-"
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return "-"
        return f"{float(x):.{digits}f}%"
    except:
        return "-"

def to_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if pd.isna(x):
            return default
        s = str(x).strip()
        if s == "":
            return default
        s = s.replace(",", "")
        s = s.replace("₩", "").replace("원", "")
        s = s.replace("%", "")
        return float(s)
    except:
        return default

def normalize_ratio(x):
    """
    Accept 0.32 / 32 / '32%' etc -> return fraction (0~1)
    """
    v = to_float(x, default=np.nan)
    if np.isnan(v):
        return np.nan
    return v / 100.0 if v > 1 else v

def normalize_shares(d: dict):
    d2 = {k: float(v or 0.0) for k, v in d.items()}
    s = sum(v for v in d2.values() if v > 0)
    if s <= 0:
        return {k: 0.0 for k in d2}
    return {k: (v/s if v > 0 else 0.0) for k, v in d2.items()}

def soft_find_col(cols, keywords):
    cols = [str(c).strip() for c in cols]
    for kw in keywords:
        for c in cols:
            if kw in c:
                return c
    return None

def safe_plotly(fig, key: str, **kwargs):
    # plotly_chart 중복 id 방지: key를 강제한다
    st.plotly_chart(fig, use_container_width=True, key=key, **kwargs)

def donut_chart(labels, values, title=None, height=280):
    dfp = pd.DataFrame({"label": labels, "value": values})
    dfp["value"] = dfp["value"].astype(float)
    fig = px.pie(dfp, values="value", names="label", hole=0.55)
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=height, margin=dict(t=40 if title else 10, b=10, l=10, r=10))
    if title:
        fig.update_layout(title=title)
    return fig

def bar_cost_chart(items, values, title=None, height=320):
    dfp = pd.DataFrame({"항목": items, "금액": values})
    fig = px.bar(dfp, x="항목", y="금액", text="금액")
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(height=height, margin=dict(t=40 if title else 10, b=10, l=10, r=10),
                      yaxis_title=None, xaxis_title=None)
    if title:
        fig.update_layout(title=title)
    return fig

# =========================================================
# Load backdata (xlsx/csv)
# =========================================================
@st.cache_data(show_spinner=False)
def load_backdata(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        raw = file.getvalue().decode("utf-8-sig", errors="replace")
        df = pd.read_csv(StringIO(raw))
        return df
    else:
        # xlsx
        df = pd.read_excel(file)
        return df

# =========================================================
# Column detection (robust)
# =========================================================
def detect_columns(df: pd.DataFrame):
    cols = list(df.columns)

    # scenario internal key
    scenario = None
    for c in cols:
        if str(c).strip() in ["시나리오명", "scenario", "Scenario", "SCENARIO"]:
            scenario = c
            break
    if scenario is None:
        # fuzzy
        scenario = soft_find_col(cols, ["시나리오", "scenario"])

    # display name: user said it's in B열(2nd col)
    display = None
    if len(cols) >= 2:
        display = cols[1]
    # but if there is explicit naming, prefer that
    disp2 = soft_find_col(cols, ["노출", "표시", "디스플레이", "한글", "시나리오명(노출)"])
    if disp2 is not None:
        display = disp2

    # stage/driver/category/position fields (optional)
    stage = soft_find_col(cols, ["ST", "단계", "Stage"])
    drv   = soft_find_col(cols, ["DRV", "드라이버", "Driver"])
    cat   = soft_find_col(cols, ["CAT", "카테고리", "Category"])
    pos   = soft_find_col(cols, ["POS", "포지", "Position", "가격"])

    # revenue channel share columns (판매채널 믹스)
    # keyword list is intentionally broad; it'll pick up cols present in your latest file.
    rev_keywords = ["자사몰", "스마트스토어", "쿠팡", "올리브영", "백화점", "면세점", "약국", "마트", "홈쇼핑", "공구", "B2B", "오픈마켓", "마켓컬리", "온라인", "오프라인"]
    rev_cols = [c for c in cols if any(k in str(c) for k in rev_keywords)]

    # media mix group columns (performance/viral/brand)
    perf_cols = [c for c in cols if ("퍼포먼스" in str(c) or "Performance" in str(c)) and ("바이럴" not in str(c)) and ("브랜드" not in str(c))]
    viral_cols = [c for c in cols if ("바이럴" in str(c) or "Viral" in str(c))]
    brand_cols = [c for c in cols if ("브랜드" in str(c) or "Brand" in str(c))]

    # If your file uses "퍼포먼스마케팅_..." "바이럴마케팅_..." style, include them too
    perf_cols += [c for c in cols if str(c).startswith("퍼포먼스마케팅_")]
    viral_cols += [c for c in cols if str(c).startswith("바이럴마케팅_")]
    brand_cols += [c for c in cols if ("기타_브랜드" in str(c) or "브랜드마케팅_" in str(c))]

    # de-dup
    perf_cols = list(dict.fromkeys(perf_cols))
    viral_cols = list(dict.fromkeys(viral_cols))
    brand_cols = list(dict.fromkeys(brand_cols))

    return {
        "scenario": scenario,
        "display": display,
        "stage": stage,
        "drv": drv,
        "cat": cat,
        "pos": pos,
        "rev_cols": rev_cols,
        "perf_cols": perf_cols,
        "viral_cols": viral_cols,
        "brand_cols": brand_cols,
    }

def scenario_options(df, col_scn, col_disp):
    tmp = df[[col_scn, col_disp]].copy()
    tmp[col_scn] = tmp[col_scn].astype(str).str.strip()
    tmp[col_disp] = tmp[col_disp].astype(str).str.strip()

    # if duplicate display names exist, append suffix to avoid ambiguity
    counts = tmp[col_disp].value_counts()
    dup = set(counts[counts > 1].index.tolist())

    key_to_disp = {}
    disp_to_key = {}

    for _, r in tmp.dropna().iterrows():
        k = str(r[col_scn]).strip()
        d = str(r[col_disp]).strip()
        if d in dup:
            d2 = f"{d}  ·  [{k}]"
        else:
            d2 = d
        key_to_disp[k] = d2
        disp_to_key[d2] = k

    disp_list = sorted(list(disp_to_key.keys()))
    return key_to_disp, disp_to_key, disp_list

# =========================================================
# Shares builders
# =========================================================
def build_rev_shares(row: pd.Series, rev_cols: list):
    d = {}
    for c in rev_cols:
        if c in row.index:
            v = normalize_ratio(row.get(c))
            if not np.isnan(v) and v > 0:
                d[str(c)] = float(v)
    return normalize_shares(d) if d else {}

def build_media_shares(row: pd.Series, perf_cols: list, viral_cols: list, brand_cols: list):
    perf = {}
    viral = {}
    brand = {}

    for c in perf_cols:
        if c in row.index:
            v = normalize_ratio(row.get(c))
            if not np.isnan(v) and v > 0:
                perf[str(c)] = float(v)

    for c in viral_cols:
        if c in row.index:
            v = normalize_ratio(row.get(c))
            if not np.isnan(v) and v > 0:
                viral[str(c)] = float(v)

    for c in brand_cols:
        if c in row.index:
            v = normalize_ratio(row.get(c))
            if not np.isnan(v) and v > 0:
                brand[str(c)] = float(v)

    # group weights are sums across group columns; then normalize across groups to show 100%
    perf_sum = sum(perf.values())
    viral_sum = sum(viral.values())
    brand_sum = sum(brand.values())
    g = perf_sum + viral_sum + brand_sum

    group = {"performance": 0.0, "viral": 0.0, "brand": 0.0}
    if g > 0:
        group = {
            "performance": perf_sum/g,
            "viral": viral_sum/g,
            "brand": brand_sum/g,
        }

    return {
        "performance": normalize_shares(perf) if perf else {},
        "viral": normalize_shares(viral) if viral else {},
        "brand": normalize_shares(brand) if brand else {},
        "group": group
    }

# =========================================================
# Core simulation (Manager / Agency / Brand)
# =========================================================
def simulate_pl_from_adspend(
    ad_spend: float,
    aov: float,
    cpc: float,
    cvr: float,
    cost_rate: float,
    logistics_per_order: float,
    labor_cost: float
):
    clicks = ad_spend / cpc if cpc > 0 else 0.0
    orders = clicks * cvr
    revenue = orders * aov

    cogs = revenue * cost_rate
    logistics = orders * logistics_per_order
    profit = revenue - (ad_spend + cogs + logistics + labor_cost)
    contrib_margin = (revenue - ad_spend - logistics - cogs) / revenue * 100 if revenue > 0 else 0.0
    roas = (revenue / ad_spend) * 100 if ad_spend > 0 else 0.0  # %
    return {
        "revenue": float(revenue),
        "ad_spend": float(ad_spend),
        "clicks": float(clicks),
        "orders": float(orders),
        "cogs": float(cogs),
        "logistics": float(logistics),
        "labor": float(labor_cost),
        "profit": float(profit),
        "contrib_margin": float(contrib_margin),
        "roas": float(roas)
    }

def simulate_pl_from_revenue(
    revenue: float,
    aov: float,
    cpc: float,
    cvr: float,
    cost_rate: float,
    logistics_per_order: float,
    labor_cost: float
):
    orders = revenue / aov if aov > 0 else 0.0
    clicks = orders / cvr if cvr > 0 else 0.0
    ad_spend = clicks * cpc

    cogs = revenue * cost_rate
    logistics = orders * logistics_per_order
    profit = revenue - (ad_spend + cogs + logistics + labor_cost)
    contrib_margin = (revenue - ad_spend - logistics - cogs) / revenue * 100 if revenue > 0 else 0.0
    roas = (revenue / ad_spend) * 100 if ad_spend > 0 else 0.0  # %
    return {
        "revenue": float(revenue),
        "ad_spend": float(ad_spend),
        "clicks": float(clicks),
        "orders": float(orders),
        "cogs": float(cogs),
        "logistics": float(logistics),
        "labor": float(labor_cost),
        "profit": float(profit),
        "contrib_margin": float(contrib_margin),
        "roas": float(roas)
    }

# =========================================================
# Viral unit pricing table (editable defaults)
# - user said: "내가 준 가격" -> for now we keep editable table.
# =========================================================
DEFAULT_VIRAL_UNIT = pd.DataFrame([
    {"구분": "네이버", "지면": "네이버_지식인", "건당비용": 100000, "지면비율(%)": 15.0},
    {"구분": "네이버", "지면": "네이버_인기글", "건당비용": 300000, "지면비율(%)": 20.0},
    {"구분": "네이버", "지면": "네이버_구매대행", "건당비용": 120060, "지면비율(%)": 25.0},
    {"구분": "네이버", "지면": "네이버_핫딜", "건당비용": 100000, "지면비율(%)": 15.0},
    {"구분": "인스타그램", "지면": "인스타그램_파워페이지", "건당비용": 400000, "지면비율(%)": 10.0},
    {"구분": "기타", "지면": "커뮤니티_핫딜", "건당비용": 200000, "지면비율(%)": 15.0},
])

def allocate_viral_counts(viral_budget, unit_df: pd.DataFrame):
    """
    Rule: budget -> allocate by placement ratio -> count = round(allocated / unit_cost)
    Sum mismatch allowed.
    """
    dfu = unit_df.copy()
    dfu["건당비용"] = dfu["건당비용"].apply(lambda x: max(to_float(x, 0.0), 0.0))
    dfu["지면비율(%)"] = dfu["지면비율(%)"].apply(lambda x: max(to_float(x, 0.0), 0.0))
    total_ratio = dfu["지면비율(%)"].sum()
    if total_ratio <= 0:
        dfu["배정예산"] = 0.0
        dfu["진행건수"] = 0
        dfu["총비용"] = 0.0
        return dfu

    dfu["배정예산"] = viral_budget * (dfu["지면비율(%)"] / total_ratio)
    # count
    def _count(row):
        c = row["건당비용"]
        if c <= 0:
            return 0
        return int(np.round(row["배정예산"] / c))
    dfu["진행건수"] = dfu.apply(_count, axis=1)
    dfu["총비용"] = dfu["진행건수"] * dfu["건당비용"]
    return dfu

# =========================================================
# Recommendation engine (kept, simplified UI)
# - Uses columns if present: stage/cat/pos/drv
# =========================================================
def recommend_simple(df, cols, filters, topn=3):
    # if metadata columns missing, just return top by "perf share" heuristic
    col_scn = cols["scenario"]
    col_disp = cols["display"]
    stage_col, cat_col, pos_col, drv_col = cols["stage"], cols["cat"], cols["pos"], cols["drv"]

    dff = df.copy()

    # Apply filters if possible
    if filters.get("stage") and stage_col in dff.columns:
        dff = dff[dff[stage_col].astype(str) == filters["stage"]]
    if filters.get("cat") and cat_col in dff.columns:
        dff = dff[dff[cat_col].astype(str) == filters["cat"]]
    if filters.get("pos") and pos_col in dff.columns:
        dff = dff[dff[pos_col].astype(str) == filters["pos"]]
    if filters.get("drv") and drv_col in dff.columns:
        dff = dff[dff[drv_col].astype(str) == filters["drv"]]

    if dff.empty:
        return pd.DataFrame()

    # score: prefer matching 판매포커스와 연결된 매체 비중이 높은 시나리오
    perf_cols = cols["perf_cols"]
    viral_cols = cols["viral_cols"]
    brand_cols = cols["brand_cols"]
    rev_cols = cols["rev_cols"]

    def score_row(r):
        media = build_media_shares(r, perf_cols, viral_cols, brand_cols)
        rev = build_rev_shares(r, rev_cols)

        # simple heuristic: if sales focus is 자사몰 -> meta-ish/performance weight
        focus = filters.get("sales_focus", "자사몰")
        s = 0.0

        # group weights
        gw = media["group"]
        s += gw.get("performance", 0) * 60
        s += gw.get("viral", 0) * 25
        s += gw.get("brand", 0) * 15

        if focus == "자사몰":
            # reward if '자사몰' in rev cols large
            for k, v in rev.items():
                if "자사몰" in k:
                    s += v * 40
        elif focus == "온라인":
            for k, v in rev.items():
                if ("쿠팡" in k) or ("스마트스토어" in k) or ("오픈마켓" in k):
                    s += v * 35
        elif focus == "홈쇼핑":
            for k, v in rev.items():
                if "홈쇼핑" in k:
                    s += v * 45
        elif focus == "공구":
            for k, v in rev.items():
                if "공구" in k or "공동구매" in k:
                    s += v * 45

        return float(s)

    dff = dff.copy()
    dff["_score"] = dff.apply(score_row, axis=1)
    out = dff.sort_values("_score", ascending=False).head(topn)

    res = pd.DataFrame({
        "노출 시나리오명": out[col_disp].astype(str).values if col_disp in out.columns else out[col_scn].astype(str).values,
        "내부키": out[col_scn].astype(str).values,
        "점수": out["_score"].values
    })
    return res

# =========================================================
# Sidebar - Upload (TOP LEVEL)  ✅ df 먼저 만든다
# =========================================================
st.sidebar.title("마케팅/유통 시뮬레이터")

uploaded = st.sidebar.file_uploader(
    "Backdata 업로드 (xlsx/csv)",
    type=["xlsx", "csv"],
    key="uploader_backdata"
)

if st.sidebar.button("업로드 초기화", key="reset_upload"):
    st.session_state.pop("uploader_backdata", None)
    st.cache_data.clear()
    st.rerun()

if uploaded is None:
    st.info("좌측에서 backdata 파일(xlsx/csv)을 업로드하세요.")
    st.stop()

with st.spinner("파일 로딩 중..."):
    df = load_backdata(uploaded)

st.sidebar.success(f"업로드 완료: {uploaded.name} / {uploaded.size/1024/1024:.2f} MB")

cols = detect_columns(df)
col_scn = cols["scenario"]
col_disp = cols["display"]

if col_scn is None or col_scn not in df.columns:
    st.error("❌ '시나리오명' 컬럼을 찾지 못했습니다. (파일 컬럼명 확인 필요)")
    st.stop()

if col_disp is None or col_disp not in df.columns:
    st.warning("⚠️ 노출용 시나리오명 컬럼이 없어, 시나리오명을 그대로 노출합니다.")
    col_disp = col_scn
    df[col_disp] = df[col_scn].astype(str)

# scenario mapping
key_to_disp, disp_to_key, disp_list = scenario_options(df, col_scn, col_disp)

# Filters
stage_col, drv_col, cat_col, pos_col = cols["stage"], cols["drv"], cols["cat"], cols["pos"]

def uniq_vals(c):
    if c is None or c not in df.columns:
        return []
    return sorted([x for x in df[c].dropna().astype(str).unique().tolist() if str(x).strip() != ""])

st.sidebar.markdown("---")
st.sidebar.markdown("### 시나리오 필터")
f_search = st.sidebar.text_input("검색(노출 시나리오명)", value="", key="f_search")

f_stage = st.sidebar.selectbox("단계(ST)", ["(전체)"] + uniq_vals(stage_col), key="f_stage")
f_cat   = st.sidebar.selectbox("카테고리", ["(전체)"] + uniq_vals(cat_col), key="f_cat")
f_pos   = st.sidebar.selectbox("가격 포지션(POS)", ["(전체)"] + uniq_vals(pos_col), key="f_pos")
f_drv   = st.sidebar.selectbox("드라이버(DRV)", ["(전체)"] + uniq_vals(drv_col), key="f_drv")

df_f = df.copy()
if f_stage != "(전체)" and stage_col in df_f.columns:
    df_f = df_f[df_f[stage_col].astype(str) == f_stage]
if f_cat != "(전체)" and cat_col in df_f.columns:
    df_f = df_f[df_f[cat_col].astype(str) == f_cat]
if f_pos != "(전체)" and pos_col in df_f.columns:
    df_f = df_f[df_f[pos_col].astype(str) == f_pos]
if f_drv != "(전체)" and drv_col in df_f.columns:
    df_f = df_f[df_f[drv_col].astype(str) == f_drv

disp_candidates = sorted(list(set(df_f[col_disp].dropna().astype(str).str.strip().tolist())))
if f_search.strip():
    s = f_search.strip()
    disp_candidates = [x for x in disp_candidates if s in x]

if not disp_candidates:
    st.sidebar.warning("필터 결과가 없습니다. 필터를 완화하세요.")
    disp_candidates = disp_list

sel_disp = st.sidebar.selectbox("시나리오 선택", options=disp_candidates, key="sel_scn")
scenario_key = disp_to_key.get(sel_disp)
if scenario_key is None:
    scenario_key = next((k0 for k0, d0 in key_to_disp.items() if d0 == sel_disp), None)
if scenario_key is None:
    st.error("❌ 선택한 시나리오를 내부키로 매칭하지 못했습니다. (노출명 중복/매핑 확인)")
    st.stop()

row = df[df[col_scn].astype(str).str.strip() == str(scenario_key).strip()]
if row.empty:
    st.error("❌ 시나리오 행을 찾지 못했습니다.")
    st.stop()
row = row.iloc[0]

# Shares from row
rev_cols = cols["rev_cols"]
perf_cols = cols["perf_cols"]
viral_cols = cols["viral_cols"]
brand_cols = cols["brand_cols"]

rev_share = build_rev_shares(row, rev_cols)
media_share = build_media_shares(row, perf_cols, viral_cols, brand_cols)
group_share = media_share["group"]

# =========================================================
# Tabs (대행 / 브랜드 / 추천엔진)
# =========================================================
tab_agency, tab_brand, tab_rec = st.tabs(["대행", "브랜드사", "추천엔진"])

# =========================================================
# TAB: Agency
# =========================================================
with tab_agency:
    st.markdown("## 대행 모드")
    submode = st.radio("버전 선택", ["외부(클라이언트 제안용)", "내부(운영/정산용)"], horizontal=True, key="agency_sub")
    st.markdown(f"<div class='smallcap'>선택 시나리오: <span class='badge'>{sel_disp}</span></div>", unsafe_allow_html=True)

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown("### 입력 (공통)")
        calc_mode = st.radio("계산 방식", ["광고비 입력 → 매출 산출", "매출 입력 → 필요 광고비 산출"], horizontal=True, key="agency_calc_mode")

        aov = st.number_input("객단가(판매가) (원)", value=50000, step=1000, key="agency_aov")
        cost_rate = st.number_input("원가율 (%)", value=30.0, step=1.0, key="agency_cost_rate") / 100.0
        logistics_per_order = st.number_input("물류비(건당) (원)", value=3000, step=500, key="agency_logi")
        labor_cost = st.number_input("인건비/고정비 (원)", value=6000000, step=500000, key="agency_labor")

        cpc = st.number_input("CPC (원)", value=300.0, step=10.0, key="agency_cpc")
        cvr = st.number_input("CVR (%)", value=2.0, step=0.1, key="agency_cvr") / 100.0

        if calc_mode.startswith("광고비"):
            total_ad_budget = st.number_input("총 광고비 (원)", value=50000000, step=1000000, key="agency_ad_budget")
            target_revenue = None
        else:
            target_revenue = st.number_input("목표 매출 (원)", value=300000000, step=10000000, key="agency_target_rev")
            total_ad_budget = None

        st.markdown("### 퍼포먼스/바이럴/브랜드 그룹 배분 (100%)")
        gw = group_share if group_share else {"performance": 0.7, "viral": 0.2, "brand": 0.1}
        fig_gw = donut_chart(["퍼포먼스", "바이럴", "브랜드"], [gw["performance"], gw["viral"], gw["brand"]],
                             title="그룹 구성(100%)", height=280)
        safe_plotly(fig_gw, key="agency_group_donut")

    # compute P&L
    if calc_mode.startswith("광고비"):
        res = simulate_pl_from_adspend(total_ad_budget, aov, cpc, cvr, cost_rate, logistics_per_order, labor_cost)
    else:
        res = simulate_pl_from_revenue(target_revenue, aov, cpc, cvr, cost_rate, logistics_per_order, labor_cost)

    # allocate ad budget to groups
    ad_total = res["ad_spend"]
    perf_budget = ad_total * gw.get("performance", 0.0)
    viral_budget = ad_total * gw.get("viral", 0.0)
    brand_budget = ad_total * gw.get("brand", 0.0)

    with right:
        st.markdown("### 결과 요약")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("예상 매출", fmt_won(res["revenue"]))
        m2.metric("예상 광고비", fmt_won(res["ad_spend"]))
        m3.metric("영업이익", fmt_won(res["profit"]))
        m4.metric("공헌이익률", fmt_pct(res["contrib_margin"], 1))

        # 비용 구조
        st.markdown("### 비용 구조")
        items = ["광고비", "원가(매출원가)", "물류비", "인건비", "영업이익"]
        vals = [res["ad_spend"], res["cogs"], res["logistics"], res["labor"], res["profit"]]
        safe_plotly(bar_cost_chart(items, vals, height=320), key="agency_cost_bar")

        # Combined chart: revenue/ad bar + roas line (secondary y)
        st.markdown("### 핵심 비교(매출/광고비 + ROAS)")
        # single-point chart just for display consistency
        dfk = pd.DataFrame([{
            "항목": "선택 시나리오",
            "매출": res["revenue"],
            "광고비": res["ad_spend"],
            "ROAS(%)": res["roas"]
        }])

        roas_axis_max = st.slider("ROAS 축 상한(%)", min_value=200, max_value=5000, value=2000, step=100, key="agency_roas_axis")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(name="매출", x=dfk["항목"], y=dfk["매출"]), secondary_y=False)
        fig.add_trace(go.Bar(name="광고비", x=dfk["항목"], y=dfk["광고비"]), secondary_y=False)
        fig.add_trace(go.Scatter(name="ROAS(%)", x=dfk["항목"], y=dfk["ROAS(%)"], mode="lines+markers"), secondary_y=True)
        fig.update_layout(height=360, margin=dict(t=10, b=10), barmode="group", legend_orientation="h")
        fig.update_yaxes(title_text="", secondary_y=False)
        fig.update_yaxes(title_text="ROAS(%)", secondary_y=True, range=[0, roas_axis_max])
        safe_plotly(fig, key="agency_combo_chart")

    st.divider()

    # =========================================================
    # Media Mix tables (Agency)
    # =========================================================
    st.markdown("### 미디어 믹스(대행용)")
    st.markdown("<div class='smallcap'>퍼포먼스는 지면별 비율로 배분(100원 단위 반올림). 바이럴은 단가 기반 건수 산출(반올림, 합계 차이 허용).</div>", unsafe_allow_html=True)

    perf_share_map = media_share.get("performance", {})
    viral_share_map = media_share.get("viral", {})
    brand_share_map = media_share.get("brand", {})

    cA, cB = st.columns([1, 1])

    # ---- Performance mix donut + table
    with cA:
        st.markdown("#### 퍼포먼스 믹스(100%)")
        if perf_share_map:
            labels = list(perf_share_map.keys())
            values = list(perf_share_map.values())
            safe_plotly(donut_chart(labels, values, height=320), key="agency_perf_donut")
        else:
            st.info("퍼포먼스 비율 컬럼을 찾지 못했습니다(파일 컬럼명 확인).")

        st.markdown("#### 퍼포먼스 배분 테이블")
        perf_df = pd.DataFrame({"매체": list(perf_share_map.keys()), "비율": list(perf_share_map.values())})
        if not perf_df.empty:
            perf_df["계획예산"] = (perf_budget * perf_df["비율"]).round(-2)  # 100원 단위
            # agency fee rate by media
            perf_df["대행수수료율(%)"] = 10.0
            perf_df["페이백률(%)"] = 0.0

            if submode.startswith("내부"):
                st.markdown("<div class='smallcap'>내부버전: 대행수수료율/페이백률을 입력하면 청구/페이백이 계산됩니다.</div>", unsafe_allow_html=True)
                edited = st.data_editor(
                    perf_df,
                    use_container_width=True,
                    hide_index=True,
                    key="agency_perf_editor"
                )
                perf_df = edited.copy()
            else:
                st.dataframe(perf_df[["매체", "비율", "계획예산"]], use_container_width=True, hide_index=True)

            # compute billing and payback (internal)
            if submode.startswith("내부") and not perf_df.empty:
                perf_df["대행수수료율(%)"] = perf_df["대행수수료율(%)"].apply(lambda x: max(to_float(x, 0.0), 0.0))
                perf_df["페이백률(%)"] = perf_df["페이백률(%)"].apply(lambda x: max(to_float(x, 0.0), 0.0))

                perf_df["청구예상비용"] = perf_df["계획예산"] * (1 + perf_df["대행수수료율(%)"]/100.0)
                perf_df["페이백예상액"] = perf_df["계획예산"] * (perf_df["페이백률(%)"]/100.0)
                perf_df["실집행비(=매체비)"] = perf_df["계획예산"]
                perf_df["마진(청구-실집행)"] = perf_df["청구예상비용"] - perf_df["실집행비(=매체비)"]

                st.markdown("##### 퍼포먼스 요약(내부)")
                t1, t2, t3 = st.columns(3)
                t1.metric("청구예상 합계", fmt_won(perf_df["청구예상비용"].sum()))
                t2.metric("페이백예상 합계", fmt_won(perf_df["페이백예상액"].sum()))
                t3.metric("마진 합계", fmt_won(perf_df["마진(청구-실집행)"].sum()))

                st.dataframe(perf_df, use_container_width=True, hide_index=True)

    # ---- Viral mix donut + table + counts
    with cB:
        st.markdown("#### 바이럴 믹스(100%)")
        if viral_share_map:
            labels = list(viral_share_map.keys())
            values = list(viral_share_map.values())
            safe_plotly(donut_chart(labels, values, height=320), key="agency_viral_donut")
        else:
            st.info("바이럴 비율 컬럼을 찾지 못했습니다(파일 컬럼명 확인).")

        st.markdown("#### 바이럴 단가/비율 설정")
        # editable unit price table (user-provided template later)
        if "viral_unit_df" not in st.session_state:
            st.session_state["viral_unit_df"] = DEFAULT_VIRAL_UNIT.copy()

        unit_df = st.data_editor(
            st.session_state["viral_unit_df"],
            use_container_width=True,
            hide_index=True,
            key="viral_unit_editor"
        )
        st.session_state["viral_unit_df"] = unit_df.copy()

        # allocation
        alloc = allocate_viral_counts(viral_budget, unit_df)

        st.markdown("#### 바이럴 건수/비용 산출")
        show_df = alloc.copy()
        show_df["배정예산"] = show_df["배정예산"].map(lambda x: f"{x:,.0f}")
        show_df["건당비용"] = show_df["건당비용"].map(lambda x: f"{x:,.0f}")
        show_df["총비용"] = show_df["총비용"].map(lambda x: f"{x:,.0f}")
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        st.markdown("##### 바이럴 요약")
        v1, v2 = st.columns(2)
        v1.metric("바이럴 계획예산", fmt_won(viral_budget))
        v2.metric("바이럴 총비용(산출)", fmt_won(alloc["총비용"].sum()))

        if submode.startswith("내부"):
            st.markdown("#### 바이럴 내부정산(실집행 입력 → 마진)")
            internal = alloc.copy()
            internal["계획비(청구비)"] = internal["총비용"]
            internal["실집행비"] = 0.0
            edited2 = st.data_editor(
                internal[["구분", "지면", "건당비용", "진행건수", "계획비(청구비)", "실집행비"]],
                use_container_width=True,
                hide_index=True,
                key="viral_internal_editor"
            )
            internal = edited2.copy()
            internal["실집행비"] = internal["실집행비"].apply(lambda x: max(to_float(x, 0.0), 0.0))
            internal["계획비(청구비)"] = internal["계획비(청구비)"].apply(lambda x: max(to_float(x, 0.0), 0.0))
            internal["마진"] = internal["계획비(청구비)"] - internal["실집행비"]

            t1, t2, t3 = st.columns(3)
            t1.metric("계획(청구) 합계", fmt_won(internal["계획비(청구비)"].sum()))
            t2.metric("실집행 합계", fmt_won(internal["실집행비"].sum()))
            t3.metric("마진 합계", fmt_won(internal["마진"].sum()))

            st.dataframe(internal, use_container_width=True, hide_index=True)

    st.divider()

    # Revenue channel mix donut
    st.markdown("### 매출 채널 믹스(100%)")
    if rev_share:
        safe_plotly(donut_chart(list(rev_share.keys()), list(rev_share.values()), height=340), key="agency_rev_donut")
    else:
        st.info("매출 채널 믹스 관련 컬럼을 찾지 못했습니다(파일 컬럼명 확인).")

# =========================================================
# TAB: Brand
# =========================================================
with tab_brand:
    st.markdown("## 브랜드사 모드")
    submode_b = st.radio("버전 선택", ["외부(클라이언트 공유용)", "내부(운영용)"], horizontal=True, key="brand_sub")
    st.markdown(f"<div class='smallcap'>선택 시나리오: <span class='badge'>{sel_disp}</span></div>", unsafe_allow_html=True)

    # Simple input
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        total_budget = st.number_input("월 총 가용 예산(원)", value=200000000, step=10000000, key="brand_total_budget")
    with c2:
        selling_price = st.number_input("판매가(원)", value=50000, step=1000, key="brand_price")
    with c3:
        landed_cost = st.number_input("제품 원가(원, 단가)", value=12000, step=500, key="brand_landed")
    with c4:
        target_units = st.number_input("월 목표 물량(Unit)", value=10000, step=100, key="brand_units")

    # Optional inputs
    with st.expander("고급 옵션(선택)"):
        cpc_b = st.number_input("CPC(원)", value=300.0, step=10.0, key="brand_cpc")
        cvr_b = st.number_input("CVR(%)", value=2.0, step=0.1, key="brand_cvr") / 100.0
        logistics_b = st.number_input("물류비(건당, 원)", value=3000, step=500, key="brand_logi")
        labor_b = st.number_input("인건비/고정비(원)", value=6000000, step=500000, key="brand_labor")

    # budget split: import + marketing (import is units*landed)
    import_cost = target_units * landed_cost
    if import_cost > total_budget and landed_cost > 0:
        affordable_units = int(total_budget // landed_cost)
        import_cost = affordable_units * landed_cost
        target_units_eff = affordable_units
    else:
        target_units_eff = target_units

    marketing_budget = max(total_budget - import_cost, 0.0)
    clicks = marketing_budget / cpc_b if cpc_b > 0 else 0.0
    orders = clicks * cvr_b
    units_sold = min(float(orders), float(target_units_eff))
    sell_through = (units_sold / target_units * 100) if target_units > 0 else 0.0

    gross_profit = units_sold * (selling_price - landed_cost)
    net_profit = gross_profit - marketing_budget - (units_sold * logistics_b) - labor_b

    st.divider()
    st.markdown("### KPI 요약")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("예상 판매량(Units)", f"{units_sold:,.0f}")
    k2.metric("완판 예상율", fmt_pct(sell_through, 1))
    k3.metric("마케팅 예산", fmt_won(marketing_budget))
    k4.metric("예상 순수익", fmt_won(net_profit))

    st.markdown("### 예산 구성(100%)")
    labels = ["제품 수입비용", "마케팅비"]
    vals = [import_cost, marketing_budget]
    # show profit/loss slice (as magnitude)
    labels.append("예상 수익" if net_profit >= 0 else "예상 손실")
    vals.append(abs(net_profit))

    safe_plotly(donut_chart(labels, vals, height=340), key="brand_budget_donut")

    st.markdown("### 유통 채널(Top) 제안")
    if rev_share:
        # top5
        s = pd.Series(rev_share).sort_values(ascending=False).head(5)
        top_df = pd.DataFrame({"채널": s.index, "비중": s.values})
        top_df["비중(%)"] = top_df["비중"] * 100

        cA, cB = st.columns([1, 1])
        with cA:
            fig = px.bar(top_df, x="채널", y="비중(%)", text="비중(%)")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=330, margin=dict(t=10), yaxis_title=None, xaxis_title=None)
            safe_plotly(fig, key="brand_top_bar")
        with cB:
            safe_plotly(px.pie(top_df, values="비중(%)", names="채널", hole=0.55).update_traces(textinfo="percent+label"),
                        key="brand_top_pie")
    else:
        st.info("매출 채널 믹스 컬럼을 찾지 못했습니다.")

    # Monthly outlook (simple, not overpromising)
    st.divider()
    st.markdown("### 월별 전망(12개월)")
    growth = st.slider("월 성장률 가정(%)", min_value=-30, max_value=30, value=0, step=1, key="brand_growth") / 100.0

    months = list(range(1, 13))
    base_rev = units_sold * selling_price
    base_ad = marketing_budget

    series_rev = []
    series_ad = []
    series_roas = []

    for i in months:
        factor = (1 + growth) ** (i - 1)
        r_i = base_rev * factor
        a_i = base_ad * factor  # scale same for simplicity
        roas_i = (r_i / a_i) * 100 if a_i > 0 else 0
        series_rev.append(r_i)
        series_ad.append(a_i)
        series_roas.append(roas_i)

    dfm = pd.DataFrame({"월": [f"{i}월" for i in months], "매출": series_rev, "광고비": series_ad, "ROAS(%)": series_roas})

    roas_axis_max_b = st.slider("ROAS 축 상한(%)", min_value=200, max_value=5000, value=2000, step=100, key="brand_roas_axis")

    figm = make_subplots(specs=[[{"secondary_y": True}]])
    figm.add_trace(go.Bar(name="매출", x=dfm["월"], y=dfm["매출"]), secondary_y=False)
    figm.add_trace(go.Bar(name="광고비", x=dfm["월"], y=dfm["광고비"]), secondary_y=False)
    figm.add_trace(go.Scatter(name="ROAS(%)", x=dfm["월"], y=dfm["ROAS(%)"], mode="lines+markers"), secondary_y=True)
    figm.update_layout(height=420, margin=dict(t=10, b=10), barmode="group", legend_orientation="h")
    figm.update_yaxes(title_text="", secondary_y=False)
    figm.update_yaxes(title_text="ROAS(%)", secondary_y=True, range=[0, roas_axis_max_b])
    safe_plotly(figm, key="brand_monthly_combo")

    if submode_b.startswith("외부"):
        st.markdown("<div class='smallcap'>외부 공유용은 과도한 확정 수치를 피하기 위해 가정(성장률) 기반의 방향성 지표로만 제공합니다.</div>", unsafe_allow_html=True)

# =========================================================
# TAB: Recommendation Engine
# =========================================================
with tab_rec:
    st.markdown("## 추천 엔진 (Top3)")
    st.markdown("<div class='smallcap'>백데이터 기반(가능한 컬럼만 사용)으로 후보를 점수화해 상위 3개를 추천합니다.</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rec_stage = st.selectbox("단계(ST)", ["(전체)"] + uniq_vals(stage_col), key="rec_stage")
    with c2:
        rec_cat = st.selectbox("카테고리", ["(전체)"] + uniq_vals(cat_col), key="rec_cat")
    with c3:
        rec_pos = st.selectbox("POS", ["(전체)"] + uniq_vals(pos_col), key="rec_pos")
    with c4:
        rec_drv = st.selectbox("DRV", ["(전체)"] + uniq_vals(drv_col), key="rec_drv")

    sales_focus = st.radio("판매 우선 목표", ["자사몰", "온라인", "홈쇼핑", "공구"], horizontal=True, key="rec_focus")

    filters = {
        "stage": None if rec_stage == "(전체)" else rec_stage,
        "cat": None if rec_cat == "(전체)" else rec_cat,
        "pos": None if rec_pos == "(전체)" else rec_pos,
        "drv": None if rec_drv == "(전체)" else rec_drv,
        "sales_focus": sales_focus
    }

    rec = recommend_simple(df, cols, filters, topn=3)
    if rec.empty:
        st.info("조건에 맞는 후보가 없습니다. (메타 컬럼이 없거나, 필터가 너무 빡셀 수 있어요)")
    else:
        for i, r in rec.iterrows():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"### #{i+1} {r['노출 시나리오명']}")
            st.markdown(f"<div class='smallcap'>내부키: {r['내부키']} · 점수: {r['점수']:.1f}</div>", unsafe_allow_html=True)

            # show quick donuts: group + top rev
            rr = df[df[col_scn].astype(str).str.strip() == str(r["내부키"]).strip()]
            if not rr.empty:
                rr = rr.iloc[0]
                rev = build_rev_shares(rr, rev_cols)
                med = build_media_shares(rr, perf_cols, viral_cols, brand_cols)

                cc1, cc2 = st.columns(2)
                with cc1:
                    gw = med["group"]
                    safe_plotly(donut_chart(["퍼포먼스", "바이럴", "브랜드"], [gw["performance"], gw["viral"], gw["brand"]], height=260),
                                key=f"rec_gw_{i}")
                with cc2:
                    if rev:
                        s = pd.Series(rev).sort_values(ascending=False).head(6)
                        safe_plotly(donut_chart(list(s.index), list(s.values), height=260),
                                    key=f"rec_rev_{i}")
                    else:
                        st.info("채널 믹스 컬럼 미탐지")

            st.markdown("</div>", unsafe_allow_html=True)
