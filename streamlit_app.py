# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
from typing import Optional, Dict, List, Tuple
from datetime import date, datetime, timedelta
import calendar

# -------------------------
# Optional dependency: Plotly
# -------------------------
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

APP_PASSWORD = "ibrsecret"

# =========================
# Page / Theme
# =========================
st.set_page_config(page_title="마케팅/유통 시뮬레이터", layout="wide")

ACCENT = "#2F6FED"

st.markdown("""
<style>
html, body, [class*="css"]{
  font-size: 14px;
}

/* -------- Card helper (works on both themes) -------- */
.card{
  border-radius: 14px;
  padding: 14px 14px;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
}
.card h3{ margin:0; }

/* light theme overrides */
@media (prefers-color-scheme: light){
  .card{
    border: 1px solid rgba(0,0,0,0.08);
    background: #ffffff;
  }
}

/* avoid unreadable text in data editor / dataframes */
div[data-testid="stDataFrame"] div,
div[data-testid="stDataFrame"] span,
div[data-testid="stDataEditor"] div,
div[data-testid="stDataEditor"] span,
div[data-baseweb="select"] * ,
input, textarea{
  opacity: 1 !important;
}

/* small caption */
.smallcap{
  opacity: .75;
  font-size: 12px;
}

/* badge */
.badge{
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 12px;
  background: rgba(47,111,237,0.14);
  color: rgb(47,111,237);
}

/* section divider look */
hr.soft{
  border: 0;
  border-top: 1px solid rgba(255,255,255,0.10);
  margin: 12px 0;
}
@media (prefers-color-scheme: light){
  hr.soft{ border-top: 1px solid rgba(0,0,0,0.08); }
}
</style>
""", unsafe_allow_html=True)

# =========================
# Early guard: plotly required
# =========================
if not HAS_PLOTLY:
    st.error(
        "❌ plotly가 설치되어 있지 않습니다.\n\n"
        "✅ 해결:\n"
        "1) 로컬/코드스페이스: `pip install plotly`\n"
        "2) Streamlit Cloud: requirements.txt에 `plotly` 추가\n"
    )
    st.stop()

# =========================
# Auth (Password gate)
# =========================
def auth_gate() -> bool:
    if st.session_state.get("auth_ok", False):
        return True

    st.sidebar.markdown("## 🔒 접근 제한")
    pw = st.sidebar.text_input("비밀번호", type="password", key="auth_pw")
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if st.button("잠금 해제", key="auth_unlock"):
            if pw == APP_PASSWORD:
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                st.sidebar.error("비밀번호가 틀립니다.")
    with col2:
        if st.button("초기화", key="auth_reset"):
            st.session_state.pop("auth_ok", None)
            st.session_state.pop("auth_pw", None)
            st.rerun()

    st.info("좌측 사이드바에서 비밀번호를 입력하세요.")
    return False

if not auth_gate():
    st.stop()

# =========================
# Helpers
# =========================
def fmt_won(x) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):,.0f} 원"
    except Exception:
        return "-"

def fmt_num(x, digits=0) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):,.{digits}f}"
    except Exception:
        return "-"

def fmt_pct(x, digits=1) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):.{digits}f}%"
    except Exception:
        return "-"

def to_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        s = str(x).strip().replace(",", "").replace("₩", "")
        s = s.replace("%", "")
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except Exception:
        return default

def normalize_ratio(x) -> float:
    """supports 0.32, 32, '32%' -> returns 0~1"""
    v = to_float(x, default=np.nan)
    if np.isnan(v):
        return np.nan
    return (v / 100.0) if v > 1 else v

def normalize_shares(d: Dict[str, float]) -> Dict[str, float]:
    d2 = {k: float(v or 0.0) for k, v in d.items()}
    s = sum(v for v in d2.values() if v > 0)
    if s <= 0:
        return {k: 0.0 for k in d2}
    return {k: (v / s if v > 0 else 0.0) for k, v in d2.items()}

def round_to_100(x) -> int:
    try:
        return int(np.round(float(x) / 100.0) * 100)
    except Exception:
        return 0

def safe_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = [str(c).strip() for c in df.columns]
    for cand in candidates:
        if cand in cols:
            return cand
    for cand in candidates:
        for c in cols:
            if cand in c:
                return c
    return None

def drop_duplicate_dot_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Excel-style duplicated columns ending with .1/.2 ..."""
    cols = list(df.columns)
    base_seen = set()
    keep = []
    for c in cols:
        cstr = str(c)
        base = re.sub(r"\.\d+$", "", cstr)
        if base in base_seen and cstr != base:
            continue
        base_seen.add(base)
        keep.append(c)
    out = df[keep].copy()
    out.columns = [re.sub(r"\.\d+$", "", str(c)).strip() for c in out.columns]
    return out

def donut_chart(labels, values, title="", height=320):
    dd = pd.DataFrame({"name": labels, "value": values})
    fig = px.pie(dd, names="name", values="value", hole=0.55)
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=height, margin=dict(t=40, b=10, l=10, r=10), title=title)
    return fig

# =========================
# Data loading (xlsx/csv)
# =========================
@st.cache_data(show_spinner=False)
def load_backdata_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()

    if name.endswith(".csv"):
        raw = file_bytes.decode("utf-8-sig", errors="replace")
        df = pd.read_csv(StringIO(raw))
        df = df.dropna(how="all")
        df = drop_duplicate_dot_columns(df)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    try:
        xls = pd.ExcelFile(pd.io.common.BytesIO(file_bytes))
    except Exception as e:
        raise RuntimeError(
            "엑셀(xlsx) 로드 실패. Streamlit Cloud라면 requirements.txt에 openpyxl 추가가 필요할 수 있습니다.\n"
            f"원인: {e}"
        )

    sheet = None
    for s in xls.sheet_names:
        s_norm = str(s).strip().lower()
        if s_norm in ("backdata", "back_data", "back data", "backdata "):
            sheet = s
            break
        if "backdata" in s_norm:
            sheet = s
            break
    if sheet is None:
        for s in xls.sheet_names:
            if str(s).strip().upper() == "BACKDATA":
                sheet = s
                break
    if sheet is None:
        sheet = xls.sheet_names[0]

    df = pd.read_excel(xls, sheet_name=sheet)
    df = df.dropna(how="all")
    df = drop_duplicate_dot_columns(df)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def load_backdata(uploaded_file) -> pd.DataFrame:
    return load_backdata_cached(uploaded_file.getvalue(), uploaded_file.name)

# =========================
# Column detection (v4 with KPI + growth/repurchase/ad_contrib)
# =========================
def detect_columns(df: pd.DataFrame) -> Dict[str, object]:
    col_scn = safe_col(df, ["시나리오명", "scenario", "Scenario"])
    col_disp = safe_col(df, ["노출 시나리오명", "노출시나리오명", "display", "표시 시나리오명"])
    if col_scn is None:
        col_scn = df.columns[0]
    if col_disp is None:
        col_disp = df.columns[1] if len(df.columns) > 1 else col_scn

    col_stage = safe_col(df, ["단계(ST)", "단계", "ST"])
    col_drv = safe_col(df, ["드라이버(DRV)", "드라이버", "DRV"])
    col_cat = safe_col(df, ["카테고리(대)", "카테고리", "CAT"])
    col_pos = safe_col(df, ["가격포지션(POS)", "가격포지션", "POS"])

    # ✅ backdata 핵심
    col_month_growth = safe_col(df, ["월성장률(%)", "월 성장률", "monthly growth"])
    col_repurchase = safe_col(df, ["재구매율(%)", "재구매율", "repurchase"])
    col_ad_contrib = safe_col(df, ["광고기여율(%)", "광고기여율", "ad contribution"])

    rev_cols = [c for c in df.columns if str(c).endswith("매출비중") and c not in [col_scn, col_disp]]

    perf_cols = [
        c for c in df.columns
        if (str(c).startswith("퍼포먼스마케팅_") or str(c) == "퍼포먼스_외부몰PA")
        and not str(c).startswith("KPI_")
    ]
    viral_cols = [c for c in df.columns if str(c).startswith("바이럴마케팅_") and not str(c).startswith("KPI_")]

    brand_cols = []
    for c in df.columns:
        s = str(c)
        if s.startswith("KPI_"):
            continue
        if s in ["브랜드 마케팅", "기타_브랜드", "기타 브랜드", "기타_브랜드%"]:
            brand_cols.append(c)
        elif ("브랜드" in s and "마케팅" in s and not s.startswith("KPI_")):
            brand_cols.append(c)

    apply_internal = safe_col(df, ["apply_internal(내부)", "apply_internal", "내부 적용"])
    apply_client = safe_col(df, ["apply_client(브랜드사)", "apply_client", "브랜드사 적용"])
    apply_agency = safe_col(df, ["apply_agency(대행)", "apply_agency", "대행 적용"])

    return {
        "scenario": col_scn,
        "display": col_disp,
        "stage": col_stage,
        "drv": col_drv,
        "cat": col_cat,
        "pos": col_pos,
        "month_growth": col_month_growth,
        "repurchase": col_repurchase,
        "ad_contrib": col_ad_contrib,
        "rev_cols": rev_cols,
        "perf_cols": perf_cols,
        "viral_cols": viral_cols,
        "brand_cols": brand_cols,
        "kpi_cols": [c for c in df.columns if str(c).startswith("KPI_")],
        "apply_internal": apply_internal,
        "apply_client": apply_client,
        "apply_agency": apply_agency,
    }

def scenario_options(df: pd.DataFrame, col_scn: str, col_disp: str):
    tmp = df[[col_scn, col_disp]].copy()
    tmp[col_scn] = tmp[col_scn].astype(str).str.strip()
    tmp[col_disp] = tmp[col_disp].astype(str).str.strip()
    tmp = tmp.dropna()

    key_to_disp = dict(zip(tmp[col_scn], tmp[col_disp]))
    disp_to_key = {}
    for kk, dd in key_to_disp.items():
        if dd in disp_to_key and disp_to_key[dd] != kk:
            disp_to_key[f"{dd} ({kk})"] = kk
        else:
            disp_to_key[dd] = kk
    disp_list = sorted(list(disp_to_key.keys()))
    return key_to_disp, disp_to_key, disp_list

# =========================
# Media pretty & buckets
# =========================
def pretty_media_name(col: str) -> str:
    c = str(col).strip()
    c = c.replace("퍼포먼스마케팅_", "")
    c = c.replace("바이럴마케팅_", "")
    c = c.replace("씨딩", "시딩")
    c = c.replace("네이버 ", "네이버")
    return c

def perf_category(media: str) -> str:
    m = str(media)
    if "SA" in m:
        return "검색 광고"
    if any(x in m for x in ["GDN", "GFA", "메타", "틱톡", "크리테오", "토스"]):
        return "디스플레이/소셜"
    if "외부몰PA" in m or "쿠팡" in m:
        return "마켓/PA"
    return "기타"

# =========================
# Viral price table (editable)
# =========================
DEFAULT_VIRAL_PRICE = pd.DataFrame([
    ["네이버", "네이버_인플루언서탭", 250000, 1.0],
    ["네이버", "네이버_스마트블록", 250000, 1.0],
    ["네이버", "네이버_지식인", 100000, 1.0],
    ["네이버", "네이버_쇼핑상위", 2000000, 1.0],
    ["네이버", "네이버_인기글", 300000, 1.0],
    ["네이버", "네이버_자동검색완성", 400000, 1.0],
    ["네이버", "네이버_카페침투바이럴", 30000, 1.0],
    ["네이버", "네이버_구매대행", 120060, 1.0],
    ["네이버", "네이버_핫딜", 100000, 1.0],
    ["인스타그램", "인스타그램_파워페이지", 400000, 1.0],
    ["인스타그램", "인스타그램_해시태그상위노출", 500000, 1.0],
    ["인스타그램", "인스타그램_계정상위노출", 400000, 1.0],
    ["오늘의집", "오늘의집_집들이", 500000, 1.0],
    ["오늘의집", "오늘의집_체험단", 400000, 1.0],
    ["오늘의집", "오늘의집_구매대행", 200952, 1.0],
    ["기타 커뮤니티", "커뮤니티_핫딜", 200000, 1.0],
], columns=["매체", "지면", "건당비용", "비율"])

# =========================
# Shares builder
# =========================
def build_rev_shares(row: pd.Series, rev_cols: List[str]) -> Dict[str, float]:
    d = {}
    for c in rev_cols:
        v = normalize_ratio(row.get(c))
        if pd.isna(v):
            v = 0.0
        name = str(c).replace("매출비중", "").strip()
        d[name] = float(v)
    return normalize_shares(d)

def build_media_shares(row: pd.Series, perf_cols: List[str], viral_cols: List[str], brand_cols: List[str]):
    perf_raw, viral_raw, brand_raw = {}, {}, {}
    for c in perf_cols:
        v = normalize_ratio(row.get(c))
        if pd.isna(v): v = 0.0
        perf_raw[pretty_media_name(c)] = float(v)
    for c in viral_cols:
        v = normalize_ratio(row.get(c))
        if pd.isna(v): v = 0.0
        viral_raw[pretty_media_name(c)] = float(v)
    for c in brand_cols:
        v = normalize_ratio(row.get(c))
        if pd.isna(v): v = 0.0
        brand_raw[pretty_media_name(c)] = float(v)

    perf_sum = sum(v for v in perf_raw.values() if v > 0)
    viral_sum = sum(v for v in viral_raw.values() if v > 0)
    brand_sum = sum(v for v in brand_raw.values() if v > 0)
    total = perf_sum + viral_sum + brand_sum

    group = {"퍼포먼스": 1.0, "바이럴": 0.0, "브랜드": 0.0} if total <= 0 else {
        "퍼포먼스": perf_sum / total,
        "바이럴": viral_sum / total,
        "브랜드": brand_sum / total
    }

    return {
        "group": group,
        "perf": normalize_shares(perf_raw),
        "viral": normalize_shares(viral_raw),
        "brand": normalize_shares(brand_raw),
        "raw_sums": {"perf": perf_sum, "viral": viral_sum, "brand": brand_sum},
    }

def viral_medium_shares(viral_share_dict: Dict[str, float]) -> Dict[str, float]:
    buckets = {"네이버": 0.0, "인스타그램": 0.0, "오늘의집": 0.0, "기타 커뮤니티": 0.0}
    for k, v in viral_share_dict.items():
        kk = str(k)
        if "네이버" in kk:
            buckets["네이버"] += v
        elif "인스타" in kk:
            buckets["인스타그램"] += v
        elif "오늘의집" in kk:
            buckets["오늘의집"] += v
        else:
            buckets["기타 커뮤니티"] += v
    return normalize_shares(buckets)

# =========================
# KPI blending (scenario-specific)
# =========================
def kpi_get(row: pd.Series, media_full: str, metric: str) -> Optional[float]:
    key = f"KPI_{metric}_{media_full}"
    if key in row.index:
        v = to_float(row.get(key), default=np.nan)
        if np.isnan(v):
            return None
        if metric in ("CTR", "CVR") and v > 1:
            v = v / 100.0
        return float(v)
    return None

def derive_cpc_from_cpm_ctr(cpm: Optional[float], ctr: Optional[float]) -> Optional[float]:
    if cpm is None or ctr is None:
        return None
    if cpm <= 0 or ctr <= 0:
        return None
    return float(cpm) / (1000.0 * float(ctr))

def blended_cpc_cvr(row: pd.Series, perf_cols: List[str]) -> Tuple[Optional[float], Optional[float]]:
    raw = {}
    for c in perf_cols:
        v = normalize_ratio(row.get(c))
        if pd.isna(v): v = 0.0
        raw[str(c)] = float(v)
    shares = normalize_shares(raw)

    cpc_vals, cvr_vals = [], []
    weights_cpc, weights_cvr = [], []

    for media_full, w in shares.items():
        if w <= 0:
            continue

        cpc = kpi_get(row, media_full, "CPC")
        if cpc is None:
            cpm = kpi_get(row, media_full, "CPM")
            ctr = kpi_get(row, media_full, "CTR")
            cpc = derive_cpc_from_cpm_ctr(cpm, ctr)

        cvr = kpi_get(row, media_full, "CVR")

        if cpc is not None and cpc > 0:
            cpc_vals.append(cpc); weights_cpc.append(w)
        if cvr is not None and cvr > 0:
            cvr_vals.append(cvr); weights_cvr.append(w)

    def wavg(vals, ws):
        if not vals or not ws:
            return None
        s = sum(ws)
        if s <= 0:
            return None
        return float(sum(v * w for v, w in zip(vals, ws)) / s)

    return wavg(cpc_vals, weights_cpc), wavg(cvr_vals, weights_cvr)

def roas_from_kpi(aov: float, cpc: float, cvr: float) -> float:
    # spend -> clicks = spend/cpc, orders=clicks*cvr, revenue=orders*aov
    # ROAS = revenue/spend = (aov*cvr)/cpc
    if cpc <= 0:
        return 0.0
    return float(aov) * float(cvr) / float(cpc)

# =========================
# Growth / Repurchase / Ad Contribution model
# =========================
def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def split_revenue(total_rev: float, repurchase_rate: float) -> Dict[str, float]:
    rr = clamp01(repurchase_rate)
    repeat = total_rev * rr
    nonrepeat = total_rev - repeat
    return {"total": total_rev, "repeat": repeat, "nonrepeat": nonrepeat}

def ad_revenue_from_total(total_rev: float, repurchase_rate: float, ad_contrib: float) -> float:
    # 광고기여율은 "신규/비재구매 매출(Non-repeat)"에 적용
    rr = clamp01(repurchase_rate)
    acr = clamp01(ad_contrib)
    nonrepeat = total_rev * (1.0 - rr)
    return nonrepeat * acr

def total_from_ad_revenue(ad_rev: float, repurchase_rate: float, ad_contrib: float) -> float:
    # ad_rev = total*(1-rr)*acr  => total = ad_rev / ((1-rr)*acr)
    rr = clamp01(repurchase_rate)
    acr = clamp01(ad_contrib)
    denom = (1.0 - rr) * acr
    if denom <= 1e-9:
        return 0.0
    return float(ad_rev) / float(denom)

# =========================
# P&L / Simulation (two-way) with ad_contrib + repurchase
# =========================
def simulate_pl_with_mix(
    calc_mode: str,
    aov: float,
    cpc: float,
    cvr: float,
    cost_rate: float,
    logistics_per_order: float,
    fixed_cost: float,
    ad_spend: Optional[float],
    revenue: Optional[float],
    repurchase_rate: float,
    ad_contrib: float,
):
    rr = clamp01(repurchase_rate)
    acr = clamp01(ad_contrib)

    if calc_mode.startswith("매출"):
        # 사용자가 입력한 revenue는 "총매출"
        total_rev = float(revenue or 0.0)

        # 총매출 -> 광고기여 매출(Non-repeat에 acr 적용)
        ad_rev = ad_revenue_from_total(total_rev, rr, acr)

        # 광고기여 매출 -> 필요한 광고비
        ad_orders = ad_rev / aov if aov > 0 else 0.0
        clicks = ad_orders / cvr if cvr > 0 else 0.0
        ad_spend = clicks * cpc

    else:
        # 사용자가 입력한 ad_spend는 광고 집행비
        ad_spend = float(ad_spend or 0.0)
        clicks = ad_spend / cpc if cpc > 0 else 0.0
        ad_orders = clicks * cvr
        ad_rev = ad_orders * aov

        # 광고기여 매출 -> 총매출 환산
        total_rev = total_from_ad_revenue(ad_rev, rr, acr)

    # 총매출 기준 주문/원가/이익 계산
    orders_total = total_rev / aov if aov > 0 else 0.0

    cogs = total_rev * cost_rate
    logistics = orders_total * logistics_per_order
    profit = total_rev - (ad_spend + cogs + logistics + fixed_cost)
    contrib_margin = ((total_rev - ad_spend - logistics - cogs) / total_rev * 100) if total_rev > 0 else 0.0
    roas = (total_rev / ad_spend) if ad_spend and ad_spend > 0 else 0.0

    split = split_revenue(total_rev, rr)

    return {
        "total_revenue": float(total_rev),
        "ad_spend": float(ad_spend),
        "orders_total": float(orders_total),
        "clicks": float(clicks),
        "ad_revenue": float(ad_rev),
        "repeat_revenue": float(split["repeat"]),
        "nonrepeat_revenue": float(split["nonrepeat"]),
        "cogs": float(cogs),
        "logistics": float(logistics),
        "fixed": float(fixed_cost),
        "profit": float(profit),
        "contrib_margin": float(contrib_margin),
        "roas": float(roas),
        "repurchase_rate": rr,
        "ad_contrib": acr,
    }

# =========================
# Mix builders
# =========================
def build_performance_mix_table(perf_share: Dict[str, float], total_perf_budget: float) -> pd.DataFrame:
    rows = []
    for media, share in perf_share.items():
        if share <= 0:
            continue
        budget = round_to_100(total_perf_budget * share)
        rows.append({
            "구분": "퍼포먼스",
            "구분2": perf_category(media),
            "매체": media,
            "지면/캠페인": "",
            "예산(계획)": budget,
            "목표 ROAS(%)": 0.0,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["구분2", "매체"]).reset_index(drop=True)

def build_viral_mix_table(
    viral_price_df: pd.DataFrame,
    medium_share: Dict[str, float],
    total_viral_budget: float,
) -> pd.DataFrame:
    rows = []
    vp = viral_price_df.copy()

    for c in ["매체", "지면"]:
        if c not in vp.columns:
            return pd.DataFrame()

    vp["건당비용"] = vp["건당비용"].apply(lambda x: to_float(x, 0.0))
    vp["비율"] = vp["비율"].apply(lambda x: to_float(x, 1.0))
    vp["비율"] = vp["비율"].replace(0, 1.0)

    for medium, mshare in medium_share.items():
        medium_budget = float(total_viral_budget) * float(mshare)
        sub = vp[vp["매체"] == medium].copy()
        if sub.empty:
            continue

        sub_w = normalize_shares(dict(zip(sub["지면"], sub["비율"])))

        for surface, w in sub_w.items():
            unit = float(sub.loc[sub["지면"] == surface, "건당비용"].iloc[0])
            planned = medium_budget * float(w)
            cnt = int(np.round(planned / unit)) if unit > 0 else 0
            total_cost = cnt * unit
            rows.append({
                "구분": "바이럴",
                "구분2": "",
                "매체": medium,
                "지면/캠페인": surface,
                "건당비용": unit,
                "진행 건수": int(cnt),
                "예산(계획)": round_to_100(total_cost),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["매체", "지면/캠페인"]).reset_index(drop=True)

def unify_mix_table(perf_df: pd.DataFrame, viral_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["구분", "구분2", "매체", "지면/캠페인", "예산(계획)"]
    out = []
    if perf_df is not None and not perf_df.empty:
        tmp = perf_df.copy()
        for c in base_cols:
            if c not in tmp.columns:
                tmp[c] = ""
        out.append(tmp[base_cols + [c for c in tmp.columns if c not in base_cols]])
    if viral_df is not None and not viral_df.empty:
        tmp = viral_df.copy()
        for c in base_cols:
            if c not in tmp.columns:
                tmp[c] = ""
        out.append(tmp[base_cols + [c for c in tmp.columns if c not in base_cols]])
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)

# =========================
# Treemap builders
# =========================
def rev_bucket(channel_name: str) -> str:
    s = str(channel_name)
    if "자사" in s:
        return "자사몰"
    if "스마트" in s or "스토어" in s or "온라인(마켓)" in s:
        return "온라인(마켓)"
    if "쿠팡" in s:
        return "온라인(마켓)"
    if any(k in s for k in ["오프라인", "면세", "리테일", "백화점", "마트", "드럭", "올리브영"]):
        return "오프라인"
    if "홈쇼핑" in s:
        return "홈쇼핑"
    if "공구" in s or "공동" in s:
        return "공구"
    if "B2B" in s or "도매" in s:
        return "B2B/도매"
    return "온라인(기타)"

def treemap_revenue(rev_share: Dict[str, float], height=380, title="매출 채널 구성(트리맵)"):
    rows = []
    for ch, v in rev_share.items():
        if v <= 0:
            continue
        rows.append({"그룹": rev_bucket(ch), "채널": ch, "비중": float(v)})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig = px.treemap(df, path=["그룹", "채널"], values="비중", color="그룹")
    fig.update_layout(height=height, margin=dict(t=50, b=10, l=10, r=10), title=title)
    fig.update_traces(texttemplate="%{label}<br>%{value:.1%}")
    return fig

def treemap_ads(perf_df: pd.DataFrame, viral_df: pd.DataFrame, height=430, title="광고 믹스(트리맵)"):
    rows = []
    if perf_df is not None and not perf_df.empty:
        for _, r in perf_df.iterrows():
            rows.append({
                "그룹": "퍼포먼스",
                "매체": r.get("매체",""),
                "지면": r.get("지면/캠페인","") or r.get("매체",""),
                "예산": float(r.get("예산(계획)",0) or 0)
            })
    if viral_df is not None and not viral_df.empty:
        for _, r in viral_df.iterrows():
            rows.append({
                "그룹": "바이럴",
                "매체": r.get("매체",""),
                "지면": r.get("지면/캠페인",""),
                "예산": float(r.get("예산(계획)",0) or 0)
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df = df[df["예산"] > 0]
    if df.empty:
        return None
    fig = px.treemap(df, path=["그룹", "매체", "지면"], values="예산", color="그룹")
    fig.update_layout(height=height, margin=dict(t=50, b=10, l=10, r=10), title=title)
    return fig

# =========================
# Editable tables (perf/viral)
# =========================
def editable_perf_table(perf_df: pd.DataFrame, submode: str, key_prefix: str) -> pd.DataFrame:
    if perf_df.empty:
        return perf_df
    perf_df = perf_df.copy()

    if submode.startswith("내부"):
        if "대행수수료율(%)" not in perf_df.columns:
            perf_df["대행수수료율(%)"] = 0.0
        if "페이백률(%)" not in perf_df.columns:
            perf_df["페이백률(%)"] = 0.0

        edited = st.data_editor(
            perf_df[["구분2", "매체", "예산(계획)", "목표 ROAS(%)", "대행수수료율(%)", "페이백률(%)"]],
            use_container_width=True,
            hide_index=True,
            disabled=["구분2", "매체"],
            key=f"{key_prefix}_perf_editor_int",
        )
        outp = perf_df.copy()
        outp.update(edited)

        outp["예산(계획)"] = outp["예산(계획)"].apply(lambda x: round_to_100(to_float(x, 0.0)))
        outp["청구예상비용"] = outp.apply(
            lambda r: round_to_100(float(r["예산(계획)"]) * (1.0 + float(r["대행수수료율(%)"]) / 100.0)), axis=1
        )
        outp["페이백예상액"] = outp.apply(
            lambda r: round_to_100(float(r["예산(계획)"]) * (float(r["페이백률(%)"]) / 100.0)), axis=1
        )

        st.dataframe(
            outp[["구분2", "매체", "예산(계획)", "목표 ROAS(%)", "대행수수료율(%)", "청구예상비용", "페이백률(%)", "페이백예상액"]],
            use_container_width=True,
            hide_index=True
        )
        return outp

    edited = st.data_editor(
        perf_df[["구분2", "매체", "예산(계획)", "목표 ROAS(%)"]],
        use_container_width=True,
        hide_index=True,
        disabled=["구분2", "매체"],
        key=f"{key_prefix}_perf_editor_ext",
    )
    outp = perf_df.copy()
    outp.update(edited)
    outp["예산(계획)"] = outp["예산(계획)"].apply(lambda x: round_to_100(to_float(x, 0.0)))
    st.dataframe(outp[["구분2", "매체", "예산(계획)", "목표 ROAS(%)"]], use_container_width=True, hide_index=True)
    return outp

def editable_viral_table(viral_df: pd.DataFrame, submode: str, key_prefix: str) -> pd.DataFrame:
    if viral_df.empty:
        return viral_df

    viral_df = viral_df.copy()
    viral_df["예산(계획)"] = viral_df["예산(계획)"].apply(lambda x: round_to_100(to_float(x, 0.0)))
    viral_df["진행 건수"] = viral_df["진행 건수"].apply(lambda x: int(to_float(x, 0.0)))

    if submode.startswith("내부"):
        if "실집행비(원)" not in viral_df.columns:
            viral_df["실집행비(원)"] = 0.0

        edited = st.data_editor(
            viral_df[["매체", "지면/캠페인", "건당비용", "진행 건수", "예산(계획)", "실집행비(원)"]],
            use_container_width=True,
            hide_index=True,
            disabled=["매체", "지면/캠페인", "건당비용"],
            key=f"{key_prefix}_viral_editor_int",
        )
        outv = viral_df.copy()
        outv.update(edited)

        outv["진행 건수"] = outv["진행 건수"].apply(lambda x: int(to_float(x, 0.0)))
        outv["예산(계획)"] = outv.apply(lambda r: round_to_100(float(r["진행 건수"]) * float(r["건당비용"])), axis=1)
        outv["실집행비(원)"] = outv["실집행비(원)"].apply(lambda x: round_to_100(to_float(x, 0.0)))
        outv["마진(원)"] = outv["예산(계획)"].astype(float) - outv["실집행비(원)"].astype(float)

        st.dataframe(
            outv[["매체", "지면/캠페인", "건당비용", "진행 건수", "예산(계획)", "실집행비(원)", "마진(원)"]],
            use_container_width=True,
            hide_index=True
        )
        return outv

    edited = st.data_editor(
        viral_df[["매체", "지면/캠페인", "건당비용", "진행 건수", "예산(계획)"]],
        use_container_width=True,
        hide_index=True,
        disabled=["매체", "지면/캠페인", "건당비용"],
        key=f"{key_prefix}_viral_editor_ext",
    )
    outv = viral_df.copy()
    outv.update(edited)
    outv["진행 건수"] = outv["진행 건수"].apply(lambda x: int(to_float(x, 0.0)))
    outv["예산(계획)"] = outv.apply(lambda r: round_to_100(float(r["진행 건수"]) * float(r["건당비용"])), axis=1)
    st.dataframe(outv[["매체", "지면/캠페인", "건당비용", "진행 건수", "예산(계획)"]], use_container_width=True, hide_index=True)
    return outv

# =========================
# Stock depletion schedule
# =========================
def month_days(yyyy: int, mm: int) -> int:
    return calendar.monthrange(yyyy, mm)[1]

def build_monthly_forecast(
    start_ym: str,
    months: int,
    base_month_rev: float,
    monthly_growth: float,
    aov: float,
    repurchase_rate: float,
    ad_contrib: float,
    roas_ad: float,  # 광고기여 매출 기준 ROAS(= 광고매출/광고비)
) -> pd.DataFrame:
    # start_ym: "YYYY-MM"
    y, m = start_ym.split("-")
    y0, m0 = int(y), int(m)

    rows = []
    for i in range(months):
        yy = y0 + (m0 - 1 + i) // 12
        mm = (m0 - 1 + i) % 12 + 1
        ym = f"{yy:04d}-{mm:02d}"

        factor = (1.0 + monthly_growth) ** i
        total_rev = float(base_month_rev) * factor

        ad_rev = ad_revenue_from_total(total_rev, repurchase_rate, ad_contrib)
        need_ad = (ad_rev / roas_ad) if roas_ad > 0 else 0.0

        units = (total_rev / aov) if aov > 0 else 0.0
        split = split_revenue(total_rev, repurchase_rate)

        rows.append({
            "월": ym,
            "총매출": total_rev,
            "광고기여매출": ad_rev,
            "필요광고비": need_ad,
            "재구매매출": split["repeat"],
            "비재구매매출": split["nonrepeat"],
            "예상판매수량(Units)": units,
        })

    dfm = pd.DataFrame(rows)
    return dfm

def calc_depletion_date(
    start_date: date,
    initial_stock: int,
    forecast_df: pd.DataFrame,
) -> Tuple[Optional[date], pd.DataFrame]:
    # 일별로 쪼개서 소진일 추정
    stock = float(initial_stock)
    cur = start_date

    rows = []
    for _, r in forecast_df.iterrows():
        ym = str(r["월"])
        yy, mm = int(ym.split("-")[0]), int(ym.split("-")[1])
        days = month_days(yy, mm)
        units_month = float(r.get("예상판매수량(Units)", 0.0) or 0.0)
        daily = units_month / days if days > 0 else 0.0

        for _d in range(days):
            if stock <= 0:
                return cur, pd.DataFrame(rows)
            stock -= daily
            rows.append({
                "일자": cur.isoformat(),
                "일판매(추정)": daily,
                "잔여재고(추정)": max(stock, 0.0),
            })
            cur += timedelta(days=1)

    # 기간 내 소진 안됨
    return None, pd.DataFrame(rows)

# =========================
# Recommendation (simple Top3 across scenarios)
# =========================
def focus_match_score(rev_share: Dict[str, float], focus: str) -> float:
    if focus == "자사몰":
        kw = ["자사"]
    elif focus == "온라인(마켓)":
        kw = ["온라인", "마켓", "스마트", "스토어", "쿠팡"]
    elif focus == "홈쇼핑":
        kw = ["홈쇼핑"]
    elif focus == "공구":
        kw = ["공구", "공동"]
    elif focus == "B2B/도매":
        kw = ["B2B", "도매"]
    else:
        kw = []
    if not kw:
        return 0.0
    s = 0.0
    for k, v in rev_share.items():
        if any(x in str(k) for x in kw):
            s += float(v)
    return float(s)

def strategy_recommendation_from_revshare(rev_share: Dict[str, float]) -> Dict[str, object]:
    # 기존 룰 기반(너가 예전에 말한 룰 반영)
    def share_contains(keyword: str) -> float:
        return sum(float(v) for k, v in rev_share.items() if keyword in str(k))

    own = share_contains("자사")
    smart = share_contains("스마트") + share_contains("스토어") + share_contains("온라인")
    coupang = share_contains("쿠팡")
    home = share_contains("홈쇼핑")
    groupbuy = share_contains("공구") + share_contains("공동")

    if home >= max(own, smart, coupang, groupbuy) and home > 0:
        title = "홈쇼핑 연계형"
        priority = [
            ("Naver SA", "홈쇼핑 유입/검색 수요 회수 중심"),
            ("네이버 블로그·콘텐츠", "검색 신뢰/후기·정보성 보강"),
            ("쿠팡 PA", "방송 후 수요를 마켓에서 흡수"),
        ]
        note = "이 케이스는 메타/구글 집행은 제외(또는 최소) 권장"
    elif groupbuy >= max(own, smart, coupang, home) and groupbuy > 0:
        title = "공구(그룹바잉) 중심형"
        priority = [
            ("인플루언서(인스타 메가)", "공구는 ‘판매자 파워/신뢰’가 매출을 좌우"),
            ("바이럴(핫딜/커뮤니티)", "구매 트리거·확산"),
            ("외부몰/제휴 PA", "공구 외 추가 판매분 흡수"),
        ]
        note = "퍼포먼스보다 ‘판매자/콘텐츠 드라이브’가 우선"
    elif coupang >= max(own, smart, home, groupbuy) and coupang > 0:
        title = "쿠팡(마켓) 중심형"
        priority = [
            ("외부몰 PA(쿠팡)", "가장 직접적인 매출 견인 레버"),
            ("메타", "리타겟/확장 및 수요 생성(보조)"),
            ("네이버 SA", "보조 검색 수요 회수"),
        ]
        note = "쿠팡 비중이 클수록 PA 1순위, 그 다음 메타가 자연스러움"
    elif smart >= max(own, coupang, home, groupbuy) and smart > 0:
        title = "스마트스토어(네이버) 중심형"
        priority = [
            ("Naver SA", "검색 기반 전환 확보"),
            ("네이버 DA/GFA", "네이버 생태계 내 확장"),
            ("바이럴(네이버 지면)", "스마트블록/콘텐츠 연계"),
        ]
        note = "스마트스토어 비중이 클수록 네이버 비중을 높이는 게 일관됨"
    elif own >= max(smart, coupang, home, groupbuy) and own > 0:
        title = "자사몰 중심형"
        priority = [
            ("메타", "자사몰은 랜딩/리타겟 설계 강점 → 효율 기대"),
            ("Google(선택)", "검색 수요 회수(상품/브랜드 검색 중심)"),
            ("네이버 SA(선택)", "국내 검색 수요 보조"),
        ]
        note = "자사몰 비중이 클수록 메타 비중을 키우는 룰이 잘 맞음"
    else:
        title = "혼합형(균형 운영)"
        priority = [
            ("Naver SA", "기본 검색 수요 회수"),
            ("메타", "수요 생성/리타겟"),
            ("마켓 PA", "보유 채널에서 매출 흡수"),
        ]
        note = "채널이 치우치지 않으면 3축 균형 운영 권장"

    top3 = sorted(rev_share.items(), key=lambda x: x[1], reverse=True)[:3]
    evidence = [f"{k}: {v*100:.1f}%" for k, v in top3 if v > 0]
    return {"title": title, "priority": priority, "note": note, "evidence": evidence}

# =========================
# Sidebar - Upload
# =========================
st.sidebar.title("마케팅/유통 시뮬레이터")

uploaded = st.sidebar.file_uploader(
    "Backdata 업로드 (xlsx/csv)",
    type=["xlsx", "csv"],
    key="backdata_uploader"
)

if st.sidebar.button("업로드 초기화", key="reset_uploader"):
    st.session_state.pop("backdata_uploader", None)
    st.cache_data.clear()
    st.rerun()

if uploaded is None:
    st.info("좌측에서 backdata 파일(xlsx/csv)을 업로드하세요.")
    st.stop()

try:
    df = load_backdata(uploaded)
except Exception as e:
    st.error(f"❌ 파일 로드 실패: {e}")
    st.stop()

cols = detect_columns(df)
col_scn = cols["scenario"]
col_disp = cols["display"]

if col_scn not in df.columns:
    st.error("❌ '시나리오명' 컬럼을 찾지 못했습니다.")
    st.stop()

if col_disp not in df.columns:
    st.warning("⚠️ '노출 시나리오명' 컬럼이 없어, 시나리오명을 그대로 노출합니다.")
    df[col_disp] = df[col_scn].astype(str)

key_to_disp, disp_to_key, disp_list = scenario_options(df, col_scn, col_disp)

stage_col, drv_col, cat_col, pos_col = cols["stage"], cols["drv"], cols["cat"], cols["pos"]

def uniq_vals(c):
    if c is None or c not in df.columns:
        return []
    return sorted([x for x in df[c].dropna().astype(str).unique().tolist() if str(x).strip() != ""])

st.sidebar.markdown("---")
st.sidebar.markdown("### 시나리오 필터")
f_search = st.sidebar.text_input("검색(노출 시나리오명)", value="", key="f_search")
f_stage = st.sidebar.selectbox("단계(ST)", ["(전체)"] + uniq_vals(stage_col), key="f_stage")
f_cat = st.sidebar.selectbox("카테고리", ["(전체)"] + uniq_vals(cat_col), key="f_cat")
f_pos = st.sidebar.selectbox("가격 포지션(POS)", ["(전체)"] + uniq_vals(pos_col), key="f_pos")
f_drv = st.sidebar.selectbox("드라이버(DRV)", ["(전체)"] + uniq_vals(drv_col), key="f_drv")

apply_internal = cols.get("apply_internal")
apply_client = cols.get("apply_client")
apply_agency = cols.get("apply_agency")

st.sidebar.markdown("### 시나리오 노출 필터(옵션)")
show_internal = st.sidebar.toggle("내부용 적용만", value=False, key="show_internal")
show_client = st.sidebar.toggle("브랜드사용 적용만", value=False, key="show_client")
show_agency = st.sidebar.toggle("대행용 적용만", value=False, key="show_agency")

df_f = df.copy()
if f_stage != "(전체)" and stage_col in df_f.columns:
    df_f = df_f[df_f[stage_col].astype(str) == f_stage]
if f_cat != "(전체)" and cat_col in df_f.columns:
    df_f = df_f[df_f[cat_col].astype(str) == f_cat]
if f_pos != "(전체)" and pos_col in df_f.columns:
    df_f = df_f[df_f[pos_col].astype(str) == f_pos]
if f_drv != "(전체)" and drv_col in df_f.columns:
    df_f = df_f[df_f[drv_col].astype(str) == f_drv]

def _apply_flag_filter(df_, flag_col):
    if flag_col and flag_col in df_.columns:
        return df_[df_[flag_col].astype(str).str.strip().isin(["1","True","TRUE","Y","y","O","o"])]
    return df_

if show_internal:
    df_f = _apply_flag_filter(df_f, apply_internal)
if show_client:
    df_f = _apply_flag_filter(df_f, apply_client)
if show_agency:
    df_f = _apply_flag_filter(df_f, apply_agency)

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
    st.error("❌ 선택한 시나리오를 내부키로 매칭하지 못했습니다.")
    st.stop()

row_df = df[df[col_scn].astype(str).str.strip() == str(scenario_key).strip()]
if row_df.empty:
    st.error("❌ 시나리오 행을 찾지 못했습니다.")
    st.stop()
row = row_df.iloc[0]

rev_cols = cols["rev_cols"]
perf_cols = cols["perf_cols"]
viral_cols = cols["viral_cols"]
brand_cols = cols["brand_cols"]

rev_share = build_rev_shares(row, rev_cols)
media_share = build_media_shares(row, perf_cols, viral_cols, brand_cols)
group_share = media_share["group"]

# ✅ backdata 핵심 파라미터(시나리오 기본값)
col_growth = cols.get("month_growth")
col_repurchase = cols.get("repurchase")
col_ad_contrib = cols.get("ad_contrib")

default_monthly_growth = to_float(row.get(col_growth), 0.0) / 100.0 if col_growth else 0.0
default_repurchase = to_float(row.get(col_repurchase), 0.0) / 100.0 if col_repurchase else 0.0
default_ad_contrib = to_float(row.get(col_ad_contrib), 0.0) / 100.0 if col_ad_contrib else 0.0

# =========================
# Main Tabs
# =========================
tab_guide, tab_agency, tab_brand, tab_rec, tab_custom, tab_plan = st.tabs(
    ["안내", "대행", "브랜드사", "추천엔진", "커스텀 시나리오", "매출 계획"]
)

# =========================
# Tab: Guide
# =========================
with tab_guide:
    st.markdown("## 사용 가이드")
    st.markdown(
        f"""
<div class="card">
<h3>이 시뮬레이터는 무엇을 하나요?</h3>
<hr class="soft"/>
<ul>
  <li><b>시나리오(backdata)</b>를 선택하면, 해당 시나리오의 <b>매출 채널 비중</b>과 <b>미디어 믹스 비중</b>을 불러옵니다.</li>
  <li><b>대행</b> 탭: 광고비↔매출 양방향 산출 + (내부) 인건비/고정비 포함 손익</li>
  <li><b>브랜드사</b> 탭: 월성장률/광고기여율/재구매율을 반영한 월별 매출 전망 + (내부) 필요 광고비/마진/재고 소진</li>
  <li><b>추천엔진</b> 탭: 매출채널 비중 기반으로 Top3 시나리오 추천(간단 점수)</li>
</ul>
<hr class="soft"/>
<h3>Backdata 반영(중요)</h3>
<ul>
  <li><b>월성장률</b>: 월별 총매출을 성장시킵니다.</li>
  <li><b>광고기여율</b>: (비재구매 매출)에 대해 광고가 만든 매출 비중으로 가정하여 필요 광고비를 역산합니다.</li>
  <li><b>재구매율</b>: 총매출 중 재구매 매출 비중을 분리해 (내부)에서 매출 구조를 보여줍니다.</li>
</ul>
</div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Tab: Agency
# =========================
with tab_agency:
    st.markdown("## 대행 모드")
    submode = st.radio("버전 선택", ["외부(클라이언트 제안용)", "내부(운영/정산용)"], horizontal=True, key="agency_sub")

    st.markdown(f"<div class='smallcap'>선택 시나리오: <span class='badge'>{sel_disp}</span></div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 입력 (시뮬레이션)")
    use_scn_kpi = st.toggle("시나리오 KPI 자동 사용(권장)", value=True, key="use_scn_kpi_ag")

    cA, cB, cC, cD = st.columns(4)
    with cA:
        calc_mode = st.radio("계산 방식", ["광고비 입력 → 매출 산출", "매출 입력 → 필요 광고비 산출"], horizontal=True, key="calc_mode_ag")
    with cB:
        aov = st.number_input("객단가(AOV=평균 판매가) (원)", value=50000, step=1000, key="aov_ag")
    with cC:
        cpc_manual = st.number_input("CPC (원) [수동]", value=300.0, step=10.0, key="cpc_ag")
    with cD:
        cvr_manual = st.number_input("CVR (%) [수동]", value=2.0, step=0.1, key="cvr_ag") / 100.0

    scn_cpc, scn_cvr = blended_cpc_cvr(row, perf_cols)
    cpc = scn_cpc if (use_scn_kpi and scn_cpc is not None) else float(cpc_manual)
    cvr = scn_cvr if (use_scn_kpi and scn_cvr is not None) else float(cvr_manual)

    # ✅ backdata 반영 파라미터(시나리오 기본값)
    c1, c2, c3 = st.columns(3)
    with c1:
        ad_contrib = st.number_input("광고기여율(%)", value=float(default_ad_contrib*100), step=1.0, key="ag_ad_contrib") / 100.0
    with c2:
        repurchase = st.number_input("재구매율(%)", value=float(default_repurchase*100), step=1.0, key="ag_repurchase") / 100.0
    with c3:
        st.caption("※ 광고기여율은 ‘비재구매 매출’에 적용(가정)")

    st.caption(
        f"현재 적용 KPI: CPC {fmt_won(cpc)} / CVR {fmt_pct(cvr*100,1)} "
        + (f"(시나리오 KPI 기반)" if use_scn_kpi and scn_cpc is not None else "(수동 입력)")
    )

    # 비용 입력(내부만)
    if submode.startswith("내부"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cost_rate = st.number_input("원가율(%)", value=30.0, step=1.0, key="cr_ag") / 100.0
        with c2:
            logistics = st.number_input("물류비(건당) (원)", value=3000, step=500, key="logi_ag")
        with c3:
            headcount = st.number_input("운영 인력(명)", value=2, step=1, min_value=0, key="hc_ag")
        with c4:
            cost_per = st.number_input("인당 고정비(원)", value=3000000, step=100000, key="cper_ag")
        fixed_cost = float(headcount) * float(cost_per)
    else:
        cost_rate = 0.0
        logistics = 0.0
        fixed_cost = 0.0

    if calc_mode.startswith("광고비"):
        ad_total = st.number_input("총 광고비(원)", value=50000000, step=1000000, key="ad_total_ag")
        rev_target = None
    else:
        rev_target = st.number_input("목표 총매출(원)", value=300000000, step=10000000, key="rev_target_ag")
        ad_total = None

    sim = simulate_pl_with_mix(
        calc_mode=calc_mode,
        aov=aov,
        cpc=cpc,
        cvr=cvr,
        cost_rate=cost_rate,
        logistics_per_order=logistics,
        fixed_cost=fixed_cost,
        ad_spend=ad_total,
        revenue=rev_target,
        repurchase_rate=repurchase,
        ad_contrib=ad_contrib,
    )

    st.divider()
    st.markdown("### 결과 요약")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("예상 총매출", fmt_won(sim["total_revenue"]))
    m2.metric("필요/집행 광고비", fmt_won(sim["ad_spend"]))
    m3.metric("예상 주문수(Units)", f"{sim['orders_total']:,.0f}")
    m4.metric("총 ROAS", f"{sim['roas']:.2f}x ({sim['roas']*100:,.0f}%)")

    # 내부는 손익 추가
    if submode.startswith("내부"):
        st.markdown("### (내부) 손익/매출 구조")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("영업이익", fmt_won(sim["profit"]))
        k2.metric("공헌이익률", f"{sim['contrib_margin']:.1f}%")
        k3.metric("재구매 매출", fmt_won(sim["repeat_revenue"]))
        k4.metric("광고기여 매출", fmt_won(sim["ad_revenue"]))
    else:
        st.markdown("### 매출 구조(참고)")
        k1, k2, k3 = st.columns(3)
        k1.metric("재구매 매출(추정)", fmt_won(sim["repeat_revenue"]))
        k2.metric("비재구매 매출(추정)", fmt_won(sim["nonrepeat_revenue"]))
        k3.metric("광고기여 매출(추정)", fmt_won(sim["ad_revenue"]))

    st.divider()

    cL, cR = st.columns(2)
    with cL:
        fig_rev_tm = treemap_revenue(rev_share, title="매출 채널 구성(트리맵)")
        if fig_rev_tm:
            st.plotly_chart(fig_rev_tm, use_container_width=True, key=f"rev_tm_ag_{scenario_key}")
        else:
            st.info("매출 채널 비중 데이터가 비어있습니다.")
    with cR:
        st.plotly_chart(
            donut_chart(
                ["퍼포먼스", "바이럴", "브랜드"],
                [group_share.get("퍼포먼스", 0), group_share.get("바이럴", 0), group_share.get("브랜드", 0)],
                title="광고비 구조(100%)",
                height=380
            ),
            use_container_width=True,
            key=f"donut_group_ag_{scenario_key}"
        )

    st.divider()
    st.markdown("## 미디어 믹스 (예산/건수 수정 가능)")

    perf_budget = float(sim["ad_spend"]) * float(group_share.get("퍼포먼스", 1.0))
    viral_budget = float(sim["ad_spend"]) * float(group_share.get("바이럴", 0.0))

    with st.expander("바이럴 단가표(편집 가능)", expanded=False):
        st.caption("지면 단가/비율 수정 → 건수/예산에 즉시 반영됩니다.")
        viral_price = st.data_editor(
            DEFAULT_VIRAL_PRICE.copy(),
            num_rows="dynamic",
            use_container_width=True,
            key=f"viral_price_editor_{scenario_key}"
        )

    perf_df = build_performance_mix_table(media_share["perf"], perf_budget)
    medium_share = viral_medium_shares(media_share["viral"])
    viral_df = build_viral_mix_table(viral_price, medium_share, viral_budget)

    st.markdown("### 퍼포먼스(예산 수정 가능)")
    if perf_df.empty:
        st.info("퍼포먼스 믹스 데이터가 비어있습니다(해당 시나리오 비율 0).")
        perf_out = perf_df
    else:
        perf_out = editable_perf_table(perf_df, submode=submode, key_prefix=f"ag_{scenario_key}")

    st.markdown("### 바이럴(건수 수정 가능)")
    if viral_df.empty:
        st.info("바이럴 믹스 데이터가 비어있습니다(해당 시나리오 비율 0).")
        viral_out = viral_df
    else:
        viral_out = editable_viral_table(viral_df, submode=submode, key_prefix=f"ag_{scenario_key}")

    st.divider()
    st.markdown("### 통합 미디어 믹스 표(퍼포먼스/바이럴)")
    mix_df = unify_mix_table(perf_out, viral_out)
    if mix_df.empty:
        st.info("통합 미디어 믹스 데이터가 없습니다.")
    else:
        st.dataframe(mix_df, use_container_width=True, hide_index=True)

    fig_ads_tm = treemap_ads(perf_out, viral_out, title="광고 믹스(트리맵: 퍼포먼스/바이럴 색 구분)")
    if fig_ads_tm:
        st.plotly_chart(fig_ads_tm, use_container_width=True, key=f"ads_tm_ag_{scenario_key}")

# =========================
# Tab: Brand (외부/내부 분리 + 성장률/광고기여율/재구매율 반영)
# =========================
with tab_brand:
    st.markdown("## 브랜드사 모드")
    submode_b = st.radio("버전 선택", ["외부(브랜드사 공유용)", "내부(브랜드 운영/검증용)"], horizontal=True, key="brand_sub")
    st.markdown(f"<div class='smallcap'>선택 시나리오: <span class='badge'>{sel_disp}</span></div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 전망 입력(월별)")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        months = st.selectbox("기간(개월)", options=[3, 6, 12], index=2, key="b_months")
    with c2:
        base_month_rev = st.number_input("월 기준 총매출(원)", value=200000000, step=10000000, key="b_base_rev")
    with c3:
        aov_b = st.number_input("예상 판매가(AOV) (원)", value=50000, step=1000, key="b_aov")
    with c4:
        start_ym = st.text_input("시작월(YYYY-MM)", value=f"{date.today().year:04d}-{date.today().month:02d}", key="b_start_ym")

    c5, c6, c7 = st.columns(3)
    with c5:
        monthly_growth = st.number_input("월 성장률(%)", value=float(default_monthly_growth*100), step=0.5, key="b_growth") / 100.0
    with c6:
        ad_contrib_b = st.number_input("광고기여율(%)", value=float(default_ad_contrib*100), step=1.0, key="b_ad_contrib") / 100.0
    with c7:
        repurchase_b = st.number_input("재구매율(%)", value=float(default_repurchase*100), step=1.0, key="b_repurchase") / 100.0

    # KPI 기반 ROAS(광고기여 매출 기준) 계산: 내부에서 필요 광고비 산출에 사용
    use_scn_kpi_b = st.toggle("시나리오 KPI로 광고 ROAS 산출(권장)", value=True, key="b_use_scn_kpi")
    cpc_manual_b = st.number_input("CPC(원) [수동]", value=300.0, step=10.0, key="b_cpc")
    cvr_manual_b = st.number_input("CVR(%) [수동]", value=2.0, step=0.1, key="b_cvr") / 100.0

    scn_cpc_b, scn_cvr_b = blended_cpc_cvr(row, perf_cols)
    cpc_b = scn_cpc_b if (use_scn_kpi_b and scn_cpc_b is not None) else float(cpc_manual_b)
    cvr_b = scn_cvr_b if (use_scn_kpi_b and scn_cvr_b is not None) else float(cvr_manual_b)
    roas_ad = roas_from_kpi(aov_b, cpc_b, cvr_b)

    st.caption(f"광고 ROAS(가정): {roas_ad:.2f}x  |  CPC {fmt_won(cpc_b)} / CVR {fmt_pct(cvr_b*100,1)}")

    # 월별 forecast
    df_fore = build_monthly_forecast(
        start_ym=start_ym,
        months=int(months),
        base_month_rev=float(base_month_rev),
        monthly_growth=float(monthly_growth),
        aov=float(aov_b),
        repurchase_rate=float(repurchase_b),
        ad_contrib=float(ad_contrib_b),
        roas_ad=float(roas_ad),
    )

    # 외부용: 대략 매출/수량/재고 소진
    # 내부용: + 필요광고비 + 마진/인건비/원가
    st.divider()
    st.markdown("### 예상 매출/수량 요약")

    total_rev_sum = float(df_fore["총매출"].sum())
    total_units_sum = float(df_fore["예상판매수량(Units)"].sum())
    total_need_ad = float(df_fore["필요광고비"].sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("기간 총매출", fmt_won(total_rev_sum))
    k2.metric("기간 총판매수량(발주 기준)", f"{total_units_sum:,.0f} 개")
    if submode_b.startswith("내부"):
        k3.metric("기간 필요 광고비(추정)", fmt_won(total_need_ad))
        k4.metric("광고기여 매출(합)", fmt_won(float(df_fore["광고기여매출"].sum())))
    else:
        k3.metric("광고기여율(가정)", fmt_pct(ad_contrib_b*100,1))
        k4.metric("재구매율(가정)", fmt_pct(repurchase_b*100,1))

    # 재고 소진
    st.markdown("### 예상 재고 소진 일정")
    cA, cB, cC = st.columns(3)
    with cA:
        initial_stock = st.number_input("초기 재고(개)", value=10000, step=100, key="b_stock")
    with cB:
        start_date_input = st.date_input("재고 시작일", value=date.today(), key="b_stock_start")
    with cC:
        st.caption("※ 월별 판매량을 ‘일 평균’으로 쪼개 소진일을 추정합니다.")

    dep_date, dep_trace = calc_depletion_date(start_date_input, int(initial_stock), df_fore)

    if dep_date is None:
        st.success("기간 내 재고가 소진되지 않습니다.")
    else:
        st.warning(f"예상 소진일: **{dep_date.isoformat()}**")

    # 외부는 월별 매출/수량만, 내부는 광고비/분해까지
    if submode_b.startswith("내부"):
        st.markdown("### (내부) 월별 매출 구조/필요 광고비")
        disp = df_fore.copy()
        disp["총매출"] = disp["총매출"].map(lambda x: f"{x:,.0f}")
        disp["광고기여매출"] = disp["광고기여매출"].map(lambda x: f"{x:,.0f}")
        disp["필요광고비"] = disp["필요광고비"].map(lambda x: f"{x:,.0f}")
        disp["재구매매출"] = disp["재구매매출"].map(lambda x: f"{x:,.0f}")
        disp["예상판매수량(Units)"] = disp["예상판매수량(Units)"].map(lambda x: f"{x:,.0f}")
        st.dataframe(disp[["월","총매출","광고기여매출","필요광고비","재구매매출","예상판매수량(Units)"]], use_container_width=True, hide_index=True)

        st.markdown("### (내부) 마진/손익(간단)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            landed_cost = st.number_input("개당 원가(원)", value=20000, step=500, key="b_landed")
        with c2:
            logistics_unit = st.number_input("물류비(건당) (원)", value=3000, step=500, key="b_logi")
        with c3:
            headcount_b = st.number_input("인원(명)", value=2, step=1, min_value=0, key="b_hc")
        with c4:
            cost_per_b = st.number_input("인당 고정비(원)", value=3000000, step=100000, key="b_cper")

        labor = float(headcount_b) * float(cost_per_b)

        revenue_calc = total_units_sum * float(aov_b)
        cogs_calc = total_units_sum * float(landed_cost)
        logistics_calc = total_units_sum * float(logistics_unit)
        profit_calc = revenue_calc - (cogs_calc + logistics_calc + total_need_ad + labor)
        margin_amt = revenue_calc - cogs_calc

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("예상 매출(판매가×수량)", fmt_won(revenue_calc))
        s2.metric("매출총이익(매출-원가)", fmt_won(margin_amt))
        s3.metric("필요 광고비(합)", fmt_won(total_need_ad))
        s4.metric("예상 순이익(간단)", fmt_won(profit_calc))

    else:
        st.markdown("### (외부) 월별 예상 매출/판매수량")
        disp = df_fore.copy()
        disp["총매출"] = disp["총매출"].map(lambda x: f"{x:,.0f}")
        disp["예상판매수량(Units)"] = disp["예상판매수량(Units)"].map(lambda x: f"{x:,.0f}")
        st.dataframe(disp[["월","총매출","예상판매수량(Units)"]], use_container_width=True, hide_index=True)

    # 간단 차트
    st.divider()
    st.markdown("### 월별 추이(차트)")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_fore["월"], y=df_fore["총매출"], name="총매출"))
    if submode_b.startswith("내부"):
        fig.add_trace(go.Bar(x=df_fore["월"], y=df_fore["필요광고비"], name="필요광고비"))
    fig.update_layout(height=420, barmode="group", margin=dict(t=30, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True, key=f"brand_fore_chart_{scenario_key}_{submode_b}")

# =========================================================
# TAB: Recommendation Engine (stable, no df_all)
# =========================================================
with tab_rec:
    st.markdown("## 추천 엔진")
    st.markdown("<div class='smallcap'>매출채널 비중 기반 Top3 시나리오 추천(간단 점수) + 선택 시나리오의 집행 우선순위</div>", unsafe_allow_html=True)
    st.divider()

    focus = st.selectbox("판매 중심 채널(추천 기준)", ["자사몰","온라인(마켓)","홈쇼핑","공구","B2B/도매"], key="rec_focus_simple")

    # top3 scenario by score
    rows = []
    for _, rr in df_f.iterrows():
        rev = build_rev_shares(rr, rev_cols)
        score = focus_match_score(rev, focus)
        rows.append({
            "시나리오": str(rr.get(col_disp, rr.get(col_scn,""))),
            "키": str(rr.get(col_scn,"")),
            "점수": score,
        })
    dfr = pd.DataFrame(rows).sort_values("점수", ascending=False)
    top3 = dfr.head(3).copy()

    c1, c2, c3 = st.columns(3)
    for i, colx in enumerate([c1,c2,c3]):
        if i >= len(top3):
            continue
        r = top3.iloc[i]
        with colx:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"### #{i+1} {r['시나리오']}")
            st.caption(r["키"])
            st.metric("채널 적합 점수", f"{r['점수']:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### (선택 시나리오) 집행 우선순위 Top3")
    rec = strategy_recommendation_from_revshare(rev_share)
    st.markdown(f"**전략 타입:** {rec['title']}")
    st.caption("근거(매출채널 상위): " + (" / ".join(rec["evidence"]) if rec["evidence"] else "-"))
    for i, (ch, why) in enumerate(rec["priority"], 1):
        st.write(f"{i}. **{ch}** — {why}")
    st.info(rec["note"])

# =========================
# Tab: Custom Scenario (그대로)
# =========================
with tab_custom:
    st.markdown("## 커스텀 시나리오")
    st.markdown("<div class='smallcap'>시나리오 자동 분배값 기반으로, 비중/예산을 직접 수정해 결과를 확인합니다.</div>", unsafe_allow_html=True)
    st.divider()

    base = st.selectbox("기준 시나리오(초기값)", options=disp_list, index=disp_list.index(sel_disp) if sel_disp in disp_list else 0, key="custom_base")
    base_key = disp_to_key.get(base)
    base_row = df[df[col_scn].astype(str).str.strip() == str(base_key).strip()].iloc[0] if base_key is not None else row

    base_media = build_media_shares(base_row, perf_cols, viral_cols, brand_cols)
    base_group = base_media["group"]

    st.markdown("### 1) 그룹 분배(퍼포먼스/바이럴/브랜드)")
    gdf = pd.DataFrame([
        {"그룹": "퍼포먼스", "비중(%)": base_group.get("퍼포먼스", 0) * 100},
        {"그룹": "바이럴", "비중(%)": base_group.get("바이럴", 0) * 100},
        {"그룹": "브랜드", "비중(%)": base_group.get("브랜드", 0) * 100},
    ])
    gdf_e = st.data_editor(gdf, use_container_width=True, hide_index=True, key="custom_group")
    group_custom = normalize_shares({r["그룹"]: to_float(r["비중(%)"], 0.0) for _, r in gdf_e.iterrows()})

    st.markdown("### 2) 퍼포먼스 채널 비중")
    pdf = pd.DataFrame([{"매체": k, "비중(%)": v*100} for k, v in base_media["perf"].items() if v > 0])
    if pdf.empty:
        st.info("기준 시나리오에 퍼포먼스 비중이 없습니다.")
        perf_custom = {}
    else:
        pdf_e = st.data_editor(pdf, use_container_width=True, hide_index=True, key="custom_perf_share")
        perf_custom = normalize_shares({r["매체"]: to_float(r["비중(%)"], 0.0) for _, r in pdf_e.iterrows()})

    st.markdown("### 3) 바이럴 비중")
    vdf = pd.DataFrame([{"바이럴 항목": k, "비중(%)": v*100} for k, v in base_media["viral"].items() if v > 0])
    if vdf.empty:
        st.info("기준 시나리오에 바이럴 비중이 없습니다.")
        viral_custom = {}
    else:
        vdf_e = st.data_editor(vdf, use_container_width=True, hide_index=True, key="custom_viral_share")
        viral_custom = normalize_shares({r["바이럴 항목"]: to_float(r["비중(%)"], 0.0) for _, r in vdf_e.iterrows()})

    with st.expander("바이럴 단가표(편집 가능)", expanded=False):
        viral_price_custom = st.data_editor(DEFAULT_VIRAL_PRICE.copy(), num_rows="dynamic", use_container_width=True, key="custom_viral_price")

    st.divider()
    st.markdown("### 커스텀 시뮬레이션 입력")
    cA, cB, cC, cD = st.columns(4)
    with cA:
        calc_mode_c = st.radio("계산 방식", ["광고비 입력 → 매출 산출", "매출 입력 → 필요 광고비 산출"], horizontal=True, key="custom_calc_mode")
    with cB:
        aov_c = st.number_input("객단가(AOV) (원)", value=50000, step=1000, key="custom_aov")
    with cC:
        cpc_c = st.number_input("CPC (원)", value=300.0, step=10.0, key="custom_cpc")
    with cD:
        cvr_c = st.number_input("CVR (%)", value=2.0, step=0.1, key="custom_cvr") / 100.0

    # 커스텀도 backdata 파라미터 입력 가능
    cE, cF = st.columns(2)
    with cE:
        ad_contrib_c = st.number_input("광고기여율(%)", value=50.0, step=1.0, key="custom_ad_contrib") / 100.0
    with cF:
        repurchase_c = st.number_input("재구매율(%)", value=10.0, step=1.0, key="custom_repurchase") / 100.0

    if calc_mode_c.startswith("광고비"):
        ad_total_c = st.number_input("총 광고비(원)", value=50000000, step=1000000, key="custom_ad_total")
        rev_target_c = None
    else:
        rev_target_c = st.number_input("목표 총매출(원)", value=300000000, step=10000000, key="custom_rev_target")
        ad_total_c = None

    sim_c = simulate_pl_with_mix(
        calc_mode=calc_mode_c,
        aov=aov_c,
        cpc=cpc_c,
        cvr=cvr_c,
        cost_rate=0.0,
        logistics_per_order=0.0,
        fixed_cost=0.0,
        ad_spend=ad_total_c,
        revenue=rev_target_c,
        repurchase_rate=repurchase_c,
        ad_contrib=ad_contrib_c,
    )

    st.markdown("### 커스텀 결과")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("예상 총매출", fmt_won(sim_c["total_revenue"]))
    m2.metric("광고비", fmt_won(sim_c["ad_spend"]))
    m3.metric("ROAS", f"{sim_c['roas']:.2f}x ({sim_c['roas']*100:,.0f}%)")
    m4.metric("예상 수량(Units)", f"{sim_c['orders_total']:,.0f}")

    st.divider()
    st.markdown("### 커스텀 미디어 믹스(예산 수정 가능)")

    perf_budget_c = float(sim_c["ad_spend"]) * float(group_custom.get("퍼포먼스", 1.0))
    viral_budget_c = float(sim_c["ad_spend"]) * float(group_custom.get("바이럴", 0.0))

    perf_df_c = build_performance_mix_table(perf_custom, perf_budget_c) if perf_custom else pd.DataFrame()
    viral_df_c = build_viral_mix_table(viral_price_custom, viral_medium_shares(viral_custom), viral_budget_c) if viral_custom else pd.DataFrame()

    st.markdown("#### 퍼포먼스")
    if perf_df_c.empty:
        st.info("커스텀 퍼포먼스 믹스가 없습니다.")
        perf_out_c = perf_df_c
    else:
        perf_out_c = editable_perf_table(perf_df_c, submode="외부", key_prefix="custom")

    st.markdown("#### 바이럴")
    if viral_df_c.empty:
        st.info("커스텀 바이럴 믹스가 없습니다.")
        viral_out_c = viral_df_c
    else:
        viral_out_c = editable_viral_table(viral_df_c, submode="외부", key_prefix="custom")

    fig_ads_tm_c = treemap_ads(perf_out_c, viral_out_c, title="커스텀 광고 믹스(트리맵)")
    if fig_ads_tm_c:
        st.plotly_chart(fig_ads_tm_c, use_container_width=True, key="custom_ads_tm")

# =========================
# Tab: Sales Plan (그대로)
# =========================
with tab_plan:
    st.markdown("## 매출 계획 (브랜드별 1~12월)")
    st.markdown("<div class='smallcap'>브랜드명/전략 입력 또는 템플릿 CSV 업로드 → 월별 계획을 한 번에 보고 편집합니다.</div>", unsafe_allow_html=True)
    st.divider()

    plan_mode = st.radio("데이터 소스", ["직접 입력", "템플릿 CSV 업로드"], horizontal=True, key="plan_mode")

    if plan_mode == "템플릿 CSV 업로드":
        up = st.file_uploader("매출 계획 템플릿 업로드(csv)", type=["csv"], key="plan_uploader")
        if up is None:
            st.info("템플릿 CSV를 업로드하면 브랜드별 월별 매출/광고비 피벗 뷰로 보여줍니다.")
        else:
            plan_raw = pd.read_csv(StringIO(up.getvalue().decode("utf-8-sig", errors="replace")))
            brand_col = safe_col(plan_raw, ["Brand", "브랜드", "brand"])
            month_col = safe_col(plan_raw, ["Month", "월", "month"])
            rev_col = safe_col(plan_raw, ["TotalRevenue", "매출", "Revenue", "totalrevenue"])
            bud_col = safe_col(plan_raw, ["Budget", "광고비", "AdSpend", "budget"])

            if brand_col is None or month_col is None or rev_col is None:
                st.error("템플릿에서 Brand/Month/TotalRevenue(매출) 컬럼을 찾지 못했습니다.")
            else:
                plan_raw[rev_col] = plan_raw[rev_col].apply(lambda x: to_float(x, 0.0))
                if bud_col and bud_col in plan_raw.columns:
                    plan_raw[bud_col] = plan_raw[bud_col].apply(lambda x: to_float(x, 0.0))

                p_rev = plan_raw.pivot_table(index=brand_col, columns=month_col, values=rev_col, aggfunc="sum").fillna(0.0)
                st.markdown("### 브랜드별 월별 매출(편집 가능)")
                st.data_editor(p_rev.reset_index(), use_container_width=True, key="plan_pivot_rev")

                if bud_col and bud_col in plan_raw.columns:
                    st.markdown("### 브랜드별 월별 광고비(편집 가능)")
                    p_bud = plan_raw.pivot_table(index=brand_col, columns=month_col, values=bud_col, aggfunc="sum").fillna(0.0)
                    st.data_editor(p_bud.reset_index(), use_container_width=True, key="plan_pivot_budget")

                totals = p_rev.sum(axis=1).reset_index()
                totals.columns = ["Brand", "TotalRevenue"]
                fig = px.bar(totals, x="Brand", y="TotalRevenue", text="TotalRevenue")
                fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
                fig.update_layout(height=380, margin=dict(t=10), yaxis_title=None, xaxis_title=None, title="브랜드별 연간 매출 합계")
                st.plotly_chart(fig, use_container_width=True, key="plan_bar_total")

    else:
        st.markdown("### 브랜드 입력(여러 개 가능)")
        seed = pd.DataFrame([
            {"Brand": "브랜드A", "전략": "Aggressive", "시작월(YYYY-MM)": "2026-01", "월매출(원)": 200000000, "월광고비(원)": 50000000, "월성장률(%)": 0.0},
        ])
        plan_in = st.data_editor(seed, num_rows="dynamic", use_container_width=True, key="plan_input")

        def month_add(ym: str, idx: int) -> str:
            y, m = ym.split("-")
            y, m = int(y), int(m)
            m2 = m + (idx - 1)
            y2 = y + (m2 - 1)//12
            m2 = (m2 - 1) % 12 + 1
            return f"{y2:04d}-{m2:02d}"

        rows = []
        for _, r in plan_in.iterrows():
            brand = str(r.get("Brand","")).strip()
            strat = str(r.get("전략","")).strip()
            start = str(r.get("시작월(YYYY-MM)","2026-01")).strip()
            base_rev = to_float(r.get("월매출(원)",0.0), 0.0)
            base_ad = to_float(r.get("월광고비(원)",0.0), 0.0)
            gr = to_float(r.get("월성장률(%)",0.0), 0.0) / 100.0
            if not brand:
                continue
            for i in range(1, 13):
                factor = (1.0 + gr) ** (i - 1)
                rows.append({
                    "Brand": brand,
                    "전략": strat,
                    "Month": month_add(start, i),
                    "매출(원)": round_to_100(base_rev * factor),
                    "광고비(원)": round_to_100(base_ad * factor),
                })

        plan_long = pd.DataFrame(rows)
        if plan_long.empty:
            st.info("브랜드를 최소 1개 입력하세요.")
        else:
            st.markdown("### 월별 계획(편집 가능)")
            plan_edit = st.data_editor(plan_long, use_container_width=True, key="plan_long_editor")

            st.markdown("### 브랜드별 월별 매출(피벗)")
            p = plan_edit.pivot_table(index="Brand", columns="Month", values="매출(원)", aggfunc="sum").fillna(0.0)
            st.data_editor(p.reset_index(), use_container_width=True, key="plan_pivot_from_manual")

            totals = p.sum(axis=1).reset_index()
            totals.columns = ["Brand", "TotalRevenue"]
            fig = px.bar(totals, x="Brand", y="TotalRevenue", text="TotalRevenue")
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_layout(height=380, margin=dict(t=10), yaxis_title=None, xaxis_title=None, title="브랜드별 연간 매출 합계")
            st.plotly_chart(fig, use_container_width=True, key="plan_bar_total_manual")
