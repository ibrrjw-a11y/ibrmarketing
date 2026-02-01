# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
from typing import Optional, Dict, List, Tuple

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

# ✅ CSS: f-string braces 문제 없도록(포맷변수 미사용) + 다크/라이트 대응
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
# Column detection (v4 with KPI + 성장/재구매/광고기여)
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

    # ✅ 추가: 성장/재구매/광고기여율
    growth_col = safe_col(df, ["월성장률(%)", "월 성장률(%)", "월성장률", "growth"])
    repeat_col = safe_col(df, ["재구매율(%)", "재구매율", "repeat"])
    ad_contrib_col = safe_col(df, ["광고기여율(%)", "광고기여율", "광고기여도", "ad_contribution"])

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
        "rev_cols": rev_cols,
        "perf_cols": perf_cols,
        "viral_cols": viral_cols,
        "brand_cols": brand_cols,
        "kpi_cols": [c for c in df.columns if str(c).startswith("KPI_")],
        "apply_internal": apply_internal,
        "apply_client": apply_client,
        "apply_agency": apply_agency,
        "growth": growth_col,
        "repeat": repeat_col,
        "ad_contrib": ad_contrib_col,
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

# =========================
# P&L / Simulation (two-way) + 성장/재구매/광고기여율 반영
# =========================
def repeat_multiplier(months: int, repeat_rate: float) -> float:
    """repeat_rate: 0~1, months>=1. 기대 구매횟수 배수(기하급수)"""
    months = max(1, int(months))
    r = float(np.clip(float(repeat_rate or 0.0), 0.0, 0.95))
    if r == 0:
        return 1.0
    # 1 + r + r^2 + ... + r^(months-1)
    return float((1.0 - (r ** months)) / (1.0 - r))

def scale_sim_over_months(sim: Dict[str, float], months: int, growth: float, fixed_is_monthly: bool = True) -> Dict[str, float]:
    """
    sim은 '1개월 기준' 결과라고 보고,
    months기간 동안 growth로 기하성장한다고 가정해 합계로 확장.
    """
    months = max(1, int(months))
    g = float(np.clip(float(growth or 0.0), -0.5, 2.0))
    # 합계 계수: 1 + (1+g) + ... + (1+g)^(m-1)
    if months == 1:
        fsum = 1.0
    else:
        if g == 0:
            fsum = float(months)
        else:
            fsum = float(((1.0 + g) ** months - 1.0) / g)

    out = sim.copy()
    # 규모에 비례하는 항목은 합계 계수로 확장
    for k in ["revenue", "ad_spend", "orders", "clicks", "cogs", "logistics", "organic_revenue"]:
        if k in out:
            out[k] = float(out[k]) * fsum

    # fixed cost는 월 고정비로 가정하면 months 곱
    if "fixed" in out and fixed_is_monthly:
        out["fixed"] = float(out["fixed"]) * float(months)

    # profit 재계산
    out["profit"] = float(out.get("revenue", 0.0)) - (
        float(out.get("ad_spend", 0.0)) +
        float(out.get("cogs", 0.0)) +
        float(out.get("logistics", 0.0)) +
        float(out.get("fixed", 0.0))
    )

    # contrib / roas 재계산
    rev = float(out.get("revenue", 0.0))
    ad = float(out.get("ad_spend", 0.0))
    if rev > 0:
        out["contrib_margin"] = float((rev - ad - float(out.get("logistics", 0.0)) - float(out.get("cogs", 0.0))) / rev * 100.0)
    else:
        out["contrib_margin"] = 0.0
    out["roas"] = float(rev / ad) if ad > 0 else 0.0

    out["assumption_months"] = int(months)
    out["assumption_growth"] = float(g)
    out["assumption_sum_factor"] = float(fsum)
    return out

def simulate_pl(
    calc_mode: str,
    aov: float,
    cpc: float,
    cvr: float,
    cost_rate: float,
    logistics_per_order: float,
    fixed_cost: float,
    ad_spend: Optional[float],
    revenue: Optional[float],
    ad_contrib_rate: float = 1.0,      # 광고기여율(0~1)
    repeat_rate: float = 0.0,          # 재구매율(0~1)
    repeat_months: int = 1,            # 재구매 반영 기간(개월)
):
    """
    핵심 반영:
    - 광고기여율: 총매출 = 광고로 만든 매출 / 광고기여율
    - 재구매율: 광고로 만든 첫구매가 기간 동안 repeat_multiplier만큼 누적 구매를 만든다고 가정
    """
    aov = float(aov or 0.0)
    cpc = float(cpc or 0.0)
    cvr = float(cvr or 0.0)

    ad_contrib_rate = float(np.clip(float(ad_contrib_rate or 1.0), 0.05, 1.0))
    rep_mult = repeat_multiplier(repeat_months, repeat_rate)

    if calc_mode.startswith("매출"):
        # 목표 총매출 -> 광고가 만들어야 하는 매출만 역산
        revenue = float(revenue or 0.0)
        paid_revenue = revenue * ad_contrib_rate

        # paid_revenue는 (광고 첫구매 * 재구매배수)로 만들어진다고 가정
        paid_orders_total = (paid_revenue / aov) if aov > 0 else 0.0
        paid_first_orders = (paid_orders_total / rep_mult) if rep_mult > 0 else 0.0

        clicks = (paid_first_orders / cvr) if cvr > 0 else 0.0
        ad_spend = clicks * cpc

        # 총 주문/클릭은 "총매출" 기준
        orders = (revenue / aov) if aov > 0 else 0.0

    else:
        # 광고비 -> 광고로 만든 매출 -> 총매출로 확장
        ad_spend = float(ad_spend or 0.0)
        clicks = (ad_spend / cpc) if cpc > 0 else 0.0

        paid_first_orders = clicks * cvr
        paid_orders_total = paid_first_orders * rep_mult
        paid_revenue = paid_orders_total * aov

        revenue = paid_revenue / ad_contrib_rate if ad_contrib_rate > 0 else paid_revenue
        orders = (revenue / aov) if aov > 0 else 0.0

    cogs = revenue * float(cost_rate)
    logistics = orders * float(logistics_per_order)
    profit = revenue - (ad_spend + cogs + logistics + float(fixed_cost))
    contrib_margin = ((revenue - ad_spend - logistics - cogs) / revenue * 100) if revenue > 0 else 0.0
    roas = (revenue / ad_spend) if ad_spend and ad_spend > 0 else 0.0

    # 참고용: paid/organic 분해
    organic_revenue = revenue * (1.0 - ad_contrib_rate)

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
        "roas": float(roas),

        # 설명용 가정
        "assumption_ad_contrib": float(ad_contrib_rate),
        "assumption_repeat_rate": float(np.clip(float(repeat_rate or 0.0), 0.0, 0.95)),
        "assumption_repeat_months": int(max(1, repeat_months)),
        "assumption_repeat_mult": float(rep_mult),
        "organic_revenue": float(organic_revenue),
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
    if "스마트" in s or "스토어" in s:
        return "스마트스토어"
    if "쿠팡" in s:
        return "쿠팡"
    if any(k in s for k in ["오프라인", "면세", "리테일", "백화점", "마트", "드럭", "올리브영"]):
        return "오프라인"
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
# Compare chart (bars + ROAS line / secondary axis)
# =========================
def compare_chart(df_cmp: pd.DataFrame, x_col: str, rev_col: str, ad_col: str, roas_col: str, height=420, title=""):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_cmp[x_col], y=df_cmp[rev_col], name="예상매출", yaxis="y1",
        hovertemplate="%{y:,.0f}원<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=df_cmp[x_col], y=df_cmp[ad_col], name="예상광고비", yaxis="y1",
        hovertemplate="%{y:,.0f}원<extra></extra>"
    ))

    roas = df_cmp[roas_col].astype(float).fillna(0.0).clip(lower=0)
    fig.add_trace(go.Scatter(
        x=df_cmp[x_col], y=roas, name="ROAS", yaxis="y2",
        mode="lines+markers",
        hovertemplate="ROAS %{y:.2f}x (%{customdata:.0f}%)<extra></extra>",
        customdata=(roas * 100.0)
    ))

    y2_min, y2_max = 1.0, 10.0
    if roas.max() > y2_max:
        y2_max = float(np.ceil(roas.max()))
    if roas.min() < y2_min and roas.min() > 0:
        y2_min = float(max(0.5, np.floor(roas.min()*2)/2))

    tickvals = list(np.linspace(y2_min, y2_max, 5))
    ticktext = [f"{v*100:.0f}%" for v in tickvals]

    fig.update_layout(
        height=height,
        barmode="group",
        title=title,
        margin=dict(t=50, b=10, l=10, r=10),
        yaxis=dict(title=None, tickformat=",.0f"),
        yaxis2=dict(
            title="ROAS(%)",
            overlaying="y",
            side="right",
            range=[y2_min, y2_max],
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
        ),
        xaxis=dict(tickangle=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

# =========================
# Recommendation (rule-based) + 성장/재구매/광고기여 반영
# =========================
def top_key(d: Dict[str, float]) -> Tuple[Optional[str], float]:
    if not d:
        return None, 0.0
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return (items[0][0], float(items[0][1])) if items else (None, 0.0)

def detect_sales_archetype(rev_share: Dict[str, float], sales_focus: str = "(무관)") -> str:
    if sales_focus and sales_focus != "(무관)":
        if sales_focus in ["자사몰", "온라인(마켓)", "홈쇼핑", "공구", "B2B/도매"]:
            return sales_focus

    k, _ = top_key(rev_share)
    if not k:
        return "기타"

    k = str(k)
    if "자사" in k:
        return "자사몰"
    if "스마트" in k or "스토어" in k:
        return "온라인(마켓)"
    if "쿠팡" in k:
        return "온라인(마켓)"
    if "홈쇼핑" in k:
        return "홈쇼핑"
    if "공구" in k or "공동" in k:
        return "공구"
    if "B2B" in k or "도매" in k:
        return "B2B/도매"
    return "기타"

def strategy_recommendation(rev_share: Dict[str, float], sales_focus: str = "(무관)") -> Dict[str, object]:
    def share_contains(keyword: str) -> float:
        s = 0.0
        for k, v in rev_share.items():
            if keyword in str(k):
                s += float(v)
        return s

    own = share_contains("자사")
    smart = share_contains("스마트") + share_contains("스토어")
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

# ✅ 시나리오 성장/재구매/광고기여율 값 로드 (없으면 default)
growth_col = cols.get("growth")
repeat_col = cols.get("repeat")
ad_contrib_col = cols.get("ad_contrib")

scn_growth = to_float(row.get(growth_col), 0.0) / 100.0 if (growth_col and growth_col in df.columns) else 0.0
scn_repeat = to_float(row.get(repeat_col), 0.0) / 100.0 if (repeat_col and repeat_col in df.columns) else 0.0
scn_ad_contrib = to_float(row.get(ad_contrib_col), 100.0) / 100.0 if (ad_contrib_col and ad_contrib_col in df.columns) else 1.0

# 안전 클램프
scn_growth = float(np.clip(scn_growth, -0.5, 2.0))
scn_repeat = float(np.clip(scn_repeat, 0.0, 0.95))
scn_ad_contrib = float(np.clip(scn_ad_contrib, 0.05, 1.0))

st.sidebar.markdown("---")
st.sidebar.markdown("### 시나리오 가정(Backdata)")
st.sidebar.caption(f"월 성장률: {scn_growth*100:.1f}%")
st.sidebar.caption(f"재구매율: {scn_repeat*100:.1f}%")
st.sidebar.caption(f"광고기여율: {scn_ad_contrib*100:.0f}%")

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
        """
<div class="card">
<h3>이 시뮬레이터는 무엇을 하나요?</h3>
<hr class="soft"/>
<ul>
  <li><b>시나리오(backdata)</b>를 선택하면, 해당 시나리오의 <b>매출 채널 비중</b>과 <b>미디어 믹스 비중</b>을 불러옵니다.</li>
  <li><b>대행</b> 탭에서는 입력값(AOV/CPC/CVR 등) 기반으로 <b>매출↔광고비를 양방향</b>으로 산출합니다.</li>
  <li><b>광고기여율/재구매율</b>을 반영해, <b>광고가 만든 매출 → 총매출(유기 포함)</b>로 확장합니다.</li>
  <li><b>월 성장률</b>은 기간(개월) 설정 시 <b>기간 합계</b>에 반영됩니다.</li>
  <li><b>미디어 믹스</b>는 시나리오 비중으로 자동 분배되며, <b>예산/건수는 사용자가 직접 수정</b>할 수 있습니다.</li>
</ul>
<hr class="soft"/>
<h3>계산식 핵심</h3>
<ul>
  <li>Clicks = 광고비/CPC → (첫구매)Orders = Clicks×CVR</li>
  <li>재구매 반영: 광고 첫구매가 기간 동안 누적 구매를 만든다고 가정(배수 적용)</li>
  <li>광고기여율: 총매출 = 광고가 만든 매출 / 광고기여율</li>
  <li>기간 성장률: 1~N개월 합계 = 기하급수 합계(성장률 기반)</li>
</ul>
<div class="smallcap">※ 입력 기반 시뮬레이션이며 실제 성과는 운영/상품/시즌 요인에 따라 달라질 수 있습니다.</div>
</div>
        """,
        unsafe_allow_html=True
    )

# =========================
# Shared editors (budget overrides)
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
# Tab: Agency
# =========================
with tab_agency:
    st.markdown("## 대행 모드")
    submode = st.radio("버전 선택", ["외부(클라이언트 제안용)", "내부(운영/정산용)"], horizontal=True, key="agency_sub")

    st.markdown(f"<div class='smallcap'>선택 시나리오: <span class='badge'>{sel_disp}</span></div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 입력 (시뮬레이션)")
    use_scn_kpi = st.toggle("시나리오 KPI 자동 사용(권장)", value=True, key="use_scn_kpi_ag")

    # ✅ 성장/재구매/광고기여율 적용
    use_scn_factors = st.toggle("시나리오 성장/재구매/광고기여율 적용(권장)", value=True, key="use_scn_factors_ag")

    cFa, cFb, cFc, cFd = st.columns(4)
    with cFa:
        months_horizon = st.selectbox("기간(개월)", options=[1, 3, 6, 12], index=0, key="ag_months_horizon")
    with cFb:
        growth_in = st.number_input("월 성장률(%)", value=float(scn_growth*100.0), step=0.5, key="ag_growth_pct") / 100.0
    with cFc:
        repeat_months = st.slider("재구매 반영 기간(개월)", 1, 12, int(months_horizon), key="ag_repeat_months")
    with cFd:
        ad_contrib_in = st.number_input("광고기여율(%)", value=float(scn_ad_contrib*100.0), step=1.0, key="ag_ad_contrib_pct") / 100.0

    repeat_in = st.number_input("재구매율(%)", value=float(scn_repeat*100.0), step=1.0, key="ag_repeat_pct") / 100.0

    cA, cB, cC, cD = st.columns(4)
    with cA:
        calc_mode = st.radio("계산 방식", ["광고비 입력 → 매출 산출", "매출 입력 → 필요 광고비 산출"], horizontal=True, key="calc_mode_ag")
    with cB:
        aov = st.number_input("객단가(AOV) (원)", value=50000, step=1000, key="aov_ag")
    with cC:
        cpc_manual = st.number_input("CPC (원) [수동]", value=300.0, step=10.0, key="cpc_ag")
    with cD:
        cvr_manual = st.number_input("CVR (%) [수동]", value=2.0, step=0.1, key="cvr_ag") / 100.0

    scn_cpc, scn_cvr = blended_cpc_cvr(row, perf_cols)
    cpc = scn_cpc if (use_scn_kpi and scn_cpc is not None) else float(cpc_manual)
    cvr = scn_cvr if (use_scn_kpi and scn_cvr is not None) else float(cvr_manual)

    st.caption(
        f"현재 적용 KPI: CPC {fmt_won(cpc)} / CVR {fmt_pct(cvr*100,1)} "
        + (f"(시나리오 KPI 기반)" if use_scn_kpi and scn_cpc is not None else "(수동 입력)")
    )

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

    if calc_mode.startswith("광고비"):
        ad_total = st.number_input("월 광고비(원)", value=50000000, step=1000000, key="ad_total_ag")
        rev_target = None
    else:
        rev_target = st.number_input("월 목표 매출(원)", value=300000000, step=10000000, key="rev_target_ag")
        ad_total = None

    ad_contrib_use = float(scn_ad_contrib) if use_scn_factors else float(ad_contrib_in)
    repeat_use = float(scn_repeat) if use_scn_factors else float(repeat_in)

    sim_1m = simulate_pl(
        calc_mode=calc_mode,
        aov=aov,
        cpc=cpc,
        cvr=cvr,
        cost_rate=cost_rate,
        logistics_per_order=logistics,
        fixed_cost=fixed_cost,
        ad_spend=ad_total,
        revenue=rev_target,
        ad_contrib_rate=ad_contrib_use,
        repeat_rate=repeat_use,
        repeat_months=int(repeat_months),
    )

    # ✅ 기간 확장(성장률 반영)
    sim = scale_sim_over_months(sim_1m, months=int(months_horizon), growth=float(growth_in), fixed_is_monthly=True)

    st.divider()
    st.markdown("### 결과 요약")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("기간 예상 매출", fmt_won(sim["revenue"]))
    m2.metric("기간 예상 광고비", fmt_won(sim["ad_spend"]))
    m3.metric("기간 영업이익", fmt_won(sim["profit"]))
    m4.metric("ROAS", f"{sim['roas']:.2f}x ({sim['roas']*100:,.0f}%)")

    st.caption(
        f"적용값: 성장률 {sim['assumption_growth']*100:.1f}% / "
        f"광고기여율 {sim_1m['assumption_ad_contrib']*100:.0f}% / "
        f"재구매율 {sim_1m['assumption_repeat_rate']*100:.0f}% "
        f"(재구매 {sim_1m['assumption_repeat_months']}개월, 배수×{sim_1m['assumption_repeat_mult']:.2f}) / "
        f"추정 유기매출(기간) {fmt_won(sim.get('organic_revenue', 0.0))}"
    )

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

    # ✅ 미디어믹스는 "기간 광고비" 기준으로 분배
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
# Tab: Brand
# =========================
with tab_brand:
    st.markdown("## 브랜드사 모드")
    submode_b = st.radio("버전 선택", ["외부(브랜드사 공유용)", "내부(브랜드 운영/검증용)"], horizontal=True, key="brand_sub")
    st.markdown(f"<div class='smallcap'>선택 시나리오: <span class='badge'>{sel_disp}</span></div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 월별 매출/광고비 전망")

    # ✅ 광고비 기반 매출 자동 추정(시나리오 KPI + 광고기여/재구매 반영)
    use_ad_based_rev = st.toggle(
        "광고비 기반 월매출 자동 추정(시나리오 KPI + 광고기여율/재구매율 반영)",
        value=True,
        key="brand_use_ad_based_rev",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        months = st.selectbox("기간(개월)", options=[3, 6, 12], index=2, key="b_months")
    with c2:
        base_month_ad = st.number_input("월 기준 광고비(원)", value=50000000, step=1000000, key="b_base_ad")
    with c3:
        growth = st.number_input("월 성장률(%)", value=float(scn_growth*100.0), step=0.5, key="b_growth") / 100.0
    with c4:
        repeat_months_b = st.slider("재구매 반영(개월)", 1, 12, int(months), key="b_repeat_months")

    # KPI 기반 추정용 입력
    c5, c6, c7 = st.columns(3)
    with c5:
        aov_b = st.number_input("객단가(AOV) (원)", value=50000, step=1000, key="b_aov")
    with c6:
        cpc_b = st.number_input("CPC(원)", value=float((blended_cpc_cvr(row, perf_cols)[0] or 300.0)), step=10.0, key="b_cpc")
    with c7:
        cvr_b = st.number_input("CVR(%)", value=float((blended_cpc_cvr(row, perf_cols)[1] or 0.02) * 100.0), step=0.1, key="b_cvr") / 100.0

    est_1m = simulate_pl(
        calc_mode="광고비 입력 → 매출 산출",
        aov=aov_b,
        cpc=cpc_b,
        cvr=cvr_b,
        cost_rate=0.0,
        logistics_per_order=0.0,
        fixed_cost=0.0,
        ad_spend=float(base_month_ad),
        revenue=None,
        ad_contrib_rate=float(scn_ad_contrib),
        repeat_rate=float(scn_repeat),
        repeat_months=int(repeat_months_b),
    )
    base_month_rev_est = round_to_100(est_1m["revenue"])

    base_month_rev_input = st.number_input("월 기준 매출(원) [수동]", value=int(base_month_rev_est), step=10000000, key="b_base_rev")
    base_month_rev = float(base_month_rev_est) if use_ad_based_rev else float(base_month_rev_input)

    months_idx = list(range(1, int(months) + 1))
    rev_list, ad_list, roas_list = [], [], []
    for i in months_idx:
        factor = (1.0 + float(growth)) ** (i - 1)
        rev_i = base_month_rev * factor
        ad_i = float(base_month_ad) * factor
        rev_list.append(rev_i)
        ad_list.append(ad_i)
        roas_list.append((rev_i / ad_i) if ad_i > 0 else 0.0)

    df_m = pd.DataFrame({"월": [f"M{i}" for i in months_idx], "예상매출": rev_list, "예상광고비": ad_list, "ROAS": roas_list})

    k1, k2, k3 = st.columns(3)
    k1.metric("기간 총매출", fmt_won(df_m["예상매출"].sum()))
    k2.metric("기간 총광고비", fmt_won(df_m["예상광고비"].sum()))
    k3.metric("평균 ROAS", f"{df_m['ROAS'].mean():.2f}x ({df_m['ROAS'].mean()*100:,.0f}%)")

    st.caption(
        f"월매출 산정: {'광고비 기반(자동)' if use_ad_based_rev else '수동 입력'} / "
        f"광고기여율 {scn_ad_contrib*100:.0f}% / 재구매율 {scn_repeat*100:.0f}% (재구매 {repeat_months_b}개월) / "
        f"월 성장률 {growth*100:.1f}%"
    )

    st.plotly_chart(
        compare_chart(df_m, "월", "예상매출", "예상광고비", "ROAS", title="월별 매출/광고비 + ROAS(보조축)"),
        use_container_width=True,
        key=f"brand_month_chart_{scenario_key}"
    )

    st.divider()
    st.markdown("### 매출 채널 구성(트리맵)")
    fig_rev_tm2 = treemap_revenue(rev_share, title="매출 채널 구성(트리맵)")
    if fig_rev_tm2:
        st.plotly_chart(fig_rev_tm2, use_container_width=True, key=f"rev_tm_brand_{scenario_key}")

# =========================================================
# Tab: Recommendation Engine (Top3) + 성장/재구매/광고기여 반영
# =========================================================
with tab_rec:
    st.markdown("## 추천 엔진")
    st.markdown("<div class='smallcap'>채널 룰 + 성장/재구매/광고기여율을 함께 반영해 Top3를 추천합니다.</div>", unsafe_allow_html=True)
    st.divider()

    def top3_text(d: Dict[str, float]) -> str:
        items = sorted(d.items(), key=lambda x: x[1], reverse=True)[:3]
        items = [(k, v) for k, v in items if v > 0]
        return ", ".join([f"{k} {v*100:.0f}%" for k, v in items]) if items else "-"

    def get_factor_vals(r: pd.Series) -> Tuple[float, float, float]:
        g = to_float(r.get(growth_col), 0.0) / 100.0 if (growth_col and growth_col in df.columns) else 0.0
        rp = to_float(r.get(repeat_col), 0.0) / 100.0 if (repeat_col and repeat_col in df.columns) else 0.0
        ac = to_float(r.get(ad_contrib_col), 100.0) / 100.0 if (ad_contrib_col and ad_contrib_col in df.columns) else 1.0
        return float(np.clip(g, -0.5, 2.0)), float(np.clip(rp, 0.0, 0.95)), float(np.clip(ac, 0.05, 1.0))

    def score_row_sales_rule(rs: Dict[str, float], ms: Dict[str, object], focus: str) -> Tuple[float, List[str]]:
        # focus별 핵심 매출 채널 점수(0~1)
        def share_contains(keyword: str) -> float:
            s = 0.0
            for k, v in rs.items():
                if keyword in str(k):
                    s += float(v)
            return s

        own = share_contains("자사")
        smart = share_contains("스마트") + share_contains("스토어")
        coupang = share_contains("쿠팡")
        home = share_contains("홈쇼핑")
        groupbuy = share_contains("공구") + share_contains("공동")

        perf = ms.get("perf", {})
        gshare = ms.get("group", {})

        meta_share = sum(v for k, v in perf.items() if "메타" in str(k))
        naver_sa_share = sum(v for k, v in perf.items() if ("네이버" in str(k) and "SA" in str(k)))
        ext_pa_share = sum(v for k, v in perf.items() if ("외부몰PA" in str(k) or "쿠팡" in str(k)))

        why = []
        score = 0.0

        if focus == "자사몰 중심":
            score = own * 0.65 + meta_share * 0.35
            why += [f"자사몰 매출비중 {own*100:.0f}%", f"메타(퍼포) 비중 {meta_share*100:.0f}%"]
        elif focus == "스마트스토어 중심":
            score = smart * 0.65 + naver_sa_share * 0.35
            why += [f"스마트스토어 매출비중 {smart*100:.0f}%", f"네이버SA(퍼포) 비중 {naver_sa_share*100:.0f}%"]
        elif focus == "쿠팡 중심":
            score = coupang * 0.70 + ext_pa_share * 0.30
            why += [f"쿠팡 매출비중 {coupang*100:.0f}%", f"외부몰PA/쿠팡(퍼포) 비중 {ext_pa_share*100:.0f}%"]
        elif focus == "홈쇼핑 중심":
            score = home * 0.75 + (1.0 - meta_share) * 0.25
            why += [f"홈쇼핑 매출비중 {home*100:.0f}%", f"메타 비중(낮을수록 적합) {meta_share*100:.0f}%"]
        elif focus == "공구 중심":
            score = groupbuy * 0.70 + float(gshare.get("바이럴", 0.0)) * 0.30
            why += [f"공구 매출비중 {groupbuy*100:.0f}%", f"바이럴 그룹 비중 {float(gshare.get('바이럴',0))*100:.0f}%"]
        else:
            score = 0.5
            why += ["포커스 미지정(기본값)"]

        return float(np.clip(score, 0.0, 1.0)), why

    def effective_cac_from_row(r: pd.Series, months: int) -> Optional[float]:
        cpc, cvr = blended_cpc_cvr(r, perf_cols)
        if cpc is None or cvr is None or cpc <= 0 or cvr <= 0:
            return None
        g, rp, ac = get_factor_vals(r)
        rep_mult = repeat_multiplier(int(months), rp)
        base_cac_first = float(cpc) / float(cvr)  # 첫구매 기준 CAC
        # 총 주문(유기 포함) 기준 유효 CAC ≈ base * ad_contrib / rep_mult
        eff = base_cac_first * float(ac) / float(rep_mult if rep_mult > 0 else 1.0)
        return float(eff)

    with st.expander("입력 조건", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            candidate_scope = st.radio(
                "후보 범위",
                ["현재 사이드바 필터 반영(df_f)", "전체 backdata(df)"],
                horizontal=True,
                key="rec_scope_v3",
            )
            focus = st.selectbox(
                "판매 중심 기준",
                ["자동(매출비중 기반)", "자사몰 중심", "스마트스토어 중심", "쿠팡 중심", "홈쇼핑 중심", "공구 중심"],
                key="rec_focus_v3",
            )

        with c2:
            rec_stage = st.selectbox("단계(ST) 필터", ["(전체)"] + uniq_vals(stage_col), key="rec_stage_v3")
            rec_cat = st.selectbox("카테고리 필터", ["(전체)"] + uniq_vals(cat_col), key="rec_cat_v3")

        with c3:
            rec_pos = st.selectbox("가격 포지션(POS) 필터", ["(전체)"] + uniq_vals(pos_col), key="rec_pos_v3")
            rec_drv = st.selectbox("드라이버(DRV) 필터", ["(전체)"] + uniq_vals(drv_col), key="rec_drv_v3")

        c4, c5 = st.columns(2)
        with c4:
            horizon_months = st.selectbox("평가 기간(개월)", options=[1, 3, 6, 12], index=1, key="rec_horizon_months")
        with c5:
            run_rec = st.button("Top3 추천 계산", use_container_width=True, key="rec_run_v3")

    if not run_rec:
        st.info("조건을 선택하고 **Top3 추천 계산**을 누르세요.")
    else:
        base_df = df_f.copy() if candidate_scope.startswith("현재") else df.copy()

        # 탭 내부 필터 적용
        if rec_stage != "(전체)" and stage_col and stage_col in base_df.columns:
            base_df = base_df[base_df[stage_col].astype(str) == rec_stage]
        if rec_cat != "(전체)" and cat_col and cat_col in base_df.columns:
            base_df = base_df[base_df[cat_col].astype(str) == rec_cat]
        if rec_pos != "(전체)" and pos_col and pos_col in base_df.columns:
            base_df = base_df[base_df[pos_col].astype(str) == rec_pos]
        if rec_drv != "(전체)" and drv_col and drv_col in base_df.columns:
            base_df = base_df[base_df[drv_col].astype(str) == rec_drv]

        if base_df.empty:
            st.warning("후보가 0개입니다. 필터를 완화하세요.")
        else:
            results = []

            for _, r in base_df.iterrows():
                rs = build_rev_shares(r, rev_cols)
                ms = build_media_shares(r, perf_cols, viral_cols, brand_cols)

                # 자동이면 archetype 기반으로 focus 결정
                if focus == "자동(매출비중 기반)":
                    arche = detect_sales_archetype(rs, "(무관)")
                    if arche == "자사몰":
                        f_use = "자사몰 중심"
                    elif arche == "온라인(마켓)":
                        coup = sum(v for k, v in rs.items() if "쿠팡" in str(k))
                        f_use = "쿠팡 중심" if coup > 0.15 else "스마트스토어 중심"
                    elif arche == "홈쇼핑":
                        f_use = "홈쇼핑 중심"
                    elif arche == "공구":
                        f_use = "공구 중심"
                    else:
                        f_use = "자사몰 중심"
                else:
                    f_use = focus

                base_rule_score01, why_rule = score_row_sales_rule(rs, ms, f_use)

                # ✅ 성장/재구매/광고기여 가점(0~1로 정규화 후 가중)
                g, rp, ac = get_factor_vals(r)
                # 성장률: -10%~+20%를 0~1로 (그 이상/이하는 클램프)
                g_norm = float(np.clip((g - (-0.10)) / (0.20 - (-0.10)), 0.0, 1.0))
                # 재구매율: 0~40%를 0~1로
                rp_norm = float(np.clip(rp / 0.40, 0.0, 1.0))
                # 광고기여율: 낮을수록(유기↑) 좋게. 90%~40%를 0~1 (40%가 최고)
                ac_norm = float(np.clip((0.90 - ac) / (0.90 - 0.40), 0.0, 1.0))

                # 가중치
                w_rule = 0.70
                w_g = 0.10
                w_rp = 0.12
                w_ac = 0.08

                total_score01 = (
                    base_rule_score01 * w_rule +
                    g_norm * w_g +
                    rp_norm * w_rp +
                    ac_norm * w_ac
                )
                total_score = float(total_score01 * 100.0)

                skey = str(r[col_scn]).strip()
                disp = str(r[col_disp]).strip() if col_disp in base_df.columns else skey

                eff_cac = effective_cac_from_row(r, months=int(horizon_months))

                results.append({
                    "scenario_key": skey,
                    "scenario_label": disp,
                    "score": total_score,
                    "why": [
                        f"룰 점수(채널/미디어): {base_rule_score01*100:.0f}점 ({f_use})",
                        f"월 성장률 {g*100:.1f}% / 재구매율 {rp*100:.0f}% / 광고기여율 {ac*100:.0f}%",
                        f"매출 Top3: {top3_text(rs)}",
                    ] + why_rule[:2],
                    "eff_cac": eff_cac,
                    "group": ms.get("group", {}),
                })

            results.sort(key=lambda x: x["score"], reverse=True)
            top3 = results[:3]

            k1, k2, k3 = st.columns(3)
            k1.metric("후보 수", f"{len(results):,} 개")
            k2.metric("추천 결과", f"{len(top3):,} 개")
            k3.metric("평가 기간", f"{int(horizon_months)} 개월")

            cols3 = st.columns(3)
            for i, item in enumerate(top3):
                with cols3[i]:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"### #{i+1} {item['scenario_label']}")
                    st.caption(item["scenario_key"])

                    m1, m2 = st.columns(2)
                    m1.metric("Score", f"{item['score']:.1f}")
                    m2.metric("유효 CAC(추정)", fmt_won(item["eff_cac"]) if item["eff_cac"] is not None else "-")

                    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
                    st.write("**요약 근거**")
                    for line in item["why"][:4]:
                        st.write(f"- {line}")

                    g0 = item.get("group", {})
                    st.write(f"- 광고 그룹: 퍼포 {float(g0.get('퍼포먼스',0))*100:.0f}% / 바이럴 {float(g0.get('바이럴',0))*100:.0f}% / 브랜드 {float(g0.get('브랜드',0))*100:.0f}%")
                    st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("전체 후보 점수표(Top10)", expanded=False):
                view = pd.DataFrame([{
                    "전략": x["scenario_label"],
                    "키": x["scenario_key"],
                    "점수": round(x["score"], 1),
                    "유효CAC": x["eff_cac"],
                } for x in results[:10]])
                if not view.empty:
                    view["유효CAC"] = view["유효CAC"].apply(lambda v: fmt_won(v) if v is not None else "-")
                    st.dataframe(view, use_container_width=True, hide_index=True)

# =========================
# Tab: Custom Scenario
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
    st.markdown("### 커스텀 시뮬레이션 입력 (성장/재구매/광고기여 반영)")

    cX1, cX2, cX3, cX4 = st.columns(4)
    with cX1:
        months_c = st.selectbox("기간(개월)", options=[1, 3, 6, 12], index=0, key="custom_months")
    with cX2:
        growth_c = st.number_input("월 성장률(%)", value=float(scn_growth*100.0), step=0.5, key="custom_growth") / 100.0
    with cX3:
        repeat_months_c = st.slider("재구매 반영(개월)", 1, 12, int(months_c), key="custom_repeat_months")
    with cX4:
        ad_contrib_c = st.number_input("광고기여율(%)", value=float(scn_ad_contrib*100.0), step=1.0, key="custom_ad_contrib") / 100.0

    repeat_c = st.number_input("재구매율(%)", value=float(scn_repeat*100.0), step=1.0, key="custom_repeat") / 100.0

    cA, cB, cC, cD = st.columns(4)
    with cA:
        calc_mode_c = st.radio("계산 방식", ["광고비 입력 → 매출 산출", "매출 입력 → 필요 광고비 산출"], horizontal=True, key="custom_calc_mode")
    with cB:
        aov_c = st.number_input("객단가(AOV) (원)", value=50000, step=1000, key="custom_aov")
    with cC:
        cpc_c = st.number_input("CPC (원)", value=300.0, step=10.0, key="custom_cpc")
    with cD:
        cvr_c = st.number_input("CVR (%)", value=2.0, step=0.1, key="custom_cvr") / 100.0

    if calc_mode_c.startswith("광고비"):
        ad_total_c = st.number_input("월 광고비(원)", value=50000000, step=1000000, key="custom_ad_total")
        rev_target_c = None
    else:
        rev_target_c = st.number_input("월 목표 매출(원)", value=300000000, step=10000000, key="custom_rev_target")
        ad_total_c = None

    sim_c_1m = simulate_pl(
        calc_mode_c, aov_c, cpc_c, cvr_c, 0.0, 0.0, 0.0,
        ad_total_c, rev_target_c,
        ad_contrib_rate=float(ad_contrib_c),
        repeat_rate=float(repeat_c),
        repeat_months=int(repeat_months_c),
    )
    sim_c = scale_sim_over_months(sim_c_1m, months=int(months_c), growth=float(growth_c), fixed_is_monthly=False)

    st.markdown("### 커스텀 결과")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("기간 예상 매출", fmt_won(sim_c["revenue"]))
    m2.metric("기간 예상 광고비", fmt_won(sim_c["ad_spend"]))
    m3.metric("ROAS", f"{sim_c['roas']:.2f}x ({sim_c['roas']*100:,.0f}%)")
    m4.metric("추정 유기매출", fmt_won(sim_c.get("organic_revenue", 0.0)))

    st.caption(
        f"적용값: 성장률 {growth_c*100:.1f}% / 광고기여율 {float(ad_contrib_c)*100:.0f}% / "
        f"재구매율 {float(repeat_c)*100:.0f}% (재구매 {repeat_months_c}개월, 배수×{sim_c_1m['assumption_repeat_mult']:.2f})"
    )

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
# Tab: Sales Plan (NEW)
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
