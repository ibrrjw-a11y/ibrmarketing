import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import re
from typing import Dict, Optional, Tuple, List

# ============================================================
# Page / Theme
# ============================================================
st.set_page_config(page_title="마케팅/유통 시뮬레이터", layout="wide")

ACCENT = "#2F6FED"
MUTED = "#6c757d"
BG = "#f8f9fa"

st.markdown(
    f"""
<style>
html, body, [class*="css"] {{
  font-size: 14px;
  color: #212529;
}}
h1, h2, h3 {{
  font-weight: 700;
  letter-spacing: -0.2px;
}}
.smallcap {{
  color: {MUTED};
  font-size: 12px;
}}
.card {{
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 14px;
  padding: 14px 14px;
  background: white;
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
.stTabs [data-baseweb="tab-list"] {{
  gap: 10px;
}}
.stTabs [data-baseweb="tab"] {{
  padding: 10px 14px;
  border-radius: 12px;
}}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Key helper (prevents StreamlitDuplicateElementId)
# ============================================================
if "_kid" not in st.session_state:
    st.session_state["_kid"] = 0


def mk(prefix="k"):
    st.session_state["_kid"] += 1
    return f"{prefix}_{st.session_state['_kid']}"


# ============================================================
# Format helpers
# ============================================================
def fmt_won(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):,.0f} 원"
    except:
        return "-"


def fmt_num(x, d=0):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):,.{d}f}"
    except:
        return "-"


def to_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        s = str(x).strip().replace(",", "")
        s = s.replace("₩", "").replace("원", "").strip()
        s = s.replace("%", "")
        if s == "" or s.lower() == "nan":
            return default
        return float(s)
    except:
        return default


def normalize_ratio(x):
    """
    ratio supports 0.32, 32, '32%', etc.
    returns 0~1 float
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
    return {k: (v / s if v > 0 else 0.0) for k, v in d2.items()}


def soft_find_key(columns, keywords):
    cols = [str(c).strip() for c in columns]
    for kw in keywords:
        for c in cols:
            if kw in c:
                return c
    return None


# ============================================================
# Data loader (xlsx / csv)
# ============================================================
def read_uploaded_to_raw_df(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    data = uploaded.getvalue()

    # CSV
    if name.endswith(".csv"):
        raw_text = data.decode("utf-8-sig", errors="replace")
        df_raw = pd.read_csv(StringIO(raw_text), header=None)
        return df_raw

    # XLSX
    if name.endswith(".xlsx") or name.endswith(".xls"):
        # Try all sheets, pick first where a cell == "시나리오명" exists
        xls = pd.ExcelFile(uploaded)
        for sh in xls.sheet_names:
            df = pd.read_excel(uploaded, sheet_name=sh, header=None)
            # detect "시나리오명" anywhere
            if (df.astype(str) == "시나리오명").any().any():
                return df
        # fallback: first sheet
        return pd.read_excel(uploaded, sheet_name=0, header=None)

    raise ValueError("지원하지 않는 파일 형식입니다. (csv/xlsx만 지원)")


# ============================================================
# preprocess_data: stacked tables split by header row where col0 == "시나리오명"
# ============================================================
def preprocess_data(df_raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # find header rows (col0 == '시나리오명')
    col0 = df_raw.iloc[:, 0].astype(str).str.strip()
    header_idx = df_raw.index[col0.eq("시나리오명")].tolist()
    if not header_idx:
        # Some data may have '시나리오명' not in col0; try search any cell row-wise
        mask = df_raw.astype(str).apply(lambda r: (r == "시나리오명").any(), axis=1)
        header_idx = df_raw.index[mask].tolist()
        if not header_idx:
            raise ValueError("스택형 데이터에서 '시나리오명' 헤더 행을 찾지 못했습니다.")

    sections = []
    for i, h in enumerate(header_idx):
        start = h
        end = header_idx[i + 1] - 1 if i + 1 < len(header_idx) else len(df_raw) - 1
        sec = df_raw.iloc[start : end + 1].copy()

        # drop all-empty columns
        non_empty_cols = [c for c in sec.columns if not sec[c].isna().all()]
        sec = sec[non_empty_cols]

        # header row
        header = sec.iloc[0].tolist()
        header = [str(x).strip() if pd.notna(x) else "" for x in header]

        # unique/clean header
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

        if "시나리오명" in body.columns:
            body["시나리오명"] = body["시나리오명"].astype(str).str.strip()

        sections.append(body)

    out = {"_sections": sections}

    def has_any(cols, keywords):
        s = " ".join([str(c) for c in cols])
        return any(k in s for k in keywords)

    # classify
    for sec in sections:
        cols = sec.columns.tolist()

        # KPI
        if "kpi" not in out and has_any(cols, ["CPC", "CTR", "CVR", "재구매율", "CPM", "목표"]):
            out["kpi"] = sec
            continue

        # channel mix
        if "channel_mix" not in out and has_any(
            cols, ["스마트스토어", "올리브영", "백화점", "쿠팡", "자사몰", "오픈마켓", "홈쇼핑", "공구", "B2B", "온라인", "오프라인"]
        ):
            out["channel_mix"] = sec
            continue

        # media mix (performance/viral/brand detailed)
        if "media_mix" not in out and has_any(cols, ["퍼포먼스", "바이럴", "브랜드", "구글", "메타", "틱톡", "네이버", "외부몰PA"]):
            out["media_mix"] = sec
            continue

        # ad alloc (simple)
        if "ad_alloc" not in out and has_any(cols, ["광고비"]):
            out["ad_alloc"] = sec
            continue

    return out


def scenario_list_from_sections(sections_dict: Dict[str, pd.DataFrame]) -> List[str]:
    names = set()
    for k, v in sections_dict.items():
        if isinstance(v, pd.DataFrame) and "시나리오명" in v.columns:
            for x in v["시나리오명"].dropna().astype(str):
                x = x.strip()
                if x and x != "시나리오명":
                    names.add(x)
    return sorted(names)


def get_row_by_scenario(df: Optional[pd.DataFrame], scenario_key: str) -> Optional[pd.Series]:
    if df is None or not isinstance(df, pd.DataFrame) or "시나리오명" not in df.columns:
        return None
    sub = df[df["시나리오명"].astype(str).str.strip() == str(scenario_key).strip()]
    if sub.empty:
        return None
    return sub.iloc[0]


# ============================================================
# Scenario display mapping (Korean labels)
# ============================================================
SCENARIO_KEY_RE = re.compile(r"^ST-(?P<st>NEW|EARLY|GROW|MATURE)__DRV-(?P<drv>[A-Z0-9]+)__CAT-(?P<cat>.+?)__POS-(?P<pos>[LMP])$")


def parse_scenario_key(name: str):
    name = str(name or "").strip()
    m = SCENARIO_KEY_RE.match(name)
    if not m:
        return None
    return {"ST": m.group("st"), "DRV": m.group("drv"), "CAT": m.group("cat"), "POS": m.group("pos")}


def find_display_col(df: Optional[pd.DataFrame]) -> Optional[str]:
    if df is None or not isinstance(df, pd.DataFrame):
        return None
    candidates = ["노출용 시나리오명", "노출 시나리오명", "시나리오명(노출)", "시나리오명_노출", "scenario_display"]
    for c in df.columns:
        if str(c).strip() in candidates:
            return c
    # fallback: contains "노출"
    for c in df.columns:
        if "노출" in str(c):
            return c
    return None


def build_scenario_display_map(sections: Dict[str, pd.DataFrame], scenario_keys: List[str]) -> Dict[str, str]:
    # Try find display column from any section
    disp_col = None
    src_df = None
    for k in ["media_mix", "channel_mix", "ad_alloc", "kpi"]:
        df = sections.get(k)
        c = find_display_col(df)
        if c:
            disp_col = c
            src_df = df
            break

    mapping = {}
    if disp_col and src_df is not None:
        tmp = src_df[["시나리오명", disp_col]].dropna()
        for _, r in tmp.iterrows():
            key = str(r["시나리오명"]).strip()
            val = str(r[disp_col]).strip()
            if key and val:
                mapping[key] = val

    # fallback: generate from key
    for key in scenario_keys:
        if key not in mapping:
            p = parse_scenario_key(key)
            if p:
                # Simple readable fallback
                pos_map = {"L": "가성비", "M": "밸류", "P": "프리미엄"}
                st_map = {"NEW": "신규", "EARLY": "초기", "GROW": "성장", "MATURE": "성숙"}
                mapping[key] = f"{st_map.get(p['ST'], p['ST'])} {p['CAT']} {pos_map.get(p['POS'], p['POS'])}"
            else:
                mapping[key] = key
    return mapping


# ============================================================
# Media mix builders
# ============================================================
def build_media_budget_shares(mm_row: Optional[pd.Series]) -> dict:
    """
    From media_mix row:
    - classify columns into performance / viral / brand
    - compute group weights and per-channel shares
    """
    out = {"performance": {}, "viral": {}, "brand": {}, "_group_weights": {"performance": 0.0, "viral": 0.0, "brand": 0.0}}
    if mm_row is None:
        return out

    cols = [c for c in mm_row.index.astype(str) if c != "시나리오명" and "노출" not in str(c)]
    # heuristics
    perf_cols = [c for c in cols if ("퍼포먼스" in c) or (c.startswith("퍼포먼스마케팅_")) or ("외부몰PA" in c)]
    viral_cols = [c for c in cols if ("바이럴" in c) or (c.startswith("바이럴마케팅_"))]
    brand_cols = [c for c in cols if ("브랜드" in c)]

    # If rows are not labeled with prefixes, try fallback:
    if not perf_cols and not viral_cols and not brand_cols:
        # Treat anything containing "SA/GDN/DA/PMAX/메타/구글/네이버/틱톡" as performance
        perf_cols = [c for c in cols if any(k in c for k in ["SA", "GDN", "DA", "PMAX", "메타", "구글", "네이버", "틱톡", "크리테오"])]
        viral_cols = [c for c in cols if any(k in c for k in ["블로그", "지식인", "인플루", "핫딜", "카페", "커뮤니티", "씨딩", "바이럴", "체험단"])]
        brand_cols = [c for c in cols if any(k in c for k in ["브랜딩", "브랜드", "PR"])]

    def extract(cols_):
        raw = {c: normalize_ratio(mm_row.get(c)) for c in cols_}
        raw = {k: (0.0 if (v is None or np.isnan(v)) else float(v)) for k, v in raw.items()}
        return raw

    perf_raw = extract(perf_cols)
    viral_raw = extract(viral_cols)
    brand_raw = extract(brand_cols)

    perf_total = sum(v for v in perf_raw.values() if v > 0)
    viral_total = sum(v for v in viral_raw.values() if v > 0)
    brand_total = sum(v for v in brand_raw.values() if v > 0)
    grand = perf_total + viral_total + brand_total

    out["performance"] = {k: v for k, v in normalize_shares(perf_raw).items() if v > 0}
    out["viral"] = {k: v for k, v in normalize_shares(viral_raw).items() if v > 0}
    out["brand"] = {k: v for k, v in normalize_shares(brand_raw).items() if v > 0}

    if grand > 0:
        out["_group_weights"]["performance"] = perf_total / grand
        out["_group_weights"]["viral"] = viral_total / grand
        out["_group_weights"]["brand"] = brand_total / grand

    return out


def donut_chart(labels, values, title="", height=320):
    df = pd.DataFrame({"label": labels, "value": values})
    df = df[df["value"] > 0]
    if df.empty:
        fig = go.Figure()
        fig.update_layout(height=height, margin=dict(t=40, l=10, r=10, b=10), title=title)
        return fig

    fig = px.pie(df, values="value", names="label", hole=0.55)
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=height, margin=dict(t=40, l=10, r=10, b=10), title=title)
    return fig


# ============================================================
# Channel mix (100% donut)
# ============================================================
def build_channel_mix_shares(ch_row: Optional[pd.Series]) -> dict:
    """
    returns shares across channels in that row (excluding scenario fields)
    """
    if ch_row is None:
        return {}
    tmp = ch_row.drop(labels=["시나리오명"], errors="ignore")
    # drop display columns if exist
    for c in list(tmp.index):
        if "노출" in str(c):
            tmp = tmp.drop(labels=[c], errors="ignore")

    vals = tmp.apply(normalize_ratio).dropna()
    vals = vals[vals > 0]
    if vals.empty:
        return {}
    return normalize_shares(vals.to_dict())


# ============================================================
# KPI utilities
# ============================================================
def kpi_value(kpi_row: Optional[pd.Series], keywords: List[str], default: float) -> float:
    if kpi_row is None:
        return default
    cols = [str(c) for c in kpi_row.index]
    # exact keyword
    for kw in keywords:
        for c in cols:
            if str(c).strip() == kw:
                return to_float(kpi_row[c], default)
    # fuzzy
    for kw in keywords:
        for c in cols:
            if kw in str(c):
                return to_float(kpi_row[c], default)
    return default


# ============================================================
# P&L simulation core
# - supports two modes:
#   1) ad -> revenue
#   2) revenue -> required ad
# - Contribution margin rule: (매출-광고-물류-원가)/매출
# ============================================================
def simulate_pl(
    mode: str,
    aov: float,
    cpc: float,
    cvr: float,
    cost_rate: float,
    logistics_per_order: float,
    headcount: int,
    cost_per_head: float,
    ad_spend: Optional[float] = None,
    revenue: Optional[float] = None,
) -> dict:
    labor = float(headcount) * float(cost_per_head)

    if mode == "광고비 입력 → 매출 산출":
        ad_spend = float(ad_spend or 0.0)
        clicks = ad_spend / cpc if cpc > 0 else 0.0
        orders = clicks * cvr
        revenue = orders * aov
    else:
        revenue = float(revenue or 0.0)
        orders = revenue / aov if aov > 0 else 0.0
        clicks = orders / cvr if cvr > 0 else 0.0
        ad_spend = clicks * cpc

    cogs = revenue * cost_rate
    logistics = orders * logistics_per_order
    profit = revenue - (ad_spend + cogs + logistics + labor)
    contrib_margin = ((revenue - ad_spend - logistics - cogs) / revenue * 100.0) if revenue > 0 else 0.0
    roas = (revenue / ad_spend) if ad_spend and ad_spend > 0 else 0.0

    return {
        "revenue": float(revenue),
        "ad_spend": float(ad_spend),
        "clicks": float(clicks),
        "orders": float(orders),
        "cogs": float(cogs),
        "logistics": float(logistics),
        "labor": float(labor),
        "profit": float(profit),
        "contrib_margin": float(contrib_margin),
        "roas": float(roas),
    }


# ============================================================
# Viral unit cost default template (editable)
# - If media_mix has viral surfaces columns, we will match by name.
# ============================================================
DEFAULT_VIRAL_PRICE = [
    {"매체": "네이버", "지면": "네이버_인플루언서탭", "건당비용": 250000},
    {"매체": "네이버", "지면": "네이버_스마트블록", "건당비용": 250000},
    {"매체": "네이버", "지면": "네이버_지식인", "건당비용": 100000},
    {"매체": "네이버", "지면": "네이버_쇼핑상위", "건당비용": 2000000},
    {"매체": "네이버", "지면": "네이버_인기글", "건당비용": 300000},
    {"매체": "네이버", "지면": "네이버_자동검색완성", "건당비용": 400000},
    {"매체": "네이버", "지면": "네이버_카페침투바이럴", "건당비용": 30000},
    {"매체": "네이버", "지면": "네이버_구매대행", "건당비용": 120060},
    {"매체": "네이버", "지면": "네이버_핫딜", "건당비용": 100000},
    {"매체": "인스타그램", "지면": "인스타그램_파워페이지", "건당비용": 400000},
    {"매체": "인스타그램", "지면": "인스타그램_해시태그상위노출", "건당비용": 500000},
    {"매체": "인스타그램", "지면": "인스타그램_계정상위노출", "건당비용": 400000},
    {"매체": "오늘의집", "지면": "오늘의집 집들이", "건당비용": 500000},
    {"매체": "오늘의집", "지면": "오늘의집 체험단", "건당비용": 400000},
    {"매체": "오늘의집", "지면": "오늘의집 구매대행", "건당비용": 200952},
    {"매체": "기타 커뮤니티", "지면": "커뮤니티_핫딜", "건당비용": 200000},
]


def extract_viral_surfaces_from_media_mix(mm_row: Optional[pd.Series]) -> List[str]:
    if mm_row is None:
        return []
    cols = [c for c in mm_row.index.astype(str) if c not in ["시나리오명"]]
    # viral columns are those containing "바이럴" OR those matching known surface tokens
    surface_like = []
    for c in cols:
        if "바이럴" in c:
            # could be "바이럴마케팅_네이버_지식인" -> keep as is
            surface_like.append(c)
        else:
            # also allow "네이버_지식인" style
            if any(token in c for token in ["네이버_", "인스타그램_", "오늘의집", "커뮤니티_"]):
                surface_like.append(c)
    # remove obvious non-surface (ex: 바이럴 총합)
    surface_like = [s for s in surface_like if "종합" not in s and "Total" not in s and "총합" not in s]
    return sorted(list(set(surface_like)))


def build_viral_allocation(
    viral_total_budget: float,
    mm_row: Optional[pd.Series],
    viral_price_df: pd.DataFrame,
    rounding_unit: int = 100,
) -> pd.DataFrame:
    """
    - allocate budget to surfaces using scenario viral ratios if available
    - compute count = round(allocated_budget / unit_cost)
    - total_cost = count * unit_cost
    - slight mismatch with budget is OK
    """
    viral_total_budget = float(viral_total_budget)

    # Determine ratio source from media_mix row:
    surface_cols = extract_viral_surfaces_from_media_mix(mm_row)
    ratios = {}

    if mm_row is not None and surface_cols:
        for c in surface_cols:
            v = normalize_ratio(mm_row.get(c))
            if not np.isnan(v) and v > 0:
                ratios[c] = float(v)

    # If ratios are empty, fallback: equal split across price list surfaces
    if not ratios:
        for s in viral_price_df["지면"].astype(str).tolist():
            ratios[s] = 1.0

    ratios = normalize_shares(ratios)

    # Allocate
    rows = []
    for surface, share in ratios.items():
        planned = viral_total_budget * share
        # round to nearest 100
        planned = int(round(planned / rounding_unit) * rounding_unit)

        # Match unit cost by "지면" contains or equals
        cost = None
        matched = viral_price_df[viral_price_df["지면"].astype(str) == str(surface)]
        if matched.empty:
            # try fuzzy: last token
            for _, r in viral_price_df.iterrows():
                if str(r["지면"]) in str(surface) or str(surface) in str(r["지면"]):
                    cost = to_float(r["건당비용"], 0.0)
                    media = r["매체"]
                    break
            if cost is None:
                cost = 0.0
                media = "기타"
        else:
            cost = to_float(matched.iloc[0]["건당비용"], 0.0)
            media = matched.iloc[0]["매체"]

        count = int(round(planned / cost)) if cost > 0 else 0
        total_cost = count * cost

        rows.append(
            {
                "구분": "바이럴",
                "매체": media,
                "지면": surface,
                "건당비용": cost,
                "진행 건수": count,
                "계획비(청구비)": planned,
                "총비용(계산)": total_cost,
            }
        )

    df = pd.DataFrame(rows)
    # order
    df = df.sort_values(["매체", "지면"]).reset_index(drop=True)
    return df


# ============================================================
# Performance allocation + billing (agency internal/external)
# ============================================================
def build_performance_allocation(
    perf_total_budget: float,
    adg: dict,
    rounding_unit: int = 100,
) -> pd.DataFrame:
    perf_total_budget = float(perf_total_budget)
    perf = adg.get("performance", {}) if isinstance(adg, dict) else {}
    if not perf:
        return pd.DataFrame(columns=["구분", "구분2", "매체", "예산(집행)"])

    perf = normalize_shares(perf)
    rows = []
    for media, share in perf.items():
        budget = perf_total_budget * share
        budget = int(round(budget / rounding_unit) * rounding_unit)
        rows.append({"구분": "퍼포먼스", "구분2": "광고", "매체": media, "예산(집행)": budget})
    df = pd.DataFrame(rows).sort_values("예산(집행)", ascending=False).reset_index(drop=True)
    return df


def apply_agency_billing(perf_df: pd.DataFrame, fee_rate: float, payback_rate: float) -> pd.DataFrame:
    """
    - fee_rate: 대행수수료율 (0~)
    - payback_rate: 페이백률 (0~)
    billed = 집행 * (1+fee_rate)
    payback = 집행 * payback_rate
    net = billed - payback
    """
    if perf_df.empty:
        return perf_df.copy()

    fee_rate = float(fee_rate)
    payback_rate = float(payback_rate)

    df = perf_df.copy()
    df["대행수수료율"] = fee_rate
    df["페이백률"] = payback_rate
    df["대행수수료(예상)"] = df["예산(집행)"] * fee_rate
    df["청구예상비용"] = df["예산(집행)"] + df["대행수수료(예상)"]
    df["페이백예상액"] = df["예산(집행)"] * payback_rate
    df["청구예상(페이백차감)"] = df["청구예상비용"] - df["페이백예상액"]
    return df


# ============================================================
# Scenario compare chart (bars + ROAS line on 0~100 secondary axis)
# ============================================================
def scenario_compare_chart(
    df_cmp: pd.DataFrame,
    view: str,
    target_roas: float,
    title: str,
) -> go.Figure:
    """
    df_cmp columns:
      시나리오, 매출, 광고비, 영업이익, 공헌이익률, ROAS
    view:
      '매출', '광고비', '영업이익', '공헌이익률', 'ROAS', '전체(3개 동시)'
    ROAS line axis: 0~100 (% of target)
    """
    df = df_cmp.copy()

    # ROAS percent of target (0~100)
    target_roas = max(float(target_roas), 1e-9)
    df["ROAS(달성률%)"] = (df["ROAS"] / target_roas) * 100.0
    df["ROAS(달성률%)"] = df["ROAS(달성률%)"].clip(0, 100)

    fig = go.Figure()

    x = df["시나리오"].tolist()

    if view in ["매출", "광고비", "영업이익"]:
        fig.add_trace(go.Bar(name=view, x=x, y=df[view], yaxis="y", text=df[view].round(0)))
    elif view == "공헌이익률":
        fig.add_trace(go.Bar(name=view, x=x, y=df[view], yaxis="y", text=df[view].round(1)))
    elif view == "ROAS":
        fig.add_trace(
            go.Scatter(
                name="ROAS 달성률(%)",
                x=x,
                y=df["ROAS(달성률%)"],
                mode="lines+markers",
                yaxis="y2",
            )
        )
    else:
        # 전체(3개 동시): 매출/광고비 막대 + ROAS 달성률 꺾은선
        fig.add_trace(go.Bar(name="매출", x=x, y=df["매출"], yaxis="y"))
        fig.add_trace(go.Bar(name="광고비", x=x, y=df["광고비"], yaxis="y"))
        fig.add_trace(
            go.Scatter(
                name="ROAS 달성률(%)",
                x=x,
                y=df["ROAS(달성률%)"],
                mode="lines+markers",
                yaxis="y2",
            )
        )

    fig.update_layout(
        title=title,
        height=420,
        margin=dict(t=60, l=10, r=10, b=10),
        barmode="group",
        xaxis=dict(tickangle=0),
        yaxis=dict(title=None, showgrid=True),
        yaxis2=dict(
            title="ROAS 달성률(0~100%)",
            overlaying="y",
            side="right",
            range=[0, 100],
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ============================================================
# Sidebar: Upload
# ============================================================
st.sidebar.title("마케팅/유통 시뮬레이터")
uploaded = st.sidebar.file_uploader("backdata 업로드 (xlsx/csv)", type=["xlsx", "xls", "csv"], key=mk("uploader"))

if uploaded is None:
    st.info("좌측에서 backdata 파일(xlsx/csv)을 업로드하세요.")
    st.stop()

try:
    df_raw = read_uploaded_to_raw_df(uploaded)
    data = preprocess_data(df_raw)
except Exception as e:
    st.error(f"❌ 데이터 로드/파싱 실패: {e}")
    st.stop()

scenarios = scenario_list_from_sections(data)
if not scenarios:
    st.error("❌ 시나리오 목록을 찾지 못했습니다. (각 섹션에 '시나리오명' 컬럼 필요)")
    st.stop()

# section dfs
ad_alloc_df = data.get("ad_alloc")
channel_mix_df = data.get("channel_mix")
media_mix_df = data.get("media_mix")
kpi_df = data.get("kpi")

# display mapping
scenario_display_map = build_scenario_display_map(data, scenarios)
display_to_key = {v: k for k, v in scenario_display_map.items()}

# ============================================================
# Top tabs
# ============================================================
tab_rec, tab_dash = st.tabs(["✅ 추천 엔진", "📊 대시보드 (대행/브랜드)"])

# ============================================================
# TAB 1) Recommendation Engine (kept, but JSON output removed)
# ============================================================
with tab_rec:
    st.markdown("## 추천 엔진")
    st.markdown('<div class="smallcap">데이터 기반 Top3 추천 (룰 기반 스코어링 + KPI가 있으면 참고)</div>', unsafe_allow_html=True)

    # Layout changed: results below inputs (better readability)
    st.markdown("### 입력")
    cA, cB, cC, cD = st.columns(4)
    with cA:
        operator = st.selectbox(
            "운영 주체",
            ["내부브랜드 운영자", "브랜드사 운영자(클라이언트)", "대행사(마케팅만)"],
            key=mk("op"),
        )
    with cB:
        stage = st.selectbox("단계(ST)", ["NEW", "EARLY", "GROW", "MATURE"], key=mk("st"))
    with cC:
        # CAT options from parsed keys
        cats = sorted({parse_scenario_key(s)["CAT"] for s in scenarios if parse_scenario_key(s)})
        category = st.selectbox("카테고리(CAT)", cats if cats else ["-"], key=mk("cat"))
    with cD:
        position = st.selectbox("가격 포지셔닝(POS)", ["L", "M", "P"], key=mk("pos"))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sales_focus_channel = st.selectbox("판매 중심", ["자사몰 중심", "온라인 중심", "홈쇼핑 중심", "공구 중심", "B2B 중심"], key=mk("sf"))
    with c2:
        target_age = st.selectbox("타겟 연령", ["10대", "20대", "30대", "40대", "50대+"], key=mk("age"))
    with c3:
        total_ad_budget_krw = st.number_input("총 광고예산(원)", value=50000000, step=1000000, min_value=1, key=mk("bud"))
    with c4:
        no_comp = st.toggle("경쟁키워드 판매의도 없음", value=True, key=mk("nocomp"))

    # Simple scoring (kept lightweight but stable)
    def simple_score(s_key: str) -> float:
        p = parse_scenario_key(s_key)
        if not p:
            return 0.0
        score = 0.0
        if p["ST"] == stage:
            score += 35
        if p["CAT"] == category:
            score += 35
        if p["POS"] == position:
            score += 20
        # age heuristic: younger => tiktok/instagram share bonus
        mm_row = get_row_by_scenario(media_mix_df, s_key)
        adg = build_media_budget_shares(mm_row)
        gw = adg["_group_weights"]
        viral = gw.get("viral", 0.0)
        perf = gw.get("performance", 0.0)
        if target_age in ["10대", "20대"]:
            score += min(10, (viral + perf) * 10)
        else:
            score += min(10, perf * 10)
        return float(score)

    run = st.button("Top3 추천", use_container_width=True, key=mk("runrec"))

    if run:
        # candidates by hard filter
        candidates = []
        for s in scenarios:
            p = parse_scenario_key(s)
            if not p:
                continue
            if p["ST"] == stage and p["CAT"] == category and p["POS"] == position:
                candidates.append(s)

        st.markdown("### 결과")
        st.metric("후보 전략 수", f"{len(candidates):,} 개", key=mk("cand"))

        if not candidates:
            st.info("조건(ST/CAT/POS)에 맞는 시나리오가 없습니다.")
        else:
            ranked = sorted([(s, simple_score(s)) for s in candidates], key=lambda x: x[1], reverse=True)[:3]

            cards = st.columns(3)
            for i, (s, sc) in enumerate(ranked):
                disp = scenario_display_map.get(s, s)
                mm_row = get_row_by_scenario(media_mix_df, s)
                adg = build_media_budget_shares(mm_row)
                gw = adg["_group_weights"]
                with cards[i]:
                    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"**#{i+1} {disp}**")
                    st.caption(s)
                    st.metric("Score", f"{sc:.1f}", key=mk("score"))
                    st.metric("퍼포먼스 비중", f"{gw.get('performance',0)*100:.0f}%", key=mk("p"))
                    st.metric("바이럴 비중", f"{gw.get('viral',0)*100:.0f}%", key=mk("v"))
                    st.metric("브랜드 비중", f"{gw.get('brand',0)*100:.0f}%", key=mk("b"))
                    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# TAB 2) Dashboard: Agency / Brand (each has internal/external)
# ============================================================
with tab_dash:
    st.sidebar.markdown("---")
    main_mode = st.sidebar.radio("모드", ["대행", "브랜드"], key=mk("mainmode"))
    sub_mode = st.sidebar.radio("버전", ["내부용", "외부용"], key=mk("submode"))

    # show scenario with display names
    display_names = [scenario_display_map[s] for s in scenarios]
    display_selected = st.sidebar.selectbox("전략 선택(노출용)", display_names, key=mk("scsel"))
    scenario_key = display_to_key.get(display_selected, scenarios[0])

    mm_row = get_row_by_scenario(media_mix_df, scenario_key)
    ch_row = get_row_by_scenario(channel_mix_df, scenario_key)
    kpi_row = get_row_by_scenario(kpi_df, scenario_key)

    # KPI defaults
    base_cpc = kpi_value(kpi_row, ["목표 평균 CPC", "CPC"], 300.0)
    base_cvr = kpi_value(kpi_row, ["목표 평균 CVR", "CVR"], 0.02)
    if base_cvr > 1:
        base_cvr = base_cvr / 100.0

    # Build grouped media shares
    adg = build_media_budget_shares(mm_row)
    gw = adg["_group_weights"]

    # Top header
    st.markdown(f"## {main_mode} · {sub_mode}")
    st.markdown(f'<div class="smallcap">선택 전략: <b>{display_selected}</b></div>', unsafe_allow_html=True)

    # ============================================================
    # Shared inputs: budget + ROAS target (for ROAS chart normalization)
    # ============================================================
    st.markdown("### 기본 입력")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        total_budget = st.number_input("총 예산(원)", value=60000000, step=1000000, key=mk("totbud"))
    with b2:
        target_roas = st.number_input("타겟 ROAS(예: 3.0)", value=3.0, step=0.1, key=mk("troas"))
    with b3:
        cpc = st.number_input("CPC(원)", value=float(base_cpc), step=10.0, key=mk("cpc"))
    with b4:
        cvr = st.number_input("CVR(%)", value=float(base_cvr * 100.0), step=0.1, key=mk("cvr")) / 100.0

    # ============================================================
    # 100% donuts: channel mix, group mix
    # ============================================================
    cL, cR = st.columns([1, 1])
    with cL:
        ch_shares = build_channel_mix_shares(ch_row)
        if ch_shares:
            fig = donut_chart(list(ch_shares.keys()), list(ch_shares.values()), title="매출 채널 구성(100%)", height=320)
            st.plotly_chart(fig, use_container_width=True, key=mk("donut_ch"))
        else:
            st.info("채널 믹스 데이터가 비어있습니다.")
    with cR:
        fig = donut_chart(["퍼포먼스", "바이럴", "브랜드"], [gw.get("performance", 0), gw.get("viral", 0), gw.get("brand", 0)], title="미디어 그룹 구성(100%)", height=320)
        st.plotly_chart(fig, use_container_width=True, key=mk("donut_group"))

    st.divider()

    # ============================================================
    # Agency mode
    # ============================================================
    if main_mode == "대행":
        # Split budgets by group weights
        perf_budget = total_budget * gw.get("performance", 0.0)
        viral_budget = total_budget * gw.get("viral", 0.0)
        brand_budget = total_budget * gw.get("brand", 0.0)

        st.markdown("### 미디어 믹스 (퍼포먼스 / 바이럴)")
        m1, m2, m3 = st.columns(3)
        m1.metric("퍼포먼스 예산", fmt_won(perf_budget), key=mk("pb"))
        m2.metric("바이럴 예산", fmt_won(viral_budget), key=mk("vb"))
        m3.metric("브랜드 예산", fmt_won(brand_budget), key=mk("bb"))

        # --- Performance table
        perf_df = build_performance_allocation(perf_budget, adg, rounding_unit=100)

        # Agency internal: allow fee/payback inputs + labor + PL
        if sub_mode == "내부용":
            st.markdown("#### 퍼포먼스 (내부용: 수수료/페이백 반영)")
            p1, p2, p3 = st.columns(3)
            with p1:
                fee_rate = st.number_input("대행 수수료율(%)", value=10.0, step=0.5, key=mk("feer")) / 100.0
            with p2:
                payback_rate = st.number_input("페이백률(%)", value=0.0, step=0.5, key=mk("pbr")) / 100.0
            with p3:
                rounding_unit = st.selectbox("예산 반올림 단위", [100, 1000, 10000], index=0, key=mk("roundp"))

            # rebuild with rounding
            perf_df = build_performance_allocation(perf_budget, adg, rounding_unit=int(rounding_unit))
            perf_bill = apply_agency_billing(perf_df, fee_rate=fee_rate, payback_rate=payback_rate)

            if not perf_bill.empty:
                show = perf_bill.copy()
                for col in ["예산(집행)", "대행수수료(예상)", "청구예상비용", "페이백예상액", "청구예상(페이백차감)"]:
                    if col in show.columns:
                        show[col] = show[col].map(lambda x: f"{x:,.0f}")
                show["대행수수료율"] = (show["대행수수료율"] * 100).map(lambda x: f"{x:.1f}%")
                show["페이백률"] = (show["페이백률"] * 100).map(lambda x: f"{x:.1f}%")
                st.dataframe(show, use_container_width=True, hide_index=True, key=mk("perf_tbl_in"))

                # performance 100% donut
                fig = donut_chart(perf_bill["매체"].tolist(), perf_bill["예산(집행)"].tolist(), title="퍼포먼스 채널 예산(100%)", height=300)
                st.plotly_chart(fig, use_container_width=True, key=mk("donut_perf"))

            # --- Viral pricing editor
            st.markdown("#### 바이럴 (내부용: 실집행 입력 + 마진)")
            st.caption("규칙: 예산을 지면별 비율로 배분 → 건수=반올림(정수) → 총비용은 건수×단가 (합계가 예산과 달라도 OK)")

            if "viral_price_df" not in st.session_state:
                st.session_state["viral_price_df"] = pd.DataFrame(DEFAULT_VIRAL_PRICE)

            viral_price_df = st.data_editor(
                st.session_state["viral_price_df"],
                use_container_width=True,
                num_rows="dynamic",
                key=mk("viral_price_editor"),
            )
            st.session_state["viral_price_df"] = viral_price_df

            v_round = st.selectbox("바이럴 예산 반올림 단위", [100, 1000, 10000], index=0, key=mk("vround"))

            viral_alloc = build_viral_allocation(viral_budget, mm_row, viral_price_df, rounding_unit=int(v_round))

            # internal: add actual spend input
            if "viral_actual" not in st.session_state:
                st.session_state["viral_actual"] = {}

            viral_alloc["실집행비용(입력)"] = 0.0
            for i in range(len(viral_alloc)):
                key = f"{viral_alloc.loc[i,'지면']}"
                viral_alloc.loc[i, "실집행비용(입력)"] = float(st.session_state["viral_actual"].get(key, 0.0))

            # render editor for actual spend only
            editable = viral_alloc[["구분", "매체", "지면", "건당비용", "진행 건수", "계획비(청구비)", "총비용(계산)", "실집행비용(입력)"]].copy()
            editable["실집행비용(입력)"] = editable["실집행비용(입력)"].astype(float)

            edited = st.data_editor(
                editable,
                use_container_width=True,
                hide_index=True,
                key=mk("viral_actual_editor"),
                column_config={
                    "실집행비용(입력)": st.column_config.NumberColumn(format="%,.0f", step=10000),
                    "계획비(청구비)": st.column_config.NumberColumn(format="%,.0f"),
                    "총비용(계산)": st.column_config.NumberColumn(format="%,.0f"),
                    "건당비용": st.column_config.NumberColumn(format="%,.0f"),
                },
                disabled=["구분", "매체", "지면", "건당비용", "진행 건수", "계획비(청구비)", "총비용(계산)"],
            )

            # save actuals back
            for _, r in edited.iterrows():
                st.session_state["viral_actual"][str(r["지면"])] = float(r["실집행비용(입력)"] or 0.0)

            # margin
            edited["마진(계획-실집행)"] = edited["계획비(청구비)"].astype(float) - edited["실집행비용(입력)"].astype(float)
            st.markdown("#### 바이럴 마진 요약")
            s1, s2, s3 = st.columns(3)
            s1.metric("바이럴 계획비 합계", fmt_won(edited["계획비(청구비)"].sum()), key=mk("vsum1"))
            s2.metric("바이럴 실집행 합계", fmt_won(edited["실집행비용(입력)"].sum()), key=mk("vsum2"))
            s3.metric("바이럴 마진 합계", fmt_won(edited["마진(계획-실집행)"].sum()), key=mk("vsum3"))

            # viral donut (100%) by surface planned
            fig = donut_chart(edited["지면"].tolist(), edited["계획비(청구비)"].tolist(), title="바이럴 지면 계획비(100%)", height=320)
            st.plotly_chart(fig, use_container_width=True, key=mk("donut_viral_surface"))

            # --- Agency internal should include labor (requested)
            st.divider()
            st.markdown("### 내부 손익(간단)")
            colA, colB, colC, colD = st.columns(4)
            with colA:
                aov = st.number_input("객단가(AOV, 원)", value=50000, step=1000, key=mk("aov_a"))
            with colB:
                cost_rate = st.number_input("원가율(%)", value=30.0, step=0.5, key=mk("cr_a")) / 100.0
            with colC:
                logistics = st.number_input("물류비/건(원)", value=3000, step=500, key=mk("lg_a"))
            with colD:
                calc_mode = st.selectbox("계산 방식", ["광고비 입력 → 매출 산출", "매출 입력 → 필요 광고비 산출"], key=mk("cm_a"))

            hc1, hc2, hc3 = st.columns(3)
            with hc1:
                headcount = st.number_input("운영 인력 수", value=2, step=1, min_value=0, key=mk("hc_a"))
            with hc2:
                cost_per_head = st.number_input("인당 고정비(원)", value=3000000, step=100000, min_value=0, key=mk("cph_a"))
            with hc3:
                if calc_mode == "광고비 입력 → 매출 산출":
                    ad_input = st.number_input("광고비(원)", value=int(total_budget), step=1000000, key=mk("adin_a"))
                    rev_input = None
                else:
                    rev_input = st.number_input("목표매출(원)", value=300000000, step=10000000, key=mk("rvin_a"))
                    ad_input = None

            pl = simulate_pl(
                mode=calc_mode,
                aov=aov,
                cpc=cpc,
                cvr=cvr,
                cost_rate=cost_rate,
                logistics_per_order=logistics,
                headcount=int(headcount),
                cost_per_head=cost_per_head,
                ad_spend=ad_input,
                revenue=rev_input,
            )

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("예상 매출", fmt_won(pl["revenue"]), key=mk("pl1"))
            k2.metric("예상 광고비", fmt_won(pl["ad_spend"]), key=mk("pl2"))
            k3.metric("영업이익", fmt_won(pl["profit"]), key=mk("pl3"))
            k4.metric("공헌이익률", f"{pl['contrib_margin']:.1f}%", key=mk("pl4"))

        # Agency external: cleaner proposal view
        else:
            st.markdown("#### 퍼포먼스 (외부용: 예산만)")
            if not perf_df.empty:
                show = perf_df.copy()
                show["예산(집행)"] = show["예산(집행)"].map(lambda x: f"{x:,.0f}")
                st.dataframe(show, use_container_width=True, hide_index=True, key=mk("perf_tbl_out"))

                fig = donut_chart(perf_df["매체"].tolist(), perf_df["예산(집행)"].tolist(), title="퍼포먼스 채널 예산(100%)", height=300)
                st.plotly_chart(fig, use_container_width=True, key=mk("donut_perf_out"))
            else:
                st.info("퍼포먼스 믹스 데이터가 비어있습니다.")

            st.markdown("#### 바이럴 (외부용: 건수 산출)")
            if "viral_price_df" not in st.session_state:
                st.session_state["viral_price_df"] = pd.DataFrame(DEFAULT_VIRAL_PRICE)
            viral_price_df = st.session_state["viral_price_df"]

            v_round = st.selectbox("바이럴 예산 반올림 단위", [100, 1000, 10000], index=0, key=mk("vround_out"))
            viral_alloc = build_viral_allocation(viral_budget, mm_row, viral_price_df, rounding_unit=int(v_round))

            # display table (no actual spend)
            disp = viral_alloc[["구분", "매체", "지면", "건당비용", "진행 건수", "계획비(청구비)", "총비용(계산)"]].copy()
            for col in ["건당비용", "계획비(청구비)", "총비용(계산)"]:
                disp[col] = disp[col].map(lambda x: f"{float(x):,.0f}")
            st.dataframe(disp, use_container_width=True, hide_index=True, key=mk("viral_tbl_out"))

            fig = donut_chart(viral_alloc["지면"].tolist(), viral_alloc["계획비(청구비)"].tolist(), title="바이럴 지면 계획비(100%)", height=320)
            st.plotly_chart(fig, use_container_width=True, key=mk("donut_viral_out"))

    # ============================================================
    # Brand mode
    # ============================================================
    else:
        st.markdown("### 브랜드 대시보드")
        st.caption("브랜드사 외부용은 과도한 약속/디테일을 피하고, 보기 좋은 수준으로 제시합니다.")

        # Brand also wants monthly revenue/adspend projection
        st.markdown("#### 월별 매출/광고비 예측")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            months = st.number_input("예측 개월수", value=6, step=1, min_value=1, max_value=24, key=mk("months"))
        with col2:
            base_revenue = st.number_input("1개월차 예상매출(원)", value=300000000, step=10000000, key=mk("brev"))
        with col3:
            monthly_growth = st.number_input("월 성장률(%)", value=5.0, step=0.5, key=mk("mgr")) / 100.0
        with col4:
            base_ad = st.number_input("1개월차 광고비(원)", value=int(total_budget), step=1000000, key=mk("bad"))

        # brand internal: include labor inputs (but do not show on dashboard if external)
        headcount = 0
        cost_per_head = 0
        if sub_mode == "내부용":
            st.markdown("#### (내부용만) 인건비 입력")
            h1, h2 = st.columns(2)
            with h1:
                headcount = st.number_input("운영 인력 수", value=2, step=1, min_value=0, key=mk("hc_b"))
            with h2:
                cost_per_head = st.number_input("인당 고정비(원)", value=3000000, step=100000, min_value=0, key=mk("cph_b"))

        # build monthly series
        rows = []
        rev = float(base_revenue)
        adsp = float(base_ad)
        for m in range(int(months)):
            month_name = f"{m+1}개월"
            rows.append({"월": month_name, "예상매출": rev, "예상광고비": adsp, "ROAS": (rev / adsp) if adsp > 0 else 0.0})
            rev *= (1.0 + monthly_growth)

        df_month = pd.DataFrame(rows)
        df_month["ROAS달성률(%)"] = (df_month["ROAS"] / max(float(target_roas), 1e-9)) * 100.0
        df_month["ROAS달성률(%)"] = df_month["ROAS달성률(%)"].clip(0, 100)

        # Chart: bars for revenue/ad, line for ROAS% (0~100 secondary axis)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="예상매출", x=df_month["월"], y=df_month["예상매출"], yaxis="y"))
        fig.add_trace(go.Bar(name="예상광고비", x=df_month["월"], y=df_month["예상광고비"], yaxis="y"))
        fig.add_trace(go.Scatter(name="ROAS 달성률(%)", x=df_month["월"], y=df_month["ROAS달성률(%)"], mode="lines+markers", yaxis="y2"))

        fig.update_layout(
            height=430,
            margin=dict(t=40, l=10, r=10, b=10),
            barmode="group",
            yaxis=dict(title=None),
            yaxis2=dict(
                title="ROAS 달성률(0~100%)",
                overlaying="y",
                side="right",
                range=[0, 100],
                showgrid=False,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True, key=mk("brand_month_chart"))

        # summary metrics (external: keep simple)
        sum_rev = df_month["예상매출"].sum()
        sum_ad = df_month["예상광고비"].sum()
        avg_roas = (sum_rev / sum_ad) if sum_ad > 0 else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("기간 합계 매출", fmt_won(sum_rev), key=mk("bm1"))
        m2.metric("기간 합계 광고비", fmt_won(sum_ad), key=mk("bm2"))
        m3.metric("평균 ROAS", f"{avg_roas:.2f}", key=mk("bm3"))
        m4.metric("ROAS 달성률(평균)", f"{min(100, (avg_roas/max(target_roas,1e-9))*100):.0f}%", key=mk("bm4"))

        if sub_mode == "내부용":
            # very light internal profit view (not too detailed)
            st.markdown("#### (내부용) 간단 손익 참고")
            # Use P&L on first month as snapshot
            aov = st.number_input("객단가(AOV, 원)", value=50000, step=1000, key=mk("aov_b"))
            cost_rate = st.number_input("원가율(%)", value=30.0, step=0.5, key=mk("cr_b")) / 100.0
            logistics = st.number_input("물류비/건(원)", value=3000, step=500, key=mk("lg_b"))

            pl = simulate_pl(
                mode="광고비 입력 → 매출 산출",
                aov=aov,
                cpc=cpc,
                cvr=cvr,
                cost_rate=cost_rate,
                logistics_per_order=logistics,
                headcount=int(headcount),
                cost_per_head=float(cost_per_head),
                ad_spend=float(base_ad),
                revenue=None,
            )
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("1개월차 예상매출", fmt_won(pl["revenue"]), key=mk("bpl1"))
            k2.metric("1개월차 광고비", fmt_won(pl["ad_spend"]), key=mk("bpl2"))
            k3.metric("1개월차 영업이익", fmt_won(pl["profit"]), key=mk("bpl3"))
            k4.metric("공헌이익률", f"{pl['contrib_margin']:.1f}%", key=mk("bpl4"))

        st.divider()

        # Scenario comparison for brand (requested earlier)
        st.markdown("### 전략 비교 (시나리오)")
        compare_display = st.multiselect(
            "비교할 전략 선택(노출용)",
            options=display_names,
            default=display_names[:3],
            key=mk("cmp_pick"),
        )
        compare_keys = [display_to_key.get(d, None) for d in compare_display]
        compare_keys = [k for k in compare_keys if k is not None]

        view = st.radio(
            "보기",
            ["매출", "광고비", "영업이익", "공헌이익률", "ROAS", "전체(3개 동시)"],
            horizontal=True,
            key=mk("cmp_view"),
        )

        # For comparison, we use same PL assumptions for all scenarios (simple & stable)
        # external: do not expose detailed PL inputs; internal: still use same so chart stays consistent
        # Use ad budget = total_budget, and compute revenue via CPC/CVR/AOV
        aov_cmp = 50000.0
        cost_rate_cmp = 0.30
        logistics_cmp = 3000.0
        headcount_cmp = int(headcount) if sub_mode == "내부용" else 0
        cost_per_head_cmp = float(cost_per_head) if sub_mode == "내부용" else 0.0

        rows = []
        for sk in compare_keys:
            disp = scenario_display_map.get(sk, sk)
            # Here: apply group weights? To keep stable, ad spend = total_budget for all
            pl = simulate_pl(
                mode="광고비 입력 → 매출 산출",
                aov=aov_cmp,
                cpc=cpc,
                cvr=cvr,
                cost_rate=cost_rate_cmp,
                logistics_per_order=logistics_cmp,
                headcount=headcount_cmp,
                cost_per_head=cost_per_head_cmp,
                ad_spend=float(total_budget),
                revenue=None,
            )
            rows.append(
                {
                    "시나리오": disp,
                    "매출": pl["revenue"],
                    "광고비": pl["ad_spend"],
                    "영업이익": pl["profit"],
                    "공헌이익률": pl["contrib_margin"],
                    "ROAS": pl["roas"],
                }
            )

        df_cmp = pd.DataFrame(rows)
        if df_cmp.empty:
            st.info("비교할 전략을 선택하세요.")
        else:
            # rename columns for chart function
            df_cmp = df_cmp.rename(columns={"매출": "매출", "광고비": "광고비", "영업이익": "영업이익", "공헌이익률": "공헌이익률", "ROAS": "ROAS"})
            # chart
            fig = scenario_compare_chart(df_cmp, view=view, target_roas=float(target_roas), title="전략 비교")
            st.plotly_chart(fig, use_container_width=True, key=mk("cmp_chart"))

            # table (no clickable behavior)
            show = df_cmp.copy()
            show["매출"] = show["매출"].map(lambda x: f"{x:,.0f}")
            show["광고비"] = show["광고비"].map(lambda x: f"{x:,.0f}")
            show["영업이익"] = show["영업이익"].map(lambda x: f"{x:,.0f}")
            show["공헌이익률"] = show["공헌이익률"].map(lambda x: f"{x:.1f}%")
            show["ROAS"] = show["ROAS"].map(lambda x: f"{x:.2f}")
            st.dataframe(show, use_container_width=True, hide_index=True, key=mk("cmp_tbl"))

# End
