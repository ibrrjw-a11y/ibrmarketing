import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import re
from datetime import datetime
import uuid

# =========================
# Page / Theme
# =========================
st.set_page_config(page_title="마케팅/유통 시뮬레이터", layout="wide")

ACCENT = "#2F6FED"
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
.badge {{
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 12px;
}}
.badge-blue {{ background: rgba(47,111,237,0.12); color: rgb(47,111,237); }}
.badge-green {{ background: rgba(25,135,84,0.12); color: rgb(25,135,84); }}
.badge-red {{ background: rgba(220,53,69,0.12); color: rgb(220,53,69); }}
</style>
""", unsafe_allow_html=True)


# =========================
# Helpers
# =========================
def k(prefix: str) -> str:
    """unique key generator to avoid StreamlitDuplicateElementId"""
    return f"{prefix}__{uuid.uuid4().hex}"

def fmt_won(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):,.0f} 원"
    except:
        return "-"

def fmt_pct(x, digits=1):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):.{digits}f}%"
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
    """supports 0.32, 32, '32%', etc. -> returns 0~1"""
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

def donut_chart(labels, values, title="", height=320):
    df = pd.DataFrame({"name": labels, "value": values})
    fig = px.pie(df, names="name", values="value", hole=0.55)
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=height, margin=dict(t=40, b=10, l=10, r=10), title=title)
    return fig

def round_to_100(x):
    try:
        return int(np.round(float(x) / 100.0) * 100)
    except:
        return 0

def safe_col(df, name_candidates):
    cols = [str(c).strip() for c in df.columns]
    for cand in name_candidates:
        if cand in cols:
            return cand
    # fuzzy contains
    for cand in name_candidates:
        for c in cols:
            if cand in c:
                return c
    return None


# =========================
# Data loading (xlsx/csv)
# =========================
def drop_duplicate_dot_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If columns have '.1', '.2' duplicates (from Excel), drop duplicates keeping the first.
    Example: '퍼포먼스마케팅_구글 SA' and '퍼포먼스마케팅_구글 SA.1'
    """
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
    # also rename 'col.1' -> 'col' when base exists? (already kept base first)
    out.columns = [re.sub(r"\.\d+$", "", str(c)) for c in out.columns]
    return out

def load_backdata(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        raw = uploaded_file.getvalue().decode("utf-8-sig", errors="replace")
        df = pd.read_csv(StringIO(raw))
        df = df.dropna(how="all")
        return drop_duplicate_dot_columns(df)

    # xlsx
    try:
        xls = pd.ExcelFile(uploaded_file)
    except Exception as e:
        # openpyxl missing etc.
        raise RuntimeError(
            "엑셀(xlsx) 로드 실패. Streamlit Cloud라면 requirements.txt에 openpyxl 추가가 필요할 수 있습니다.\n"
            f"원인: {e}"
        )

    sheet = None
    for s in xls.sheet_names:
        if str(s).strip().lower() in ("backdata", "back_data", "back data", "backdata ", "backdata\n", "backdata\t"):
            sheet = s
            break
        if "backdata" in str(s).strip().lower():
            sheet = s
            break
        if "back" in str(s).strip().lower() and "data" in str(s).strip().lower():
            sheet = s
            break
    if sheet is None:
        # many of your files used 'BACKDATA'
        for s in xls.sheet_names:
            if str(s).strip().upper() == "BACKDATA":
                sheet = s
                break
    if sheet is None:
        sheet = xls.sheet_names[0]

    df = pd.read_excel(xls, sheet_name=sheet)
    df = df.dropna(how="all")
    df = drop_duplicate_dot_columns(df)
    # strip column names
    df.columns = [str(c).strip() for c in df.columns]
    return df


# =========================
# Column grouping (based on your v4 file)
# =========================
def detect_columns(df: pd.DataFrame):
    # scenario key / display name
    col_scn = safe_col(df, ["시나리오명", "scenario", "Scenario"])
    col_disp = safe_col(df, ["노출 시나리오명", "노출시나리오명", "display", "표시 시나리오명"])

    # fallback if missing
    if col_scn is None:
        col_scn = df.columns[0]
    if col_disp is None:
        # if 2nd col exists, treat as display
        col_disp = df.columns[1] if len(df.columns) > 1 else col_scn

    # meta filter columns
    col_stage = safe_col(df, ["단계(ST)", "단계", "ST"])
    col_drv = safe_col(df, ["드라이버(DRV)", "드라이버", "DRV"])
    col_cat = safe_col(df, ["카테고리(대)", "카테고리", "CAT"])
    col_pos = safe_col(df, ["가격포지션(POS)", "가격포지션", "POS"])

    # revenue channel mix: endswith '매출비중'
    rev_cols = [c for c in df.columns if str(c).endswith("매출비중") and c not in [col_scn, col_disp]]

    # media mix: performance/viral/brand
    perf_cols = [c for c in df.columns if str(c).startswith("퍼포먼스마케팅_") or str(c) in ["퍼포먼스_외부몰PA"]]
    viral_cols = [c for c in df.columns if str(c).startswith("바이럴마케팅_")]
    brand_cols = [c for c in df.columns if str(c) in ["브랜드 마케팅", "기타_브랜드", "기타 브랜드"] or ("브랜드" in str(c) and "마케팅" in str(c))]

    # KPI columns
    kpi_cols = [c for c in df.columns if str(c).startswith("KPI_")]

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
        "kpi_cols": kpi_cols,
    }


def scenario_options(df: pd.DataFrame, col_scn: str, col_disp: str):
    tmp = df[[col_scn, col_disp]].copy()
    tmp[col_scn] = tmp[col_scn].astype(str).str.strip()
    tmp[col_disp] = tmp[col_disp].astype(str).str.strip()
    tmp = tmp.dropna()
    # mapping
    key_to_disp = dict(zip(tmp[col_scn], tmp[col_disp]))
    disp_to_key = {}
    for kk, dd in key_to_disp.items():
        # handle duplicates display by appending short key
        if dd in disp_to_key and disp_to_key[dd] != kk:
            disp_to_key[f"{dd} ({kk})"] = kk
        else:
            disp_to_key[dd] = kk
    disp_list = list(disp_to_key.keys())
    disp_list.sort()
    return key_to_disp, disp_to_key, disp_list


# =========================
# Media name mapping (pretty)
# =========================
def pretty_media_name(col: str) -> str:
    c = str(col)
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
# Viral price table (default; you can edit in UI)
# =========================
DEFAULT_VIRAL_PRICE = pd.DataFrame([
    # medium, surface, unit_cost, weight
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
# Build shares from scenario row
# =========================
def build_rev_shares(row: pd.Series, rev_cols: list):
    d = {}
    for c in rev_cols:
        v = normalize_ratio(row.get(c))
        if pd.isna(v):
            v = 0.0
        d[str(c).replace(" 매출비중", "").replace("매출비중", "")] = float(v)
    return normalize_shares(d)

def build_media_shares(row: pd.Series, perf_cols: list, viral_cols: list, brand_cols: list):
    perf = {}
    viral = {}
    brand = {}

    for c in perf_cols:
        v = normalize_ratio(row.get(c))
        if pd.isna(v):
            v = 0.0
        perf[pretty_media_name(c)] = float(v)

    for c in viral_cols:
        v = normalize_ratio(row.get(c))
        if pd.isna(v):
            v = 0.0
        viral[pretty_media_name(c)] = float(v)

    for c in brand_cols:
        v = normalize_ratio(row.get(c))
        if pd.isna(v):
            v = 0.0
        brand[pretty_media_name(c)] = float(v)

    # totals
    perf_sum = sum(v for v in perf.values() if v > 0)
    viral_sum = sum(v for v in viral.values() if v > 0)
    brand_sum = sum(v for v in brand.values() if v > 0)
    total = perf_sum + viral_sum + brand_sum

    if total <= 0:
        group = {"퍼포먼스": 1.0, "바이럴": 0.0, "브랜드": 0.0}
    else:
        group = {"퍼포먼스": perf_sum / total, "바이럴": viral_sum / total, "브랜드": brand_sum / total}

    return {
        "group": group,
        "perf": normalize_shares(perf),
        "viral": normalize_shares(viral),
        "brand": normalize_shares(brand),
    }

def viral_medium_shares(viral_share_dict: dict):
    """
    scenario viral columns -> medium shares (네이버/인스타/오늘의집/기타 커뮤니티)
    """
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
# P&L / Simulation engine (two-way)
# =========================
def simulate_pl(
    calc_mode: str,
    aov: float,
    cpc: float,
    cvr: float,
    cost_rate: float,
    logistics_per_order: float,
    fixed_cost: float,
    ad_spend: float | None,
    revenue: float | None,
):
    """
    calc_mode:
      - "광고비 입력 → 매출 산출"
      - "매출 입력 → 필요 광고비 산출"
    """
    if calc_mode.startswith("매출"):
        revenue = float(revenue or 0.0)
        orders = revenue / aov if aov > 0 else 0.0
        clicks = orders / cvr if cvr > 0 else 0.0
        ad_spend = clicks * cpc
    else:
        ad_spend = float(ad_spend or 0.0)
        clicks = ad_spend / cpc if cpc > 0 else 0.0
        orders = clicks * cvr
        revenue = orders * aov

    cogs = revenue * cost_rate
    logistics = orders * logistics_per_order
    profit = revenue - (ad_spend + cogs + logistics + fixed_cost)
    contrib_margin = ((revenue - ad_spend - logistics - cogs) / revenue * 100) if revenue > 0 else 0.0
    roas = (revenue / ad_spend) if ad_spend > 0 else 0.0

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
    }


# =========================
# Agency media mix table builder
# =========================
def build_performance_mix_table(perf_share: dict, total_perf_budget: float):
    """
    returns DataFrame with:
      구분(퍼포먼스), 구분2(검색/디스플레이...), 매체, 예산(계획)
    """
    rows = []
    for media, share in perf_share.items():
        if share <= 0:
            continue
        budget = round_to_100(total_perf_budget * share)
        rows.append({
            "구분": "퍼포먼스",
            "구분2": perf_category(media),
            "매체": media,
            "예산(계획)": budget,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # subtotal rows
    df = df.sort_values(["구분2", "매체"]).reset_index(drop=True)
    return df

def build_viral_mix_table(
    viral_price_df: pd.DataFrame,
    medium_share: dict,
    total_viral_budget: float,
):
    """
    Budget -> medium allocation -> surface allocation by '비율'
    Count = round(planned / unit_cost) (integer)
    Total cost = count * unit_cost (budget mismatch allowed)
    """
    rows = []
    for medium, mshare in medium_share.items():
        medium_budget = total_viral_budget * mshare
        sub = viral_price_df[viral_price_df["매체"] == medium].copy()
        if sub.empty:
            continue
        sub["비율"] = sub["비율"].astype(float).fillna(1.0)
        sub_w = normalize_shares(dict(zip(sub["지면"], sub["비율"])))
        for surface, w in sub_w.items():
            unit = float(sub.loc[sub["지면"] == surface, "건당비용"].iloc[0])
            planned = medium_budget * w
            cnt = int(np.round(planned / unit)) if unit > 0 else 0
            total_cost = cnt * unit
            rows.append({
                "구분": "바이럴",
                "매체": medium,
                "지면": surface,
                "건당비용": unit,
                "진행 건수": cnt,
                "총비용(계획)": total_cost,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["매체", "지면"]).reset_index(drop=True)
    return df


# =========================
# Scenario compare chart (bars + ROAS line on secondary axis)
# =========================
def compare_chart(df_cmp: pd.DataFrame, roas_col="ROAS", height=420, title=""):
    """
    df_cmp columns: ["시나리오", "예상매출", "예상광고비", "ROAS"]
    ROAS displayed on secondary axis with fixed range 100%~1000% (i.e., 1~10)
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_cmp["시나리오"],
        y=df_cmp["예상매출"],
        name="예상매출",
        yaxis="y1",
        hovertemplate="%{y:,.0f}원<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=df_cmp["시나리오"],
        y=df_cmp["예상광고비"],
        name="예상광고비",
        yaxis="y1",
        hovertemplate="%{y:,.0f}원<extra></extra>"
    ))

    roas = df_cmp[roas_col].astype(float).fillna(0.0)
    roas = roas.clip(lower=0)

    fig.add_trace(go.Scatter(
        x=df_cmp["시나리오"],
        y=roas,
        name="ROAS",
        yaxis="y2",
        mode="lines+markers",
        hovertemplate="ROAS %{y:.2f}x (%{customdata:.0f}%)<extra></extra>",
        customdata=(roas * 100.0)
    ))

    # secondary axis range: 1~10 (100%~1000%)
    y2_min, y2_max = 1.0, 10.0
    # if outside range, expand a bit but keep readable
    if roas.max() > y2_max:
        y2_max = float(np.ceil(roas.max()))
    if roas.min() < y2_min and roas.min() > 0:
        y2_min = float(np.floor(roas.min()))

    fig.update_layout(
        height=height,
        barmode="group",
        title=title,
        margin=dict(t=50, b=10, l=10, r=10),
        yaxis=dict(title=None, tickformat=",.0f"),
        yaxis2=dict(
            title="ROAS (x)",
            overlaying="y",
            side="right",
            range=[y2_min, y2_max],
            tickformat=".0%",
        ),
        xaxis=dict(tickangle=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    # Plotly tickformat ".0%" expects 0~1 scale; we used x-scale.
    # So we'll manually set ticktext via tickvals:
    tickvals = list(np.linspace(y2_min, y2_max, 5))
    fig.update_layout(yaxis2=dict(
        tickmode="array",
        tickvals=tickvals,
        ticktext=[f"{v*100:.0f}%" for v in tickvals]
    ))
    return fig


# =========================
# Sidebar - Upload
# =========================
st.sidebar.title("마케팅/유통 시뮬레이터")
uploaded = st.sidebar.file_uploader("Backdata 업로드 (xlsx/csv)", type=["xlsx", "csv"], key=k("uploader"))

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

# make sure scenario cols exist
if col_scn not in df.columns:
    st.error("❌ '시나리오명' 컬럼을 찾지 못했습니다. (파일 컬럼명 확인 필요)")
    st.stop()
if col_disp not in df.columns:
    st.warning("⚠️ '노출 시나리오명' 컬럼이 없어, 시나리오명을 그대로 노출합니다.")
    df[col_disp] = df[col_scn].astype(str)

# scenario mappings
key_to_disp, disp_to_key, disp_list = scenario_options(df, col_scn, col_disp)

# filters
stage_col, drv_col, cat_col, pos_col = cols["stage"], cols["drv"], cols["cat"], cols["pos"]

def uniq_vals(c):
    if c is None or c not in df.columns:
        return []
    return sorted([x for x in df[c].dropna().astype(str).unique().tolist() if str(x).strip() != ""])

st.sidebar.markdown("---")
st.sidebar.markdown("### 시나리오 필터")
f_search = st.sidebar.text_input("검색(노출 시나리오명)", value="", key=k("search"))

f_stage = st.sidebar.selectbox("단계(ST)", ["(전체)"] + uniq_vals(stage_col), key=k("f_stage"))
f_cat = st.sidebar.selectbox("카테고리", ["(전체)"] + uniq_vals(cat_col), key=k("f_cat"))
f_pos = st.sidebar.selectbox("가격 포지션(POS)", ["(전체)"] + uniq_vals(pos_col), key=k("f_pos"))
f_drv = st.sidebar.selectbox("드라이버(DRV)", ["(전체)"] + uniq_vals(drv_col), key=k("f_drv"))

# build filtered display list
df_f = df.copy()
if f_stage != "(전체)" and stage_col in df_f.columns:
    df_f = df_f[df_f[stage_col].astype(str) == f_stage]
if f_cat != "(전체)" and cat_col in df_f.columns:
    df_f = df_f[df_f[cat_col].astype(str) == f_cat]
if f_pos != "(전체)" and pos_col in df_f.columns:
    df_f = df_f[df_f[pos_col].astype(str) == f_pos]
if f_drv != "(전체)" and drv_col in df_f.columns:
    df_f = df_f[df_f[drv_col].astype(str) == f_drv]

disp_candidates = []
for _, r in df_f[[col_scn, col_disp]].dropna().iterrows():
    disp_candidates.append(str(r[col_disp]).strip())
disp_candidates = sorted(list(set(disp_candidates)))

if f_search.strip():
    s = f_search.strip()
    disp_candidates = [x for x in disp_candidates if s in x]

if not disp_candidates:
    st.sidebar.warning("필터 결과가 없습니다. 필터를 완화하세요.")
    disp_candidates = disp_list

sel_disp = st.sidebar.selectbox("시나리오 선택", options=disp_candidates, key=k("sel_scn"))
scenario_key = disp_to_key.get(sel_disp, None)
if scenario_key is None:
    # fallback: try reverse map by exact
    scenario_key = next((k0 for k0, d0 in key_to_disp.items() if d0 == sel_disp), None)

if scenario_key is None:
    st.error("❌ 선택한 시나리오를 내부키로 매칭하지 못했습니다. (노출명 중복/매핑 확인)")
    st.stop()

row = df[df[col_scn].astype(str).str.strip() == str(scenario_key).strip()]
if row.empty:
    st.error("❌ 시나리오 행을 찾지 못했습니다.")
    st.stop()
row = row.iloc[0]

# group columns
rev_cols = cols["rev_cols"]
perf_cols = cols["perf_cols"]
viral_cols = cols["viral_cols"]
brand_cols = cols["brand_cols"]

rev_share = build_rev_shares(row, rev_cols)
media_share = build_media_shares(row, perf_cols, viral_cols, brand_cols)
group_share = media_share["group"]

# =========================
# Tabs
# =========================
tab_agency, tab_brand, tab_rec = st.tabs(["대행", "브랜드사", "추천엔진"])


# =========================
# Tab: Agency (internal/external)
# =========================
with tab_agency:
    st.markdown("## 대행 모드")
    submode = st.radio("버전 선택", ["외부(클라이언트 제안용)", "내부(운영/정산용)"], horizontal=True, key=k("agency_sub"))

    st.markdown(f"<div class='smallcap'>선택 시나리오: <span class='badge badge-blue'>{sel_disp}</span></div>", unsafe_allow_html=True)

    st.divider()

    # Common inputs
    cA, cB, cC, cD = st.columns(4)
    with cA:
        calc_mode = st.radio("계산 방식", ["광고비 입력 → 매출 산출", "매출 입력 → 필요 광고비 산출"], horizontal=True, key=k("calc_mode_ag"))
    with cB:
        aov = st.number_input("객단가(AOV) (원)", value=50000, step=1000, key=k("aov_ag"))
    with cC:
        cpc = st.number_input("CPC (원)", value=300.0, step=10.0, key=k("cpc_ag"))
    with cD:
        cvr = st.number_input("CVR (%)", value=2.0, step=0.1, key=k("cvr_ag")) / 100.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cost_rate = st.number_input("원가율(%)", value=30.0, step=1.0, key=k("cr_ag")) / 100.0
    with c2:
        logistics = st.number_input("물류비(건당) (원)", value=3000, step=500, key=k("logi_ag"))
    with c3:
        # 내부용은 인건비 입력만 받고 화면에는 과하게 노출 안함
        headcount = st.number_input("운영 인력(명)", value=2, step=1, min_value=0, key=k("hc_ag"))
    with c4:
        cost_per = st.number_input("인당 고정비(원)", value=3000000, step=100000, key=k("cper_ag"))

    fixed_cost = float(headcount) * float(cost_per)

    if calc_mode.startswith("광고비"):
        ad_total = st.number_input("총 광고비(원)", value=50000000, step=1000000, key=k("ad_total_ag"))
        rev_target = None
    else:
        rev_target = st.number_input("목표 매출(원)", value=300000000, step=10000000, key=k("rev_target_ag"))
        ad_total = None

    sim = simulate_pl(
        calc_mode=calc_mode,
        aov=aov,
        cpc=cpc,
        cvr=cvr,
        cost_rate=cost_rate,
        logistics_per_order=logistics,
        fixed_cost=fixed_cost,
        ad_spend=ad_total,
        revenue=rev_target
    )

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("예상 매출", fmt_won(sim["revenue"]))
    m2.metric("예상 광고비", fmt_won(sim["ad_spend"]))
    m3.metric("영업이익", fmt_won(sim["profit"]))
    m4.metric("ROAS", f"{sim['roas']:.2f}x ({sim['roas']*100:,.0f}%)")

    st.divider()

    # 100% donut: group mix
    gcol1, gcol2 = st.columns([1, 1])
    with gcol1:
        st.plotly_chart(
            donut_chart(
                ["퍼포먼스", "바이럴", "브랜드"],
                [group_share.get("퍼포먼스", 0), group_share.get("바이럴", 0), group_share.get("브랜드", 0)],
                title="광고비 구조(100%)",
                height=320
            ),
            use_container_width=True,
            key=k("donut_group_ag")
        )
    with gcol2:
        # revenue channel donut
        r_labels = list(rev_share.keys())
        r_vals = list(rev_share.values())
        st.plotly_chart(
            donut_chart(r_labels, r_vals, title="매출 채널 구성(100%)", height=320),
            use_container_width=True,
            key=k("donut_rev_ag")
        )

    st.divider()

    # =========== Media Mix Proposal ===========
    st.markdown("### 미디어 믹스 제안 (퍼포먼스 / 바이럴)")
    perf_budget = sim["ad_spend"] * group_share.get("퍼포먼스", 1.0)
    viral_budget = sim["ad_spend"] * group_share.get("바이럴", 0.0)

    perf_df = build_performance_mix_table(media_share["perf"], perf_budget)

    # Viral price editor
    with st.expander("바이럴 단가표(편집 가능)", expanded=False):
        st.caption("지면 단가는 여기서 수정하면, 건수/총비용 산출에 바로 반영됩니다.")
        viral_price = st.data_editor(
            DEFAULT_VIRAL_PRICE.copy(),
            num_rows="dynamic",
            use_container_width=True,
            key=k("viral_price_editor")
        )

    medium_share = viral_medium_shares(media_share["viral"])
    viral_df = build_viral_mix_table(viral_price, medium_share, viral_budget)

    # Performance table (external/internal)
    st.markdown("#### 퍼포먼스")
    if perf_df.empty:
        st.info("퍼포먼스 믹스 데이터가 비어있습니다(해당 시나리오 비율 0).")
    else:
        # target ROAS + fee/payback inputs
        perf_df = perf_df.copy()
        perf_df["목표 ROAS(%)"] = 0

        if submode.startswith("내부"):
            perf_df["대행수수료율(%)"] = 0.0
            perf_df["청구예상비용"] = 0
            perf_df["페이백률(%)"] = 0.0
            perf_df["페이백예상액"] = 0

            # editable
            edited = st.data_editor(
                perf_df[["구분", "구분2", "매체", "예산(계획)", "목표 ROAS(%)", "대행수수료율(%)", "페이백률(%)"]],
                use_container_width=True,
                key=k("perf_editor_internal"),
                disabled=["구분", "구분2", "매체"]
            )
            # compute billing/payback
            perf_out = perf_df.copy()
            perf_out.update(edited)

            perf_out["청구예상비용"] = perf_out.apply(
                lambda r: round_to_100(float(r["예산(계획)"]) * (1.0 + float(r["대행수수료율(%)"]) / 100.0)), axis=1
            )
            perf_out["페이백예상액"] = perf_out.apply(
                lambda r: round_to_100(float(r["예산(계획)"]) * (float(r["페이백률(%)"]) / 100.0)), axis=1
            )

            st.dataframe(
                perf_out[["구분2", "매체", "예산(계획)", "목표 ROAS(%)", "대행수수료율(%)", "청구예상비용", "페이백률(%)", "페이백예상액"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            edited = st.data_editor(
                perf_df[["구분", "구분2", "매체", "예산(계획)", "목표 ROAS(%)"]],
                use_container_width=True,
                key=k("perf_editor_external"),
                disabled=["구분", "구분2", "매체"]
            )
            st.dataframe(edited, use_container_width=True, hide_index=True)

        st.plotly_chart(
            donut_chart(
                perf_df["매체"].tolist(),
                perf_df["예산(계획)"].astype(float).tolist(),
                title="퍼포먼스 예산 분배(100%)",
                height=320
            ),
            use_container_width=True,
            key=k("donut_perf_only")
        )

    st.divider()

    # Viral table (external/internal)
    st.markdown("#### 바이럴")
    if viral_df.empty:
        st.info("바이럴 믹스 데이터가 비어있습니다(해당 시나리오 비율 0).")
    else:
        viral_df = viral_df.copy()
        viral_df["총비용(계획)"] = viral_df["총비용(계획)"].astype(float).apply(round_to_100)
        if submode.startswith("내부"):
            # allow actual cost input, margin = planned(billed) - actual
            viral_df["실집행비(원)"] = 0
            viral_df["마진(원)"] = 0

            edited = st.data_editor(
                viral_df[["매체", "지면", "건당비용", "진행 건수", "총비용(계획)", "실집행비(원)"]],
                use_container_width=True,
                key=k("viral_editor_internal"),
                disabled=["매체", "지면", "건당비용", "진행 건수", "총비용(계획)"]
            )
            outv = viral_df.copy()
            outv.update(edited)
            outv["마진(원)"] = outv["총비용(계획)"].astype(float) - outv["실집행비(원)"].astype(float)

            st.dataframe(
                outv[["매체", "지면", "건당비용", "진행 건수", "총비용(계획)", "실집행비(원)", "마진(원)"]],
                use_container_width=True,
                hide_index=True
            )

            # summary margin
            cS1, cS2, cS3 = st.columns(3)
            cS1.metric("바이럴 계획비 합계", fmt_won(outv["총비용(계획)"].sum()))
            cS2.metric("바이럴 실집행 합계", fmt_won(outv["실집행비(원)"].sum()))
            cS3.metric("바이럴 마진 합계", fmt_won(outv["마진(원)"].sum()))
        else:
            st.dataframe(
                viral_df[["매체", "지면", "건당비용", "진행 건수", "총비용(계획)"]],
                use_container_width=True,
                hide_index=True
            )

        # donut by medium (100%)
        med_sum = viral_df.groupby("매체")["총비용(계획)"].sum().reset_index()
        st.plotly_chart(
            donut_chart(med_sum["매체"].tolist(), med_sum["총비용(계획)"].tolist(), title="바이럴 예산 분배(100%)", height=320),
            use_container_width=True,
            key=k("donut_viral_only")
        )

    st.divider()

    # Scenario compare
    st.markdown("### 시나리오 비교 (매출/광고비 막대 + ROAS 꺾은선/보조축)")
    pick = st.multiselect("비교 시나리오 선택", options=disp_list, default=disp_list[:3], key=k("cmp_ag"))
    if pick:
        rows_cmp = []
        for disp in pick:
            key_ = disp_to_key.get(disp, None)
            if key_ is None:
                continue
            # NOTE: same inputs, only scenario mix differs in mix tables.
            # P&L itself doesn't use scenario row except mix display. So compare uses same sim results.
            # If you want scenario-specific KPI to affect revenue later, we can wire KPI columns next.
            rows_cmp.append({
                "시나리오": disp,
                "예상매출": sim["revenue"],
                "예상광고비": sim["ad_spend"],
                "ROAS": sim["roas"],
            })
        df_cmp = pd.DataFrame(rows_cmp)
        if not df_cmp.empty:
            st.plotly_chart(compare_chart(df_cmp, title="시나리오 비교"), use_container_width=True, key=k("cmp_chart_ag"))


# =========================
# Tab: Brand (internal/external)
# =========================
with tab_brand:
    st.markdown("## 브랜드사 모드")
    submode_b = st.radio("버전 선택", ["외부(브랜드사 공유용)", "내부(브랜드 운영/검증용)"], horizontal=True, key=k("brand_sub"))
    st.markdown(f"<div class='smallcap'>선택 시나리오: <span class='badge badge-blue'>{sel_disp}</span></div>", unsafe_allow_html=True)

    st.divider()

    # Brand mode inputs (simplified + monthly projection)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        months = st.selectbox("기간(개월)", options=[3, 6, 12], index=2, key=k("b_months"))
    with c2:
        base_month_rev = st.number_input("월 기준 매출(원)", value=200000000, step=10000000, key=k("b_base_rev"))
    with c3:
        base_month_ad = st.number_input("월 기준 광고비(원)", value=50000000, step=1000000, key=k("b_base_ad"))
    with c4:
        growth = st.number_input("월 성장률(%)", value=0.0, step=0.5, key=k("b_growth")) / 100.0

    # internal-only cost inputs
    if submode_b.startswith("내부"):
        cI1, cI2, cI3 = st.columns(3)
        with cI1:
            cost_rate_b = st.number_input("원가율(%)", value=30.0, step=1.0, key=k("b_cr")) / 100.0
        with cI2:
            logistics_b = st.number_input("물류비(건당) (원)", value=3000, step=500, key=k("b_logi"))
        with cI3:
            headcount_b = st.number_input("운영 인력(명)", value=2, step=1, min_value=0, key=k("b_hc"))
        cost_per_b = st.number_input("인당 고정비(원)", value=3000000, step=100000, key=k("b_cper"))
        fixed_b = float(headcount_b) * float(cost_per_b)
        # approximate orders from AOV
        aov_b = st.number_input("객단가(AOV) (원)", value=50000, step=1000, key=k("b_aov"))
    else:
        cost_rate_b, logistics_b, fixed_b, aov_b = 0.0, 0.0, 0.0, 50000

    # monthly table
    months_idx = list(range(1, int(months) + 1))
    rev_list = []
    ad_list = []
    roas_list = []
    for i in months_idx:
        factor = (1.0 + growth) ** (i - 1)
        rev_i = base_month_rev * factor
        ad_i = base_month_ad * factor
        rev_list.append(rev_i)
        ad_list.append(ad_i)
        roas_list.append((rev_i / ad_i) if ad_i > 0 else 0.0)

    df_m = pd.DataFrame({
        "월": [f"M{i}" for i in months_idx],
        "예상매출": rev_list,
        "예상광고비": ad_list,
        "ROAS": roas_list,
    })

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("기간 총매출", fmt_won(df_m["예상매출"].sum()))
    k2.metric("기간 총광고비", fmt_won(df_m["예상광고비"].sum()))
    k3.metric("평균 ROAS", f"{df_m['ROAS'].mean():.2f}x ({df_m['ROAS'].mean()*100:,.0f}%)")
    if submode_b.startswith("내부"):
        # rough P&L
        # orders approx from AOV
        total_orders = df_m["예상매출"].sum() / aov_b if aov_b > 0 else 0.0
        cogs = df_m["예상매출"].sum() * cost_rate_b
        logistics_cost = total_orders * logistics_b
        profit = df_m["예상매출"].sum() - (df_m["예상광고비"].sum() + cogs + logistics_cost + fixed_b)
        k4.metric("추정 영업이익", fmt_won(profit))
    else:
        k4.metric("채널 추천 상위", "아래 차트 참조")

    st.divider()

    # monthly chart (bars + ROAS line / secondary axis)
    st.plotly_chart(compare_chart(df_m.rename(columns={"월": "시나리오"}).assign(시나리오=df_m["월"]),
                                  roas_col="ROAS",
                                  title="월별 매출/광고비 + ROAS(보조축)"),
                    use_container_width=True,
                    key=k("brand_month_chart"))

    st.divider()

    # Channel recommendation (top 5 from revenue mix)
    st.markdown("### 유통 채널 추천 (매출 비중 상위 5)")
    rev_items = sorted(rev_share.items(), key=lambda x: x[1], reverse=True)
    top5 = [(k, v) for k, v in rev_items if v > 0][:5]
    if not top5:
        st.info("매출 채널 비중 데이터가 비어있습니다.")
    else:
        top_df = pd.DataFrame({"채널": [x[0] for x in top5], "비중(%)": [x[1] * 100 for x in top5]})
        cL, cR = st.columns(2)
        with cL:
            fig = px.bar(top_df, x="채널", y="비중(%)", text="비중(%)")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=340, margin=dict(t=10), xaxis_tickangle=0, yaxis_title=None, xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True, key=k("brand_top_bar"))
        with cR:
            st.plotly_chart(
                donut_chart(top_df["채널"].tolist(), top_df["비중(%)"].tolist(), title="Top5 채널 구성", height=340),
                use_container_width=True,
                key=k("brand_top_donut")
            )

    # keep it "not too promising" for external
    if submode_b.startswith("외부"):
        st.markdown("<div class='smallcap'>※ 본 화면의 수치는 입력값 기반의 시뮬레이션이며, 실제 성과는 집행/운영 변수에 따라 달라질 수 있습니다.</div>", unsafe_allow_html=True)


# =========================
# Tab: Recommendation Engine (kept, but UI improved)
# =========================
with tab_rec:
    st.markdown("## 추천 엔진")
    st.markdown("<div class='smallcap'>데이터 기반 Top3 추천 (룰 기반 스코어링 + KPI 기반 예상 CAC)</div>", unsafe_allow_html=True)

    # --- Inputs (top) ---
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            operator = st.selectbox("운영 주체", ["내부브랜드 운영자", "브랜드사 운영자(클라이언트)", "대행사(마케팅만)"], key=k("op"))
        with c2:
            stage = st.selectbox("단계(ST)", ["NEW", "EARLY", "GROW", "MATURE"], key=k("st"))
        with c3:
            category = st.selectbox("카테고리", ["뷰티", "건강", "푸드", "리빙", "기타"], key=k("cat"))
        with c4:
            position = st.selectbox("가격 포지셔닝(POS)", ["L", "M", "P"], key=k("pos"))

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            sales_focus = st.selectbox("판매 중심 채널", ["자사몰 중심", "온라인 중심", "홈쇼핑 중심", "공구 중심", "B2B 중심"], key=k("sf"))
        with c6:
            target_age = st.selectbox("타겟 연령대", ["10대", "20대", "30대", "40대", "50대+"], key=k("age"))
        with c7:
            no_comp = st.toggle("경쟁키워드 판매의도 없음", value=True, key=k("no_comp"))
        with c8:
            total_ad_budget = st.number_input("총 광고예산(원)", value=50000000, step=1000000, min_value=1, key=k("rec_budget"))

        c9, c10 = st.columns(2)
        with c9:
            brand_keyword_level = st.selectbox(
                "브랜드 키워드(인지도) 검색량",
                ["매우낮음(~300)", "낮음(300~1,000)", "중간(1,000~4,000)", "높음(4,000~8,000)", "매우높음(8,000~)"],
                key=k("brand_kw")
            )
        with c10:
            include_viral_conv = st.toggle("바이럴 KPI 없더라도 전환 포함(권장X)", value=False, key=k("viral_conv"))

        run = st.button("Top3 추천", use_container_width=True, key=k("run_rec"))

    # --- Simple, practical rule-based recommendations (no JSON output) ---
    # NOTE: this is intentionally lightweight; later we can wire ST/CAT/POS from backdata directly.
    def quick_score(row_: pd.Series) -> float:
        score = 0.0
        # match stage/category/pos if available
        if cols["stage"] and cols["stage"] in df.columns:
            if str(row_.get(cols["stage"], "")).strip() == stage:
                score += 25
        if cols["cat"] and cols["cat"] in df.columns:
            if str(row_.get(cols["cat"], "")).strip() == category:
                score += 25
        if cols["pos"] and cols["pos"] in df.columns:
            if str(row_.get(cols["pos"], "")).strip() == position:
                score += 20

        # sales focus vs revenue mix
        rs = build_rev_shares(row_, rev_cols)
        focus_map = {
            "자사몰 중심": "자사몰",
            "온라인 중심": "온라인(마켓)",
            "홈쇼핑 중심": "홈쇼핑",
            "공구 중심": "공구",
            "B2B 중심": "B2B/도매",
        }
        tgt = focus_map.get(sales_focus)
        if tgt:
            # fuzzy key
            best = 0.0
            for kname, vv in rs.items():
                if tgt in kname:
                    best = max(best, vv)
            score += best * 30

        # operator
        if operator == "대행사(마케팅만)":
            score += 5

        # age -> prefer tiktok/insta vs naver
        ms = build_media_shares(row_, perf_cols, viral_cols, brand_cols)
        perf = ms["perf"]
        viral = ms["viral"]
        if target_age in ("10대", "20대"):
            score += (perf.get("틱톡", 0) + perf.get("메타", 0) + viral.get("인스타그램 시딩(메가)", 0)) * 20
        else:
            score += (perf.get("네이버 SA", 0) + viral.get("네이버 블로그", 0)) * 20

        return float(score)

    def expected_cac_simple(total_budget_: float, cpc_=300.0, cvr_=0.02):
        clicks = total_budget_ / cpc_ if cpc_ > 0 else 0
        conv = clicks * cvr_
        return (total_budget_ / conv) if conv > 0 else None, conv

    if run:
        cand = []
        for _, rr in df.iterrows():
            sc = quick_score(rr)
            cand.append((sc, rr))

        cand.sort(key=lambda x: x[0], reverse=True)
        top = cand[:3]

        st.divider()
        st.markdown("### 추천 결과 (Top3)")
        if not top:
            st.info("추천 후보가 없습니다. (데이터/컬럼 확인 필요)")
        else:
            # show as 3 cards in a row (readable)
            cards = st.columns(3)
            for i, (sc, rr) in enumerate(top):
                with cards[i]:
                    disp = str(rr.get(col_disp, rr.get(col_scn, ""))).strip()
                    key_ = str(rr.get(col_scn, "")).strip()

                    cac, conv = expected_cac_simple(total_ad_budget, cpc_=300.0, cvr_=0.02)
                    st.markdown(f"<div class='card'><h3 style='margin:0;'>#{i+1} {disp}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<div class='smallcap'>{key_}</div>", unsafe_allow_html=True)
                    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
                    st.metric("Score", f"{sc:.1f}")
                    st.metric("예상 CAC", fmt_won(cac))
                    st.metric("예상 전환(단순)", f"{conv:,.1f}")
                    st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("<div class='smallcap'>※ 추천엔진은 현재 '실무적 룰 기반'으로 구성되어 있고, 추후 KPI 섹션을 정확히 연결해 CAC를 더 정교하게 개선할 수 있습니다.</div>", unsafe_allow_html=True)
