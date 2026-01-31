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

/* ✅ white-on-white 방지: 카드/데이터프레임/입력 위젯 텍스트 강제 다크 */
.card, .card * {{ color: #212529 !important; background: white; }}

div[data-testid="metric-container"] * {{ color: #212529 !important; }}

/* ---------- 기본 텍스트 ---------- */
html, body, [class*="css"] {{
  font-size: 14px;
}}

/* ---------- 카드(흰 배경) 전용: 카드 안의 글씨만 검정 ---------- */
.card {{
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 14px;
  padding: 14px 14px;
  background: white;
  color: #212529 !important;
}}
.card * {{
  color: #212529 !important;
  background: transparent;   /* ✅ 자식까지 흰색 배경 강제하지 말기(깨짐 방지) */
}}

/* ---------- 다크 UI 위젯(셀렉트/인풋) 글씨가 검게 먹는 문제 복구 ---------- */
/* 셀렉트 박스(드롭다운) */
div[data-baseweb="select"] * {{
  color: rgba(255,255,255,0.92) !important;
}}
/* 셀렉트/인풋의 배경(너무 검으면 살짝 띄움) */
div[data-baseweb="select"] > div {{
  background: rgba(255,255,255,0.06) !important;
}}

/* 텍스트 입력/숫자 입력 */
div[data-baseweb="base-input"] input,
div[data-baseweb="base-input"] textarea {{
  color: rgba(255,255,255,0.92) !important;
}}
div[data-baseweb="base-input"] > div {{
  background: rgba(255,255,255,0.06) !important;
}}

/* placeholder 글씨 */
input::placeholder, textarea::placeholder {{
  color: rgba(255,255,255,0.45) !important;
}}

/* 라벨(필드명) */
label, .stSelectbox label, .stTextInput label, .stNumberInput label {{
  color: rgba(255,255,255,0.70) !important;
}}
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
# Helpers
# =========================
def fmt_won(x) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):,.0f} 원"
    except Exception:
        return "-"
def top_key(d: Dict[str, float]) -> Tuple[Optional[str], float]:
    if not d:
        return None, 0.0
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return (items[0][0], float(items[0][1])) if items else (None, 0.0)

def detect_sales_archetype(rev_share: Dict[str, float], sales_focus: str = "(무관)") -> str:
    """
    rev_share key examples contain: 자사몰, 스마트스토어, 쿠팡, 홈쇼핑, 공구, B2B 등
    sales_focus(추천탭 입력)가 있으면 우선 반영, 없으면 rev_share로 추정
    """
    if sales_focus and sales_focus != "(무관)":
        # UI 입력을 그대로 archetype으로 사용
        if sales_focus in ["자사몰", "온라인(마켓)", "홈쇼핑", "공구", "B2B/도매"]:
            return sales_focus

    k, _ = top_key(rev_share)
    if not k:
        return "기타"

    k = str(k)
    if "자사" in k:
        return "자사몰"
    if "스마트" in k or "스스" in k or "스토어" in k:
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

def strategy_recommendation(
    rev_share: Dict[str, float],
    sales_focus: str = "(무관)",
) -> Dict[str, object]:
    """
    너가 말한 룰을 그대로 반영:
    - 자사몰 크면: 메타 비중 커야
    - 스마트스토어 크면: 네이버 비중
    - 쿠팡 크면: 외부몰PA가 제일 크고 그 다음 메타
    - 홈쇼핑 크면: 네이버SA/블로그 + 쿠팡PA가 커야, 이 경우 메타/구글은 안함
    - 공구(그룹바잉) 위주면: 인플루언서(인스타 메가)가 제일 커야
    """
    # 채널 존재 추정(키워드 매칭)
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

    archetype = detect_sales_archetype(rev_share, sales_focus=sales_focus)

    # 우선순위/룰 세팅
    if home >= max(own, smart, coupang, groupbuy) and home > 0:
        title = "홈쇼핑 연계형"
        priority = [
            ("Naver SA", "홈쇼핑 유입/검색 수요 회수 중심"),
            ("네이버 블로그·콘텐츠", "검색 신뢰/후기·정보성 보강"),
            ("쿠팡 PA", "방송 후 수요를 마켓에서 흡수"),
        ]
        note = "이 케이스는 약속 리스크를 줄이기 위해 메타/구글 집행은 제외(또는 최소)하는 방향을 권장"
    elif groupbuy >= max(own, smart, coupang, home) and groupbuy > 0:
        title = "공구(그룹바잉) 중심형"
        priority = [
            ("인플루언서(인스타 메가)", "공구는 매체 효율보다 ‘판매자 파워/신뢰’가 매출을 좌우"),
            ("바이럴(핫딜/커뮤니티)", "구매 트리거·확산"),
            ("외부몰/제휴 PA", "공구 외 추가 판매분 흡수"),
        ]
        note = "이 케이스는 퍼포먼스보다 ‘판매자/콘텐츠 드라이브’가 우선"
    elif coupang >= max(own, smart, home, groupbuy) and coupang > 0:
        title = "쿠팡(마켓) 중심형"
        priority = [
            ("외부몰 PA(쿠팡)", "가장 직접적인 매출 견인 레버"),
            ("메타", "리타겟/확장 및 수요 생성(보조)"),
            ("네이버 SA", "보조 검색 수요 회수"),
        ]
        note = "쿠팡 비중이 클수록 PA가 1순위, 그 다음 메타가 자연스러움"
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
            ("메타", "자사몰은 랜딩/리타겟 설계가 강점 → 매출 효율 기대"),
            ("Google(선택)", "검색 수요 회수(상품/브랜드 검색 중심)"),
            ("네이버 SA(선택)", "국내 검색 수요 보조"),
        ]
        note = "자사몰 매출 비중이 클수록 메타 비중을 키우는 룰이 가장 잘 맞음"
    else:
        title = "혼합형(균형 운영)"
        priority = [
            ("Naver SA", "기본 검색 수요 회수"),
            ("메타", "수요 생성/리타겟"),
            ("마켓 PA", "보유 채널에서 매출 흡수"),
        ]
        note = "매출 채널이 명확히 치우치지 않으면 3축 균형 운영을 권장"

    # 근거(상위 매출채널 Top3)
    top3 = sorted(rev_share.items(), key=lambda x: x[1], reverse=True)[:3]
    evidence = [f"{k}: {v*100:.1f}%" for k, v in top3 if v > 0]

    return {
        "title": title,
        "archetype": archetype,
        "priority": priority,
        "note": note,
        "evidence": evidence,
        "signals": {
            "자사몰": own, "스마트스토어": smart, "쿠팡": coupang, "홈쇼핑": home, "공구": groupbuy
        }
    }

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
    """
    If columns have '.1', '.2' duplicates (from Excel), drop duplicates keeping the first base.
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

    # xlsx
    try:
        xls = pd.ExcelFile(StringIO(""), engine=None)  # dummy to satisfy type checkers
    except Exception:
        pass

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
    file_bytes = uploaded_file.getvalue()
    return load_backdata_cached(file_bytes, uploaded_file.name)

# =========================
# Column detection (v4 with KPI)
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

    # revenue channel mix
    rev_cols = [c for c in df.columns if str(c).endswith("매출비중") and c not in [col_scn, col_disp]]

    # media mix (exclude KPI_ columns)
    perf_cols = [
        c for c in df.columns
        if (str(c).startswith("퍼포먼스마케팅_") or str(c) == "퍼포먼스_외부몰PA")
        and not str(c).startswith("KPI_")
    ]
    viral_cols = [c for c in df.columns if str(c).startswith("바이럴마케팅_") and not str(c).startswith("KPI_")]

    # brand mix columns (strict)
    brand_cols = []
    for c in df.columns:
        s = str(c)
        if s.startswith("KPI_"):
            continue
        if s in ["브랜드 마케팅", "기타_브랜드", "기타 브랜드", "기타_브랜드%"]:
            brand_cols.append(c)
        elif ("브랜드" in s and "마케팅" in s and not s.startswith("KPI_")):
            brand_cols.append(c)

    kpi_cols = [c for c in df.columns if str(c).startswith("KPI_")]

    # apply flags (optional)
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
        "kpi_cols": kpi_cols,
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
        if pd.isna(v):
            v = 0.0
        perf_raw[pretty_media_name(c)] = float(v)

    for c in viral_cols:
        v = normalize_ratio(row.get(c))
        if pd.isna(v):
            v = 0.0
        viral_raw[pretty_media_name(c)] = float(v)

    for c in brand_cols:
        v = normalize_ratio(row.get(c))
        if pd.isna(v):
            v = 0.0
        brand_raw[pretty_media_name(c)] = float(v)

    perf_sum = sum(v for v in perf_raw.values() if v > 0)
    viral_sum = sum(v for v in viral_raw.values() if v > 0)
    brand_sum = sum(v for v in brand_raw.values() if v > 0)
    total = perf_sum + viral_sum + brand_sum

    if total <= 0:
        group = {"퍼포먼스": 1.0, "바이럴": 0.0, "브랜드": 0.0}
    else:
        group = {"퍼포먼스": perf_sum / total, "바이럴": viral_sum / total, "브랜드": brand_sum / total}

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
    """
    media_full is original column name base: e.g. '퍼포먼스마케팅_네이버 SA'
    metric in {'CPM','CTR','CVR','CPC'}
    In your file: KPI_CPM_퍼포먼스마케팅_네이버 SA ...
    """
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
    """
    Blend CPC/CVR across performance mix shares.
    Uses KPI_CPC if exists; else CPM/CTR -> CPC.
    CVR uses KPI_CVR if exists.
    """
    # shares based on raw performance columns (not pretty names)
    raw = {}
    for c in perf_cols:
        v = normalize_ratio(row.get(c))
        if pd.isna(v):
            v = 0.0
        raw[str(c)] = float(v)
    shares = normalize_shares(raw)

    cpc_vals = []
    cvr_vals = []
    weights_cpc = []
    weights_cvr = []

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
            cpc_vals.append(cpc)
            weights_cpc.append(w)
        if cvr is not None and cvr > 0:
            cvr_vals.append(cvr)
            weights_cvr.append(w)

    def wavg(vals, ws):
        if not vals or not ws:
            return None
        s = sum(ws)
        if s <= 0:
            return None
        return float(sum(v * w for v, w in zip(vals, ws)) / s)

    return wavg(cpc_vals, weights_cpc), wavg(cvr_vals, weights_cvr)

# =========================
# P&L / Simulation (two-way)
# =========================
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
):
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
    roas = (revenue / ad_spend) if ad_spend and ad_spend > 0 else 0.0

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
# Agency media mix tables
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

    # sanitize
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
            cnt = int(np.round(planned / unit)) if unit > 0 else 0  # ✅ 정수(반올림)
            total_cost = cnt * unit  # ✅ 예산 mismatch 허용
            rows.append({
                "구분": "바이럴",
                "매체": medium,
                "지면": surface,
                "건당비용": unit,
                "진행 건수": cnt,
                "총비용(계획)": round_to_100(total_cost),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["매체", "지면"]).reset_index(drop=True)

# =========================
# Compare chart (bars + ROAS line / secondary axis 100~1000%)
# =========================
def compare_chart(df_cmp: pd.DataFrame, x_col: str, rev_col: str, ad_col: str, roas_col: str, height=420, title=""):
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_cmp[x_col],
        y=df_cmp[rev_col],
        name="예상매출",
        yaxis="y1",
        hovertemplate="%{y:,.0f}원<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=df_cmp[x_col],
        y=df_cmp[ad_col],
        name="예상광고비",
        yaxis="y1",
        hovertemplate="%{y:,.0f}원<extra></extra>"
    ))

    roas = df_cmp[roas_col].astype(float).fillna(0.0).clip(lower=0)

    fig.add_trace(go.Scatter(
        x=df_cmp[x_col],
        y=roas,
        name="ROAS",
        yaxis="y2",
        mode="lines+markers",
        hovertemplate="ROAS %{y:.2f}x (%{customdata:.0f}%)<extra></extra>",
        customdata=(roas * 100.0)
    ))

    # ✅ 보조축: 100%~1000% 범위(=1~10x) 중심
    y2_min, y2_max = 1.0, 10.0
    if roas.max() > y2_max:
        y2_max = float(np.ceil(roas.max()))
    if roas.min() < y2_min and roas.min() > 0:
        y2_min = float(max(0.5, np.floor(roas.min()*2)/2))  # 0.5 단위 완화

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
# Sidebar - Upload
# =========================
st.sidebar.title("마케팅/유통 시뮬레이터")

uploaded = st.sidebar.file_uploader(
    "Backdata 업로드 (xlsx/csv)",
    type=["xlsx", "csv"],
    key="backdata_uploader"  # ✅ 고정 key (업로드 안되는 현상 방지)
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

f_search = st.sidebar.text_input("검색(노출 시나리오명)", value="", key="f_search")
f_stage = st.sidebar.selectbox("단계(ST)", ["(전체)"] + uniq_vals(stage_col), key="f_stage")
f_cat = st.sidebar.selectbox("카테고리", ["(전체)"] + uniq_vals(cat_col), key="f_cat")
f_pos = st.sidebar.selectbox("가격 포지션(POS)", ["(전체)"] + uniq_vals(pos_col), key="f_pos")
f_drv = st.sidebar.selectbox("드라이버(DRV)", ["(전체)"] + uniq_vals(drv_col), key="f_drv")

# optional apply filters
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

if show_internal and apply_internal in df_f.columns:
    df_f = df_f[df_f[apply_internal].astype(str).str.strip().isin(["1","True","TRUE","Y","y","O","o"])]
if show_client and apply_client in df_f.columns:
    df_f = df_f[df_f[apply_client].astype(str).str.strip().isin(["1","True","TRUE","Y","y","O","o"])]
if show_agency and apply_agency in df_f.columns:
    df_f = df_f[df_f[apply_agency].astype(str).str.strip().isin(["1","True","TRUE","Y","y","O","o"])]

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

sel_disp = st.sidebar.selectbox("시나리오 선택", options=disp_candidates, key="sel_scn")

scenario_key = disp_to_key.get(sel_disp)
if scenario_key is None:
    # fallback
    scenario_key = next((k0 for k0, d0 in key_to_disp.items() if d0 == sel_disp), None)

if scenario_key is None:
    st.error("❌ 선택한 시나리오를 내부키로 매칭하지 못했습니다. (노출명 중복/매핑 확인)")
    st.stop()

row_df = df[df[col_scn].astype(str).str.strip() == str(scenario_key).strip()]
if row_df.empty:
    st.error("❌ 시나리오 행을 찾지 못했습니다.")
    st.stop()
row = row_df.iloc[0]

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
tab_guide, tab_agency, tab_brand, tab_rec = st.tabs(["가이드(사용법)", "대행", "브랜드사", "추천엔진"])


# =========================
# Tab: Agency
# =========================

with tab_guide:
    st.markdown("# 마케팅/유통 시뮬레이터 사용 가이드")

    st.markdown("## 1) 이 앱이 하는 일")
    st.write("""
- 백데이터(시나리오별 비중)를 업로드하면, 선택한 시나리오의 **매출채널 믹스 / 미디어 믹스**를 자동으로 불러옵니다.
- 입력한 가정(AOV, CPC, CVR 등)을 기반으로 **매출↔광고비를 상호 계산**합니다.
- 대행/브랜드사/추천엔진 모드에 따라, 보여주는 항목(정산/수수료/페이백/바이럴 건수 등)이 달라집니다.
""")

    st.markdown("## 2) 기본 흐름(처음 사용자 추천 순서)")
    st.write("""
1) 좌측에서 백데이터(xlsx/csv) 업로드  
2) 좌측에서 시나리오 필터 후 시나리오 선택  
3) '대행' 또는 '브랜드사' 탭에서 입력값(AOV/CPC/CVR 등) 설정  
4) 결과 요약 → 미디어 믹스(제안/정산) 확인  
5) 필요 시 '추천엔진'에서 조건 입력 → Top 추천 확인  
""")

    st.markdown("## 3) 계산 방식 설명")
    st.markdown("### A. 광고비 입력 → 매출 산출")
    st.write("""
- 광고비 → 클릭수 → 주문수 → 매출로 계산합니다.
- 클릭수 = 광고비 / CPC
- 주문수 = 클릭수 × CVR
- 매출 = 주문수 × AOV
""")

    st.markdown("### B. 매출 입력 → 필요 광고비 산출")
    st.write("""
- 목표 매출 → 주문수 → 클릭수 → 광고비로 역산합니다.
- 주문수 = 목표매출 / AOV
- 클릭수 = 주문수 / CVR
- 광고비 = 클릭수 × CPC
""")

    st.markdown("### C. ROAS")
    st.write("""
- ROAS = 매출 / 광고비
- 차트에서 매출/광고비와 축이 겹치지 않도록 ROAS는 **보조축(100%~1000%)**으로 표시합니다.
""")

    st.markdown("## 4) 미디어 믹스(제안/정산) 규칙")
    st.markdown("### A. 퍼포먼스 예산 배분")
    st.write("""
- 시나리오에 있는 퍼포먼스 매체 비중(100% 기준)을 사용해 예산을 배분합니다.
- 예산은 **100원 단위 반올림**합니다.
- 내부(운영/정산용)에서는:
  - 대행수수료율(%) 입력 → **청구예상비용** 반영
  - 페이백률(%) 입력 → **페이백예상액** 표시
""")

    st.markdown("### B. 바이럴 예산 → 건수 산출")
    st.write("""
- “예산을 지면별 비율로 배분 → 각 지면 건수 계산” 규칙을 사용합니다.
- 건수 = round(지면배정예산 / 건당비용) (소수 불가)
- 건수 반올림으로 인해 **총비용 합계가 예산과 일부 차이날 수 있으며 허용**합니다.
- 내부(운영/정산용)에서는:
  - 실집행비 입력 가능
  - 마진 = 계획비(또는 청구비) - 실집행비
""")

    st.markdown("## 5) 추천엔진(전략 추천) 기준")
    st.write("""
- 매출채널 비중(자사몰/스마트스토어/쿠팡/홈쇼핑/공구 등)에 따라 채널 우선순위를 룰 기반으로 추천합니다.
- 추가로 단계(ST)/카테고리/포지션/타겟연령/운영주체(내부/대행/브랜드사) 등을 반영해 Top N 시나리오를 점수화합니다.
""")

    st.markdown("## 6) 주의사항(외부 공유용)")
    st.info("이 화면의 수치는 입력값 기반 시뮬레이션이며, 실제 성과는 집행/운영 변수에 따라 달라질 수 있습니다.")


with tab_agency:
    st.markdown("## 대행 모드")
    submode = st.radio("버전 선택", ["외부(클라이언트 제안용)", "내부(운영/정산용)"], horizontal=True, key="agency_sub")

    st.markdown(
        f"<div class='smallcap'>선택 시나리오: <span class='badge badge-blue'>{sel_disp}</span></div>",
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 입력 (시뮬레이션)")
    use_scn_kpi = st.toggle("시나리오 KPI 자동 사용(권장)", value=True, key="use_scn_kpi_ag")

    cA, cB, cC, cD = st.columns(4)
    with cA:
        calc_mode = st.radio("계산 방식", ["광고비 입력 → 매출 산출", "매출 입력 → 필요 광고비 산출"], horizontal=True, key="calc_mode_ag")
    with cB:
        aov = st.number_input("객단가(AOV) (원)", value=50000, step=1000, key="aov_ag")
    with cC:
        cpc_manual = st.number_input("CPC (원) [수동]", value=300.0, step=10.0, key="cpc_ag")
    with cD:
        cvr_manual = st.number_input("CVR (%) [수동]", value=2.0, step=0.1, key="cvr_ag") / 100.0

    # blended KPI
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
        ad_total = st.number_input("총 광고비(원)", value=50000000, step=1000000, key="ad_total_ag")
        rev_target = None
    else:
        rev_target = st.number_input("목표 매출(원)", value=300000000, step=10000000, key="rev_target_ag")
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

    st.divider()

    st.markdown("### 결과 요약")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("예상 매출", fmt_won(sim["revenue"]))
    m2.metric("예상 광고비", fmt_won(sim["ad_spend"]))
    m3.metric("영업이익", fmt_won(sim["profit"]))
    m4.metric("ROAS", f"{sim['roas']:.2f}x ({sim['roas']*100:,.0f}%)")

    # 100% donuts
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.plotly_chart(
            donut_chart(
                ["퍼포먼스", "바이럴", "브랜드"],
                [group_share.get("퍼포먼스", 0), group_share.get("바이럴", 0), group_share.get("브랜드", 0)],
                title="광고비 구조(100%)",
                height=320
            ),
            use_container_width=True,
            key=f"donut_group_ag_{scenario_key}"
        )
    with gcol2:
        st.plotly_chart(
            donut_chart(list(rev_share.keys()), list(rev_share.values()), title="매출 채널 구성(100%)", height=320),
            use_container_width=True,
            key=f"donut_rev_ag_{scenario_key}"
        )

    st.divider()

    # =========== Media Mix Proposal (통합표) ===========
st.markdown("## 미디어 믹스 (제안/정산)")

perf_budget = float(sim["ad_spend"]) * float(group_share.get("퍼포먼스", 1.0))
viral_budget = float(sim["ad_spend"]) * float(group_share.get("바이럴", 0.0))

# 바이럴 단가표(기존 기능 유지)
with st.expander("바이럴 단가표(편집 가능)", expanded=False):
    st.caption("지면 단가/비율 수정 → 건수/총비용에 즉시 반영됩니다.")
    viral_price = st.data_editor(
        DEFAULT_VIRAL_PRICE.copy(),
        num_rows="dynamic",
        use_container_width=True,
        key=f"viral_price_editor_{scenario_key}"
    )

# 원본 계산(기존 로직 유지)
perf_df = build_performance_mix_table(media_share["perf"], perf_budget)
medium_share = viral_medium_shares(media_share["viral"])
viral_df = build_viral_mix_table(viral_price, medium_share, viral_budget)

# ✅ 퍼포먼스/바이럴 통합 테이블로 변환
mix_rows = []

# 퍼포먼스 -> 통합행
if not perf_df.empty:
    for _, r in perf_df.iterrows():
        mix_rows.append({
            "유형": "퍼포먼스",
            "구분": r.get("구분2", ""),
            "매체": r.get("매체", ""),
            "지면": "",
            "계획비(원)": float(r.get("예산(계획)", 0.0)),
            "목표 ROAS(%)": float(r.get("목표 ROAS(%)", 0.0)),
            "대행수수료율(%)": 0.0,
            "페이백률(%)": 0.0,
            "청구예상비용(원)": 0.0,
            "페이백예상액(원)": 0.0,
            "건당비용(원)": np.nan,
            "진행 건수": np.nan,
            "실집행비(원)": np.nan,
            "마진(원)": np.nan,
        })

# 바이럴 -> 통합행
if not viral_df.empty:
    for _, r in viral_df.iterrows():
        mix_rows.append({
            "유형": "바이럴",
            "구분": r.get("매체", ""),
            "매체": r.get("매체", ""),
            "지면": r.get("지면", ""),
            "계획비(원)": float(r.get("총비용(계획)", 0.0)),
            "목표 ROAS(%)": np.nan,
            "대행수수료율(%)": np.nan,
            "페이백률(%)": np.nan,
            "청구예상비용(원)": np.nan,
            "페이백예상액(원)": np.nan,
            "건당비용(원)": float(r.get("건당비용", np.nan)),
            "진행 건수": float(r.get("진행 건수", np.nan)),
            "실집행비(원)": 0.0 if submode.startswith("내부") else np.nan,
            "마진(원)": 0.0 if submode.startswith("내부") else np.nan,
        })

mix_df = pd.DataFrame(mix_rows)

if mix_df.empty:
    st.info("미디어 믹스 데이터가 비어있습니다(해당 시나리오 비율 0).")
else:
    # 보기 좋은 정렬
    mix_df["유형"] = pd.Categorical(mix_df["유형"], categories=["퍼포먼스", "바이럴"], ordered=True)
    mix_df = mix_df.sort_values(["유형", "구분", "매체", "지면"]).reset_index(drop=True)

    # 내부/외부 모드별 편집 정책
    if submode.startswith("내부"):
        st.markdown("### 통합 미디어 믹스 표 (내부 정산용)")
        # 편집 가능한 컬럼(열 단위: data_editor는 행 단위 disable이 불가하므로 열로 통제)
        editable_cols = [
            "유형", "구분", "매체", "지면", "계획비(원)",
            "목표 ROAS(%)", "대행수수료율(%)", "페이백률(%)",
            "청구예상비용(원)", "페이백예상액(원)",
            "건당비용(원)", "진행 건수",
            "실집행비(원)", "마진(원)"
        ]
        disabled_cols = [
            "유형", "구분", "매체", "지면",
            "계획비(원)", "청구예상비용(원)", "페이백예상액(원)",
            "건당비용(원)", "진행 건수", "마진(원)"
        ]

        edited = st.data_editor(
            mix_df[editable_cols],
            use_container_width=True,
            hide_index=True,
            disabled=disabled_cols,
            key=f"mix_editor_internal_{scenario_key}"
        )
        out = mix_df.copy()
        out.update(edited)

        # ✅ 퍼포먼스 정산 계산
        m_perf = out["유형"].astype(str) == "퍼포먼스"
        out.loc[m_perf, "대행수수료율(%)"] = out.loc[m_perf, "대행수수료율(%)"].apply(lambda x: to_float(x, 0.0))
        out.loc[m_perf, "페이백률(%)"] = out.loc[m_perf, "페이백률(%)"].apply(lambda x: to_float(x, 0.0))

        out.loc[m_perf, "청구예상비용(원)"] = out.loc[m_perf].apply(
            lambda r: round_to_100(to_float(r["계획비(원)"], 0.0) * (1.0 + to_float(r["대행수수료율(%)"], 0.0) / 100.0)),
            axis=1
        )
        out.loc[m_perf, "페이백예상액(원)"] = out.loc[m_perf].apply(
            lambda r: round_to_100(to_float(r["계획비(원)"], 0.0) * (to_float(r["페이백률(%)"], 0.0) / 100.0)),
            axis=1
        )

        # ✅ 바이럴 마진 계산
        m_viral = out["유형"].astype(str) == "바이럴"
        out.loc[m_viral, "실집행비(원)"] = out.loc[m_viral, "실집행비(원)"].apply(lambda x: to_float(x, 0.0))
        out.loc[m_viral, "마진(원)"] = out.loc[m_viral].apply(
            lambda r: round_to_100(to_float(r["계획비(원)"], 0.0) - to_float(r["실집행비(원)"], 0.0)),
            axis=1
        )

        st.dataframe(out[editable_cols], use_container_width=True, hide_index=True)

        # 합계 요약(기존 기능 유지/확장)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("퍼포먼스 계획비 합계", fmt_won(out.loc[m_perf, "계획비(원)"].sum()))
        s2.metric("퍼포먼스 청구예상 합계", fmt_won(out.loc[m_perf, "청구예상비용(원)"].sum()))
        s3.metric("바이럴 계획비 합계", fmt_won(out.loc[m_viral, "계획비(원)"].sum()))
        s4.metric("바이럴 마진 합계", fmt_won(out.loc[m_viral, "마진(원)"].sum()))

    else:
        st.markdown("### 통합 미디어 믹스 표 (외부 제안용)")
        view_cols = ["유형", "구분", "매체", "지면", "계획비(원)", "목표 ROAS(%)", "건당비용(원)", "진행 건수"]
        disabled_cols = ["유형", "구분", "매체", "지면", "계획비(원)", "건당비용(원)", "진행 건수"]

        # ✅ 여기서 ‘중복 표’가 생기지 않게: data_editor 하나만 사용 (dataframe로 재출력 X)
        st.data_editor(
            mix_df[view_cols],
            use_container_width=True,
            hide_index=True,
            disabled=disabled_cols,
            key=f"mix_editor_external_{scenario_key}"
        )

    # 도넛은 기존대로 유지(요청은 “표 통합”, 차트까지 합치라는 건 아니었음)
    cL, cR = st.columns(2)
    with cL:
        if not perf_df.empty:
            st.plotly_chart(
                donut_chart(
                    perf_df["매체"].tolist(),
                    perf_df["예산(계획)"].astype(float).tolist(),
                    title="퍼포먼스 예산 분배(100%)",
                    height=300
                ),
                use_container_width=True,
                key=f"donut_perf_{scenario_key}"
            )
    with cR:
        if not viral_df.empty:
            med_sum = viral_df.groupby("매체")["총비용(계획)"].sum().reset_index()
            st.plotly_chart(
                donut_chart(
                    med_sum["매체"].tolist(),
                    med_sum["총비용(계획)"].tolist(),
                    title="바이럴 예산 분배(100%)",
                    height=300
                ),
                use_container_width=True,
                key=f"donut_viral_{scenario_key}"
            )

    # ======================
    # Scenario compare (scenario KPI 반영)
    # ======================
    st.markdown("## 시나리오 비교 (매출/광고비 막대 + ROAS 꺾은선/보조축)")
    pick = st.multiselect("비교 시나리오 선택", options=disp_list, default=disp_list[:3], key="cmp_ag")

    if pick:
        rows_cmp = []
        for disp in pick:
            key_ = disp_to_key.get(disp)
            if key_ is None:
                continue
            rr_df = df[df[col_scn].astype(str).str.strip() == str(key_).strip()]
            if rr_df.empty:
                continue
            rr = rr_df.iloc[0]

            # KPI by scenario (or fallback to current input)
            cpc_s, cvr_s = blended_cpc_cvr(rr, perf_cols)
            cpc_use = cpc_s if (use_scn_kpi and cpc_s is not None) else float(cpc_manual)
            cvr_use = cvr_s if (use_scn_kpi and cvr_s is not None) else float(cvr_manual)

            sim_i = simulate_pl(
                calc_mode=calc_mode,
                aov=aov,
                cpc=cpc_use,
                cvr=cvr_use,
                cost_rate=cost_rate,
                logistics_per_order=logistics,
                fixed_cost=fixed_cost,
                ad_spend=ad_total,
                revenue=rev_target
            )
            rows_cmp.append({
                "시나리오": disp,
                "예상매출": sim_i["revenue"],
                "예상광고비": sim_i["ad_spend"],
                "ROAS": sim_i["roas"],
            })

        df_cmp = pd.DataFrame(rows_cmp)
        if not df_cmp.empty:
            st.plotly_chart(
                compare_chart(df_cmp, "시나리오", "예상매출", "예상광고비", "ROAS", title="시나리오 비교"),
                use_container_width=True,
                key="cmp_chart_ag"
            )

# =========================
# Tab: Brand
# =========================
with tab_brand:
    st.markdown("## 브랜드사 모드")
    submode_b = st.radio("버전 선택", ["외부(브랜드사 공유용)", "내부(브랜드 운영/검증용)"], horizontal=True, key="brand_sub")
    st.markdown(
        f"<div class='smallcap'>선택 시나리오: <span class='badge badge-blue'>{sel_disp}</span></div>",
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("### 월별 매출/광고비 전망")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        months = st.selectbox("기간(개월)", options=[3, 6, 12], index=2, key="b_months")
    with c2:
        base_month_rev = st.number_input("월 기준 매출(원)", value=200000000, step=10000000, key="b_base_rev")
    with c3:
        base_month_ad = st.number_input("월 기준 광고비(원)", value=50000000, step=1000000, key="b_base_ad")
    with c4:
        growth = st.number_input("월 성장률(%)", value=0.0, step=0.5, key="b_growth") / 100.0

    # internal-only (risk: external에는 과한 약속이 될 수 있어 숨김)
    if submode_b.startswith("내부"):
        st.markdown("### 내부 검증용 비용 입력(브랜드 내부만)")
        i1, i2, i3, i4 = st.columns(4)
        with i1:
            cost_rate_b = st.number_input("원가율(%)", value=30.0, step=1.0, key="b_cr") / 100.0
        with i2:
            logistics_b = st.number_input("물류비(건당) (원)", value=3000, step=500, key="b_logi")
        with i3:
            headcount_b = st.number_input("운영 인력(명)", value=2, step=1, min_value=0, key="b_hc")
        with i4:
            cost_per_b = st.number_input("인당 고정비(원)", value=3000000, step=100000, key="b_cper")
        fixed_b = float(headcount_b) * float(cost_per_b)
        aov_b = st.number_input("객단가(AOV) (원)", value=50000, step=1000, key="b_aov")
    else:
        cost_rate_b, logistics_b, fixed_b, aov_b = 0.0, 0.0, 0.0, 50000

    months_idx = list(range(1, int(months) + 1))
    rev_list, ad_list, roas_list = [], [], []
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

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("기간 총매출", fmt_won(df_m["예상매출"].sum()))
    k2.metric("기간 총광고비", fmt_won(df_m["예상광고비"].sum()))
    k3.metric("평균 ROAS", f"{df_m['ROAS'].mean():.2f}x ({df_m['ROAS'].mean()*100:,.0f}%)")

    if submode_b.startswith("내부"):
        total_orders = df_m["예상매출"].sum() / aov_b if aov_b > 0 else 0.0
        cogs = df_m["예상매출"].sum() * cost_rate_b
        logistics_cost = total_orders * logistics_b
        profit = df_m["예상매출"].sum() - (df_m["예상광고비"].sum() + cogs + logistics_cost + fixed_b)
        k4.metric("추정 영업이익(내부)", fmt_won(profit))
    else:
        k4.metric("요약", "월별 추세 확인")

    st.plotly_chart(
        compare_chart(df_m, "월", "예상매출", "예상광고비", "ROAS", title="월별 매출/광고비 + ROAS(보조축 100~1000%)"),
        use_container_width=True,
        key=f"brand_month_chart_{scenario_key}"
    )

    st.divider()

    # Channel recommendation (Top5)
    st.markdown("### 유통 채널 (매출 비중 Top5)")
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
            fig.update_layout(height=340, margin=dict(t=10), yaxis_title=None, xaxis_title=None)
            st.plotly_chart(fig, use_container_width=True, key=f"brand_top_bar_{scenario_key}")
        with cR:
            st.plotly_chart(
                donut_chart(top_df["채널"].tolist(), top_df["비중(%)"].tolist(), title="Top5 채널 구성(100%)", height=340),
                use_container_width=True,
                key=f"brand_top_donut_{scenario_key}"
            )

    if submode_b.startswith("외부"):
        st.markdown(
            "<div class='smallcap'>※ 본 화면의 수치는 입력값 기반 시뮬레이션이며, 실제 성과는 집행/운영 변수에 따라 달라질 수 있습니다.</div>",
            unsafe_allow_html=True
        )

# =========================
# Tab: Recommendation Engine
# =========================
with tab_rec:
    st.markdown("## 추천 엔진")
    st.markdown("<div class='smallcap'>backdata의 단계/카테고리/포지션/매출비중/미디어믹스를 기반으로 Top3를 추천합니다.</div>", unsafe_allow_html=True)

    # Inputs (make result appear below, not right)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        want_stage = st.selectbox("단계(ST)", ["(무관)"] + uniq_vals(stage_col), key="rec_stage")
    with c2:
        want_cat = st.selectbox("카테고리", ["(무관)"] + uniq_vals(cat_col), key="rec_cat")
    with c3:
        want_pos = st.selectbox("가격 포지션(POS)", ["(무관)"] + uniq_vals(pos_col), key="rec_pos")
    with c4:
        sales_focus = st.selectbox("판매 중심 채널", ["자사몰", "온라인(마켓)", "홈쇼핑", "공구", "B2B/도매", "(무관)"], key="rec_focus")

    c5, c6, c7 = st.columns(3)
    with c5:
        target_age = st.selectbox("타겟 연령대", ["10대/20대", "30대+", "(무관)"], key="rec_age")
    with c6:
        operator = st.selectbox("운영 주체", ["내부", "대행", "브랜드사", "(무관)"], key="rec_op")
    with c7:
        topn = st.selectbox("추천 개수", [3, 5, 10], index=0, key="rec_topn")

    run = st.button("추천 실행", use_container_width=True, key="rec_run")
    st.divider()
    st.markdown("### 전략 추천(채널 우선순위)")

    reco = strategy_recommendation(rev_share, sales_focus=sales_focus)

    st.markdown(
        f"<div class='card'>"
        f"<h3 style='margin:0;'>추천 전략: {reco['title']}</h3>"
        f"<div class='smallcap'>판단 근거(매출채널 Top): {' · '.join(reco['evidence']) if reco['evidence'] else '데이터 부족'}</div>"
        f"<hr class='soft'/>"
        f"<div style='font-weight:700;margin-bottom:6px;'>우선순위</div>"
        + "".join([f"<div>• <b>{a}</b> — {b}</div>" for a,b in reco["priority"]])
        + f"<hr class='soft'/>"
        f"<div class='smallcap'>메모: {reco['note']}</div>"
        f"</div>",
        unsafe_allow_html=True
    )
    def score_row(rr: pd.Series) -> float:
        score = 0.0

        # 1) stage/cat/pos match
        if want_stage != "(무관)" and stage_col in df.columns:
            score += 25 if str(rr.get(stage_col, "")).strip() == want_stage else 0
        if want_cat != "(무관)" and cat_col in df.columns:
            score += 25 if str(rr.get(cat_col, "")).strip() == want_cat else 0
        if want_pos != "(무관)" and pos_col in df.columns:
            score += 20 if str(rr.get(pos_col, "")).strip() == want_pos else 0

        # 2) sales focus match via revenue mix
        rs = build_rev_shares(rr, rev_cols)
        if sales_focus != "(무관)":
            best = 0.0
            for kname, vv in rs.items():
                if sales_focus in kname:
                    best = max(best, vv)
            score += best * 30.0

        # 3) age vs media tendency
        ms = build_media_shares(rr, perf_cols, viral_cols, brand_cols)
        perf = ms["perf"]
        viral = ms["viral"]
        if target_age == "10대/20대":
            score += (perf.get("틱톡", 0) + perf.get("메타", 0) + viral.get("인스타그램 시딩(메가)", 0)) * 20.0
        elif target_age == "30대+":
            score += (perf.get("네이버 SA", 0) + viral.get("네이버 블로그", 0)) * 20.0

        # 4) operator apply flags (if exists)
        if operator != "(무관)":
            if operator == "내부" and apply_internal in df.columns:
                score += 5 if str(rr.get(apply_internal, "")).strip() in ["1","True","TRUE","Y","y","O","o"] else 0
            if operator == "브랜드사" and apply_client in df.columns:
                score += 5 if str(rr.get(apply_client, "")).strip() in ["1","True","TRUE","Y","y","O","o"] else 0
            if operator == "대행" and apply_agency in df.columns:
                score += 5 if str(rr.get(apply_agency, "")).strip() in ["1","True","TRUE","Y","y","O","o"] else 0

        return float(score)

    if run:
        # candidate pool
        cand = []
        for _, rr in df.iterrows():
            cand.append((score_row(rr), rr))
        cand.sort(key=lambda x: x[0], reverse=True)
        top = cand[:int(topn)]

        st.divider()
        st.markdown("### 추천 결과")
        if not top:
            st.info("추천 후보가 없습니다. (데이터/컬럼 확인 필요)")
        else:
            cols_cards = st.columns(3)
            for i, (sc, rr) in enumerate(top[:3]):
                with cols_cards[i]:
                    disp = str(rr.get(col_disp, rr.get(col_scn, ""))).strip()
                    key_ = str(rr.get(col_scn, "")).strip()
                    # quick KPI info
                    cpc_s, cvr_s = blended_cpc_cvr(rr, perf_cols)
                    st.markdown(f"<div class='card'><h3 style='margin:0;'>#{i+1} {disp}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<div class='smallcap'>{key_}</div>", unsafe_allow_html=True)
                    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
                    st.metric("추천 점수", f"{sc:.1f}")
                    st.metric("시나리오 CPC(추정)", fmt_won(cpc_s) if cpc_s is not None else "-")
                    st.metric("시나리오 CVR(추정)", fmt_pct((cvr_s*100) if cvr_s is not None else np.nan, 1))
                    st.markdown("</div>", unsafe_allow_html=True)

            if len(top) > 3:
                st.markdown("### Top 리스트(요약)")
                rows = []
                for sc, rr in top:
                    rows.append({
                        "노출 시나리오명": str(rr.get(col_disp, "")).strip(),
                        "시나리오명": str(rr.get(col_scn, "")).strip(),
                        "점수": sc
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
