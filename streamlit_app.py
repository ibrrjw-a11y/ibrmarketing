import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import re
import json

# =========================================================
# Page / Theme
# =========================================================
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
section.main > div {{
  gap: 2rem;
}}
.smallcap {{
  color: {MUTED};
  font-size: 12px;
}}
.badge {{
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 12px;
}}
.badge-green {{ background: rgba(25,135,84,0.12); color: rgb(25,135,84); }}
.badge-yellow {{ background: rgba(255,193,7,0.15); color: rgb(161,118,0); }}
.badge-red {{ background: rgba(220,53,69,0.12); color: rgb(220,53,69); }}
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
.kpirow {{
  display:flex; gap:10px; align-items:center; flex-wrap:wrap;
}}
.kpibox {{
  padding:8px 10px; border:1px solid rgba(0,0,0,0.08); border-radius:12px; background:white;
}}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# Helpers
# =========================================================
def fmt_won(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):,.0f} 원"
    except Exception:
        return "-"

def fmt_num(x, digits=1):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):,.{digits}f}"
    except Exception:
        return "-"

def to_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        s = str(x).strip().replace(",", "")
        s = s.replace("원", "").strip()
        if s.endswith("%"):
            s = s[:-1]
        if s == "":
            return default
        return float(s)
    except Exception:
        return default

def normalize_ratio(x):
    """Supports 0.32, 32, '32%', '0.32' etc -> 0~1"""
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

def safe_str_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

# =========================================================
# Loader: supports XLSX(all-in-one) and CSV(all-in-one),
# plus legacy "stacked" format fallback
# =========================================================
REQUIRED_ALLINONE = ["시나리오명", "노출 시나리오명"]

def read_uploaded(uploaded):
    name = (uploaded.name or "").lower()

    if name.endswith(".xlsx"):
        # read first sheet by default
        df = pd.read_excel(uploaded)
        df = safe_str_cols(df)
        return df, "xlsx"
    else:
        raw = uploaded.getvalue()
        text = raw.decode("utf-8-sig", errors="replace")
        # try header
        df = pd.read_csv(StringIO(text))
        df = safe_str_cols(df)
        return df, "csv"

# -------- legacy stacked preprocess (kept for compatibility) --------
def preprocess_stacked(df_raw: pd.DataFrame):
    """
    Stacked format: multiple sections vertically; each section begins with a header row
    whose first cell equals '시나리오명'. We split sections and try to classify them.
    """
    col0 = df_raw.iloc[:, 0].astype(str).str.strip()
    header_idx = df_raw.index[col0.eq("시나리오명")].tolist()
    if not header_idx:
        raise ValueError("스택형 데이터에서 '시나리오명' 헤더 행을 찾지 못했습니다.")

    sections = []
    for i, h in enumerate(header_idx):
        start = h
        end = header_idx[i + 1] - 1 if i + 1 < len(header_idx) else len(df_raw) - 1
        sec = df_raw.iloc[start : end + 1].copy()

        non_empty_cols = [c for c in sec.columns if not sec[c].isna().all()]
        sec = sec[non_empty_cols]

        header = sec.iloc[0].tolist()
        header = [str(x).strip() if pd.notna(x) else "" for x in header]

        seen = {}
        clean_header = []
        for j, nm in enumerate(header):
            if nm == "" or str(nm).lower().startswith("unnamed"):
                nm = f"_COL_{j+1}"
            if nm in seen:
                seen[nm] += 1
                nm = f"{nm}_{seen[nm]}"
            else:
                seen[nm] = 1
            clean_header.append(nm)

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

    for sec in sections:
        cols = sec.columns.tolist()
        # More strict heuristics
        if "ad_alloc" not in out and has_any(cols, ["네이버", "쿠팡"]) and has_any(cols, ["광고비", "%"]):
            out["ad_alloc"] = sec
            continue
        if "channel_mix" not in out and has_any(cols, ["스마트스토어", "올리브영", "백화점", "쿠팡", "자사몰", "홈쇼핑", "공구", "B2B"]):
            out["channel_mix"] = sec
            continue
        if "media_mix" not in out and (has_any(cols, ["퍼포먼스마케팅_", "바이럴마케팅_", "브랜드 마케팅"]) or has_any(cols, ["구글", "메타", "틱톡"])):
            out["media_mix"] = sec
            continue
        if "kpi" not in out and has_any(cols, ["CPC", "CTR", "CVR", "재구매율", "CPM", "CLICKRATE", "CONVRATE"]):
            out["kpi"] = sec
            continue

    return out

def scenario_list_from_df(df: pd.DataFrame):
    if df is None or "시나리오명" not in df.columns:
        return []
    s = df["시나리오명"].dropna().astype(str).str.strip()
    s = [x for x in s if x and x != "시나리오명"]
    return sorted(list(dict.fromkeys(s)))

# =========================================================
# Scenario Key Parser (your ST-...__DRV-... format)
# =========================================================
SCENARIO_KEY_RE = re.compile(
    r"^ST-(?P<st>NEW|EARLY|GROW|MATURE)__DRV-(?P<drv>[^_]+)__CAT-(?P<cat>.+?)__POS-(?P<pos>[LMP])$"
)

def parse_scenario_key(name: str):
    name = str(name or "").strip()
    m = SCENARIO_KEY_RE.match(name)
    if not m:
        return None
    return {"ST": m.group("st"), "DRV": m.group("drv"), "CAT": m.group("cat"), "POS": m.group("pos")}

# =========================================================
# KPI parsing (all-in-one: KPI_<TOKEN>_<MEDIA> columns)
# =========================================================
TOKEN_ALIASES = {
    "CPM": ["CPM"],
    "CTR": ["CTR", "CLICKRATE", "클릭률"],
    "CVR": ["CVR", "CONVRATE", "전환율"],
    "CPC": ["CPC"],
}

def pick_kpi_for_media_from_row(row: pd.Series, media: str):
    """
    Supports:
    - KPI_CPM_<media>, KPI_CTR_<media>, KPI_CVR_<media>, KPI_CPC_<media>
    - <media>_CPM, <media> CTR ... (legacy)
    - Fuzzy contains(media) and contains(token)
    Returns token->value (CTR/CVR are in 0~1)
    """
    if row is None:
        return {}

    idx = list(row.index.astype(str))
    out = {}

    for token, aliases in TOKEN_ALIASES.items():
        found = None

        # 1) exact: KPI_TOKEN_media
        for al in aliases:
            exact = f"KPI_{al}_{media}"
            if exact in idx:
                found = exact
                break
        # 2) exact: media_TOKEN
        if not found:
            for al in aliases:
                exact2 = f"{media}_{al}"
                if exact2 in idx:
                    found = exact2
                    break
        # 3) fuzzy: contains media and contains any alias (plus KPI_ optional)
        if not found:
            for c in idx:
                cc = str(c)
                if (media in cc) and any(al in cc for al in aliases):
                    found = c
                    break

        if found:
            v = to_float(row[found], default=np.nan)
            if not np.isnan(v):
                if token in ("CTR", "CVR") and v > 1:
                    v = v / 100.0
                out[token] = float(v)

    return out

def fallback_kpi_for_media(media: str):
    m = str(media or "")
    if ("네이버" in m and "SA" in m) or ("구글" in m and "SA" in m):
        return {"CPC": 900.0, "CVR": 0.03}
    if ("외부몰PA" in m) or ("쿠팡" in m):
        return {"CPC": 700.0, "CVR": 0.025}
    if ("메타" in m) or ("틱톡" in m) or ("크리테오" in m) or ("GDN" in m) or ("GFA" in m) or ("유튜브" in m):
        return {"CPM": 9000.0, "CTR": 0.012, "CVR": 0.02}
    return {"CPM": 10000.0, "CTR": 0.008, "CVR": 0.01}

def derive_cpc(kpi: dict):
    if kpi.get("CPC") and kpi["CPC"] > 0:
        return float(kpi["CPC"])
    cpm = kpi.get("CPM")
    ctr = kpi.get("CTR")
    if cpm and ctr and cpm > 0 and ctr > 0:
        return float(cpm) / (1000.0 * float(ctr))
    return None

# =========================================================
# Build mixes from all-in-one row
# =========================================================
def build_channel_mix_from_row(row: pd.Series):
    """
    Use all columns ending with '매출비중' (per your request).
    Returns normalized shares: {channel_col_name: share(0~1)}
    """
    if row is None:
        return {}

    cols = [c for c in row.index.astype(str) if str(c).endswith("매출비중")]
    raw = {}
    for c in cols:
        v = normalize_ratio(row.get(c))
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if float(v) > 0:
            raw[c.replace(" 매출비중", "").strip()] = float(v)
    return normalize_shares(raw)

def build_media_grouped_from_row(row: pd.Series):
    """
    Uses columns like:
      퍼포먼스마케팅_*, 바이럴마케팅_*, '브랜드 마케팅'
    IMPORTANT: exclude KPI_* columns.
    Returns:
      {
        "performance": {...},
        "viral": {...},
        "brand": {...},
        "_group_weights": {...}
      }
    """
    out = {"performance": {}, "viral": {}, "brand": {}, "_group_weights": {"performance": 0, "viral": 0, "brand": 0}}
    if row is None:
        return out

    cols = [c for c in row.index.astype(str) if c not in ("시나리오명", "노출 시나리오명") and not str(c).startswith("KPI_")]

    perf_cols = [c for c in cols if str(c).startswith("퍼포먼스마케팅_") or str(c).startswith("퍼포먼스_")]
    viral_cols = [c for c in cols if str(c).startswith("바이럴마케팅_")]
    # brand: allow '브랜드 마케팅' and '기타 브랜드' etc, but avoid '브랜드 키워드' 같은 입력이 있다면 필터 필요
    brand_cols = [c for c in cols if ("브랜드" in str(c) and ("마케팅" in str(c) or "브랜드 마케팅" in str(c))) or str(c).startswith("기타_브랜드")]

    perf_raw = {}
    for c in perf_cols:
        v = normalize_ratio(row.get(c))
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if float(v) > 0:
            perf_raw[c] = float(v)

    viral_raw = {}
    for c in viral_cols:
        v = normalize_ratio(row.get(c))
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if float(v) > 0:
            viral_raw[c] = float(v)

    brand_raw = {}
    for c in brand_cols:
        v = normalize_ratio(row.get(c))
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if float(v) > 0:
            brand_raw[c] = float(v)

    perf_total = sum(perf_raw.values())
    viral_total = sum(viral_raw.values())
    brand_total = sum(brand_raw.values())
    grand = perf_total + viral_total + brand_total

    out["performance"] = normalize_shares(perf_raw) if perf_raw else {}
    out["viral"] = normalize_shares(viral_raw) if viral_raw else {}
    out["brand"] = normalize_shares(brand_raw) if brand_raw else {}

    if grand > 0:
        out["_group_weights"]["performance"] = perf_total / grand
        out["_group_weights"]["viral"] = viral_total / grand
        out["_group_weights"]["brand"] = brand_total / grand

    return out

def overall_media_share(adg, media):
    gw = adg.get("_group_weights", {"performance": 0, "viral": 0, "brand": 0})
    if media in adg.get("performance", {}):
        return gw["performance"] * adg["performance"][media]
    if media in adg.get("viral", {}):
        return gw["viral"] * adg["viral"][media]
    if media in adg.get("brand", {}):
        return gw["brand"] * adg["brand"][media]
    return 0.0

# =========================================================
# Expected CAC from media mix + KPI
# =========================================================
def calc_expected_cac(total_budget, adg, kpi_row, include_viral_if_kpi_missing=False):
    """
    Mix-weighted estimate:
      clicks = sum(budget_i / CPC_i)
      conversions = sum(clicks_i * CVR_i)
      CAC = total_budget / conversions
    """
    if total_budget <= 0:
        return {"expected_clicks": 0.0, "expected_conversions": 0.0, "expected_CAC": None, "media_contrib": []}

    gw = adg.get("_group_weights", {"performance": 0, "viral": 0, "brand": 0})

    overall = {}
    for m, v in adg.get("performance", {}).items():
        overall[m] = overall.get(m, 0.0) + gw["performance"] * v
    for m, v in adg.get("viral", {}).items():
        overall[m] = overall.get(m, 0.0) + gw["viral"] * v
    for m, v in adg.get("brand", {}).items():
        overall[m] = overall.get(m, 0.0) + gw["brand"] * v

    overall = normalize_shares(overall)

    contrib = []
    total_clicks = 0.0
    total_convs = 0.0

    for media, share in overall.items():
        budget_i = total_budget * share
        if budget_i <= 0:
            continue

        kpi = pick_kpi_for_media_from_row(kpi_row, media)
        kpi_is_fallback = False
        if not kpi:
            kpi = fallback_kpi_for_media(media)
            kpi_is_fallback = True

        is_viral = str(media).startswith("바이럴마케팅_")
        if is_viral and kpi_is_fallback and not include_viral_if_kpi_missing:
            contrib.append(
                {
                    "channel": media,
                    "budget": budget_i,
                    "CPC": None,
                    "clicks": 0.0,
                    "conversions": 0.0,
                    "note": "viral_kpi_missing_excluded",
                }
            )
            continue

        cpc = derive_cpc(kpi)
        cvr = float(kpi.get("CVR", 0.0) or 0.0)
        if (cpc is None) or cpc <= 0 or cvr <= 0:
            contrib.append(
                {
                    "channel": media,
                    "budget": budget_i,
                    "CPC": cpc,
                    "clicks": 0.0,
                    "conversions": 0.0,
                    "note": "kpi_insufficient",
                }
            )
            continue

        clicks = budget_i / float(cpc)
        convs = clicks * float(cvr)

        total_clicks += clicks
        total_convs += convs

        contrib.append(
            {
                "channel": media,
                "budget": budget_i,
                "CPC": float(cpc),
                "clicks": clicks,
                "conversions": convs,
                "note": "fallback_kpi" if kpi_is_fallback else "ok",
            }
        )

    expected_cac = (total_budget / total_convs) if total_convs > 0 else None
    return {
        "expected_clicks": total_clicks,
        "expected_conversions": total_convs,
        "expected_CAC": expected_cac,
        "media_contrib": contrib,
    }

# =========================================================
# Recommendation scoring (rule-based + your earlier logic)
# =========================================================
WEIGHTS = {
    "channel_match": 45.0,
    "drv_bonus": 25.0,
    "channel_ad_link": 20.0,
    "demo_keyword": 10.0,
}

DRV_PRIMARY = {
    "자사몰 중심": "D2C",
    "온라인 중심": "COM",
    "홈쇼핑 중심": "HSP",
    "공구 중심": "GB",
    "B2B 중심": "B2B",
}
DRV_SECONDARY = {
    "자사몰 중심": "PERF",
    "온라인 중심": "PERF",
    "홈쇼핑 중심": None,
    "공구 중심": None,
    "B2B 중심": None,
}

LEVEL_SCORE = {
    "매우낮음(~3,000)": 0.0,
    "낮음(3,000~10,000)": 0.25,
    "중간(10,000~20,000)": 0.5,
    "높음(20,000~30,000)": 0.75,
    "매우높음(35,000~)": 1.0,
    "매우낮음(~300)": 0.0,
    "낮음(300~1,000)": 0.25,
    "중간(1,000~4,000)": 0.5,
    "높음(4,000~8,000)": 0.75,
    "매우높음(8,000~)": 1.0,
}

def score_channel_match(channel_mix_norm, sales_focus):
    # Here we match by "high share" on a group label; since we now use file channels,
    # we use simple keyword mapping.
    target_kw = {
        "자사몰 중심": ["자사몰"],
        "온라인 중심": ["온라인", "스마트스토어", "쿠팡", "오픈마켓", "마켓"],
        "홈쇼핑 중심": ["홈쇼핑"],
        "공구 중심": ["공구", "공동구매"],
        "B2B 중심": ["B2B", "도매"],
    }.get(sales_focus, [])

    if not target_kw:
        return 0.0

    # Take the max share among matched channels
    best = 0.0
    for ch, v in channel_mix_norm.items():
        if any(kw in ch for kw in target_kw):
            best = max(best, float(v))
    return float(best)

def score_drv_bonus(drv, sales_focus, operator):
    drv = str(drv or "").strip()
    primary = DRV_PRIMARY.get(sales_focus)
    secondary = DRV_SECONDARY.get(sales_focus)

    operator_bonus = 0.0
    if operator == "대행사(마케팅만)" and drv in ("PERF", "VIR", "COM", "D2C"):
        operator_bonus = 0.15

    if primary and drv == primary:
        return min(1.0, 1.0 + operator_bonus)
    if secondary and drv == secondary:
        return min(1.0, 0.6 + operator_bonus)
    return max(0.15, 0.25 + operator_bonus)

def score_channel_ad_link(channel_mix_norm, adg, sales_focus, online_market_focus):
    # Media keys (from your columns) — if your file uses different spellings, adjust here once.
    meta = overall_media_share(adg, "퍼포먼스마케팅_메타")
    ext_pa = overall_media_share(adg, "퍼포먼스_외부몰PA")
    naver_sa = overall_media_share(adg, "퍼포먼스마케팅_네이버 SA")
    google_sa = overall_media_share(adg, "퍼포먼스마케팅_구글 SA")
    naver_blog = overall_media_share(adg, "바이럴마케팅_네이버 블로그")
    ig_mega = overall_media_share(adg, "바이럴마케팅_인스타그램 씨딩(메가)")
    google_gdn = overall_media_share(adg, "퍼포먼스마케팅_구글 GDN")
    tiktok = overall_media_share(adg, "퍼포먼스마케팅_틱톡")

    score = 0.0
    if sales_focus == "자사몰 중심":
        score += min(1.0, meta * 3.0) * 0.6

    elif sales_focus == "온라인 중심":
        if online_market_focus == "스마트스토어 중심":
            score += min(1.0, naver_sa * 3.0) * 0.45
            score += min(1.0, meta * 3.0) * 0.35
            score += min(1.0, google_sa * 3.0) * 0.2
        else:  # 쿠팡/마켓 중심
            score += min(1.0, ext_pa * 3.0) * 0.6
            score += 0.4 if ext_pa >= meta else 0.15

    elif sales_focus == "홈쇼핑 중심":
        core = naver_sa + naver_blog + ext_pa
        score += min(1.0, core * 2.5) * 0.7
        penalty = meta + google_gdn + tiktok
        score += max(0.0, 1.0 - penalty * 2.0) * 0.3

    elif sales_focus == "공구 중심":
        # If 공구 share exists and IG mega exists
        score += min(1.0, score_channel_match(channel_mix_norm, "공구 중심") * 1.8) * 0.5
        score += min(1.0, ig_mega * 4.0) * 0.5

    elif sales_focus == "B2B 중심":
        brand_share = sum(adg.get("brand", {}).values()) if isinstance(adg.get("brand"), dict) else 0.0
        score += min(1.0, naver_sa * 3.0) * 0.6
        score += min(1.0, brand_share * 3.0) * 0.4

    return float(max(0.0, min(1.0, score)))

def score_demo_keyword(adg, payload):
    gw = adg.get("_group_weights", {"performance": 0, "viral": 0, "brand": 0})
    perf_sum = gw["performance"]
    viral_sum = gw["viral"]
    brand_sum = gw["brand"]

    no_comp = payload["no_competitor_intent"]
    comp_lv = payload.get("competitor_keyword_level")
    brand_lv = payload["brand_keyword_level"]
    age = payload["target_age"]

    score = 0.0
    if no_comp:
        score += min(1.0, (viral_sum + brand_sum) * 1.7) * 0.4
    else:
        comp_v = LEVEL_SCORE.get(comp_lv, 0.5)
        score += min(1.0, perf_sum * (1.0 + comp_v)) * 0.4

    brand_v = LEVEL_SCORE.get(brand_lv, 0.5)
    if brand_v <= 0.25:
        score += min(1.0, (viral_sum + brand_sum) * 1.5) * 0.3
    elif brand_v >= 0.75:
        score += min(1.0, perf_sum * 1.3) * 0.3
    else:
        score += 0.15

    if age in ("10대", "20대"):
        score += min(
            1.0,
            (
                overall_media_share(adg, "퍼포먼스마케팅_틱톡")
                + overall_media_share(adg, "바이럴마케팅_인스타그램 씨딩(메가)")
                + overall_media_share(adg, "바이럴마케팅_인스타그램 씨딩(노말)")
            )
            * 3.0,
        ) * 0.3
    else:
        score += min(
            1.0,
            (
                overall_media_share(adg, "퍼포먼스마케팅_네이버 SA")
                + overall_media_share(adg, "바이럴마케팅_네이버 블로그")
            )
            * 3.0,
        ) * 0.3

    return float(max(0.0, min(1.0, score)))

def build_why(channel_mix_norm, adg):
    top_rev = sorted(channel_mix_norm.items(), key=lambda x: x[1], reverse=True)[:3]
    rev_txt = ", ".join([f"{k} {v:.0%}" for k, v in top_rev if v > 0]) or "-"

    gw = adg.get("_group_weights", {"performance": 0, "viral": 0, "brand": 0})
    overall = {}
    for m, v in adg.get("performance", {}).items():
        overall[m] = overall.get(m, 0.0) + gw["performance"] * v
    for m, v in adg.get("viral", {}).items():
        overall[m] = overall.get(m, 0.0) + gw["viral"] * v
    for m, v in adg.get("brand", {}).items():
        overall[m] = overall.get(m, 0.0) + gw["brand"] * v
    overall = normalize_shares(overall)

    top_ad = sorted(overall.items(), key=lambda x: x[1], reverse=True)[:3]
    ad_txt = ", ".join([f"{k} {v:.0%}" for k, v in top_ad if v > 0]) or "-"

    return [
        f"매출채널 상위: {rev_txt}",
        f"미디어믹스 상위: {ad_txt}",
        f"그룹 비중: 퍼포 {gw.get('performance',0):.0%} / 바이럴 {gw.get('viral',0):.0%} / 브랜드 {gw.get('brand',0):.0%}",
    ]

def recommend_top3_allinone(payload, df_all: pd.DataFrame, key_to_label: dict):
    scenarios = scenario_list_from_df(df_all)
    meta_map = {s: parse_scenario_key(s) for s in scenarios}

    # hard filter first; if empty, fallback to all
    candidates = []
    for s in scenarios:
        m = meta_map.get(s)
        if m and m.get("ST"):
            if m["ST"] == payload["stage"] and m["CAT"] == payload["category"] and m["POS"] == payload["position"]:
                candidates.append(s)
    if not candidates:
        candidates = scenarios[:]

    results = []
    for s in candidates:
        row = df_all[df_all["시나리오명"].astype(str).str.strip() == str(s).strip()]
        if row.empty:
            continue
        row = row.iloc[0]

        m = meta_map.get(s) or {}
        drv = m.get("DRV")

        channel_mix_norm = build_channel_mix_from_row(row)
        adg = build_media_grouped_from_row(row)

        a = score_channel_match(channel_mix_norm, payload["sales_focus_channel"])
        b = score_drv_bonus(drv, payload["sales_focus_channel"], payload["operator"])
        c = score_channel_ad_link(channel_mix_norm, adg, payload["sales_focus_channel"], payload.get("online_market_focus"))
        d = score_demo_keyword(adg, payload)

        total = (
            a * WEIGHTS["channel_match"] +
            b * WEIGHTS["drv_bonus"] +
            c * WEIGHTS["channel_ad_link"] +
            d * WEIGHTS["demo_keyword"]
        ) / sum(WEIGHTS.values()) * 100.0

        expected = calc_expected_cac(
            total_budget=float(payload["total_ad_budget_krw"]),
            adg=adg,
            kpi_row=row,  # all-in-one row contains KPI columns
            include_viral_if_kpi_missing=bool(payload.get("include_viral_conversions_if_kpi_missing", False)),
        )

        results.append({
            "scenario_key": s,
            "scenario_label": key_to_label.get(s, s),
            "score": float(max(0.0, min(100.0, total))),
            "why": build_why(channel_mix_norm, adg),
            "expected_metrics": {
                "expected_clicks": expected["expected_clicks"],
                "expected_conversions": expected["expected_conversions"],
                "expected_CAC": expected["expected_CAC"],
                "media_contrib": expected["media_contrib"],
            }
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"input": payload, "candidate_count": len(candidates), "recommendations": results[:3]}

# =========================================================
# Sidebar: Upload
# =========================================================
st.sidebar.title("마케팅/유통 시뮬레이터")

with st.sidebar.expander("📌 안내 (권장 운영)", expanded=False):
    st.write("- **XLSX 권장**, CSV도 지원합니다.")
    st.write("- CSV는 **UTF-8-SIG** 저장을 권장합니다.")
    st.write("- Streamlit Cloud 사용 시 requirements.txt에 `openpyxl` 포함 필요할 수 있어요.")

uploaded = st.sidebar.file_uploader("Backdata 업로드 (XLSX/CSV)", type=["xlsx", "csv"])

if uploaded is None:
    st.info("좌측에서 backdata 파일(xlsx/csv)을 업로드하세요.")
    st.stop()

# =========================================================
# Load & Detect Format
# =========================================================
try:
    df_loaded, fmt = read_uploaded(uploaded)
except Exception as e:
    st.error(f"❌ 파일 로드 실패: {e}")
    st.stop()

is_all_in_one = all(c in df_loaded.columns for c in REQUIRED_ALLINONE)

data_mode = "all_in_one" if is_all_in_one else "stacked"

if data_mode == "all_in_one":
    df_all = df_loaded.copy()
    # normalize key/label
    df_all["시나리오명"] = df_all["시나리오명"].astype(str).str.strip()
    df_all["노출 시나리오명"] = df_all["노출 시나리오명"].astype(str).str.strip()

    scenarios = scenario_list_from_df(df_all)
    if not scenarios:
        st.error("❌ 시나리오 목록이 비어있습니다. (시나리오명 컬럼 확인)")
        st.stop()

    key_to_label = dict(zip(df_all["시나리오명"], df_all["노출 시나리오명"]))
    # if labels duplicated, append key for UI uniqueness
    label_counts = pd.Series(list(key_to_label.values())).value_counts().to_dict()
    key_to_label_ui = {}
    for k, v in key_to_label.items():
        if label_counts.get(v, 0) > 1:
            key_to_label_ui[k] = f"{v}  ({k})"
        else:
            key_to_label_ui[k] = v

    # category options from CAT in key
    parsed = [parse_scenario_key(s) for s in scenarios]
    cat_options = sorted(list({p["CAT"] for p in parsed if p and p.get("CAT")})) or ["(카테고리 파싱 실패)"]

else:
    # legacy stacked fallback (csv with header=None or badly formed)
    try:
        # for stacked, we need header=None reading
        if fmt == "csv":
            raw = uploaded.getvalue().decode("utf-8-sig", errors="replace")
            df_raw = pd.read_csv(StringIO(raw), header=None)
        else:
            df_raw = pd.read_excel(uploaded, header=None)
        sections = preprocess_stacked(df_raw)
    except Exception as e:
        st.error(f"❌ 스택형 파싱 실패: {e}")
        st.stop()

    # stitch minimal fields
    scenarios = set()
    for v in sections.values():
        if isinstance(v, pd.DataFrame) and "시나리오명" in v.columns:
            scenarios |= set(v["시나리오명"].dropna().astype(str).str.strip().tolist())
    scenarios = sorted([s for s in scenarios if s and s != "시나리오명"])
    if not scenarios:
        st.error("❌ 스택형 데이터에서 시나리오를 찾지 못했습니다.")
        st.stop()

    # In stacked, we don't have '노출 시나리오명'. Use key itself.
    key_to_label = {s: s for s in scenarios}
    key_to_label_ui = key_to_label.copy()
    parsed = [parse_scenario_key(s) for s in scenarios]
    cat_options = sorted(list({p["CAT"] for p in parsed if p and p.get("CAT")})) or ["(카테고리 파싱 실패)"]

# =========================================================
# Tabs
# =========================================================
tab_rec, tab_dash = st.tabs(["✅ 추천 엔진 (Top3 + CAC)", "📊 대시보드 (내부/브랜드사/대행사)"])

# =========================================================
# TAB 1) Recommendation Engine
# =========================================================
with tab_rec:
    st.markdown("## 추천 엔진 (Top3 + CAC 계산)")
    st.markdown('<div class="smallcap">ST/CAT/POS 필터(없으면 전체 fallback) → 룰 스코어링 → KPI 기반 예상 CAC</div>', unsafe_allow_html=True)

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown("### 입력 체크리스트")
        operator = st.selectbox("운영 주체", ["내부브랜드 운영자", "브랜드사 운영자(클라이언트)", "대행사(마케팅만)"])
        stage = st.selectbox("단계(ST)", ["NEW", "EARLY", "GROW", "MATURE"])
        category = st.selectbox("카테고리(CAT)", cat_options)
        position = st.selectbox("가격 포지셔닝(POS)", ["L", "M", "P"])
        sales_focus_channel = st.selectbox("판매 중심 채널", ["자사몰 중심", "온라인 중심", "홈쇼핑 중심", "공구 중심", "B2B 중심"])

        online_market_focus = None
        if sales_focus_channel == "온라인 중심":
            online_market_focus = st.selectbox(
                "온라인 마켓 포커스(옵션)",
                [None, "쿠팡 중심", "스마트스토어 중심"],
                format_func=lambda x: "미지정(자동)" if x is None else x,
            )

        no_comp = st.toggle("경쟁키워드 판매의도 없음", value=True)
        competitor_keyword_level = None
        if not no_comp:
            competitor_keyword_level = st.selectbox(
                "경쟁키워드 검색량 구간",
                ["매우낮음(~3,000)", "낮음(3,000~10,000)", "중간(10,000~20,000)", "높음(20,000~30,000)", "매우높음(35,000~)"],
            )

        brand_keyword_level = st.selectbox(
            "브랜드 키워드(인지도) 검색량 구간",
            ["매우낮음(~300)", "낮음(300~1,000)", "중간(1,000~4,000)", "높음(4,000~8,000)", "매우높음(8,000~)"],
        )

        target_age = st.selectbox("주요 타겟 연령대", ["10대", "20대", "30대", "40대", "50대+"])
        total_ad_budget_krw = st.number_input("총 광고예산(원)", value=50_000_000, step=1_000_000, min_value=1)

        include_viral_if_missing = st.toggle("바이럴 KPI 없더라도 전환 포함(권장X)", value=False)
        run = st.button("Top3 추천 + CAC 계산", use_container_width=True)

    with right:
        st.markdown("### 출력")
        if run:
            payload = {
                "operator": operator,
                "stage": stage,
                "category": category,
                "position": position,
                "sales_focus_channel": sales_focus_channel,
                "online_market_focus": online_market_focus,
                "no_competitor_intent": bool(no_comp),
                "competitor_keyword_level": competitor_keyword_level,
                "brand_keyword_level": brand_keyword_level,
                "target_age": target_age,
                "total_ad_budget_krw": float(total_ad_budget_krw),
                "include_viral_conversions_if_kpi_missing": bool(include_viral_if_missing),
            }

            if data_mode == "all_in_one":
                out = recommend_top3_allinone(payload=payload, df_all=df_all, key_to_label=key_to_label)
            else:
                st.warning("현재 업로드는 스택형 데이터로 인식되었습니다. 추천 엔진은 all-in-one에서 가장 정확합니다.")
                out = {"input": payload, "candidate_count": 0, "recommendations": []}

            st.metric("후보 시나리오 수", f"{out.get('candidate_count', 0):,} 개")
            recs = out.get("recommendations", [])

            if not recs:
                st.info("조건에 맞는 추천을 만들지 못했습니다. (데이터 구조/시나리오 키/컬럼 확인 필요)")
            else:
                for i, r in enumerate(recs, 1):
                    title = f"#{i}. {r['scenario_label']}"
                    sub = r["scenario_key"]
                    st.markdown(f"<div class='card'><h3 style='margin:0;'>{title}</h3><div class='smallcap'>{sub}</div>", unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Score", f"{r['score']:.1f}")
                    c2.metric("예상 전환(Conversions)", f"{r['expected_metrics']['expected_conversions']:.1f}")
                    c3.metric("예상 CAC", fmt_won(r['expected_metrics']['expected_CAC']))

                    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
                    st.write("**Why (3줄)**")
                    for line in r["why"]:
                        st.write(f"- {line}")

                    contrib = pd.DataFrame(r["expected_metrics"]["media_contrib"])
                    if not contrib.empty:
                        st.write("**매체별 기여(테이블)**")
                        disp = contrib.copy()
                        for col in ["budget", "clicks", "conversions", "CPC"]:
                            if col in disp.columns:
                                disp[col] = pd.to_numeric(disp[col], errors="coerce")
                        st.dataframe(disp, use_container_width=True, hide_index=True)

                        top_conv = disp.sort_values("conversions", ascending=False).head(12)
                        fig = px.bar(top_conv, x="channel", y="conversions", text="conversions")
                        fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                        fig.update_layout(height=320, xaxis_title=None, yaxis_title=None, margin=dict(t=10))
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                st.divider()
                st.write("**반환 JSON(복사/다운로드용)**")
                st.code(json.dumps(out, ensure_ascii=False, indent=2), language="json")

# =========================================================
# TAB 2) Dashboard
# =========================================================
with tab_dash:
    st.sidebar.markdown("---")
    mode = st.sidebar.radio("모드 선택", ["내부 실무용", "브랜드사(임원용)", "대행사(제안용)"])

    # Scenario selection (show Korean label)
    if data_mode == "all_in_one":
        scenario_key = st.sidebar.selectbox(
            "전략 선택",
            options=scenarios,
            format_func=lambda k: key_to_label_ui.get(k, k),
        )
        row = df_all[df_all["시나리오명"].astype(str).str.strip() == str(scenario_key).strip()]
        row = row.iloc[0] if not row.empty else None
        scenario_label = key_to_label_ui.get(scenario_key, scenario_key)
    else:
        scenario_key = st.sidebar.selectbox("전략 선택", options=scenarios)
        scenario_label = scenario_key
        row = None  # stacked support omitted in dashboard for simplicity

    if data_mode != "all_in_one":
        st.error("현재 대시보드는 all-in-one(backdata_filled_all_in_one_*.xlsx/csv) 포맷을 기준으로 구성되어 있습니다.")
        st.stop()

    # Build mixes from row
    channel_mix = build_channel_mix_from_row(row)
    adg = build_media_grouped_from_row(row)

    # =========================================================
    # Common KPI helpers (for funnel / exec mode)
    # =========================================================
    def get_any_kpi_scalar(row, token, default):
        """Try to infer a single KPI value across performance media by weighted average."""
        if row is None:
            return default

        # Build overall media shares
        gw = adg.get("_group_weights", {"performance": 0, "viral": 0, "brand": 0})
        overall = {}
        for m, v in adg.get("performance", {}).items():
            overall[m] = overall.get(m, 0.0) + gw["performance"] * v
        for m, v in adg.get("viral", {}).items():
            overall[m] = overall.get(m, 0.0) + gw["viral"] * v
        for m, v in adg.get("brand", {}).items():
            overall[m] = overall.get(m, 0.0) + gw["brand"] * v
        overall = normalize_shares(overall)

        num = 0.0
        den = 0.0
        for media, share in overall.items():
            k = pick_kpi_for_media_from_row(row, media)
            if not k:
                k = fallback_kpi_for_media(media)
            val = k.get(token)
            if val is None:
                # derive CPC if token is CPC
                if token == "CPC":
                    val = derive_cpc(k)
            if val is None or val <= 0:
                continue
            num += float(val) * float(share)
            den += float(share)

        return (num / den) if den > 0 else default

    base_cpc = get_any_kpi_scalar(row, "CPC", 300.0)
    base_ctr = get_any_kpi_scalar(row, "CTR", 0.012)
    base_cvr = get_any_kpi_scalar(row, "CVR", 0.02)

    st.markdown(f"## {scenario_label}")
    st.markdown(f"<div class='smallcap'>{scenario_key}</div>", unsafe_allow_html=True)

    # =========================================================
    # Mode A: 내부 실무용
    # =========================================================
    if mode == "내부 실무용":
        st.markdown('<div class="smallcap">정교한 손익 분석 + 전략 비교 (시나리오별 KPI/믹스를 반영)</div>', unsafe_allow_html=True)

        left, right = st.columns([1.05, 1])

        with left:
            st.markdown("### 입력")
            calc_mode = st.radio(
                "계산 방식",
                ["광고비 입력 → 매출 산출", "매출 입력 → 필요 광고비 산출"],
                horizontal=True,
            )

            aov = st.number_input("객단가(판매가) (원)", value=50_000, step=1_000)
            cost_rate = st.number_input("원가율 (%)", value=30.0) / 100.0
            logistics_per_order = st.number_input("물류비(건당) (원)", value=3_000, step=500)
            fixed_cost = st.number_input("고정비(인건비 등) (원)", value=6_000_000, step=500_000)

            st.markdown("##### KPI (시나리오 DB에서 자동 추정, 필요 시 수정)")
            cpc = st.number_input("CPC (원)", value=float(base_cpc), step=10.0)
            cvr = st.number_input("CVR (%)", value=float(base_cvr * 100.0), step=0.1) / 100.0

            # Scenario CAC estimate (more realistic than single CPC/CVR)
            include_viral_conv = st.toggle("바이럴 KPI 없더라도 전환 포함(권장X)", value=False)
            est = calc_expected_cac(
                total_budget=1_000_000.0,  # scale-free for CAC
                adg=adg,
                kpi_row=row,
                include_viral_if_kpi_missing=bool(include_viral_conv),
            )
            expected_cac = est["expected_CAC"]
            use_mix_cac = st.toggle("시나리오 믹스 기반 CAC 사용(추천)", value=True)

            st.markdown(
                "<div class='kpirow'>"
                f"<div class='kpibox'><span class='smallcap'>추정 CAC</span><div><b>{fmt_won(expected_cac)}</b></div></div>"
                f"<div class='kpibox'><span class='smallcap'>추정 CPC</span><div><b>{fmt_won(base_cpc)}</b></div></div>"
                f"<div class='kpibox'><span class='smallcap'>추정 CVR</span><div><b>{base_cvr*100:.2f}%</b></div></div>"
                "</div>",
                unsafe_allow_html=True,
            )

            if calc_mode.startswith("광고비 입력"):
                marketing_budget = st.number_input("총 광고비 (원)", value=50_000_000, step=1_000_000)
                target_revenue = None
            else:
                target_revenue = st.number_input("목표 매출 (원)", value=300_000_000, step=10_000_000)
                marketing_budget = None

        def simulate_manager(ad_spend=None, revenue=None):
            # Decide CAC / CPC-CVR mode
            if use_mix_cac and (expected_cac is not None) and expected_cac > 0:
                cac = float(expected_cac)
                if revenue is not None:
                    orders = revenue / aov if aov > 0 else 0
                    ad_spend = orders * cac
                else:
                    orders = ad_spend / cac if cac > 0 else 0
                    revenue = orders * aov
                # clicks/cvr are secondary; show rough using provided cvr & cpc
                clicks = orders / cvr if cvr > 0 else 0
                # If provided cpc causes inconsistency, keep it as indicative
            else:
                # classic CPC/CVR mode
                if revenue is not None:
                    orders = revenue / aov if aov > 0 else 0
                    clicks = orders / cvr if cvr > 0 else 0
                    ad_spend = clicks * cpc
                else:
                    clicks = ad_spend / cpc if cpc > 0 else 0
                    orders = clicks * cvr
                    revenue = orders * aov

            cogs = revenue * cost_rate
            logistics = orders * logistics_per_order
            profit = revenue - (ad_spend + cogs + logistics + fixed_cost)
            contrib_margin = (revenue - ad_spend - logistics - cogs) / revenue * 100 if revenue > 0 else 0
            roas = revenue / ad_spend if ad_spend and ad_spend > 0 else 0.0

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

        res = simulate_manager(ad_spend=marketing_budget, revenue=target_revenue)

        with right:
            st.markdown("### 결과")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("예상 매출", fmt_won(res["revenue"]))
            m2.metric("예상 광고비", fmt_won(res["ad_spend"]))
            m3.metric("영업이익", fmt_won(res["profit"]))
            m4.metric("공헌이익률", f"{res['contrib_margin']:.1f}%")

            st.markdown("### 비용 구조")
            cost_df = pd.DataFrame(
                {
                    "항목": ["광고비", "원가(매출원가)", "물류비", "고정비", "영업이익"],
                    "금액": [res["ad_spend"], res["cogs"], res["logistics"], res["fixed"], res["profit"]],
                }
            )
            fig_cost = px.bar(cost_df, x="항목", y="금액", text="금액")
            fig_cost.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig_cost.update_layout(height=320, yaxis_title=None, xaxis_title=None, margin=dict(t=10, b=10))
            st.plotly_chart(fig_cost, use_container_width=True)

        st.divider()

        # --- Media group pie: performance/viral/brand (restore feature) ---
        st.markdown("### 퍼포먼스 / 바이럴 / 브랜드 구성")
        gw = adg.get("_group_weights", {"performance": 0, "viral": 0, "brand": 0})
        group_df = pd.DataFrame(
            {"그룹": ["퍼포먼스", "바이럴", "브랜드"], "비중": [gw["performance"], gw["viral"], gw["brand"]]}
        )
        cA, cB = st.columns([1, 1])
        with cA:
            fig_grp = px.pie(group_df, values="비중", names="그룹", hole=0.5)
            fig_grp.update_traces(textinfo="percent+label")
            fig_grp.update_layout(height=320, margin=dict(t=10))
            st.plotly_chart(fig_grp, use_container_width=True)
        with cB:
            # channel mix pie (restore "매출 채널 구성" using file columns)
            if channel_mix:
                ch_df = pd.DataFrame({"채널": list(channel_mix.keys()), "비중": [v * 100 for v in channel_mix.values()]})
                fig_ch = px.pie(ch_df, values="비중", names="채널", hole=0.5)
                fig_ch.update_traces(textinfo="percent+label")
                fig_ch.update_layout(height=320, margin=dict(t=10))
                st.plotly_chart(fig_ch, use_container_width=True)
            else:
                st.info("채널 믹스(…매출비중) 컬럼을 찾지 못했습니다.")

        st.divider()

        # =====================================================
        # Scenario Comparison (bars: revenue/ad, line: ROAS) + 3개 동시 옵션
        # =====================================================
        st.markdown("### 시나리오 비교 (막대: 매출/광고비, 꺾은선: ROAS)")

        default_compare = scenarios[:3] if len(scenarios) >= 3 else scenarios
        compare_keys = st.multiselect(
            "비교할 전략 선택",
            options=scenarios,
            default=default_compare,
            format_func=lambda k: key_to_label_ui.get(k, k),
        )

        view_mode = st.radio(
            "표시 방식",
            ["전체(매출+광고비+ROAS)", "매출/광고비만(막대)", "ROAS만(꺾은선)"],
            horizontal=True,
        )

        rows = []
        for k in compare_keys:
            rrow = df_all[df_all["시나리오명"].astype(str).str.strip() == str(k).strip()]
            if rrow.empty:
                continue
            rrow = rrow.iloc[0]
            adg_k = build_media_grouped_from_row(rrow)
            est_k = calc_expected_cac(
                total_budget=1_000_000.0,
                adg=adg_k,
                kpi_row=rrow,
                include_viral_if_kpi_missing=bool(include_viral_conv),
            )
            cac_k = est_k["expected_CAC"]

            # simulate with same input (budget or revenue) but scenario-specific CAC if enabled
            if use_mix_cac and cac_k and cac_k > 0:
                if calc_mode.startswith("광고비 입력"):
                    ad_spend_k = float(res["ad_spend"])
                    orders_k = ad_spend_k / float(cac_k)
                    revenue_k = orders_k * aov
                else:
                    revenue_k = float(res["revenue"])
                    orders_k = revenue_k / aov if aov > 0 else 0
                    ad_spend_k = orders_k * float(cac_k)
                roas_k = revenue_k / ad_spend_k if ad_spend_k > 0 else 0
            else:
                # fallback: same cpc/cvr -> same results, but keep label
                sim = simulate_manager(ad_spend=res["ad_spend"] if calc_mode.startswith("광고비") else None,
                                       revenue=res["revenue"] if calc_mode.startswith("매출") else None)
                revenue_k = sim["revenue"]; ad_spend_k = sim["ad_spend"]; roas_k = sim["roas"]

            rows.append({
                "시나리오키": k,
                "전략": key_to_label_ui.get(k, k),
                "예상매출": revenue_k,
                "예상광고비": ad_spend_k,
                "ROAS": roas_k,
            })

        cmp_df = pd.DataFrame(rows)

        if cmp_df.empty:
            st.info("비교할 시나리오를 선택하세요.")
        else:
            # Chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            if view_mode in ("전체(매출+광고비+ROAS)", "매출/광고비만(막대)"):
                fig.add_trace(go.Bar(x=cmp_df["전략"], y=cmp_df["예상매출"], name="예상매출"), secondary_y=False)
                fig.add_trace(go.Bar(x=cmp_df["전략"], y=cmp_df["예상광고비"], name="예상광고비"), secondary_y=False)
            if view_mode in ("전체(매출+광고비+ROAS)", "ROAS만(꺾은선)"):
                fig.add_trace(go.Scatter(x=cmp_df["전략"], y=cmp_df["ROAS"], name="ROAS", mode="lines+markers"), secondary_y=True)

            fig.update_layout(
                height=420,
                barmode="group",
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                xaxis_title=None,
            )
            fig.update_yaxes(title_text=None, secondary_y=False)
            fig.update_yaxes(title_text=None, secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

            # Table
            disp = cmp_df.copy()
            disp["예상매출"] = disp["예상매출"].map(lambda x: f"{x:,.0f}")
            disp["예상광고비"] = disp["예상광고비"].map(lambda x: f"{x:,.0f}")
            disp["ROAS"] = disp["ROAS"].map(lambda x: f"{x:.2f}")
            st.dataframe(disp[["전략", "예상매출", "예상광고비", "ROAS"]], use_container_width=True, hide_index=True)

    # =========================================================
    # Mode B: 브랜드사(임원용)
    # =========================================================
    elif mode == "브랜드사(임원용)":
        st.markdown('<div class="smallcap">수입 유통사 관점 의사결정: 판매량/완판율/예산 소진</div>', unsafe_allow_html=True)

        st.markdown("### 입력")
        c1, c2, c3 = st.columns(3)
        with c1:
            total_budget = st.number_input("총 가용 예산 (원) (수입+마케팅 포함)", value=200_000_000, step=10_000_000)
        with c2:
            target_units = st.number_input("목표 수입 물량 (Total Unit)", value=10_000, step=100)
        with c3:
            landed_cost = st.number_input("개당 수입 원가 (Landed Cost, 원)", value=12_000, step=500)

        with st.expander("고급 옵션 (선택)", expanded=False):
            price_mult = st.slider("예상 판매가 배수(판매가 = Landed Cost × 배수)", min_value=1.2, max_value=4.0, value=2.0, step=0.1)
            selling_price = landed_cost * price_mult
            st.caption(f"예상 판매가(추정): {selling_price:,.0f} 원")

            cpc = st.number_input("CPC (원)", value=float(base_cpc), step=10.0)
            cvr = st.number_input("CVR (%)", value=float(base_cvr * 100.0), step=0.1) / 100.0

        import_cost = target_units * landed_cost
        affordable_units = target_units
        if import_cost > total_budget and landed_cost > 0:
            affordable_units = int(total_budget // landed_cost)
            import_cost = affordable_units * landed_cost
        marketing_budget = max(total_budget - import_cost, 0.0)

        clicks = marketing_budget / cpc if cpc > 0 else 0
        orders = clicks * cvr
        units_sold = min(float(orders), float(affordable_units))
        sell_through = (units_sold / target_units * 100) if target_units > 0 else 0

        unit_margin = max(selling_price - landed_cost, 0)
        net_profit = units_sold * unit_margin - marketing_budget

        st.markdown("### 예상 판매 성과")
        k1, k2, k3 = st.columns([1, 1, 1])

        with k1:
            st.metric("총 예상 판매량 (Units Sold)", f"{units_sold:,.0f} 개")

        with k2:
            if sell_through >= 100:
                badge = "badge-green"
            elif sell_through >= 80:
                badge = "badge-yellow"
            else:
                badge = "badge-red"
            st.markdown(f"완판 예상율: <span class='badge {badge}'>{sell_through:.1f}%</span>", unsafe_allow_html=True)
            st.caption("목표 물량 대비 예상 판매량")

        with k3:
            st.metric("예상 순수익 (Net Profit)", fmt_won(net_profit))

        st.divider()

        st.markdown("### 유통 채널 구성 (파일의 …매출비중 기준, 상위 8)")
        if not channel_mix:
            st.info("…매출비중 컬럼을 찾지 못했습니다.")
        else:
            top = sorted(channel_mix.items(), key=lambda x: x[1], reverse=True)[:8]
            ch_df = pd.DataFrame({"채널": [k for k, _ in top], "비중(%)": [v * 100 for _, v in top]})

            colA, colB = st.columns([1, 1])
            with colA:
                fig_bar = px.bar(ch_df, x="채널", y="비중(%)", text="비중(%)")
                fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_bar.update_layout(height=340, yaxis_title=None, xaxis_title=None, margin=dict(t=10))
                st.plotly_chart(fig_bar, use_container_width=True)
            with colB:
                fig_pie = px.pie(ch_df, values="비중(%)", names="채널", hole=0.45)
                fig_pie.update_traces(textinfo="percent+label")
                fig_pie.update_layout(height=340, margin=dict(t=10))
                st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()

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

    # =========================================================
    # Mode C: 대행사(제안용)
    # =========================================================
    else:
        st.markdown('<div class="smallcap">상세 미디어 믹스 + 퍼널(보수/평범/긍정)</div>', unsafe_allow_html=True)

        st.markdown("### 상세 미디어 믹스")
        # Build overall (performance+viral+brand) shares
        gw = adg.get("_group_weights", {"performance": 0, "viral": 0, "brand": 0})
        overall = {}
        for m, v in adg.get("performance", {}).items():
            overall[m] = overall.get(m, 0.0) + gw["performance"] * v
        for m, v in adg.get("viral", {}).items():
            overall[m] = overall.get(m, 0.0) + gw["viral"] * v
        for m, v in adg.get("brand", {}).items():
            overall[m] = overall.get(m, 0.0) + gw["brand"] * v
        overall = normalize_shares(overall)

        if not overall:
            st.info("미디어 믹스(퍼포먼스/바이럴/브랜드) 컬럼을 찾지 못했습니다.")
        else:
            mm_long = pd.DataFrame({"채널": list(overall.keys()), "비중": list(overall.values())})
            mm_long = mm_long.sort_values("비중", ascending=False)
            if len(mm_long) > 18:
                top18 = mm_long.head(18)
                other = mm_long.iloc[18:]["비중"].sum()
                mm_long = pd.concat([top18, pd.DataFrame([{"채널": "기타", "비중": other}])], ignore_index=True)

            col1, col2 = st.columns([1, 1])
            with col1:
                fig_tm = px.treemap(mm_long, path=["채널"], values="비중")
                fig_tm.update_layout(height=420, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_tm, use_container_width=True)
            with col2:
                fig_p = px.pie(mm_long, values="비중", names="채널", hole=0.45)
                fig_p.update_traces(textinfo="percent+label")
                fig_p.update_layout(height=420, margin=dict(t=10))
                st.plotly_chart(fig_p, use_container_width=True)

        st.divider()

        st.markdown("### 퍼널 시뮬레이션 (노출 → 유입 → 전환)")
        left, right = st.columns([1, 1])

        with left:
            budget = st.number_input("투입 예산 (원)", value=50_000_000, step=1_000_000)
            cpc = st.number_input("CPC (원)", value=float(base_cpc), step=10.0)
            ctr = st.number_input("CTR (%)", value=float(base_ctr * 100.0), step=0.1) / 100.0
            cvr = st.number_input("CVR (%)", value=float(base_cvr * 100.0), step=0.1) / 100.0

        with right:
            scenario_type = st.radio("가정 선택", ["보수적", "평범", "긍정적"], horizontal=True)

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

        funnel_df = pd.DataFrame(
            {"단계": ["노출(Impressions)", "유입(Clicks)", "전환(Conversions)"], "값": [impressions, clicks, conversions]}
        )

        fig_funnel = go.Figure(
            go.Funnel(
                y=funnel_df["단계"],
                x=funnel_df["값"],
                textinfo="value+percent initial",
            )
        )
        fig_funnel.update_layout(height=420, margin=dict(t=10, b=10), font=dict(size=13))
        st.plotly_chart(fig_funnel, use_container_width=True)
