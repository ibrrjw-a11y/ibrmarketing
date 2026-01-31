import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import re

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
  border-radius: 16px;
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
  padding:10px 12px; border:1px solid rgba(0,0,0,0.08); border-radius:14px; background:white;
}}
.dim {{
  color:{MUTED};
  font-size:12px;
}}
.section-title {{
  margin-top: 2px;
  margin-bottom: 8px;
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
    """Supports 0.32, 32, '32%', etc -> 0~1"""
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

def safe_str_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def topN_plus_other(d: dict, n=8, other_label="기타"):
    """return (labels, values) as shares, with topN and the rest as other"""
    if not d:
        return [], []
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    top = items[:n]
    other = sum(v for _, v in items[n:])
    labels = [k for k, _ in top]
    vals = [v for _, v in top]
    if other > 0:
        labels.append(other_label)
        vals.append(other)
    # normalize in case
    s = sum(vals)
    if s > 0:
        vals = [v / s for v in vals]
    return labels, vals

def donut_chart(labels, values, title=None, height=320):
    df = pd.DataFrame({"라벨": labels, "비중": values})
    fig = px.pie(df, values="비중", names="라벨", hole=0.52)
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=height, margin=dict(t=30 if title else 10, b=10, l=10, r=10), title=title)
    return fig

# =========================================================
# Loader: supports XLSX(all-in-one) and CSV(all-in-one)
# =========================================================
REQUIRED_ALLINONE = ["시나리오명", "노출 시나리오명"]

def read_uploaded(uploaded):
    name = (uploaded.name or "").lower()
    if name.endswith(".xlsx"):
        df = pd.read_excel(uploaded)
        df = safe_str_cols(df)
        return df, "xlsx"
    else:
        raw = uploaded.getvalue()
        text = raw.decode("utf-8-sig", errors="replace")
        df = pd.read_csv(StringIO(text))
        df = safe_str_cols(df)
        return df, "csv"

def scenario_list_from_df(df: pd.DataFrame):
    if df is None or "시나리오명" not in df.columns:
        return []
    s = df["시나리오명"].dropna().astype(str).str.strip()
    s = [x for x in s if x and x != "시나리오명"]
    return sorted(list(dict.fromkeys(s)))

# =========================================================
# Scenario Key Parser
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
# KPI parsing
# =========================================================
TOKEN_ALIASES = {
    "CPM": ["CPM"],
    "CTR": ["CTR", "CLICKRATE", "클릭률"],
    "CVR": ["CVR", "CONVRATE", "전환율"],
    "CPC": ["CPC"],
}

def pick_kpi_for_media_from_row(row: pd.Series, media: str):
    if row is None:
        return {}
    idx = list(row.index.astype(str))
    out = {}
    for token, aliases in TOKEN_ALIASES.items():
        found = None
        # 1) exact KPI_TOKEN_media
        for al in aliases:
            exact = f"KPI_{al}_{media}"
            if exact in idx:
                found = exact
                break
        # 2) exact media_TOKEN
        if not found:
            for al in aliases:
                exact2 = f"{media}_{al}"
                if exact2 in idx:
                    found = exact2
                    break
        # 3) fuzzy
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
# Mix builders from all-in-one row
# =========================================================
def build_channel_mix_from_row(row: pd.Series):
    """Use columns ending with '매출비중' -> normalized shares"""
    if row is None:
        return {}
    cols = [c for c in row.index.astype(str) if str(c).endswith("매출비중")]
    raw = {}
    for c in cols:
        v = normalize_ratio(row.get(c))
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if float(v) > 0:
            raw[c.replace("매출비중", "").strip()] = float(v)
    return normalize_shares(raw)

def build_media_grouped_from_row(row: pd.Series):
    """
    Uses columns like:
      퍼포먼스마케팅_*, 바이럴마케팅_*, 브랜드 마케팅 / 기타_브랜드*
    Excludes KPI_* and scenario columns.
    """
    out = {"performance": {}, "viral": {}, "brand": {}, "_group_weights": {"performance": 0, "viral": 0, "brand": 0}}
    if row is None:
        return out

    cols = [c for c in row.index.astype(str) if c not in ("시나리오명", "노출 시나리오명") and not str(c).startswith("KPI_")]

    perf_cols = [c for c in cols if str(c).startswith("퍼포먼스마케팅_") or str(c).startswith("퍼포먼스_")]
    viral_cols = [c for c in cols if str(c).startswith("바이럴마케팅_")]
    brand_cols = [c for c in cols if ("브랜드" in str(c) and "마케팅" in str(c)) or str(c).startswith("기타_브랜드")]

    perf_raw, viral_raw, brand_raw = {}, {}, {}

    for c in perf_cols:
        v = normalize_ratio(row.get(c))
        if not (v is None or (isinstance(v, float) and np.isnan(v))) and float(v) > 0:
            perf_raw[c] = float(v)

    for c in viral_cols:
        v = normalize_ratio(row.get(c))
        if not (v is None or (isinstance(v, float) and np.isnan(v))) and float(v) > 0:
            viral_raw[c] = float(v)

    for c in brand_cols:
        v = normalize_ratio(row.get(c))
        if not (v is None or (isinstance(v, float) and np.isnan(v))) and float(v) > 0:
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
# Expected CAC
# =========================================================
def calc_expected_cac(total_budget, adg, kpi_row, include_viral_if_kpi_missing=False):
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
            contrib.append({"channel": media, "budget": budget_i, "CPC": None, "clicks": 0.0, "conversions": 0.0, "note": "viral_kpi_missing_excluded"})
            continue

        cpc = derive_cpc(kpi)
        cvr = float(kpi.get("CVR", 0.0) or 0.0)
        if (cpc is None) or cpc <= 0 or cvr <= 0:
            contrib.append({"channel": media, "budget": budget_i, "CPC": cpc, "clicks": 0.0, "conversions": 0.0, "note": "kpi_insufficient"})
            continue

        clicks = budget_i / float(cpc)
        convs = clicks * float(cvr)

        total_clicks += clicks
        total_convs += convs

        contrib.append({"channel": media, "budget": budget_i, "CPC": float(cpc), "clicks": clicks, "conversions": convs, "note": "fallback_kpi" if kpi_is_fallback else "ok"})

    expected_cac = (total_budget / total_convs) if total_convs > 0 else None
    return {"expected_clicks": total_clicks, "expected_conversions": total_convs, "expected_CAC": expected_cac, "media_contrib": contrib}

# =========================================================
# Recommendation scoring (rule-based)
# =========================================================
WEIGHTS = {"channel_match": 45.0, "drv_bonus": 25.0, "channel_ad_link": 20.0, "demo_keyword": 10.0}

DRV_PRIMARY = {
    "자사몰 중심": "D2C",
    "온라인 중심": "COM",
    "홈쇼핑 중심": "HSP",
    "공구 중심": "GB",
    "B2B 중심": "B2B",
}
DRV_SECONDARY = {"자사몰 중심": "PERF", "온라인 중심": "PERF", "홈쇼핑 중심": None, "공구 중심": None, "B2B 중심": None}

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
    target_kw = {
        "자사몰 중심": ["자사몰"],
        "온라인 중심": ["온라인", "스마트스토어", "쿠팡", "오픈마켓", "마켓"],
        "홈쇼핑 중심": ["홈쇼핑"],
        "공구 중심": ["공구", "공동구매"],
        "B2B 중심": ["B2B", "도매"],
    }.get(sales_focus, [])
    if not target_kw:
        return 0.0
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
        else:
            score += min(1.0, ext_pa * 3.0) * 0.6
            score += 0.4 if ext_pa >= meta else 0.15

    elif sales_focus == "홈쇼핑 중심":
        core = naver_sa + naver_blog + ext_pa
        score += min(1.0, core * 2.5) * 0.7
        penalty = meta + google_gdn + tiktok
        score += max(0.0, 1.0 - penalty * 2.0) * 0.3

    elif sales_focus == "공구 중심":
        score += min(1.0, score_channel_match(channel_mix_norm, "공구 중심") * 1.8) * 0.5
        score += min(1.0, ig_mega * 4.0) * 0.5

    elif sales_focus == "B2B 중심":
        brand_share = sum(adg.get("brand", {}).values()) if isinstance(adg.get("brand"), dict) else 0.0
        score += min(1.0, naver_sa * 3.0) * 0.6
        score += min(1.0, brand_share * 3.0) * 0.4

    return float(max(0.0, min(1.0, score)))

def score_demo_keyword(adg, payload):
    gw = adg.get("_group_weights", {"performance": 0, "viral": 0, "brand": 0})
    perf_sum, viral_sum, brand_sum = gw["performance"], gw["viral"], gw["brand"]

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
            (overall_media_share(adg, "퍼포먼스마케팅_네이버 SA") + overall_media_share(adg, "바이럴마케팅_네이버 블로그")) * 3.0,
        ) * 0.3

    return float(max(0.0, min(1.0, score)))

def build_why(channel aligning + 3 lines):
    # (We keep it short; no AI comment generation)
    # In UI we show: top channels + top media + group share
    pass

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
        rowdf = df_all[df_all["시나리오명"].astype(str).str.strip() == str(s).strip()]
        if rowdf.empty:
            continue
        row = rowdf.iloc[0]

        m = meta_map.get(s) or {}
        drv = m.get("DRV")

        channel_mix_norm = build_channel_mix_from_row(row)
        adg = build_media_grouped_from_row(row)

        a = score_channel_match(channel_mix_norm, payload["sales_focus_channel"])
        b = score_drv_bonus(drv, payload["sales_focus_channel"], payload["operator"])
        c = score_channel_ad_link(channel_mix_norm, adg, payload["sales_focus_channel"], payload.get("online_market_focus"))
        d = score_demo_keyword(adg, payload)

        total = (
            a * WEIGHTS["channel_match"] + b * WEIGHTS["drv_bonus"] + c * WEIGHTS["channel_ad_link"] + d * WEIGHTS["demo_keyword"]
        ) / sum(WEIGHTS.values()) * 100.0

        expected = calc_expected_cac(
            total_budget=float(payload["total_ad_budget_krw"]),
            adg=adg,
            kpi_row=row,
            include_viral_if_kpi_missing=bool(payload.get("include_viral_conversions_if_kpi_missing", False)),
        )

        results.append(
            {
                "scenario_key": s,
                "scenario_label": key_to_label.get(s, s),
                "score": float(max(0.0, min(100.0, total))),
                "why": build_why(channel_mix_norm, adg),
                "expected_metrics": expected,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"input": payload, "candidate_count": len(candidates), "recommendations": results[:3]}

# =========================================================
# Sidebar: Upload
# =========================================================
st.sidebar.title("마케팅/유통 시뮬레이터")
uploaded = st.sidebar.file_uploader("Backdata 업로드 (XLSX/CSV)", type=["xlsx", "csv"], key="uploader_main")

if uploaded is None:
    st.info("좌측에서 backdata 파일(xlsx/csv)을 업로드하세요.")
    st.stop()

try:
    df_loaded, fmt = read_uploaded(uploaded)
except Exception as e:
    st.error(f"❌ 파일 로드 실패: {e}")
    st.stop()

is_all_in_one = all(c in df_loaded.columns for c in REQUIRED_ALLINONE)
if not is_all_in_one:
    st.error("❌ 현재 코드는 all-in-one 포맷(시나리오명/노출 시나리오명 포함)을 기준으로 동작합니다. 파일 구조를 확인해 주세요.")
    st.stop()

df_all = df_loaded.copy()
df_all["시나리오명"] = df_all["시나리오명"].astype(str).str.strip()
df_all["노출 시나리오명"] = df_all["노출 시나리오명"].astype(str).str.strip()

scenarios = scenario_list_from_df(df_all)
if not scenarios:
    st.error("❌ 시나리오 목록이 비어있습니다.")
    st.stop()

key_to_label = dict(zip(df_all["시나리오명"], df_all["노출 시나리오명"]))

# labels uniqueness
label_counts = pd.Series(list(key_to_label.values())).value_counts().to_dict()
key_to_label_ui = {}
for k, v in key_to_label.items():
    key_to_label_ui[k] = f"{v}  ({k})" if label_counts.get(v, 0) > 1 else v

# category options from CAT in key (fallback: show raw)
parsed = [parse_scenario_key(s) for s in scenarios]
cat_options = sorted(list({p["CAT"] for p in parsed if p and p.get("CAT")})) or ["(카테고리 파싱 실패)"]

# =========================================================
# Global layout: Main tabs = [추천엔진] + [대시보드]
# with Dashboard = [대행사] [브랜드사], each has internal/external sub-tabs
# =========================================================
tab_rec, tab_dash = st.tabs(["✅ 추천 엔진", "📊 대시보드 (대행/브랜드)"])

# =========================================================
# TAB 1) Recommendation Engine (UI cleaned)
# =========================================================
with tab_rec:
    st.markdown("## 추천 엔진")
    st.markdown('<div class="smallcap">데이터 기반 Top3 추천 (룰 기반 스코어링 + KPI 기반 예상 CAC)</div>', unsafe_allow_html=True)

    left, right = st.columns([0.95, 1.05])

    with left:
        st.markdown("### 입력")
        operator = st.selectbox("운영 주체", ["내부브랜드 운영자", "브랜드사 운영자(클라이언트)", "대행사(마케팅만)"], key="rec_operator")
        stage = st.selectbox("단계(ST)", ["NEW", "EARLY", "GROW", "MATURE"], key="rec_stage")
        category = st.selectbox("카테고리(CAT)", cat_options, key="rec_cat")
        position = st.selectbox("가격 포지셔닝(POS)", ["L", "M", "P"], key="rec_pos")
        sales_focus_channel = st.selectbox("판매 중심 채널", ["자사몰 중심", "온라인 중심", "홈쇼핑 중심", "공구 중심", "B2B 중심"], key="rec_sales")

        online_market_focus = None
        if sales_focus_channel == "온라인 중심":
            online_market_focus = st.selectbox("온라인 마켓 포커스(옵션)", [None, "쿠팡 중심", "스마트스토어 중심"], format_func=lambda x: "미지정(자동)" if x is None else x, key="rec_online_focus")

        no_comp = st.toggle("경쟁키워드 판매의도 없음", value=True, key="rec_no_comp")
        competitor_keyword_level = None
        if not no_comp:
            competitor_keyword_level = st.selectbox(
                "경쟁키워드 검색량 구간",
                ["매우낮음(~3,000)", "낮음(3,000~10,000)", "중간(10,000~20,000)", "높음(20,000~30,000)", "매우높음(35,000~)"],
                key="rec_comp_lv",
            )

        brand_keyword_level = st.selectbox(
            "브랜드 키워드(인지도) 검색량 구간",
            ["매우낮음(~300)", "낮음(300~1,000)", "중간(1,000~4,000)", "높음(4,000~8,000)", "매우높음(8,000~)"],
            key="rec_brand_lv",
        )

        target_age = st.selectbox("주요 타겟 연령대", ["10대", "20대", "30대", "40대", "50대+"], key="rec_age")
        total_ad_budget_krw = st.number_input("총 광고예산(원)", value=50_000_000, step=1_000_000, min_value=1, key="rec_budget")

        include_viral_if_missing = st.toggle("바이럴 KPI 없더라도 전환 포함(권장X)", value=False, key="rec_include_viral")
        run = st.button("Top3 추천 계산", use_container_width=True, key="rec_run")

    with right:
        st.markdown("### 결과")
        if not run:
            st.info("좌측 조건을 설정하고 **Top3 추천 계산**을 누르세요.")
        else:
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

            out = recommend_top3_allinone(payload=payload, df_all=df_all, key_to_label=key_to_label)

            c1, c2 = st.columns(2)
            c1.metric("후보 전략 수", f"{out.get('candidate_count', 0):,} 개")
            c2.metric("추천 결과", f"{len(out.get('recommendations', []))} 개")

            recs = out.get("recommendations", [])
            if not recs:
                st.warning("추천 결과가 없습니다. (시나리오 키 규칙/카테고리 파싱/데이터 확인 필요)")
            else:
                cols = st.columns(3)
                for i, r in enumerate(recs):
                    with cols[i]:
                        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                        st.markdown(f"#### #{i+1} {r['scenario_label']}")
                        st.caption(r["scenario_key"])
                        st.metric("Score", f"{r['score']:.1f}")
                        st.metric("예상 CAC", fmt_won(r["expected_metrics"]["expected_CAC"]))
                        st.metric("예상 전환", f"{r['expected_metrics']['expected_conversions']:.1f}")

                        st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
                        for line in r["why"]:
                            st.write(f"- {line}")

                        # Mix donuts (100% 기준)
                        # channel mix
                        rowdf = df_all[df_all["시나리오명"].astype(str).str.strip() == str(r["scenario_key"]).strip()]
                        row0 = rowdf.iloc[0] if not rowdf.empty else None
                        ch = build_channel_mix_from_row(row0)
                        adg_r = build_media_grouped_from_row(row0)
                        gw = adg_r.get("_group_weights", {"performance": 0, "viral": 0, "brand": 0})

                        with st.expander("상세(믹스/기여)", expanded=False):
                            # Group donut
                            fig_g = donut_chart(["퍼포먼스", "바이럴", "브랜드"], [gw["performance"], gw["viral"], gw["brand"]], title="그룹 구성(100%)", height=280)
                            st.plotly_chart(fig_g, use_container_width=True)

                            if ch:
                                lab, val = topN_plus_other(ch, n=8)
                                fig_ch = donut_chart(lab, val, title="매출 채널 구성(100%)", height=280)
                                st.plotly_chart(fig_ch, use_container_width=True)

                            # media overall donut
                            overall = {}
                            for m, v in adg_r.get("performance", {}).items():
                                overall[m] = overall.get(m, 0.0) + gw["performance"] * v
                            for m, v in adg_r.get("viral", {}).items():
                                overall[m] = overall.get(m, 0.0) + gw["viral"] * v
                            for m, v in adg_r.get("brand", {}).items():
                                overall[m] = overall.get(m, 0.0) + gw["brand"] * v
                            overall = normalize_shares(overall)
                            if overall:
                                lab2, val2 = topN_plus_other(overall, n=10)
                                fig_mm = donut_chart(lab2, val2, title="미디어 믹스(100%)", height=280)
                                st.plotly_chart(fig_mm, use_container_width=True)

                        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 2) Dashboard
# =========================================================
with tab_dash:
    # Top-level: Agency / Brand
    main_mode = st.sidebar.radio("대시보드 선택", ["대행사", "브랜드사"], key="dash_main_mode")
    sub_mode = st.sidebar.radio("버전 선택", ["내부", "외부"], horizontal=True, key="dash_sub_mode")

    scenario_key = st.sidebar.selectbox("전략 선택", options=scenarios, format_func=lambda k: key_to_label_ui.get(k, k), key="dash_scenario")
    rowdf = df_all[df_all["시나리오명"].astype(str).str.strip() == str(scenario_key).strip()]
    row = rowdf.iloc[0] if not rowdf.empty else None
    scenario_label = key_to_label_ui.get(scenario_key, scenario_key)

    channel_mix = build_channel_mix_from_row(row)
    adg = build_media_grouped_from_row(row)
    gw = adg.get("_group_weights", {"performance": 0, "viral": 0, "brand": 0})

    # KPI base from mix-weighted averages
    def get_any_kpi_scalar(row, token, default):
        if row is None:
            return default
        overall = {}
        for m, v in adg.get("performance", {}).items():
            overall[m] = overall.get(m, 0.0) + gw["performance"] * v
        for m, v in adg.get("viral", {}).items():
            overall[m] = overall.get(m, 0.0) + gw["viral"] * v
        for m, v in adg.get("brand", {}).items():
            overall[m] = overall.get(m, 0.0) + gw["brand"] * v
        overall = normalize_shares(overall)

        num, den = 0.0, 0.0
        for media, share in overall.items():
            k = pick_kpi_for_media_from_row(row, media)
            if not k:
                k = fallback_kpi_for_media(media)
            val = k.get(token)
            if val is None and token == "CPC":
                val = derive_cpc(k)
            if val is None or val <= 0:
                continue
            num += float(val) * float(share)
            den += float(share)
        return (num / den) if den > 0 else default

    base_cpc = get_any_kpi_scalar(row, "CPC", 300.0)
    base_ctr = get_any_kpi_scalar(row, "CTR", 0.012)
    base_cvr = get_any_kpi_scalar(row, "CVR", 0.02)

    st.markdown(f"## {main_mode} · {sub_mode}")
    st.markdown(f"### {scenario_label}")
    st.markdown(f"<div class='smallcap'>{scenario_key}</div>", unsafe_allow_html=True)

    # =====================================================
    # Common Mix Donuts (100% 기준) - requested
    # =====================================================
    st.markdown("#### 믹스 요약(100%)")
    cA, cB, cC = st.columns(3)
    with cA:
        fig_g = donut_chart(["퍼포먼스", "바이럴", "브랜드"], [gw["performance"], gw["viral"], gw["brand"]], title="그룹 구성", height=300)
        st.plotly_chart(fig_g, use_container_width=True)
    with cB:
        if channel_mix:
            lab, val = topN_plus_other(channel_mix, n=8)
            st.plotly_chart(donut_chart(lab, val, title="매출 채널", height=300), use_container_width=True)
        else:
            st.info("…매출비중 컬럼 없음")
    with cC:
        overall = {}
        for m, v in adg.get("performance", {}).items():
            overall[m] = overall.get(m, 0.0) + gw["performance"] * v
        for m, v in adg.get("viral", {}).items():
            overall[m] = overall.get(m, 0.0) + gw["viral"] * v
        for m, v in adg.get("brand", {}).items():
            overall[m] = overall.get(m, 0.0) + gw["brand"] * v
        overall = normalize_shares(overall)
        if overall:
            lab2, val2 = topN_plus_other(overall, n=10)
            st.plotly_chart(donut_chart(lab2, val2, title="미디어 믹스", height=300), use_container_width=True)
        else:
            st.info("미디어 믹스 컬럼 없음")

    st.divider()

    # =====================================================
    # Agency
    # =====================================================
    if main_mode == "대행사":
        if sub_mode == "외부":
            st.markdown("#### 외부(클라이언트 제안용) — 광고비/효율/추천 믹스 중심")
            left, right = st.columns([1, 1])
            with left:
                budget = st.number_input("예산(원)", value=50_000_000, step=1_000_000, key="ag_ext_budget")
                include_viral = st.toggle("바이럴 KPI 없더라도 전환 포함(권장X)", value=False, key="ag_ext_include_viral")

                cpc = st.number_input("CPC (원)", value=float(base_cpc), step=10.0, key="ag_ext_cpc")
                ctr = st.number_input("CTR (%)", value=float(base_ctr * 100.0), step=0.1, key="ag_ext_ctr") / 100.0
                cvr = st.number_input("CVR (%)", value=float(base_cvr * 100.0), step=0.1, key="ag_ext_cvr") / 100.0

                funnel_profile = st.radio("가정", ["보수적", "평범", "긍정적"], horizontal=True, key="ag_ext_funnel_profile")

            with right:
                # Expected CAC (mix+KPI)
                est = calc_expected_cac(total_budget=float(budget), adg=adg, kpi_row=row, include_viral_if_kpi_missing=bool(include_viral))
                expected_conv = est["expected_conversions"]
                expected_cac = est["expected_CAC"]

                m1, m2, m3 = st.columns(3)
                m1.metric("예상 전환", f"{expected_conv:,.1f}")
                m2.metric("예상 CAC", fmt_won(expected_cac))
                m3.metric("예상 클릭", f"{est['expected_clicks']:,.0f}")

                # Funnel 3 profiles (요청: 긍정/보수/평범)
                if funnel_profile == "보수적":
                    m_ctr, m_cvr, m_cpc = 0.85, 0.85, 1.10
                elif funnel_profile == "긍정적":
                    m_ctr, m_cvr, m_cpc = 1.15, 1.15, 0.90
                else:
                    m_ctr, m_cvr, m_cpc = 1.00, 1.00, 1.00

                ctr2 = max(ctr * m_ctr, 1e-6)
                cvr2 = max(cvr * m_cvr, 1e-6)
                cpc2 = max(cpc * m_cpc, 1e-6)

                clicks = float(budget) / cpc2
                impressions = clicks / ctr2
                conversions = clicks * cvr2

                funnel_df = pd.DataFrame({"단계": ["노출(Impressions)", "유입(Clicks)", "전환(Conversions)"], "값": [impressions, clicks, conversions]})
                fig_funnel = go.Figure(go.Funnel(y=funnel_df["단계"], x=funnel_df["값"], textinfo="value+percent initial"))
                fig_funnel.update_layout(height=360, margin=dict(t=10, b=10))
                st.plotly_chart(fig_funnel, use_container_width=True)

            st.caption("※ 외부용 화면은 ‘추정치’ 중심으로 단정적인 약속이 되지 않게 구성했습니다. (마진/인건비 노출 없음)")

        else:
            st.markdown("#### 내부(제안 제작용) — 광고비/마진/인건비 입력 포함")
            left, right = st.columns([1.05, 1])

            with left:
                calc_mode = st.radio("계산 방식", ["광고비 입력 → 매출 산출", "매출 입력 → 필요 광고비 산출"], horizontal=True, key="ag_int_calc_mode")

                aov = st.number_input("객단가(판매가) (원)", value=50_000, step=1_000, key="ag_int_aov")
                cost_rate = st.number_input("원가율 (%)", value=30.0, key="ag_int_cost_rate") / 100.0
                logistics_per_order = st.number_input("물류비(건당) (원)", value=3_000, step=500, key="ag_int_logi")
                # 인건비/고정비 입력은 내부에서만 (요청 반영)
                fixed_cost = st.number_input("인건비/고정비 (원)", value=6_000_000, step=500_000, key="ag_int_fixed")

                include_viral = st.toggle("바이럴 KPI 없더라도 전환 포함(권장X)", value=False, key="ag_int_include_viral")
                est_unit = calc_expected_cac(total_budget=1_000_000.0, adg=adg, kpi_row=row, include_viral_if_kpi_missing=bool(include_viral))
                expected_cac = est_unit["expected_CAC"]
                use_mix_cac = st.toggle("시나리오 믹스 기반 CAC 사용(추천)", value=True, key="ag_int_use_mix_cac")

                if calc_mode.startswith("광고비 입력"):
                    marketing_budget = st.number_input("총 광고비 (원)", value=50_000_000, step=1_000_000, key="ag_int_budget")
                    target_revenue = None
                else:
                    target_revenue = st.number_input("목표 매출 (원)", value=300_000_000, step=10_000_000, key="ag_int_rev")
                    marketing_budget = None

                # KPI overrides for fallback mode
                st.markdown("##### KPI(참고)")
                cpc = st.number_input("CPC (원)", value=float(base_cpc), step=10.0, key="ag_int_cpc")
                cvr = st.number_input("CVR (%)", value=float(base_cvr * 100.0), step=0.1, key="ag_int_cvr") / 100.0

            def simulate_pl(ad_spend=None, revenue=None):
                if use_mix_cac and expected_cac and expected_cac > 0:
                    cac = float(expected_cac)
                    if revenue is not None:
                        orders = revenue / aov if aov > 0 else 0
                        ad_spend = orders * cac
                    else:
                        orders = ad_spend / cac if cac > 0 else 0
                        revenue = orders * aov
                    clicks = orders / cvr if cvr > 0 else 0
                else:
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
                # 공헌이익률 요청식: (매출 - 광고비 - 물류비 - 원가) / 매출
                contrib_margin = (revenue - ad_spend - logistics - cogs) / revenue * 100 if revenue > 0 else 0
                roas = revenue / ad_spend if ad_spend and ad_spend > 0 else 0.0
                return dict(revenue=float(revenue), ad=float(ad_spend), orders=float(orders), clicks=float(clicks), cogs=float(cogs), logistics=float(logistics), fixed=float(fixed_cost), profit=float(profit), contrib=float(contrib_margin), roas=float(roas))

            res = simulate_pl(ad_spend=marketing_budget, revenue=target_revenue)

            with right:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("예상 매출", fmt_won(res["revenue"]))
                m2.metric("예상 광고비", fmt_won(res["ad"]))
                m3.metric("영업이익", fmt_won(res["profit"]))
                m4.metric("공헌이익률", f"{res['contrib']:.1f}%")

                # 비용 구조(내부만 노출)
                cost_df = pd.DataFrame({"항목": ["광고비", "원가", "물류비", "인건비/고정비", "영업이익"], "금액": [res["ad"], res["cogs"], res["logistics"], res["fixed"], res["profit"]]})
                fig_cost = px.bar(cost_df, x="항목", y="금액", text="금액")
                fig_cost.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
                fig_cost.update_layout(height=340, yaxis_title=None, xaxis_title=None, margin=dict(t=10, b=10))
                st.plotly_chart(fig_cost, use_container_width=True)

            st.divider()

            # 시나리오 비교(내부만) : 막대(매출/광고비) + 선(ROAS) + 전체/부분 보기
            st.markdown("#### 전략 비교 (내부)")
            default_compare = scenarios[:3] if len(scenarios) >= 3 else scenarios
            compare_keys = st.multiselect("비교할 전략", options=scenarios, default=default_compare, format_func=lambda k: key_to_label_ui.get(k, k), key="ag_int_compare")
            view_mode = st.radio("표시", ["전체(매출+광고비+ROAS)", "매출/광고비만", "ROAS만"], horizontal=True, key="ag_int_viewmode")

            rows = []
            for k in compare_keys:
                rrowdf = df_all[df_all["시나리오명"].astype(str).str.strip() == str(k).strip()]
                if rrowdf.empty:
                    continue
                rrow = rrowdf.iloc[0]
                adg_k = build_media_grouped_from_row(rrow)
                est_k = calc_expected_cac(total_budget=1_000_000.0, adg=adg_k, kpi_row=rrow, include_viral_if_kpi_missing=bool(include_viral))
                cac_k = est_k["expected_CAC"]

                # simulate same input (budget/rev) with scenario-specific CAC when enabled
                if use_mix_cac and cac_k and cac_k > 0:
                    if calc_mode.startswith("광고비 입력"):
                        ad_spend_k = float(res["ad"])
                        orders_k = ad_spend_k / float(cac_k)
                        revenue_k = orders_k * aov
                    else:
                        revenue_k = float(res["revenue"])
                        orders_k = revenue_k / aov if aov > 0 else 0
                        ad_spend_k = orders_k * float(cac_k)
                    roas_k = revenue_k / ad_spend_k if ad_spend_k > 0 else 0
                else:
                    revenue_k, ad_spend_k, roas_k = res["revenue"], res["ad"], res["roas"]

                rows.append({"전략": key_to_label_ui.get(k, k), "예상매출": revenue_k, "예상광고비": ad_spend_k, "ROAS": roas_k})

            cmp_df = pd.DataFrame(rows)
            if cmp_df.empty:
                st.info("비교할 전략을 선택하세요.")
            else:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                if view_mode in ("전체(매출+광고비+ROAS)", "매출/광고비만"):
                    fig.add_trace(go.Bar(x=cmp_df["전략"], y=cmp_df["예상매출"], name="예상매출"), secondary_y=False)
                    fig.add_trace(go.Bar(x=cmp_df["전략"], y=cmp_df["예상광고비"], name="예상광고비"), secondary_y=False)
                if view_mode in ("전체(매출+광고비+ROAS)", "ROAS만"):
                    fig.add_trace(go.Scatter(x=cmp_df["전략"], y=cmp_df["ROAS"], name="ROAS", mode="lines+markers"), secondary_y=True)
                fig.update_layout(height=420, barmode="group", margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h", y=1.02, x=0))
                st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # Brand
    # =====================================================
    else:
        # Brand requires monthly view (sales/ad). If file doesn't have monthly weights,
        # we use a safe "ramp" distribution (can be replaced later by your data columns).
        st.markdown("#### 브랜드사 — 월별 매출/광고비(추정) 포함")

        # inputs (internal/external both may share, but keep external minimal)
        if sub_mode == "외부":
            c1, c2 = st.columns(2)
            with c1:
                total_budget = st.number_input("총 가용 예산(원)", value=200_000_000, step=10_000_000, key="br_ext_total_budget")
                target_units = st.number_input("목표 물량(Unit)", value=10_000, step=100, key="br_ext_units")
            with c2:
                landed_cost = st.number_input("개당 수입원가(원)", value=12_000, step=500, key="br_ext_landed")
                price_mult = st.slider("예상 판매가 배수", 1.2, 4.0, 2.0, 0.1, key="br_ext_mult")
            selling_price = landed_cost * price_mult

            st.caption("※ 외부용은 ‘추정치/범위’ 중심으로 과도한 약속이 되지 않게 구성하는 것을 권장합니다.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                total_budget = st.number_input("총 가용 예산(원)", value=200_000_000, step=10_000_000, key="br_int_total_budget")
            with c2:
                target_units = st.number_input("목표 물량(Unit)", value=10_000, step=100, key="br_int_units")
            with c3:
                landed_cost = st.number_input("개당 수입원가(원)", value=12_000, step=500, key="br_int_landed")

            with st.expander("내부 설정(비공개 입력)", expanded=False):
                fixed_cost = st.number_input("인건비/고정비 (원)", value=6_000_000, step=500_000, key="br_int_fixed")  # 입력은 가능, 대시보드에선 과도 노출 X
                price_mult = st.slider("예상 판매가 배수", 1.2, 4.0, 2.0, 0.1, key="br_int_mult")
            selling_price = landed_cost * price_mult

        # Budget split: import vs marketing
        import_cost = target_units * landed_cost
        affordable_units = target_units
        if import_cost > total_budget and landed_cost > 0:
            affordable_units = int(total_budget // landed_cost)
            import_cost = affordable_units * landed_cost
        marketing_budget = max(total_budget - import_cost, 0.0)

        # Use base KPI to approximate sell-through (simple, intentionally)
        cpc = float(base_cpc)
        cvr = float(base_cvr)

        clicks = marketing_budget / cpc if cpc > 0 else 0
        orders = clicks * cvr
        units_sold = min(float(orders), float(affordable_units))
        sell_through = (units_sold / target_units * 100) if target_units > 0 else 0

        unit_margin = max(selling_price - landed_cost, 0)
        net_profit = units_sold * unit_margin - marketing_budget

        st.markdown("### KPI 요약")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("예상 판매량", f"{units_sold:,.0f} 개")
        if sell_through >= 100:
            badge = "badge-green"
        elif sell_through >= 80:
            badge = "badge-yellow"
        else:
            badge = "badge-red"
        k2.markdown(f"완판 예상율<br><span class='badge {badge}'>{sell_through:.1f}%</span>", unsafe_allow_html=True)
        k3.metric("수입비용", fmt_won(import_cost))
        k4.metric("마케팅 예산", fmt_won(marketing_budget))

        st.divider()

        # Donut: import / marketing / profit(loss)
        st.markdown("### 예산 소진(도넛)")
        donut_labels = ["제품 수입비용", "마케팅 집행비"]
        donut_vals = [import_cost, marketing_budget]
        if net_profit >= 0:
            donut_labels.append("예상 수익")
            donut_vals.append(net_profit)
        else:
            donut_labels.append("예상 손실")
            donut_vals.append(abs(net_profit))
        fig_budget = px.pie(pd.DataFrame({"구성": donut_labels, "금액": donut_vals}), values="금액", names="구성", hole=0.52)
        fig_budget.update_layout(height=340, margin=dict(t=10, b=10))
        st.plotly_chart(fig_budget, use_container_width=True)

        st.divider()

        # Channel recommendation: top channels by 매출비중 (donut only)
        st.markdown("### 유통 채널(상위)")
        if channel_mix:
            lab, val = topN_plus_other(channel_mix, n=8)
            st.plotly_chart(donut_chart(lab, val, title="매출 채널 구성(100%)", height=360), use_container_width=True)
        else:
            st.info("…매출비중 컬럼이 없어 채널 차트를 그릴 수 없습니다.")

        st.divider()

        # Monthly forecast (sales/ad) - safe distribution templates
        st.markdown("### 월별 매출/광고비(추정)")
        # choose profile for distribution
        profile = st.radio("월별 분배 가정", ["보수적", "기본", "공격"], horizontal=True, key=f"br_month_profile_{sub_mode}")

        # 12-month weights: conservative (front-light), base (mid-heavy), aggressive (front-heavy)
        if profile == "보수적":
            w = np.array([0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.10, 0.10, 0.09, 0.08, 0.07, 0.06])
        elif profile == "공격":
            w = np.array([0.10, 0.10, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07, 0.07, 0.06, 0.05, 0.03])
        else:
            w = np.array([0.06, 0.07, 0.08, 0.08, 0.09, 0.09, 0.09, 0.09, 0.08, 0.07, 0.06, 0.04])
        w = w / w.sum()

        # total expected revenue from units_sold * selling_price (simple)
        total_revenue = units_sold * selling_price
        monthly_revenue = total_revenue * w
        monthly_ad = marketing_budget * w

        month_df = pd.DataFrame({
            "월": [f"{i}월" for i in range(1, 13)],
            "예상 매출": monthly_revenue,
            "예상 광고비": monthly_ad,
        })
        month_df["ROAS"] = month_df["예상 매출"] / month_df["예상 광고비"].replace(0, np.nan)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=month_df["월"], y=month_df["예상 매출"], name="예상 매출"), secondary_y=False)
        fig.add_trace(go.Bar(x=month_df["월"], y=month_df["예상 광고비"], name="예상 광고비"), secondary_y=False)
        fig.add_trace(go.Scatter(x=month_df["월"], y=month_df["ROAS"], name="ROAS", mode="lines+markers"), secondary_y=True)
        fig.update_layout(height=420, barmode="group", margin=dict(t=10, b=10, l=10, r=10), legend=dict(orientation="h", y=1.02, x=0))
        st.plotly_chart(fig, use_container_width=True)

        # External: keep table minimal; Internal: can show table
        if sub_mode == "내부":
            disp = month_df.copy()
            disp["예상 매출"] = disp["예상 매출"].map(lambda x: f"{x:,.0f}")
            disp["예상 광고비"] = disp["예상 광고비"].map(lambda x: f"{x:,.0f}")
            disp["ROAS"] = disp["ROAS"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
            st.dataframe(disp, use_container_width=True, hide_index=True)
