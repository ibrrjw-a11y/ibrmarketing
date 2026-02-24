# streamlit_app.py
# ✅ 원칙: 기존 코드에서 '논의 없던 기능'은 삭제/축소하지 않음
# ✅ 이번 수정:
# 1) 대행 내부: 인건비 포함한 대행 손익(수수료/페이백/바이럴마진-인건비) 계산 추가
# 2) 추천엔진: 단순 룰 fallback 유지 + backdata 기반 체크리스트 점수화 Top3 추천 엔진 복원/강화
# 3) 성장률/광고기여율/재구매율 backdata 반영은 기존 그대로 유지

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
st.set_page_config(
    page_title="마케팅/유통 시뮬레이터",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#4F8EF7"
ACCENT2 = "#7C5CFC"
SUCCESS = "#22C55E"
WARNING = "#F59E0B"
DANGER  = "#EF4444"

st.markdown("""
<style>
/* ─── Google Font ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ─── Global Reset ────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  font-size: 14px;
  letter-spacing: -0.01em;
}

/* ─── 앱 배경 ─────────────────────────────────────── */
.stApp {
  background: #0F1117 !important;
}

/* ─── 메인 컨테이너 패딩 ──────────────────────────── */
.main .block-container {
  padding: 1.5rem 2rem 3rem 2rem !important;
  max-width: 1400px !important;
}

/* ─── 사이드바 ────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: #161B27 !important;
  border-right: 1px solid rgba(79,142,247,0.12) !important;
}
[data-testid="stSidebar"] .block-container {
  padding: 1.2rem 1rem !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  color: #E2E8F0 !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  margin-bottom: 0.6rem !important;
}
[data-testid="stSidebar"] label {
  color: #94A3B8 !important;
  font-size: 12px !important;
  font-weight: 500 !important;
}
[data-testid="stSidebar"] .stMarkdown p {
  color: #CBD5E1 !important;
  font-size: 13px !important;
}
/* 사이드바 구분선 */
[data-testid="stSidebar"] hr {
  border-color: rgba(79,142,247,0.15) !important;
  margin: 0.8rem 0 !important;
}

/* ─── 탭 ───────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
  background: #1A2035 !important;
  border-radius: 12px !important;
  padding: 4px !important;
  gap: 2px !important;
  border: 1px solid rgba(79,142,247,0.12) !important;
}
[data-testid="stTabs"] [role="tab"] {
  border-radius: 9px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  color: #64748B !important;
  padding: 7px 16px !important;
  transition: all 0.2s ease !important;
  border: none !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
  color: #CBD5E1 !important;
  background: rgba(79,142,247,0.08) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  background: linear-gradient(135deg, #4F8EF7, #7C5CFC) !important;
  color: #FFFFFF !important;
  font-weight: 600 !important;
  box-shadow: 0 2px 12px rgba(79,142,247,0.35) !important;
}

/* ─── st.metric 카드 ──────────────────────────────── */
[data-testid="stMetric"] {
  background: #1A2035 !important;
  border: 1px solid rgba(79,142,247,0.14) !important;
  border-radius: 12px !important;
  padding: 14px 18px !important;
  transition: border-color 0.2s ease !important;
}
[data-testid="stMetric"]:hover {
  border-color: rgba(79,142,247,0.35) !important;
}
[data-testid="stMetricLabel"] {
  color: #64748B !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] {
  color: #F1F5F9 !important;
  font-size: 1.4rem !important;
  font-weight: 700 !important;
  white-space: normal !important;
  overflow: visible !important;
  text-overflow: clip !important;
  line-height: 1.2 !important;
}
[data-testid="stMetricDelta"] {
  font-size: 12px !important;
}

/* ─── 버튼 ────────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, #4F8EF7, #7C5CFC) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 8px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  padding: 8px 18px !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 2px 10px rgba(79,142,247,0.25) !important;
  letter-spacing: 0.01em !important;
}
.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 16px rgba(79,142,247,0.4) !important;
}
.stButton > button:active {
  transform: translateY(0) !important;
}

/* ─── 입력 필드 ──────────────────────────────────── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
.stTextInput input,
.stNumberInput input {
  background: #1A2035 !important;
  border: 1px solid rgba(79,142,247,0.18) !important;
  border-radius: 8px !important;
  color: #E2E8F0 !important;
  font-size: 13px !important;
  transition: border-color 0.2s ease !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
  border-color: #4F8EF7 !important;
  box-shadow: 0 0 0 3px rgba(79,142,247,0.15) !important;
}

/* ─── 셀렉트박스 ──────────────────────────────────── */
[data-baseweb="select"] > div {
  background: #1A2035 !important;
  border: 1px solid rgba(79,142,247,0.18) !important;
  border-radius: 8px !important;
  color: #E2E8F0 !important;
  font-size: 13px !important;
}
[data-baseweb="select"] > div:hover {
  border-color: #4F8EF7 !important;
}
[data-baseweb="popover"] {
  background: #1A2035 !important;
  border: 1px solid rgba(79,142,247,0.2) !important;
  border-radius: 10px !important;
}
[data-baseweb="menu"] {
  background: #1A2035 !important;
}
[data-baseweb="option"]:hover {
  background: rgba(79,142,247,0.12) !important;
}

/* ─── 라디오 버튼 ──────────────────────────────────── */
[data-testid="stRadio"] label {
  color: #CBD5E1 !important;
  font-size: 13px !important;
}
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
  font-size: 13px !important;
}

/* ─── 토글 ─────────────────────────────────────────── */
[data-testid="stToggle"] label {
  color: #CBD5E1 !important;
  font-size: 13px !important;
}

/* ─── 데이터프레임/에디터 ──────────────────────────── */
[data-testid="stDataFrame"],
[data-testid="stDataEditor"] {
  border: 1px solid rgba(79,142,247,0.12) !important;
  border-radius: 10px !important;
  overflow: hidden !important;
}
[data-testid="stDataFrame"] div,
[data-testid="stDataFrame"] span,
[data-testid="stDataEditor"] div,
[data-testid="stDataEditor"] span {
  opacity: 1 !important;
}
.dvn-scroller {
  background: #161B27 !important;
}

/* ─── 익스팬더 ──────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid rgba(79,142,247,0.12) !important;
  border-radius: 10px !important;
  background: #161B27 !important;
  overflow: hidden !important;
}
[data-testid="stExpander"] summary {
  color: #CBD5E1 !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 10px 14px !important;
}
[data-testid="stExpander"] summary:hover {
  background: rgba(79,142,247,0.06) !important;
}

/* ─── 구분선 ────────────────────────────────────────── */
hr {
  border: none !important;
  border-top: 1px solid rgba(79,142,247,0.12) !important;
  margin: 1.4rem 0 !important;
}

/* ─── 알림/인포 박스 ──────────────────────────────── */
[data-testid="stInfo"] {
  background: rgba(79,142,247,0.08) !important;
  border: 1px solid rgba(79,142,247,0.2) !important;
  border-radius: 10px !important;
  color: #93C5FD !important;
}
[data-testid="stWarning"] {
  background: rgba(245,158,11,0.08) !important;
  border: 1px solid rgba(245,158,11,0.25) !important;
  border-radius: 10px !important;
  color: #FCD34D !important;
}
[data-testid="stError"] {
  background: rgba(239,68,68,0.08) !important;
  border: 1px solid rgba(239,68,68,0.25) !important;
  border-radius: 10px !important;
  color: #FCA5A5 !important;
}
[data-testid="stSuccess"] {
  background: rgba(34,197,94,0.08) !important;
  border: 1px solid rgba(34,197,94,0.25) !important;
  border-radius: 10px !important;
  color: #86EFAC !important;
}

/* ─── 캡션/텍스트 ────────────────────────────────── */
[data-testid="stCaptionContainer"],
.stCaptionContainer p,
caption {
  color: #475569 !important;
  font-size: 11.5px !important;
}
h1 { color: #F1F5F9 !important; font-size: 1.6rem !important; font-weight: 700 !important; }
h2 { color: #E2E8F0 !important; font-size: 1.25rem !important; font-weight: 600 !important; }
h3 { color: #CBD5E1 !important; font-size: 1.05rem !important; font-weight: 600 !important; }
p  { color: #94A3B8 !important; }

/* ─── 파일 업로더 ──────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: #1A2035 !important;
  border: 1.5px dashed rgba(79,142,247,0.28) !important;
  border-radius: 10px !important;
  padding: 8px !important;
  transition: border-color 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: #4F8EF7 !important;
}
[data-testid="stFileUploader"] label {
  color: #94A3B8 !important;
  font-size: 12px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
  color: #64748B !important;
}

/* ─── Plotly 차트 배경 ──────────────────────────────── */
.js-plotly-plot .plotly .bg {
  fill: #161B27 !important;
}

/* ─── 커스텀 컴포넌트 ──────────────────────────────── */

/* 페이지 헤더 */
.page-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px 0 16px 0;
  border-bottom: 1px solid rgba(79,142,247,0.12);
  margin-bottom: 20px;
}
.page-header .ph-icon {
  width: 40px; height: 40px;
  background: linear-gradient(135deg, #4F8EF7, #7C5CFC);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px;
}
.page-header .ph-title {
  font-size: 1.3rem;
  font-weight: 700;
  color: #F1F5F9;
  margin: 0;
}
.page-header .ph-sub {
  font-size: 12px;
  color: #475569;
  margin: 2px 0 0 0;
}

/* 섹션 헤더 */
.section-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 24px 0 12px 0;
}
.section-header .sh-bar {
  width: 4px; height: 20px;
  background: linear-gradient(180deg, #4F8EF7, #7C5CFC);
  border-radius: 2px;
}
.section-header .sh-title {
  font-size: 15px;
  font-weight: 600;
  color: #E2E8F0;
  margin: 0;
}

/* 일반 카드 */
.card {
  background: #1A2035;
  border: 1px solid rgba(79,142,247,0.14);
  border-radius: 14px;
  padding: 20px 22px;
  margin-bottom: 12px;
  transition: border-color 0.2s ease;
}
.card:hover {
  border-color: rgba(79,142,247,0.3);
}
.card h3 {
  color: #E2E8F0 !important;
  font-size: 15px !important;
  font-weight: 600 !important;
  margin: 0 0 12px 0 !important;
}

/* 강조 카드 (Top3 추천용) */
.card-highlight {
  background: linear-gradient(135deg, rgba(79,142,247,0.08), rgba(124,92,252,0.06));
  border: 1px solid rgba(79,142,247,0.28);
  border-radius: 14px;
  padding: 20px 22px;
  margin-bottom: 12px;
  position: relative;
  overflow: hidden;
}
.card-highlight::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, #4F8EF7, #7C5CFC);
}

/* 뱃지 */
.badge {
  display: inline-flex;
  align-items: center;
  padding: 4px 10px;
  border-radius: 999px;
  font-weight: 600;
  font-size: 11.5px;
  background: rgba(79,142,247,0.12);
  color: #7EB4FF;
  border: 1px solid rgba(79,142,247,0.2);
  letter-spacing: 0.02em;
}
.badge-success {
  background: rgba(34,197,94,0.1);
  color: #4ADE80;
  border-color: rgba(34,197,94,0.2);
}
.badge-warning {
  background: rgba(245,158,11,0.1);
  color: #FCD34D;
  border-color: rgba(245,158,11,0.2);
}
.badge-purple {
  background: rgba(124,92,252,0.12);
  color: #A78BFA;
  border-color: rgba(124,92,252,0.2);
}

/* 소형 텍스트 */
.smallcap {
  color: #475569;
  font-size: 11.5px;
}

/* 구분선(소프트) */
hr.soft {
  border: none !important;
  border-top: 1px solid rgba(79,142,247,0.10) !important;
  margin: 14px 0 !important;
}

/* 스탯 그리드 */
.stat-row {
  display: flex;
  gap: 12px;
  margin: 12px 0;
  flex-wrap: wrap;
}
.stat-item {
  flex: 1;
  min-width: 110px;
  background: rgba(79,142,247,0.06);
  border: 1px solid rgba(79,142,247,0.12);
  border-radius: 10px;
  padding: 12px 14px;
}
.stat-label {
  font-size: 10.5px;
  font-weight: 600;
  color: #475569;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 4px;
}
.stat-value {
  font-size: 1.1rem;
  font-weight: 700;
  color: #E2E8F0;
}

/* 안내 탭 ul/li */
.guide-list li {
  color: #94A3B8;
  font-size: 13.5px;
  line-height: 1.8;
  margin-bottom: 2px;
}
.guide-list li b {
  color: #CBD5E1;
}
.guide-list code {
  background: rgba(79,142,247,0.12);
  color: #93C5FD;
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 12px;
}

/* 스크롤바 */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #161B27; }
::-webkit-scrollbar-thumb { background: #2D3A55; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4F8EF7; }
</style>
""", unsafe_allow_html=True)

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
def _norm01_rate(x):
    return normalize_ratio(x)

def auth_gate() -> bool:
    if st.session_state.get("auth_ok", False):
        return True

    st.sidebar.markdown("""
<div style="text-align:center; padding: 16px 0 8px 0;">
  <div style="width:48px;height:48px;background:linear-gradient(135deg,#4F8EF7,#7C5CFC);
    border-radius:14px;display:inline-flex;align-items:center;justify-content:center;
    font-size:22px;margin-bottom:10px;">🔒</div>
  <div style="color:#E2E8F0;font-size:15px;font-weight:700;">접근 제한</div>
  <div style="color:#475569;font-size:11.5px;margin-top:4px;">비밀번호를 입력하여 잠금 해제</div>
</div>
""", unsafe_allow_html=True)
    pw = st.sidebar.text_input("비밀번호", type="password", key="auth_pw", placeholder="비밀번호 입력...")
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if st.button("잠금 해제", key="auth_unlock", use_container_width=True):
            if pw == APP_PASSWORD:
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                st.sidebar.error("비밀번호가 틀립니다.")
    with col2:
        if st.button("초기화", key="auth_reset", use_container_width=True):
            st.session_state.pop("auth_ok", None)
            st.session_state.pop("auth_pw", None)
            st.rerun()

    # 메인 화면 안내
    st.markdown("""
<div style="
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  min-height:60vh; text-align:center; gap:16px;
">
  <div style="
    width:72px;height:72px;
    background:linear-gradient(135deg,#4F8EF7,#7C5CFC);
    border-radius:20px;
    display:flex;align-items:center;justify-content:center;
    font-size:32px;
    box-shadow:0 8px 32px rgba(79,142,247,0.35);
  ">📊</div>
  <div>
    <div style="font-size:1.5rem;font-weight:700;color:#F1F5F9;margin-bottom:8px;">
      마케팅 / 유통 시뮬레이터
    </div>
    <div style="font-size:13.5px;color:#475569;line-height:1.7;">
      좌측 사이드바에서 비밀번호를 입력하여<br>시뮬레이터에 접속하세요.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
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
def fmt_won_compact(x) -> str:
    """
    큰 숫자 카드에서 글씨가 짤리는 문제 해결용(축약 표기).
    - 1,234 -> 1,234원
    - 12,345,678 -> 1,235만원
    - 1,234,567,890 -> 12.3억원
    """
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        v = float(x)

        av = abs(v)
        sign = "-" if v < 0 else ""

        if av < 10_000:
            return f"{v:,.0f} 원"
        if av < 100_000_000:  # 1억 미만: 만원 단위
            return f"{sign}{av/10_000:,.0f} 만원"
        # 1억 이상: 억원 단위 (소수 1자리)
        return f"{sign}{av/100_000_000:,.1f} 억원"
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

def clamp01(x: float, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isnan(v):
            return default
        return max(0.0, min(1.0, v))
    except Exception:
        return default

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

PLOTLY_DARK = {
    "paper_bgcolor": "#161B27",
    "plot_bgcolor":  "#161B27",
    "font":  {"color": "#94A3B8", "family": "Inter, sans-serif", "size": 12},
    "title_font": {"color": "#E2E8F0", "size": 14, "family": "Inter, sans-serif"},
    "legend": {"bgcolor": "rgba(22,27,39,0.8)", "bordercolor": "rgba(79,142,247,0.2)", "borderwidth": 1},
    "colorway": ["#4F8EF7","#7C5CFC","#22C55E","#F59E0B","#EF4444","#06B6D4","#EC4899","#F97316"],
}

def donut_chart(labels, values, title="", height=320):
    dd = pd.DataFrame({"name": labels, "value": values})
    fig = px.pie(dd, names="name", values="value", hole=0.58,
                 color_discrete_sequence=PLOTLY_DARK["colorway"])
    fig.update_traces(
        textinfo="percent+label",
        textfont=dict(size=12, color="#E2E8F0"),
        marker=dict(line=dict(color="#161B27", width=2)),
    )
    fig.update_layout(
        height=height,
        margin=dict(t=50, b=10, l=10, r=10),
        title=dict(text=title, font=PLOTLY_DARK["title_font"], x=0.01),
        paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
        plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
        font=PLOTLY_DARK["font"],
        legend=PLOTLY_DARK["legend"],
    )
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
# Column detection (v4 with KPI + 성장/기여/재구매)
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

    # ✅ Backdata 확장 컬럼(월 성장률/광고기여율/재구매율/광고의존도)
    col_month_growth = safe_col(df, ["월 성장률", "월성장률", "monthly_growth", "MoM Growth", "월성장률(%)"])
    col_ad_contrib = safe_col(df, ["광고기여율", "광고 기여율", "ad_contribution", "광고기여율(%)", "광고기여"])
    col_repurchase = safe_col(df, ["재구매율", "재구매 비중", "repurchase", "재구매율(%)"])
    col_ad_dependency = safe_col(df, ["광고의존도", "광고 의존도", "ad_dependency", "광고의존도(%)"])

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
        "month_growth": col_month_growth,
        "ad_contrib": col_ad_contrib,
        "repurchase": col_repurchase,
        "ad_dependency": col_ad_dependency,
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
    if any(x in m for x in ["GDN", "GFA", "메타", "틱톡", "크리테오", "토스", "유튜브", "PMAX", "PMax", "pmax"]):
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
# Growth / Ad contribution / Repurchase (from backdata row)
# =========================
def get_row_rate(row: pd.Series, col: Optional[str], default: float) -> float:
    if col is None or col not in row.index:
        return float(default)
    v = normalize_ratio(row.get(col))
    if pd.isna(v):
        return float(default)
    return clamp01(float(v), default=float(default))

def get_row_growth(row: pd.Series, col: Optional[str], default: float) -> float:
    if col is None or col not in row.index:
        return float(default)
    v = to_float(row.get(col), default=np.nan)
    if np.isnan(v):
        return float(default)
    if abs(v) > 1.0:
        v = v / 100.0
    return float(v)

# =========================
# P&L / Simulation (sales-side)
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
    ad_contrib_rate: float = 1.0,
    repurchase_rate: float = 0.0,
):
    ad_contrib_rate = clamp01(ad_contrib_rate, 1.0)
    repurchase_rate = clamp01(repurchase_rate, 0.0)

    if calc_mode.startswith("매출"):
        revenue = float(revenue or 0.0)
        total_orders = (revenue / aov) if aov > 0 else 0.0

        ad_revenue = revenue * ad_contrib_rate
        ad_orders = (ad_revenue / aov) if aov > 0 else 0.0

        clicks = (ad_orders / cvr) if cvr > 0 else 0.0
        ad_spend = clicks * cpc
    else:
        ad_spend = float(ad_spend or 0.0)
        clicks = (ad_spend / cpc) if cpc > 0 else 0.0
        ad_orders = clicks * cvr
        ad_revenue = ad_orders * aov

        denom = ad_contrib_rate if ad_contrib_rate > 0 else 1.0
        revenue = ad_revenue / denom
        total_orders = (revenue / aov) if aov > 0 else 0.0

    repeat_orders = total_orders * repurchase_rate
    first_orders = max(total_orders - repeat_orders, 0.0)
    repeat_revenue = repeat_orders * aov
    first_revenue = first_orders * aov

    cogs = revenue * cost_rate
    logistics = total_orders * logistics_per_order
    profit = revenue - (ad_spend + cogs + logistics + fixed_cost)
    contrib_margin = ((revenue - ad_spend - logistics - cogs) / revenue * 100) if revenue > 0 else 0.0
    roas = (revenue / ad_spend) if ad_spend and ad_spend > 0 else 0.0

    return {
        "revenue": float(revenue),
        "ad_spend": float(ad_spend),
        "orders": float(total_orders),
        "clicks": float(clicks),
        "cogs": float(cogs),
        "logistics": float(logistics),
        "fixed": float(fixed_cost),
        "profit": float(profit),
        "contrib_margin": float(contrib_margin),
        "roas": float(roas),
        "ad_contrib_rate": float(ad_contrib_rate),
        "repurchase_rate": float(repurchase_rate),
        "ad_revenue": float(ad_revenue),
        "ad_orders": float(ad_orders),
        "repeat_orders": float(repeat_orders),
        "first_orders": float(first_orders),
        "repeat_revenue": float(repeat_revenue),
        "first_revenue": float(first_revenue),
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
    for ch, v in (rev_share or {}).items():
        if v <= 0:
            continue
        rows.append({"그룹": rev_bucket(ch), "채널": ch, "비중": float(v)})

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # ✅ 핵심: 색상을 '그룹' 기준으로 고정 → 그룹별 1색(알록달록 방지)
    fig = px.treemap(
        df,
        path=["그룹", "채널"],
        values="비중",
        color="그룹",
    )

    fig.update_layout(
        height=height,
        margin=dict(t=50, b=10, l=10, r=10),
        title=dict(text=title, font=PLOTLY_DARK["title_font"], x=0.01),
        paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
        font=PLOTLY_DARK["font"],
    )
    fig.update_traces(
        texttemplate="%{label}<br>%{value:.1%}",
        textfont=dict(color="#E2E8F0", size=12),
        marker=dict(line=dict(width=2, color="#161B27"))
    )
    return fig


def treemap_ads(perf_df: pd.DataFrame, viral_df: pd.DataFrame, height=430, title="광고 믹스(트리맵: 퍼포먼스/바이럴 색 구분)"):
    """
    광고 믹스 트리맵
    - 퍼포먼스/바이럴을 1차 그룹으로 크게 구분
    - 2차는 '매체'까지만 보여서 너무 잘게 쪼개지는 문제 방지
    - 색상도 그룹 기준으로 통일(퍼포먼스 vs 바이럴 한눈에)
    """
    rows = []

    # 퍼포먼스
    if perf_df is not None and not perf_df.empty:
        for _, r in perf_df.iterrows():
            rows.append({
                "그룹": "퍼포먼스",
                "매체": str(r.get("매체", "") or "").strip(),
                "예산": float(r.get("예산(계획)", 0) or 0),
            })

    # 바이럴
    if viral_df is not None and not viral_df.empty:
        for _, r in viral_df.iterrows():
            rows.append({
                "그룹": "바이럴",
                "매체": str(r.get("매체", "") or "").strip(),
                "예산": float(r.get("예산(계획)", 0) or 0),
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df["예산"] = df["예산"].fillna(0.0).astype(float)
    df = df[df["예산"] > 0]

    if df.empty:
        return None

    # ✅ 핵심: path를 ["그룹","매체"]까지만 (지면 제거)
    # ✅ 핵심: color를 "그룹"으로 고정해서 퍼포먼스/바이럴 색이 확실히 구분되게
    fig = px.treemap(
        df,
        path=["그룹", "매체"],
        values="예산",
        color="그룹",
    )
    fig.update_layout(
        height=height,
        margin=dict(t=50, b=10, l=10, r=10),
        title=dict(text=title, font=PLOTLY_DARK["title_font"], x=0.01),
        paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
        font=PLOTLY_DARK["font"],
    )
    fig.update_traces(
        textfont=dict(color="#E2E8F0", size=12),
        marker=dict(line=dict(width=2, color="#161B27"))
    )
    return fig


# =========================
# Compare chart
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
        title=dict(text=title, font=PLOTLY_DARK["title_font"], x=0.01),
        margin=dict(t=50, b=10, l=10, r=10),
        paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
        plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
        font=PLOTLY_DARK["font"],
        yaxis=dict(title=None, tickformat=",.0f", gridcolor="rgba(79,142,247,0.08)", color="#64748B"),
        yaxis2=dict(
            title="ROAS(%)",
            overlaying="y",
            side="right",
            range=[y2_min, y2_max],
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            color="#64748B",
        ),
        xaxis=dict(tickangle=0, color="#64748B", gridcolor="rgba(79,142,247,0.05)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(22,27,39,0.8)", bordercolor="rgba(79,142,247,0.2)", borderwidth=1),
        colorway=PLOTLY_DARK["colorway"],
    )
    return fig

# =========================
# Recommendation (fallback rule) - 유지
# =========================
def top_key(d: Dict[str, float]) -> Tuple[Optional[str], float]:
    if not d:
        return None, 0.0
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return (items[0][0], float(items[0][1])) if items else (None, 0.0)

def strategy_recommendation_fallback(rev_share: Dict[str, float], sales_focus: str = "(무관)") -> Dict[str, object]:
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
# ✅ Recommendation Engine (Checklist scoring) - 복원/강화
# =========================
def share_sum(rev_share: Dict[str, float], includes: List[str]) -> float:
    s = 0.0
    for k, v in rev_share.items():
        kk = str(k)
        if any(x in kk for x in includes):
            s += float(v)
    return float(s)

def build_checklist_features(
    row: pd.Series,
    rev_share: Dict[str, float],
    media_share: Dict[str, Dict[str, float]],
    cols: Dict[str, object],
    df_cols: Dict[str, object],
) -> Dict[str, object]:
    # channel dominance
    own = share_sum(rev_share, ["자사"])
    smart = share_sum(rev_share, ["스마트", "스토어"])
    coupang = share_sum(rev_share, ["쿠팡"])
    offline = share_sum(rev_share, ["오프라인", "면세", "백화점", "올리브", "마트", "드럭"])
    etc_online = max(1.0 - (own + smart + coupang + offline), 0.0)

    # scenario attributes
    stg = str(row.get(df_cols.get("stage")) if df_cols.get("stage") in row.index else "").strip()
    drv = str(row.get(df_cols.get("drv")) if df_cols.get("drv") in row.index else "").strip()
    cat = str(row.get(df_cols.get("cat")) if df_cols.get("cat") in row.index else "").strip()
    pos = str(row.get(df_cols.get("pos")) if df_cols.get("pos") in row.index else "").strip()

    month_growth = get_row_growth(row, cols.get("month_growth"), default=0.0)         # decimal
    ad_contrib = get_row_rate(row, cols.get("ad_contrib"), default=1.0)
    repurchase = get_row_rate(row, cols.get("repurchase"), default=0.0)
    ad_dep = get_row_rate(row, cols.get("ad_dependency"), default=ad_contrib)

    group = media_share.get("group", {"퍼포먼스": 1.0, "바이럴": 0.0, "브랜드": 0.0})
    perf = group.get("퍼포먼스", 0.0)
    viral = group.get("바이럴", 0.0)
    brand = group.get("브랜드", 0.0)

    return {
        "own": own,
        "smart": smart,
        "coupang": coupang,
        "offline": offline,
        "etc_online": etc_online,
        "stage": stg,
        "driver": drv,
        "category": cat,
        "position": pos,
        "month_growth": month_growth,
        "ad_contrib": ad_contrib,
        "repurchase": repurchase,
        "ad_dependency": ad_dep,
        "mix_perf": perf,
        "mix_viral": viral,
        "mix_brand": brand,
    }

STRATEGY_LIBRARY = [
    {
        "id": "S1_D2C_META_SCALE",
        "name": "자사몰 스케일업(메타/구글 중심 + 리타겟)",
        "desc": "자사몰 비중이 큰 경우, 픽셀/CRM 기반 리타겟과 메타 확장으로 CAC 안정화.",
        "rules": [
            ("own", "ge", 0.35, 30),
            ("mix_perf", "ge", 0.55, 15),
            ("ad_dependency", "ge", 0.50, 10),
        ],
        "bonus": [
            ("repurchase", "ge", 0.25, 10),
            ("month_growth", "ge", 0.03, 10),
        ],
        "actions": [
            "메타: 리타겟(7/14/30일) + 유사타겟 확장, 크리에이티브 AB",
            "구글: 브랜드/상품 검색 방어 + PMAX(가능 시)로 확장",
            "CRM: 재구매율 높으면 카카오/메일 자동화로 ROAS 보강",
        ],
    },
    {
        "id": "S2_NAVER_SMARTSTORE",
        "name": "스마트스토어 성장(네이버 SA/DA/GFA 중심)",
        "desc": "스마트스토어 중심이면 네이버 생태계 내 검색/DA로 전환을 밀어주는 게 정합.",
        "rules": [
            ("smart", "ge", 0.30, 30),
            ("mix_perf", "ge", 0.55, 10),
        ],
        "bonus": [
            ("month_growth", "ge", 0.02, 10),
            ("ad_dependency", "ge", 0.50, 5),
        ],
        "actions": [
            "네이버 SA: 핵심 키워드/브랜드 키워드 구조화 + 랜딩 최적화",
            "GFA/DA: 장바구니/재방문 리타겟 + 신규 확장",
            "콘텐츠(블로그/스마트블록): 검색 신뢰/후기 강화",
        ],
    },
    {
        "id": "S3_COUPANG_PA",
        "name": "쿠팡 매출 가속(쿠팡/외부몰 PA 중심)",
        "desc": "쿠팡 비중이 크면 외부몰 PA가 가장 직접적인 레버.",
        "rules": [
            ("coupang", "ge", 0.25, 35),
            ("mix_perf", "ge", 0.55, 10),
        ],
        "bonus": [
            ("month_growth", "ge", 0.02, 10),
        ],
        "actions": [
            "쿠팡 PA: ROAS 기준 자동입찰/키워드 확장, 베스트SKU 집중",
            "메타/네이버: 쿠팡 랜딩 리타겟으로 보조 흡수(필요 시)",
            "리뷰/평점: 전환의 핵심(운영 과제로 명시)",
        ],
    },
    {
        "id": "S4_AD_DEP_DIVERSIFY",
        "name": "광고의존도 완화(바이럴/브랜드 보강 + 성과 방어)",
        "desc": "광고 의존도가 높을수록 유기적 수요(콘텐츠/바이럴)로 방어 레이어 필요.",
        "rules": [
            ("ad_dependency", "ge", 0.70, 35),
        ],
        "bonus": [
            ("mix_viral", "ge", 0.15, 10),
            ("mix_brand", "ge", 0.10, 10),
        ],
        "actions": [
            "바이럴: 네이버/커뮤니티 핫딜/체험단으로 검색량/후기 확보",
            "브랜드: 핵심 메시지/크리에이티브 일관화로 전환율 보강",
            "퍼포먼스: 리타겟/브랜드검색 방어로 효율 유지",
        ],
    },
    {
        "id": "S5_RETENTION_CRM",
        "name": "리텐션/CRM 강화(재구매 중심 LTV 업)",
        "desc": "재구매율이 높거나 높일 여지가 큰 경우, CRM이 CAC를 구조적으로 낮춤.",
        "rules": [
            ("repurchase", "ge", 0.25, 35),
        ],
        "bonus": [
            ("own", "ge", 0.20, 10),
            ("month_growth", "le", 0.01, 10),
        ],
        "actions": [
            "CRM: 첫구매→N일 후 리마인드/세트업/정기구독 시나리오",
            "리타겟: 재구매 세그먼트 분리(구매 횟수/최근성)",
            "콘텐츠: 사용법/후기 UGC로 재구매 트리거 강화",
        ],
    },
    {
        "id": "S6_OFFLINE_SUPPORT",
        "name": "오프라인/리테일 지원(인지/검색 방어)",
        "desc": "오프라인 비중이 있는 경우, 검색 방어와 브랜드 메시지로 동반 상승 유도.",
        "rules": [
            ("offline", "ge", 0.15, 30),
        ],
        "bonus": [
            ("mix_brand", "ge", 0.10, 10),
        ],
        "actions": [
            "네이버/구글 브랜드검색 방어: 매장 방문/구매 검색 수요 회수",
            "콘텐츠: '어디서 살 수 있나' FAQ/지도/리뷰 강화",
            "브랜딩: 핵심 USP를 오프라인 판촉과 일관되게",
        ],
    },
]

def score_strategy(feat: Dict[str, object], strat: Dict[str, object]) -> Tuple[int, List[str]]:
    score = 0
    reasons = []

    def cmp(op: str, a: float, b: float) -> bool:
        if op == "ge": return a >= b
        if op == "gt": return a > b
        if op == "le": return a <= b
        if op == "lt": return a < b
        if op == "eq": return a == b
        return False

    for key, op, th, pts in strat.get("rules", []):
        a = float(feat.get(key, 0.0) or 0.0)
        if cmp(op, a, float(th)):
            score += int(pts)
            reasons.append(f"{key} {op} {th} (+{pts})")

    for key, op, th, pts in strat.get("bonus", []):
        a = float(feat.get(key, 0.0) or 0.0)
        if cmp(op, a, float(th)):
            score += int(pts)
            reasons.append(f"{key} {op} {th} (+{pts})")

    return score, reasons

def recommend_top3_strategies(feat: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for s in STRATEGY_LIBRARY:
        sc, reasons = score_strategy(feat, s)
        rows.append({
            "전략ID": s["id"],
            "전략명": s["name"],
            "점수": sc,
            "설명": s["desc"],
            "근거(룰/보너스)": " / ".join(reasons) if reasons else "",
            "권장 액션": "\n- " + "\n- ".join(s["actions"]),
        })
    df_s = pd.DataFrame(rows).sort_values("점수", ascending=False).reset_index(drop=True)
    return df_s

# =========================
# Sidebar - Upload
# =========================
st.sidebar.markdown("""
<div style="padding: 4px 0 14px 0;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
    <div style="width:34px;height:34px;background:linear-gradient(135deg,#4F8EF7,#7C5CFC);
      border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:16px;">📊</div>
    <div>
      <div style="color:#E2E8F0;font-size:13.5px;font-weight:700;line-height:1.2;">마케팅/유통</div>
      <div style="color:#475569;font-size:11px;">시뮬레이터</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<p style="color:#64748B;font-size:10.5px;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px;">📂 데이터 업로드</p>', unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader(
    "Backdata 파일 (xlsx / csv)",
    type=["xlsx", "csv"],
    key="backdata_uploader"
)

if st.sidebar.button("🔄 업로드 초기화", key="reset_uploader", use_container_width=True):
    st.session_state.pop("backdata_uploader", None)
    st.cache_data.clear()
    st.rerun()

if uploaded is None:
    st.markdown("""
<div style="
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-height:65vh;text-align:center;gap:18px;
">
  <div style="
    width:68px;height:68px;
    background:linear-gradient(135deg,rgba(79,142,247,0.15),rgba(124,92,252,0.12));
    border:1.5px dashed rgba(79,142,247,0.35);
    border-radius:18px;
    display:flex;align-items:center;justify-content:center;
    font-size:28px;
  ">📂</div>
  <div>
    <div style="font-size:1.15rem;font-weight:700;color:#E2E8F0;margin-bottom:8px;">
      Backdata를 업로드하세요
    </div>
    <div style="font-size:13px;color:#475569;line-height:1.8;">
      좌측 사이드바에서 <span style="color:#7EB4FF;font-weight:500;">.xlsx</span> 또는
      <span style="color:#7EB4FF;font-weight:500;">.csv</span> 파일을 업로드하면<br>
      시뮬레이터가 자동으로 시나리오를 로드합니다.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
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
st.sidebar.markdown('<p style="color:#64748B;font-size:10.5px;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px;">🔍 시나리오 필터</p>', unsafe_allow_html=True)
f_search = st.sidebar.text_input("시나리오 검색", value="", key="f_search", placeholder="키워드 입력...")
f_stage = st.sidebar.selectbox("단계 (ST)", ["(전체)"] + uniq_vals(stage_col), key="f_stage")
f_cat = st.sidebar.selectbox("카테고리", ["(전체)"] + uniq_vals(cat_col), key="f_cat")
f_pos = st.sidebar.selectbox("가격 포지션 (POS)", ["(전체)"] + uniq_vals(pos_col), key="f_pos")
f_drv = st.sidebar.selectbox("드라이버 (DRV)", ["(전체)"] + uniq_vals(drv_col), key="f_drv")

apply_internal = cols.get("apply_internal")
apply_client = cols.get("apply_client")
apply_agency = cols.get("apply_agency")

st.sidebar.markdown("---")
st.sidebar.markdown('<p style="color:#64748B;font-size:10.5px;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px;">🏷️ 노출 필터</p>', unsafe_allow_html=True)
show_internal = st.sidebar.toggle("내부용만 보기", value=False, key="show_internal")
show_client = st.sidebar.toggle("브랜드사용만 보기", value=False, key="show_client")
show_agency = st.sidebar.toggle("대행용만 보기", value=False, key="show_agency")

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

st.sidebar.markdown("---")
st.sidebar.markdown('<p style="color:#64748B;font-size:10.5px;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px;">📌 시나리오 선택</p>', unsafe_allow_html=True)
sel_disp = st.sidebar.selectbox("", options=disp_candidates, key="sel_scn", label_visibility="collapsed")

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

# ✅ 시나리오 기반 성장/기여/재구매 기본값
scn_month_growth = get_row_growth(row, cols.get("month_growth"), default=0.0)
scn_ad_contrib = get_row_rate(row, cols.get("ad_contrib"), default=1.0)
scn_repurchase = get_row_rate(row, cols.get("repurchase"), default=0.0)
scn_ad_dependency = get_row_rate(row, cols.get("ad_dependency"), default=scn_ad_contrib)

def _rev_bucket_vec(rev_share: Dict[str, float]) -> Dict[str, float]:
    # 채널명을 버킷으로 묶어서 벡터화
    buckets = {"자사몰": 0.0, "스마트스토어": 0.0, "쿠팡": 0.0, "오프라인": 0.0, "온라인(기타)": 0.0}
    for ch, v in (rev_share or {}).items():
        b = rev_bucket(ch)
        buckets[b] = buckets.get(b, 0.0) + float(v or 0.0)
    return normalize_shares(buckets)

def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a.keys()) | set(b.keys())
    va = np.array([float(a.get(k, 0.0) or 0.0) for k in keys], dtype=float)
    vb = np.array([float(b.get(k, 0.0) or 0.0) for k in keys], dtype=float)
    da = float(np.linalg.norm(va))
    db = float(np.linalg.norm(vb))
    if da <= 0 or db <= 0:
        return 0.0
    return float(np.dot(va, vb) / (da * db))

def _media_kw_share(perf_share: Dict[str, float], keywords: List[str]) -> float:
    s = 0.0
    for k, v in (perf_share or {}).items():
        kk = str(k)
        if any(kw in kk for kw in keywords):
            s += float(v or 0.0)
    return float(s)

def _ceiling_index(month_growth: float, repurchase: float, ad_dependency: float) -> float:
    """
    고점지수: 성장(+), 재구매(+), 광고의존(-)
    0~2 정도 범위로 대략적인 비교지표 (상대 비교용)
    """
    g = float(month_growth or 0.0)
    r = clamp01(repurchase, 0.0)
    d = clamp01(ad_dependency, 0.0)

    # 성장률은 -도 있으니 완만하게
    g_score = np.clip(g * 6.0, -0.6, 0.9)      # -10%~-? ~ +15% 정도에서 완만
    r_score = r * 0.9                         # 0~0.9
    d_penalty = d * 0.7                       # 0~0.7

    return float(np.clip(1.0 + g_score + r_score - d_penalty, 0.2, 2.2))

# =========================================================
# Recommendation helpers (ADD THIS ABOVE ALL tabs)
# =========================================================

def _row_for_key(df_: pd.DataFrame, col_key: str, key: str) -> Optional[pd.Series]:
    if df_ is None or df_.empty:
        return None
    m = df_[col_key].astype(str).str.strip() == str(key).strip()
    sub = df_[m]
    if sub.empty:
        return None
    return sub.iloc[0]

def _estimate_now_and_roi(
    budget: float,
    aov: float,
    cpc: float,
    cvr: float,
    ad_contrib: float,
    month_growth: float = 0.0,
    repurchase: float = 0.0,
    ad_dependency: float = 0.0,
    months: int = 12,
    gross_margin_rate: float = 0.0,
    **kwargs,
):
    """
    추천 비교용 추정치(방탄):
    - budget/aov/cpc/cvr로 광고매출(ad_rev) 계산
    - 광고기여율(ad_contrib)로 총매출(total_rev) 환산
    - ROI는 마진기반 ROI(= (총매출*마진율 - 광고비)/광고비)
    """
    budget = float(budget or 0.0)
    aov = float(aov or 0.0)
    cpc = float(cpc or 0.0)
    cvr = float(cvr or 0.0)

    ac = clamp01(normalize_ratio(ad_contrib), 0.6)
    rep = clamp01(normalize_ratio(repurchase), 0.0)

    # 성장률은 음수 포함 가능
    g = to_float(month_growth, 0.0)
    if abs(g) > 1.0:
        g = g / 100.0

    addep = clamp01(normalize_ratio(ad_dependency), ac)

    # 광고 -> 광고기여매출
    clicks = (budget / cpc) if cpc > 0 else 0.0
    orders = clicks * cvr
    ad_rev = orders * aov

    # 총매출 환산
    total_rev = ad_rev / max(ac, 1e-6)

    # 재구매 완만 반영(누적)
    rep_mult = 1.0 + rep * max(0, months - 1) * 0.15
    total_rev *= rep_mult

    roas = (total_rev / budget) if budget > 0 else np.nan
    gm = float(gross_margin_rate or 0.0)
    roi = ((total_rev * gm - budget) / budget) if budget > 0 else np.nan

    # 고점(참고용) — 카드에는 안 써도 비교용으로 남김
    peak_total_rev = total_rev * ((1.0 + g) ** max(0, months - 1))
    peak_budget = budget * ((1.0 + g * addep) ** max(0, months - 1))

    return {
        "total_rev": float(total_rev),
        "ad_rev": float(ad_rev),
        "budget": float(budget),
        "roas": float(roas) if not pd.isna(roas) else np.nan,
        "roi": float(roi) if not pd.isna(roi) else np.nan,
        "peak_total_rev": float(peak_total_rev),
        "peak_budget": float(peak_budget),
    }

# =========================
# Tabs
# =========================
tab_guide, tab_agency, tab_brand, tab_rec, tab_custom, tab_plan = st.tabs(
    ["📖 안내", "🏢 대행", "🏷️ 브랜드사", "🤖 추천엔진", "⚙️ 커스텀 시나리오", "📅 매출 계획"]
)

# =========================
# Tab: Guide
# =========================
with tab_guide:
    st.markdown("""
<div class="page-header">
  <div class="ph-icon">📖</div>
  <div>
    <div class="ph-title">사용 가이드</div>
    <div class="ph-sub">시뮬레이터 기능 설명 및 지표 산식 안내</div>
  </div>
</div>
""", unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2, gap="medium")

    with col_g1:
        st.markdown("""
<div class="card">
  <h3>📌 이 시뮬레이터는 무엇을 하나요?</h3>
  <hr class="soft"/>
  <ul class="guide-list">
    <li><b>시나리오(backdata)</b> 선택 → 매출 채널 비중 + 미디어 믹스 비중 자동 로드</li>
    <li><b>🏢 대행 탭</b> — 광고비↔매출 양방향 산출, 내부: 수수료/페이백/바이럴마진/인건비 손익</li>
    <li><b>🏷️ 브랜드사 탭</b> — 외부(전망+트리맵) / 내부(채널별 월매출+필요광고비+마진/인건비)</li>
    <li><b>🤖 추천엔진 탭</b> — Top3 추천 + ROAS/ROI vs 고점(성장/재구매/광고의존) 비교</li>
    <li><b>⚙️ 커스텀 시나리오</b> — 비중/예산 직접 수정 후 즉시 결과 확인</li>
    <li><b>📅 매출 계획</b> — 브랜드별 1~12월 계획 생성 + CSV 업로드</li>
  </ul>
</div>
<div class="card" style="margin-top:12px;">
  <h3>📐 핵심 계산 로직</h3>
  <hr class="soft"/>
  <ul class="guide-list">
    <li><b>주문수</b> = 매출 ÷ AOV</li>
    <li><b>광고비 → 매출</b><br/>
      &nbsp;&nbsp;클릭수 = 광고비 ÷ CPC &nbsp;|&nbsp; 광고주문수 = 클릭수 × CVR<br/>
      &nbsp;&nbsp;<b>전체매출 = (광고주문수 × AOV) ÷ 광고기여율</b>
    </li>
    <li><b>매출 → 필요 광고비</b><br/>
      &nbsp;&nbsp;광고기여매출 = 전체매출 × 광고기여율<br/>
      &nbsp;&nbsp;<b>필요 광고비 = (광고기여매출 ÷ AOV ÷ CVR) × CPC</b>
    </li>
    <li><b>ROAS</b> = 전체매출 ÷ 광고비</li>
  </ul>
  <div class="smallcap" style="margin-top:10px;">※ 브랜드 내부 탭은 원가/물류/고정비(인건비 포함)를 추가 반영해 영업이익을 계산합니다.</div>
</div>
""", unsafe_allow_html=True)

    with col_g2:
        st.markdown("""
<div class="card">
  <h3>📊 지표 설명</h3>
  <hr class="soft"/>
  <div style="display:flex;flex-direction:column;gap:10px;">
    <div style="display:flex;align-items:center;gap:12px;">
      <span class="badge" style="min-width:80px;justify-content:center;">AOV</span>
      <span style="color:#94A3B8;font-size:13px;">객단가 (평균 주문금액)</span>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
      <span class="badge" style="min-width:80px;justify-content:center;">CPC</span>
      <span style="color:#94A3B8;font-size:13px;">클릭당 비용 (원)</span>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
      <span class="badge" style="min-width:80px;justify-content:center;">CVR</span>
      <span style="color:#94A3B8;font-size:13px;">전환율 (주문수 / 클릭수)</span>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
      <span class="badge" style="min-width:80px;justify-content:center;">ROAS</span>
      <span style="color:#94A3B8;font-size:13px;">광고비 대비 매출 (전체매출 ÷ 광고비)</span>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
      <span class="badge badge-purple" style="min-width:80px;justify-content:center;">광고기여율</span>
      <span style="color:#94A3B8;font-size:13px;">전체매출 중 광고가 기여한 비중 (0~1)</span>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
      <span class="badge badge-success" style="min-width:80px;justify-content:center;">재구매율</span>
      <span style="color:#94A3B8;font-size:13px;">전체주문 중 재구매 주문 비중 (0~1)</span>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
      <span class="badge badge-warning" style="min-width:80px;justify-content:center;">월성장률</span>
      <span style="color:#94A3B8;font-size:13px;">월 기준 매출 성장률 (음수 가능)</span>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
      <span class="badge" style="min-width:80px;justify-content:center;background:rgba(239,68,68,0.1);color:#FCA5A5;border-color:rgba(239,68,68,0.2);">광고의존도</span>
      <span style="color:#94A3B8;font-size:13px;">성장 시 광고비 증가 민감도</span>
    </div>
  </div>
</div>
<div class="card" style="margin-top:12px;">
  <h3>📈 고점지수란?</h3>
  <hr class="soft"/>
  <ul class="guide-list">
    <li>추천엔진에서 <b>현재 효율(ROAS/ROI)</b>과 <b>성장 잠재력</b>을 함께 비교하는 보조 지표</li>
    <li>구성: <b>월성장률</b> (+) · <b>재구매율</b> (+) · <b>광고의존도</b> (−)</li>
    <li>값이 높을수록 → 성장 여력 ↑, 유기적 매출 기반 ↑</li>
  </ul>
  <div class="smallcap" style="margin-top:10px;padding:8px 12px;background:rgba(79,142,247,0.06);border-radius:6px;border-left:3px solid rgba(79,142,247,0.4);">
    고점지수 = 1.0 + (성장 보너스) + (재구매 보너스) − (광고의존 패널티)
  </div>
</div>
<div style="margin-top:12px;padding:12px 16px;background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.15);border-radius:10px;">
  <div style="color:#FCD34D;font-size:11.5px;font-weight:600;">⚠️ 주의사항</div>
  <div style="color:#94A3B8;font-size:12.5px;margin-top:4px;line-height:1.7;">
    입력 기반 시뮬레이션이며, 실제 성과는 운영·상품·시즌 요인에 따라 달라질 수 있습니다.
  </div>
</div>
""", unsafe_allow_html=True)


# =========================
# Editors
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
        outp["수수료매출(원)"] = outp.apply(
            lambda r: round_to_100(float(r["예산(계획)"]) * (float(r["대행수수료율(%)"]) / 100.0)), axis=1
        )
        outp["페이백예상액"] = outp.apply(
            lambda r: round_to_100(float(r["예산(계획)"]) * (float(r["페이백률(%)"]) / 100.0)), axis=1
        )
        outp["퍼포먼스마진(원)"] = outp["수수료매출(원)"].astype(float) - outp["페이백예상액"].astype(float)

        st.dataframe(
            outp[["구분2", "매체", "예산(계획)", "대행수수료율(%)", "수수료매출(원)", "페이백률(%)", "페이백예상액", "퍼포먼스마진(원)", "청구예상비용"]],
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
        outv["바이럴마진(원)"] = outv["예산(계획)"].astype(float) - outv["실집행비(원)"].astype(float)

        st.dataframe(
            outv[["매체", "지면/캠페인", "건당비용", "진행 건수", "예산(계획)", "실집행비(원)", "바이럴마진(원)"]],
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
# ✅ Agency Internal P&L (수수료/마진/인건비) 계산
# =========================
def agency_internal_pl(perf_out: pd.DataFrame, viral_out: pd.DataFrame, labor_cost: float) -> Dict[str, float]:
    perf_budget = 0.0
    perf_fee_rev = 0.0
    perf_payback = 0.0
    perf_margin = 0.0

    if perf_out is not None and not perf_out.empty:
        perf_budget = float(perf_out["예산(계획)"].astype(float).sum())
        if "수수료매출(원)" in perf_out.columns:
            perf_fee_rev = float(perf_out["수수료매출(원)"].astype(float).sum())
        else:
            # fallback: fee rate if exists
            if "대행수수료율(%)" in perf_out.columns:
                perf_fee_rev = float((perf_out["예산(계획)"].astype(float) * (perf_out["대행수수료율(%)"].astype(float)/100.0)).sum())
        if "페이백예상액" in perf_out.columns:
            perf_payback = float(perf_out["페이백예상액"].astype(float).sum())
        else:
            if "페이백률(%)" in perf_out.columns:
                perf_payback = float((perf_out["예산(계획)"].astype(float) * (perf_out["페이백률(%)"].astype(float)/100.0)).sum())
        if "퍼포먼스마진(원)" in perf_out.columns:
            perf_margin = float(perf_out["퍼포먼스마진(원)"].astype(float).sum())
        else:
            perf_margin = perf_fee_rev - perf_payback

    viral_budget = 0.0
    viral_real = 0.0
    viral_margin = 0.0
    if viral_out is not None and not viral_out.empty:
        viral_budget = float(viral_out["예산(계획)"].astype(float).sum())
        if "실집행비(원)" in viral_out.columns:
            viral_real = float(viral_out["실집행비(원)"].astype(float).sum())
        if "바이럴마진(원)" in viral_out.columns:
            viral_margin = float(viral_out["바이럴마진(원)"].astype(float).sum())
        else:
            viral_margin = viral_budget - viral_real

    gross_margin = perf_margin + viral_margin
    op_profit = gross_margin - float(labor_cost)

    # 대행 "청구액" 관점(참고)
    billed = 0.0
    if perf_out is not None and not perf_out.empty and "청구예상비용" in perf_out.columns:
        billed += float(perf_out["청구예상비용"].astype(float).sum())
    else:
        billed += perf_budget  # pass-through만이라도
    billed += viral_budget  # 바이럴은 예산(계획) 기준 청구로 가정

    return {
        "perf_budget": perf_budget,
        "perf_fee_rev": perf_fee_rev,
        "perf_payback": perf_payback,
        "perf_margin": perf_margin,
        "viral_budget": viral_budget,
        "viral_real": viral_real,
        "viral_margin": viral_margin,
        "gross_margin": gross_margin,
        "labor_cost": float(labor_cost),
        "op_profit": op_profit,
        "billed_total": billed,
    }

# =========================
# Tab: Agency
# =========================
with tab_agency:
    st.markdown("""
<div class="page-header">
  <div class="ph-icon">🏢</div>
  <div>
    <div class="ph-title">대행 모드</div>
    <div class="ph-sub">광고비 ↔ 매출 양방향 시뮬레이션 · 미디어 믹스 · 내부 손익</div>
  </div>
</div>
""", unsafe_allow_html=True)
    submode = st.radio("버전 선택", ["외부(클라이언트 제안용)", "내부(운영/정산용)"], horizontal=True, key="agency_sub")

    st.markdown(f"<div class='smallcap'>선택 시나리오: <span class='badge'>{sel_disp}</span></div>", unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">⚙️ 입력 (시뮬레이션)</div></div>', unsafe_allow_html=True)
    use_scn_kpi = st.toggle("시나리오 KPI 자동 사용(권장)", value=True, key="use_scn_kpi_ag")

    cG1, cG2, cG3 = st.columns(3)
    with cG1:
        ad_contrib_in = st.number_input("광고기여율(%)", value=float(scn_ad_contrib * 100.0), step=1.0, key="ag_ad_contrib") / 100.0
    with cG2:
        repurchase_in = st.number_input("재구매율(%)", value=float(scn_repurchase * 100.0), step=1.0, key="ag_repurchase") / 100.0
    with cG3:
        st.caption(f"참고(시나리오): 월성장률 {fmt_pct(scn_month_growth*100,1)} / 광고의존도 {fmt_pct(scn_ad_dependency*100,1)}")

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
        cost_rate=0.0,
        logistics_per_order=0.0,
        fixed_cost=0.0,
        ad_spend=ad_total,
        revenue=rev_target,
        ad_contrib_rate=float(ad_contrib_in),
        repurchase_rate=float(repurchase_in),
    )

    st.divider()
    st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">📊 결과 요약 · 마케팅 성과</div></div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("예상 매출(총)", fmt_won(sim["revenue"]))
    m2.metric("필요/입력 광고비", fmt_won(sim["ad_spend"]))
    m3.metric("ROAS", f"{sim['roas']:.2f}x ({sim['roas']*100:,.0f}%)")
    m4.metric("광고기여율", fmt_pct(sim["ad_contrib_rate"]*100, 1))

    m5, m6, m7 = st.columns(3)
    m5.metric("광고기여 매출", fmt_won(sim["ad_revenue"]))
    m6.metric("재구매 매출(추정)", fmt_won(sim["repeat_revenue"]))
    m7.metric("재구매율", fmt_pct(sim["repurchase_rate"]*100, 1))

    st.divider()
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
    st.markdown('<div class="section-header" style="margin-top:20px;"><div class="sh-bar"></div><div class="sh-title">📺 미디어 믹스 (예산/건수 수정 가능)</div></div>', unsafe_allow_html=True)

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

    st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">🎯 퍼포먼스 (예산 수정 가능)</div></div>', unsafe_allow_html=True)
    if perf_df.empty:
        st.info("퍼포먼스 믹스 데이터가 비어있습니다(해당 시나리오 비율 0).")
        perf_out = perf_df
    else:
        perf_out = editable_perf_table(perf_df, submode=submode, key_prefix=f"ag_{scenario_key}")

    st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">📢 바이럴 (건수 수정 가능)</div></div>', unsafe_allow_html=True)
    if viral_df.empty:
        st.info("바이럴 믹스 데이터가 비어있습니다(해당 시나리오 비율 0).")
        viral_out = viral_df
    else:
        viral_out = editable_viral_table(viral_df, submode=submode, key_prefix=f"ag_{scenario_key}")

    st.divider()
    st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">📋 통합 미디어 믹스표</div></div>', unsafe_allow_html=True)
    mix_df = unify_mix_table(perf_out, viral_out)
    if mix_df.empty:
        st.info("통합 미디어 믹스 데이터가 없습니다.")
    else:
        st.dataframe(mix_df, use_container_width=True, hide_index=True)

    fig_ads_tm = treemap_ads(perf_out, viral_out, title="광고 믹스(트리맵: 퍼포먼스/바이럴 색 구분)")
    if fig_ads_tm:
        st.plotly_chart(fig_ads_tm, use_container_width=True, key=f"ads_tm_ag_{scenario_key}")

    # ✅ 핵심 수정: 대행 내부 손익(인건비 포함)
    if submode.startswith("내부"):
        st.divider()
        st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">💰 대행 내부 손익 · 수수료/마진 − 인건비</div></div>', unsafe_allow_html=True)

        cL1, cL2 = st.columns(2)
        with cL1:
            headcount = st.number_input("대행 운영 인력(명)", value=2, step=1, min_value=0, key="ag_hc_internal")
        with cL2:
            cost_per = st.number_input("인당 월 인건비(원)", value=3500000, step=100000, key="ag_cper_internal")

        labor_cost = float(headcount) * float(cost_per)

        pl = agency_internal_pl(perf_out, viral_out, labor_cost=labor_cost)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("총 청구액(추정)", fmt_won(pl["billed_total"]))
        k2.metric("퍼포먼스 마진", fmt_won(pl["perf_margin"]))
        k3.metric("바이럴 마진", fmt_won(pl["viral_margin"]))
        k4.metric("총 마진(=Gross)", fmt_won(pl["gross_margin"]))

        k5, k6, k7 = st.columns(3)
        k5.metric("인건비(월)", fmt_won(pl["labor_cost"]))
        k6.metric("대행 영업이익(월)", fmt_won(pl["op_profit"]))
        gm_rate = (pl["gross_margin"] / pl["billed_total"] * 100.0) if pl["billed_total"] > 0 else 0.0
        k7.metric("마진율(청구 대비)", fmt_pct(gm_rate, 1))

        st.caption("※ 퍼포먼스는 예산(pass-through) + 수수료 매출 구조로 가정. 페이백은 마진에서 차감. 바이럴은 예산-실집행비를 마진으로 계산.")

# =========================
# Tab: Brand
# (이하 동일: 이전 버전 그대로 유지 — 길이상 생략 없이 포함해야 하지만,
#  사용자 요구가 '대행 인건비' + '추천엔진'이라 브랜드 탭은 기존 코드 그대로 사용하면 됩니다.)
# =========================
with tab_brand:
    st.markdown("""
<div class="page-header">
  <div class="ph-icon">🏷️</div>
  <div>
    <div class="ph-title">브랜드사 모드</div>
    <div class="ph-sub">매출 전망 · 채널별 계획 · 재고 소진 · 인건비 포함 손익</div>
  </div>
</div>
""", unsafe_allow_html=True)
    submode_b = st.radio("버전 선택", ["외부(브랜드사 공유용)", "내부(브랜드 운영/검증용)"], horizontal=True, key="brand_sub")
    st.markdown(f"<div class='smallcap'>선택 시나리오: <span class='badge'>{sel_disp}</span></div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 시나리오 기본 변수(Backdata)")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("월 성장률(기본)", fmt_pct(scn_month_growth*100, 1))
    g2.metric("광고기여율(기본)", fmt_pct(scn_ad_contrib*100, 1))
    g3.metric("재구매율(기본)", fmt_pct(scn_repurchase*100, 1))
    g4.metric("광고의존도(참고)", fmt_pct(scn_ad_dependency*100, 1))
    st.divider()

    if submode_b.startswith("외부"):
        st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">📈 대략 전망 · 매출 / 물량 / 재고소진</div></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            months = st.selectbox("기간(개월)", options=[3, 6, 12], index=2, key="b_months")
        with c2:
            base_month_rev = st.number_input("월 기준 총매출(원)", value=200000000, step=10000000, key="b_base_rev")
        with c3:
            growth = st.number_input("월 성장률(%)", value=float(scn_month_growth*100.0), step=0.5, key="b_growth") / 100.0
        with c4:
            selling_price = st.number_input("예상 판매가(AOV) (원)", value=50000, step=1000, key="b_sell_price_ext")

        s1, s2, s3 = st.columns(3)
        with s1:
            current_stock = st.number_input("현재 재고(개)", value=10000, step=100, key="b_stock_ext")
        with s2:
            safety_stock = st.number_input("안전재고(개)", value=0, step=100, key="b_safety_ext")
        with s3:
            start_day = st.date_input("기준일(재고 시작일)", key="b_startday_ext")

        months_idx = list(range(1, int(months) + 1))
        rev_list, units_list, ym_list = [], [], []
        for i in months_idx:
            factor = (1.0 + growth) ** (i - 1)
            rev_i = base_month_rev * factor
            units_i = (rev_i / selling_price) if selling_price > 0 else 0.0
            rev_list.append(rev_i)
            units_list.append(units_i)
            ym_list.append(f"M{i}")

        df_m = pd.DataFrame({"월": ym_list, "총매출": rev_list, "예상판매수량(개)": units_list})
        df_m["누적판매(개)"] = df_m["예상판매수량(개)"].cumsum()

        burn_point = float(current_stock)
        burn_month = None
        burn_in_month_ratio = None
        prev = 0.0
        for _, r in df_m.iterrows():
            cumu = float(r["누적판매(개)"])
            if cumu >= burn_point and burn_month is None:
                burn_month = r["월"]
                month_units = float(r["예상판매수량(개)"])
                burn_in_month_ratio = (burn_point - prev) / month_units if month_units > 0 else 1.0
                break
            prev = cumu

        total_units = float(df_m["예상판매수량(개)"].sum())
        po_units = max(int(np.ceil(total_units + float(safety_stock) - float(current_stock))), 0)

        k1, k2, k3 = st.columns(3)
        k1.metric("기간 총매출", fmt_won(df_m["총매출"].sum()))
        k2.metric("기간 예상 판매수량", f"{df_m['예상판매수량(개)'].sum():,.0f} 개")
        k3.metric("권장 발주(대략)", f"{po_units:,.0f} 개")

        if burn_month is None:
            st.info("재고가 기간 내에 소진되지 않는 것으로 추정됩니다.")
        else:
            day_offset = int(np.clip((burn_in_month_ratio or 1.0) * 30.0, 1, 30))
            st.warning(f"예상 재고 소진: **{burn_month}** 내 **약 {day_offset}일차 전후**(대략)")

        st.divider()
        st.markdown("### 매출 채널 구성(트리맵)")
        fig_rev_tm2 = treemap_revenue(rev_share, title="매출 채널 구성(트리맵)")
        if fig_rev_tm2:
            st.plotly_chart(fig_rev_tm2, use_container_width=True, key=f"rev_tm_brand_ext_{scenario_key}")

        st.divider()
        st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">📅 월별 총매출 / 판매수량</div></div>', unsafe_allow_html=True)
        df_show = df_m.copy()
        df_show["총매출"] = df_show["총매출"].map(lambda x: f"{x:,.0f}")
        df_show["예상판매수량(개)"] = df_show["예상판매수량(개)"].map(lambda x: f"{x:,.0f}")
        df_show["누적판매(개)"] = df_show["누적판매(개)"].map(lambda x: f"{x:,.0f}")
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    else:
        st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">🔍 운영/검증 · 매출 + 필요광고비 + 마진/인건비</div></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            months = st.selectbox("기간(개월)", options=[3, 6, 12], index=2, key="b_months_int")
        with c2:
            base_month_rev = st.number_input("월 기준 총매출(원)", value=200000000, step=10000000, key="b_base_rev_int")
        with c3:
            growth = st.number_input("월 성장률(%)", value=float(scn_month_growth*100.0), step=0.5, key="b_growth_int") / 100.0
        with c4:
            selling_price = st.number_input("예상 판매가(AOV) (원)", value=50000, step=1000, key="b_sell_price_int")

        a1, a2, a3, a4 = st.columns(4)
        with a1:
            use_scn_kpi_b = st.toggle("시나리오 KPI 자동 사용(권장)", value=True, key="use_scn_kpi_brand")
        with a2:
            ad_contrib_in = st.number_input("광고기여율(%)", value=float(scn_ad_contrib*100.0), step=1.0, key="b_ad_contrib_int") / 100.0
        with a3:
            repurchase_in = st.number_input("재구매율(%)", value=float(scn_repurchase*100.0), step=1.0, key="b_repurchase_int") / 100.0
        with a4:
            st.caption("필요 광고비는 '총매출→광고기여 매출→주문→클릭→CPC'로 역산")

        b1, b2 = st.columns(2)
        with b1:
            cpc_manual_b = st.number_input("CPC (원) [수동]", value=300.0, step=10.0, key="b_cpc_manual")
        with b2:
            cvr_manual_b = st.number_input("CVR (%) [수동]", value=2.0, step=0.1, key="b_cvr_manual") / 100.0

        scn_cpc, scn_cvr = blended_cpc_cvr(row, perf_cols)
        cpc_b = scn_cpc if (use_scn_kpi_b and scn_cpc is not None) else float(cpc_manual_b)
        cvr_b = scn_cvr if (use_scn_kpi_b and scn_cvr is not None) else float(cvr_manual_b)

        st.caption(
            f"현재 적용 KPI: CPC {fmt_won(cpc_b)} / CVR {fmt_pct(cvr_b*100,1)} "
            + (f"(시나리오 KPI 기반)" if use_scn_kpi_b and scn_cpc is not None else "(수동 입력)")
        )

        cost1, cost2, cost3, cost4 = st.columns(4)
        with cost1:
            cost_rate = st.number_input("원가율(%)", value=30.0, step=1.0, key="b_cost_rate") / 100.0
        with cost2:
            logistics = st.number_input("물류비(건당) (원)", value=3000, step=500, key="b_logi")
        with cost3:
            headcount = st.number_input("운영 인력(명)", value=2, step=1, min_value=0, key="b_hc")
        with cost4:
            cost_per = st.number_input("인당 고정비(원)", value=3000000, step=100000, key="b_cper")

        fixed_cost = float(headcount) * float(cost_per)

        s1, s2, s3 = st.columns(3)
        with s1:
            current_stock = st.number_input("현재 재고(개)", value=10000, step=100, key="b_stock_int")
        with s2:
            safety_stock = st.number_input("안전재고(개)", value=0, step=100, key="b_safety_int")
        with s3:
            start_day = st.date_input("기준일(재고 시작일)", key="b_startday_int")

        months_idx = list(range(1, int(months) + 1))
        ym_list = [f"M{i}" for i in months_idx]
        rev_list = []
        for i in months_idx:
            factor = (1.0 + growth) ** (i - 1)
            rev_list.append(base_month_rev * factor)

        rows = []
        for ym, rev_i in zip(ym_list, rev_list):
            sim_i = simulate_pl(
                calc_mode="매출 입력 → 필요 광고비 산출",
                aov=float(selling_price),
                cpc=float(cpc_b),
                cvr=float(cvr_b),
                cost_rate=float(cost_rate),
                logistics_per_order=float(logistics),
                fixed_cost=float(fixed_cost)/max(int(months),1),
                ad_spend=None,
                revenue=float(rev_i),
                ad_contrib_rate=float(ad_contrib_in),
                repurchase_rate=float(repurchase_in),
            )
            units_i = (float(rev_i) / float(selling_price)) if selling_price > 0 else 0.0
            rows.append({
                "월": ym,
                "총매출": float(sim_i["revenue"]),
                "필요광고비": float(sim_i["ad_spend"]),
                "ROAS": float(sim_i["roas"]),
                "예상판매수량(개)": float(units_i),
                "광고기여매출": float(sim_i["ad_revenue"]),
                "재구매매출": float(sim_i["repeat_revenue"]),
                "영업이익(월)": float(sim_i["profit"]),
            })

        df_fore = pd.DataFrame(rows)
        df_fore["누적판매(개)"] = df_fore["예상판매수량(개)"].cumsum()

        burn_point = float(current_stock)
        burn_month = None
        burn_in_month_ratio = None
        prev = 0.0
        for _, r in df_fore.iterrows():
            cumu = float(r["누적판매(개)"])
            if cumu >= burn_point and burn_month is None:
                burn_month = r["월"]
                month_units = float(r["예상판매수량(개)"])
                burn_in_month_ratio = ((burn_point - prev) / month_units) if month_units > 0 else 1.0
                break
            prev = cumu

        total_units = float(df_fore["예상판매수량(개)"].sum())
        po_units = max(int(np.ceil(total_units + float(safety_stock) - float(current_stock))), 0)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("기간 총매출", fmt_won(df_fore["총매출"].sum()))
        k2.metric("기간 필요 광고비", fmt_won(df_fore["필요광고비"].sum()))
        k3.metric("기간 예상 판매수량", f"{total_units:,.0f} 개")
        k4.metric("권장 발주(필요)", f"{po_units:,.0f} 개")

        if burn_month is None:
            st.info("재고가 기간 내에 소진되지 않는 것으로 추정됩니다.")
        else:
            day_offset = int(np.clip((burn_in_month_ratio or 1.0) * 30.0, 1, 30))
            st.warning(f"예상 재고 소진: **{burn_month}** 내 **약 {day_offset}일차 전후**(대략)")

        st.divider()
        df_chart = df_fore.copy()
        df_chart["ROAS"] = df_chart["총매출"] / df_chart["필요광고비"].replace(0, np.nan)
        st.plotly_chart(
            compare_chart(df_chart, "월", "총매출", "필요광고비", "ROAS", title="월별 총매출/필요광고비 + ROAS"),
            use_container_width=True,
            key=f"brand_int_month_chart_{scenario_key}"
        )

        st.divider()
        st.markdown("### 매출 채널 구성(트리맵)")
        fig_rev_tm2 = treemap_revenue(rev_share, title="매출 채널 구성(트리맵)")
        if fig_rev_tm2:
            st.plotly_chart(fig_rev_tm2, use_container_width=True, key=f"rev_tm_brand_int_{scenario_key}")

        st.divider()
        st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">🏪 판매채널별 월별 매출 계획</div></div>', unsafe_allow_html=True)
        ch_rows = []
        for _, r in df_fore.iterrows():
            ym = r["월"]
            total_rev = float(r["총매출"])
            for ch, share in rev_share.items():
                if share <= 0:
                    continue
                ch_rows.append({
                    "월": ym,
                    "채널": ch,
                    "매출(원)": round_to_100(total_rev * float(share)),
                    "비중(%)": float(share) * 100.0,
                })
        df_ch = pd.DataFrame(ch_rows)
        if df_ch.empty:
            st.info("판매채널 비중 데이터가 비어있습니다.")
        else:
            st.dataframe(
                df_ch.sort_values(["월", "매출(원)"], ascending=[True, False]),
                use_container_width=True,
                hide_index=True
            )

        st.divider()
        st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">📋 월별 상세 테이블</div></div>', unsafe_allow_html=True)
        disp = df_fore.copy()
        for c in ["총매출","필요광고비","광고기여매출","재구매매출","영업이익(월)"]:
            disp[c] = disp[c].map(lambda x: f"{x:,.0f}")
        disp["ROAS"] = (df_fore["총매출"] / df_fore["필요광고비"].replace(0, np.nan)).map(lambda x: "-" if pd.isna(x) else f"{x:.2f}x")
        disp["예상판매수량(개)"] = df_fore["예상판매수량(개)"].map(lambda x: f"{x:,.0f}")
        disp["누적판매(개)"] = df_fore["누적판매(개)"].map(lambda x: f"{x:,.0f}")
        st.dataframe(disp, use_container_width=True, hide_index=True)

# =========================
# Tab: Recommendation (Classic Top3 + Compare Panel)
# =========================
with tab_rec:
    st.markdown("""
<div class="page-header">
  <div class="ph-icon">🤖</div>
  <div>
    <div class="ph-title">추천 엔진</div>
    <div class="ph-sub">시나리오 Top3 추천 · ROAS/ROI vs 고점지수 비교 분석</div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.markdown("<div class='smallcap'>예전 방식 Top3 추천 + ROI/고점(성장·재구매·광고의존) 비교</div>", unsafe_allow_html=True)
    st.divider()

    # ---- backdata의 성장/재구매/광고 관련 컬럼 자동 탐지 ----
    col_m_growth = safe_col(df, ["월 성장률", "월성장률", "monthly_growth", "MoM 성장률", "MoM"])
    col_ad_contrib = safe_col(df, ["광고기여율", "광고 기여율", "ad_contrib", "ad_contribution", "광고기여"])
    col_repurchase = safe_col(df, ["재구매율", "재구매", "repurchase", "repeat_rate"])
    col_ad_dep = safe_col(df, ["광고의존도", "광고 의존도", "ad_dependency", "ads_dependency", "ad_dep"])


    # ---- 입력(예전 느낌 유지하되, 비교에 필요한 최소 입력만 추가) ----
    with st.expander("입력 조건", expanded=True):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            operator = st.selectbox(
                "운영 주체",
                ["내부브랜드 운영자", "브랜드사 운영자(클라이언트)", "대행사(마케팅만)"],
                key="rec_operator",
            )
            stage_in = st.selectbox("단계(ST)", ["(무관)"] + uniq_vals(stage_col), key="rec_stage_in") if stage_col in df.columns else "(무관)"
            cat_in = st.selectbox("카테고리", ["(무관)"] + uniq_vals(cat_col), key="rec_cat_in") if cat_col in df.columns else "(무관)"

        with c2:
            pos_in = st.selectbox("가격 포지션(POS)", ["(무관)"] + uniq_vals(pos_col), key="rec_pos_in") if pos_col in df.columns else "(무관)"
            drv_in = st.selectbox("드라이버(DRV)", ["(무관)"] + uniq_vals(drv_col), key="rec_drv_in") if drv_col in df.columns else "(무관)"

        with c3:
            sales_focus = st.selectbox(
                "판매 중심 채널(브랜드사 중요 / 대행사는 참고만)",
                ["(무관)", "자사몰", "스마트스토어", "쿠팡", "오프라인", "온라인(기타)"],
                key="rec_sales_focus",
            )
            total_budget = st.number_input("총 광고예산(원)", value=50_000_000, step=1_000_000, min_value=1, key="rec_budget2")

        with c4:
            # 비교/ROI에 필요한 최소 입력(삭제 아님: 추천엔진에만 추가)
            aov = st.number_input("객단가(AOV) (원)", value=50_000, step=1_000, key="rec_aov")
            gross_margin_pct = st.number_input("마진율(%) (ROI 계산용)", value=50.0, step=1.0, key="rec_gm") / 100.0
            use_scn_kpi = st.toggle("시나리오 KPI 자동 사용(권장)", value=True, key="rec_use_kpi")
            run = st.button("Top3 추천 계산", use_container_width=True, key="rec_run2")

    if not run:
        st.info("입력 조건을 설정하고 **Top3 추천 계산**을 누르세요.")
        st.stop()

    # ---- 후보 시나리오 구성(전체 df 기준, 입력조건 반영) ----
    cand = df.copy()

    if stage_col in cand.columns and stage_in != "(무관)":
        cand = cand[cand[stage_col].astype(str) == stage_in]
    if cat_col in cand.columns and cat_in != "(무관)":
        cand = cand[cand[cat_col].astype(str) == cat_in]
    if pos_col in cand.columns and pos_in != "(무관)":
        cand = cand[cand[pos_col].astype(str) == pos_in]
    if drv_col in cand.columns and drv_in != "(무관)":
        cand = cand[cand[drv_col].astype(str) == drv_in]

    if cand.empty:
        st.warning("조건에 맞는 후보가 없어 전체 시나리오에서 추천합니다.")
        cand = df.copy()

    # ---- 스코어링(예전 감성 유지: 채널/미디어 정합 + 메타데이터 보너스) ----
    #   - 브랜드사는 판매채널 가중치 ↑
    #   - 대행사는 판매채널 가중치 ↓ (참고만)
    W_META = 25.0
    W_SALES = 35.0 if operator != "대행사(마케팅만)" else 15.0
    W_MEDIA = 25.0
    W_RISK = 15.0  # 광고의존/광고기여율(효율) 가벼운 반영(비교패널이 메인)

    target_vec = {"자사몰":0, "스마트스토어":0, "쿠팡":0, "오프라인":0, "온라인(기타)":0}
    if sales_focus != "(무관)":
        target_vec[sales_focus] = 1.0
    target_vec = normalize_shares(target_vec)

    rows = []
    for _, r in cand.iterrows():
        # shares
        rs = build_rev_shares(r, rev_cols)
        ms = build_media_shares(r, perf_cols, viral_cols, brand_cols)

        # 1) 메타데이터 매칭
        meta_score = 0.0
        if stage_col in df.columns and stage_in != "(무관)" and str(r.get(stage_col)) == stage_in: meta_score += 1.0
        if cat_col in df.columns and cat_in != "(무관)" and str(r.get(cat_col)) == cat_in: meta_score += 1.0
        if pos_col in df.columns and pos_in != "(무관)" and str(r.get(pos_col)) == pos_in: meta_score += 1.0
        if drv_col in df.columns and drv_in != "(무관)" and str(r.get(drv_col)) == drv_in: meta_score += 1.0
        meta_score = meta_score / 4.0  # 0~1

        # 2) 판매채널 정합(코사인 유사도)
        sales_sim = _cosine(_rev_bucket_vec(rs), target_vec) if sales_focus != "(무관)" else 0.5

        # 3) 미디어 정합(판매포커스별 “있어야 할 매체” 가산)
        perf = ms.get("perf", {})
        viral = ms.get("viral", {})
        # 키워드 기반 간단 가산(기존 삭제 아님: 비교패널이 최종 판단)
        media_score = 0.5
        if sales_focus == "자사몰":
            media_score = np.clip(_media_kw_share(perf, ["메타"]) * 2.0 + _media_kw_share(perf, ["구글", "Google"]) * 1.0, 0, 1)
        elif sales_focus == "스마트스토어":
            media_score = np.clip(_media_kw_share(perf, ["네이버", "SA"]) * 2.0 + _media_kw_share(perf, ["GFA","GDN","DA"]) * 1.0, 0, 1)
        elif sales_focus == "쿠팡":
            media_score = np.clip(_media_kw_share(perf, ["외부몰PA","쿠팡","PA"]) * 2.2 + _media_kw_share(perf, ["메타"]) * 1.0, 0, 1)
        else:
            media_score = 0.6 + np.clip(ms["group"].get("퍼포먼스",0)*0.4, 0, 0.4)

        # 4) 리스크/효율(가볍게만 점수에 반영하고, 비교패널에서 상세 확인)
        ad_dep = _norm01_rate(r.get(col_ad_dep)) if col_ad_dep else np.nan
        ad_contrib = _norm01_rate(r.get(col_ad_contrib)) if col_ad_contrib else np.nan
        risk_score = 0.5
        if not np.isnan(ad_dep):
            risk_score = 1.0 - ad_dep  # 의존도 낮을수록 좋음
        if not np.isnan(ad_contrib):
            # 광고기여율이 너무 높으면 “광고 없으면 매출 유지 어려움” -> 가산 낮춤
            risk_score = (risk_score * 0.6) + ((1.0 - ad_contrib) * 0.4)

        score = (meta_score * W_META) + (sales_sim * W_SALES) + (float(media_score) * W_MEDIA) + (float(risk_score) * W_RISK)

        rows.append({
            "scenario_key": str(r[col_scn]).strip(),
            "scenario_disp": str(r[col_disp]).strip(),
            "score": float(score),
            "rev_share": rs,
            "media_share": ms,
            "row": r
        })

    rows = sorted(rows, key=lambda x: x["score"], reverse=True)
    top = rows[:3]
    top10 = rows[:10]

    # ---- Top3 카드(예전처럼 3컬럼) ----
    st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">🥇 Top 3 추천 시나리오</div></div>', unsafe_allow_html=True)
    cA, cB, cC = st.columns(3)

    def _render_card(col, idx, item):
        r = item["row"]
        rs = item["rev_share"]
        ms = item["media_share"]

        # KPI (시나리오 KPI 있으면 사용, 없으면 현재 선택 시나리오 blended를 fallback)
        scn_cpc, scn_cvr = blended_cpc_cvr(r, perf_cols)
        if (not use_scn_kpi) or (scn_cpc is None) or (scn_cvr is None):
            # fallback: 현재 선택 시나리오 KPI
            fb_cpc, fb_cvr = blended_cpc_cvr(row, perf_cols)
            scn_cpc = fb_cpc if fb_cpc is not None else 300.0
            scn_cvr = fb_cvr if fb_cvr is not None else 0.02

        # 광고기여/재구매/광고의존
        mg = _norm01_rate(r.get(col_m_growth)) if col_m_growth else np.nan
        ac = _norm01_rate(r.get(col_ad_contrib)) if col_ad_contrib else np.nan
        rep = _norm01_rate(r.get(col_repurchase)) if col_repurchase else np.nan
        addep = _norm01_rate(r.get(col_ad_dep)) if col_ad_dep else np.nan

        est = _estimate_now_and_roi(
            budget=float(total_budget),
            aov=float(aov),
            cpc=float(scn_cpc),
            cvr=float(scn_cvr),
            ad_contrib=float(ac if not np.isnan(ac) else 0.6),
            month_growth=float(mg if not np.isnan(mg) else 0.0),
            repurchase=float(rep if not np.isnan(rep) else 0.0),
            ad_dependency=float(addep if not np.isnan(addep) else float(ac if not np.isnan(ac) else 0.6)),
            months=12,
            gross_margin_rate=float(gross_margin_pct),
            )
        ceil = _ceiling_index(
            month_growth=float(mg if not np.isnan(mg) else 0.0),
            repurchase=float(rep if not np.isnan(rep) else 0.0),
            ad_dependency=float(addep if not np.isnan(addep) else 0.0),
        )

        # 카드
        with col:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"### #{idx} {item['scenario_disp']}")
            st.caption(item["scenario_key"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Score", f"{item['score']:.1f}")
            m2.metric("예상 ROAS", f"{est['roas']:.2f}x")
            m3.metric("고점지수", f"{ceil:.2f}")

            m4, m5, m6 = st.columns(3)
            m4.metric("예상 매출", fmt_won_compact(est["total_rev"]))
            m5.metric("광고매출", fmt_won_compact(est["ad_rev"]))
            m6.metric("ROI(마진)", "-" if np.isnan(est["roi"]) else f"{est['roi']*100:.0f}%")
            st.caption(f"풀값: 예상매출 {fmt_won(est['total_rev'])} / 광고매출 {fmt_won(est['ad_rev'])}")

            st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
            # 근거 3줄(예전 스타일)
            top_rev = sorted(_rev_bucket_vec(rs).items(), key=lambda x: x[1], reverse=True)[:2]
            top_perf = sorted((ms.get("perf") or {}).items(), key=lambda x: x[1], reverse=True)[:2]
            st.write("**요약 근거(3줄)**")
            st.write(f"- 판매채널: " + ", ".join([f"{k} {v:.0%}" for k, v in top_rev if v > 0]) if top_rev else "-")
            st.write(f"- 퍼포먼스: " + ", ".join([f"{k} {v:.0%}" for k, v in top_perf if v > 0]) if top_perf else "-")
            st.write(f"- 성장/재구매/의존: "
                     f"성장 {('-' if np.isnan(mg) else f'{mg*100:.1f}%')} / "
                     f"재구매 {('-' if np.isnan(rep) else f'{rep*100:.0f}%')} / "
                     f"광고의존 {('-' if np.isnan(addep) else f'{addep*100:.0f}%')}"
                    )

            # 선택 버튼(바로 사이드바 시나리오로 이동)
            if st.button("이 시나리오로 선택", key=f"pick_{item['scenario_key']}"):
                # sel_scn는 sidebar selectbox key
                st.session_state["sel_scn"] = item["scenario_disp"]
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    if len(top) >= 1: _render_card(cA, 1, top[0])
    if len(top) >= 2: _render_card(cB, 2, top[1])
    if len(top) >= 3: _render_card(cC, 3, top[2])

    # ---- 비교 패널(여기서 “ROI 낮지만 고점 높은 것”까지 한눈에) ----
    st.divider()
    st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">📊 시나리오 비교 (Top 10)</div></div>', unsafe_allow_html=True)
    st.caption("‘지금 ROAS/ROI’ vs ‘고점(성장·재구매·광고의존)’을 동시에 보고 최종 선택하세요.")

    cmp_rows = []
    for item in top10:
        r = item["row"]
        scn_cpc, scn_cvr = blended_cpc_cvr(r, perf_cols)
        if (not use_scn_kpi) or (scn_cpc is None) or (scn_cvr is None):
            fb_cpc, fb_cvr = blended_cpc_cvr(row, perf_cols)
            scn_cpc = fb_cpc if fb_cpc is not None else 300.0
            scn_cvr = fb_cvr if fb_cvr is not None else 0.02

        mg = _norm01_rate(r.get(col_m_growth)) if col_m_growth else np.nan
        ac = _norm01_rate(r.get(col_ad_contrib)) if col_ad_contrib else np.nan
        rep = _norm01_rate(r.get(col_repurchase)) if col_repurchase else np.nan
        addep = _norm01_rate(r.get(col_ad_dep)) if col_ad_dep else np.nan

        est = _estimate_now_and_roi(
            budget=float(total_budget),
            aov=float(aov),
            cpc=float(scn_cpc),
            cvr=float(scn_cvr),
            ad_contrib=float(ac if not np.isnan(ac) else 0.6),
            month_growth=float(mg if not np.isnan(mg) else 0.0),
            repurchase=float(rep if not np.isnan(rep) else 0.0),
            ad_dependency=float(addep if not np.isnan(addep) else float(ac if not np.isnan(ac) else 0.6)),
            months=12,
            gross_margin_rate=float(gross_margin_pct),
            )

        ceil = _ceiling_index(
            month_growth=float(mg if not np.isnan(mg) else 0.0),
            repurchase=float(rep if not np.isnan(rep) else 0.0),
            ad_dependency=float(addep if not np.isnan(addep) else 0.0),
        )

        cmp_rows.append({
            "시나리오": item["scenario_disp"],
            "키": item["scenario_key"],
            "Score": item["score"],
            "예상매출(총)": est["total_rev"],
            "ROAS": est["roas"],
            "ROI(마진)": est["roi"],
            "고점지수": ceil,
            "월성장률": mg,
            "재구매율": rep,
            "광고기여율": ac,
            "광고의존도": addep,
        })

    df_cmp = pd.DataFrame(cmp_rows)
    if df_cmp.empty:
        st.info("비교할 후보가 없습니다.")
    else:
        # 보기 좋은 표시용
        disp = df_cmp.copy()
        disp["예상매출(총)"] = disp["예상매출(총)"].map(lambda x: f"{x:,.0f}")
        disp["ROAS"] = disp["ROAS"].map(lambda x: f"{x:.2f}x")
        disp["ROI(마진)"] = disp["ROI(마진)"].map(lambda x: "-" if (pd.isna(x) or np.isnan(x)) else f"{x*100:.0f}%")
        disp["고점지수"] = disp["고점지수"].map(lambda x: f"{x:.2f}")
        for c in ["월성장률","재구매율","광고기여율","광고의존도"]:
            disp[c] = disp[c].map(lambda x: "-" if (pd.isna(x) or np.isnan(x)) else f"{x*100:.1f}%")

        st.dataframe(
            disp[["시나리오","Score","예상매출(총)","ROAS","ROI(마진)","고점지수","월성장률","재구매율","광고의존도","광고기여율"]],
            use_container_width=True,
            hide_index=True
        )

        # 산점도: x=ROAS, y=고점지수, size=예상매출, 색=광고의존도(있으면)
        plot_df = df_cmp.copy()
        plot_df["광고의존도"] = plot_df["광고의존도"].fillna(0.0)
        fig = px.scatter(
            plot_df,
            x="ROAS",
            y="고점지수",
            size="예상매출(총)",
            color="광고의존도",
            hover_name="시나리오",
            hover_data=["키", "Score", "월성장률", "재구매율", "광고기여율"],
            title="ROAS(현재 효율) vs 고점지수(성장·재구매·의존 리스크)"
        )
        fig.update_layout(
            height=420, margin=dict(t=50, b=10, l=10, r=10),
            paper_bgcolor=PLOTLY_DARK["paper_bgcolor"],
            plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
            font=PLOTLY_DARK["font"],
            title_font=PLOTLY_DARK["title_font"],
            xaxis=dict(gridcolor="rgba(79,142,247,0.08)", color="#64748B"),
            yaxis=dict(gridcolor="rgba(79,142,247,0.08)", color="#64748B"),
        )
        st.plotly_chart(fig, use_container_width=True, key="rec_compare_scatter")

        # 최종 선택(테이블에서 바로)
        pick = st.selectbox("비교 후 최종 선택", options=plot_df["시나리오"].tolist(), key="rec_final_pick")
        if st.button("선택한 시나리오로 이동", key="rec_go_pick"):
            st.session_state["sel_scn"] = pick
            st.rerun()

# =========================
# Tab: Custom Scenario (기존 유지)
# =========================
with tab_custom:
    st.markdown("""
<div class="page-header">
  <div class="ph-icon">⚙️</div>
  <div>
    <div class="ph-title">커스텀 시나리오</div>
    <div class="ph-sub">시나리오 기반 비중/예산을 직접 수정해 즉시 결과 확인</div>
  </div>
</div>
""", unsafe_allow_html=True)
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
    st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">⚙️ 시뮬레이션 입력</div></div>', unsafe_allow_html=True)
    cA, cB, cC, cD = st.columns(4)
    with cA:
        calc_mode_c = st.radio("계산 방식", ["광고비 입력 → 매출 산출", "매출 입력 → 필요 광고비 산출"], horizontal=True, key="custom_calc_mode")
    with cB:
        aov_c = st.number_input("객단가(AOV) (원)", value=50000, step=1000, key="custom_aov")
    with cC:
        cpc_c = st.number_input("CPC (원)", value=300.0, step=10.0, key="custom_cpc")
    with cD:
        cvr_c = st.number_input("CVR (%)", value=2.0, step=0.1, key="custom_cvr") / 100.0

    cX1, cX2 = st.columns(2)
    with cX1:
        ad_contrib_c = st.number_input("광고기여율(%)", value=float(scn_ad_contrib*100.0), step=1.0, key="custom_ad_contrib") / 100.0
    with cX2:
        repurchase_c = st.number_input("재구매율(%)", value=float(scn_repurchase*100.0), step=1.0, key="custom_repurchase") / 100.0

    if calc_mode_c.startswith("광고비"):
        ad_total_c = st.number_input("총 광고비(원)", value=50000000, step=1000000, key="custom_ad_total")
        rev_target_c = None
    else:
        rev_target_c = st.number_input("목표 매출(원)", value=300000000, step=10000000, key="custom_rev_target")
        ad_total_c = None

    sim_c = simulate_pl(
        calc_mode=calc_mode_c,
        aov=aov_c,
        cpc=cpc_c,
        cvr=cvr_c,
        cost_rate=0.0,
        logistics_per_order=0.0,
        fixed_cost=0.0,
        ad_spend=ad_total_c,
        revenue=rev_target_c,
        ad_contrib_rate=float(ad_contrib_c),
        repurchase_rate=float(repurchase_c),
    )

    st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">📊 시뮬레이션 결과</div></div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("예상 매출(총)", fmt_won(sim_c["revenue"]))
    m2.metric("예상 광고비", fmt_won(sim_c["ad_spend"]))
    m3.metric("ROAS", f"{sim_c['roas']:.2f}x ({sim_c['roas']*100:,.0f}%)")
    m4.metric("광고기여 매출", fmt_won(sim_c["ad_revenue"]))

    st.divider()
    st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">📺 커스텀 미디어 믹스</div></div>', unsafe_allow_html=True)

    perf_budget_c = float(sim_c["ad_spend"]) * float(group_custom.get("퍼포먼스", 1.0))
    viral_budget_c = float(sim_c["ad_spend"]) * float(group_custom.get("바이럴", 0.0))

    perf_df_c = build_performance_mix_table(perf_custom, perf_budget_c) if perf_custom else pd.DataFrame()
    viral_df_c = build_viral_mix_table(viral_price_custom, viral_medium_shares(viral_custom), viral_budget_c) if viral_custom else pd.DataFrame()

    st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">🎯 퍼포먼스</div></div>', unsafe_allow_html=True)
    if perf_df_c.empty:
        st.info("커스텀 퍼포먼스 믹스가 없습니다.")
        perf_out_c = perf_df_c
    else:
        perf_out_c = editable_perf_table(perf_df_c, submode="외부", key_prefix="custom")

    st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">📢 바이럴</div></div>', unsafe_allow_html=True)
    if viral_df_c.empty:
        st.info("커스텀 바이럴 믹스가 없습니다.")
        viral_out_c = viral_df_c
    else:
        viral_out_c = editable_viral_table(viral_df_c, submode="외부", key_prefix="custom")

    fig_ads_tm_c = treemap_ads(perf_out_c, viral_out_c, title="커스텀 광고 믹스(트리맵)")
    if fig_ads_tm_c:
        st.plotly_chart(fig_ads_tm_c, use_container_width=True, key="custom_ads_tm")

# =========================
# Tab: Sales Plan (기존 유지)
# =========================
with tab_plan:
    st.markdown("""
<div class="page-header">
  <div class="ph-icon">📅</div>
  <div>
    <div class="ph-title">매출 계획</div>
    <div class="ph-sub">브랜드별 1~12월 계획 생성 · CSV 업로드 · 피벗 뷰</div>
  </div>
</div>
""", unsafe_allow_html=True)
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
                st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">📈 브랜드별 월별 매출</div></div>', unsafe_allow_html=True)
                st.data_editor(p_rev.reset_index(), use_container_width=True, key="plan_pivot_rev")

                if bud_col and bud_col in plan_raw.columns:
                    st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">💸 브랜드별 월별 광고비</div></div>', unsafe_allow_html=True)
                    p_bud = plan_raw.pivot_table(index=brand_col, columns=month_col, values=bud_col, aggfunc="sum").fillna(0.0)
                    st.data_editor(p_bud.reset_index(), use_container_width=True, key="plan_pivot_budget")

                totals = p_rev.sum(axis=1).reset_index()
                totals.columns = ["Brand", "TotalRevenue"]
                fig = px.bar(totals, x="Brand", y="TotalRevenue", text="TotalRevenue")
                fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
                fig.update_layout(
                    height=380, margin=dict(t=50, b=10, l=10, r=10),
                    yaxis_title=None, xaxis_title=None,
                    title=dict(text="브랜드별 연간 매출 합계", font=PLOTLY_DARK["title_font"], x=0.01),
                    paper_bgcolor=PLOTLY_DARK["paper_bgcolor"], plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
                    font=PLOTLY_DARK["font"],
                    xaxis=dict(gridcolor="rgba(79,142,247,0.08)", color="#64748B"),
                    yaxis=dict(gridcolor="rgba(79,142,247,0.08)", color="#64748B"),
                    colorway=PLOTLY_DARK["colorway"],
                )
                st.plotly_chart(fig, use_container_width=True, key="plan_bar_total")

    else:
        st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">✏️ 브랜드 입력</div></div>', unsafe_allow_html=True)
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
            st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">📅 월별 계획</div></div>', unsafe_allow_html=True)
            plan_edit = st.data_editor(plan_long, use_container_width=True, key="plan_long_editor")

            st.markdown('<div class="section-header"><div class="sh-bar"></div><div class="sh-title">📊 브랜드별 월별 매출 (피벗)</div></div>', unsafe_allow_html=True)
            p = plan_edit.pivot_table(index="Brand", columns="Month", values="매출(원)", aggfunc="sum").fillna(0.0)
            st.data_editor(p.reset_index(), use_container_width=True, key="plan_pivot_from_manual")

            totals = p.sum(axis=1).reset_index()
            totals.columns = ["Brand", "TotalRevenue"]
            fig = px.bar(totals, x="Brand", y="TotalRevenue", text="TotalRevenue")
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig.update_layout(
                height=380, margin=dict(t=50, b=10, l=10, r=10),
                yaxis_title=None, xaxis_title=None,
                title=dict(text="브랜드별 연간 매출 합계", font=PLOTLY_DARK["title_font"], x=0.01),
                paper_bgcolor=PLOTLY_DARK["paper_bgcolor"], plot_bgcolor=PLOTLY_DARK["plot_bgcolor"],
                font=PLOTLY_DARK["font"],
                xaxis=dict(gridcolor="rgba(79,142,247,0.08)", color="#64748B"),
                yaxis=dict(gridcolor="rgba(79,142,247,0.08)", color="#64748B"),
                colorway=PLOTLY_DARK["colorway"],
            )
            st.plotly_chart(fig, use_container_width=True, key="plan_bar_total_manual")
