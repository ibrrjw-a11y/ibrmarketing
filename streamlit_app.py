# streamlit_app.py
# 🎯 IBR 마케팅 시뮬레이터 v2.0 - 완전 리팩터링 버전

import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO, BytesIO
from typing import Optional, Dict, List, Tuple

# =========================
# 라이브러리 체크
# =========================
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    st.error("❌ plotly가 필요합니다: pip install plotly")
    st.stop()

try:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False

APP_PASSWORD = "ibrsecret"

# =========================
# 페이지 설정 및 스타일
# =========================
st.set_page_config(
    page_title="IBR 마케팅 시뮬레이터 Pro", 
    layout="wide", 
    page_icon="🚀"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    margin-bottom: 20px;
}
.metric-card {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #dee2e6;
    margin: 5px 0;
}
.success-box {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
div[data-testid="stMetricValue"] {
    font-size: 1.2rem !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 유틸리티 함수
# =========================
def auth_gate():
    if st.session_state.get("auth_ok", False):
        return True
    
    st.sidebar.markdown("## 🔒 접근 제한")
    pw = st.sidebar.text_input("비밀번호", type="password")
    
    if st.sidebar.button("잠금 해제"):
        if pw == APP_PASSWORD:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.sidebar.error("비밀번호가 틀립니다.")
    
    st.info("👈 좌측 사이드바에서 비밀번호를 입력하세요.")
    return False

def to_float(x, default=0.0):
    try:
        if pd.isna(x): return default
        s = str(x).replace(",", "").replace("₩", "").replace("%", "").strip()
        return float(s) if s else default
    except: return default

def fmt_won(x):
    try: return f"{float(x):,.0f}원"
    except: return "-"

def fmt_won_compact(x):
    try:
        v = float(x)
        if abs(v) < 10_000: return f"{v:,.0f}원"
        if abs(v) < 100_000_000: return f"{v/10_000:,.0f}만원"
        return f"{v/100_000_000:,.1f}억원"
    except: return "-"

def fmt_pct(x): 
    try: return f"{float(x):.1f}%"
    except: return "-"

def normalize_ratio(x):
    v = to_float(x, default=np.nan)
    if np.isnan(v): return np.nan
    return (v / 100.0) if v > 1 else v

def normalize_shares(d):
    d2 = {k: float(v or 0.0) for k, v in d.items()}
    s = sum(v for v in d2.values() if v > 0)
    if s <= 0: return {k: 0.0 for k in d2}
    return {k: (v / s if v > 0 else 0.0) for k, v in d2.items()}

# =========================
# 인증
# =========================
if not auth_gate():
    st.stop()

# =========================
# 데이터 로딩
# =========================
@st.cache_data
def load_backdata(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8-sig')
        else:
            # 엑셀 파일에서 backdata 시트 찾기
            xls = pd.ExcelFile(file)
            sheet = None
            for s in xls.sheet_names:
                if "backdata" in str(s).lower():
                    sheet = s
                    break
            sheet = sheet or xls.sheet_names[0]
            df = pd.read_excel(xls, sheet_name=sheet)
        
        df = df.dropna(how="all")
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"파일 로드 오류: {e}")
        return pd.DataFrame()

def detect_columns(df):
    def safe_col(candidates):
        for c in candidates:
            if c in df.columns: return c
        for c in candidates:
            for col in df.columns:
                if c in str(col): return col
        return None
    
    return {
        "scenario": safe_col(["시나리오명", "scenario"]) or df.columns[0],
        "display": safe_col(["노출 시나리오명", "display"]) or df.columns[0],
        "stage": safe_col(["단계(ST)", "단계", "ST"]),
        "cat": safe_col(["카테고리(대)", "카테고리", "CAT"]),
        "pos": safe_col(["가격포지션(POS)", "가격포지션", "POS"]),
        "rev_cols": [c for c in df.columns if "매출비중" in str(c)],
        "perf_cols": [c for c in df.columns if "퍼포먼스마케팅_" in str(c) and "KPI_" not in str(c)],
        "viral_cols": [c for c in df.columns if "바이럴마케팅_" in str(c) and "KPI_" not in str(c)],
        "growth": safe_col(["월 성장률", "월성장률", "monthly_growth"]),
        "ad_contrib": safe_col(["광고기여율", "ad_contribution"]),
        "repurchase": safe_col(["재구매율", "repurchase"]),
    }

# =========================
# 핵심 비즈니스 로직
# =========================
def build_shares(row, cols):
    # 매출 채널 비중
    rev_share = {}
    for c in cols["rev_cols"]:
        v = normalize_ratio(row.get(c, 0))
        if not np.isnan(v) and v > 0:
            name = str(c).replace("매출비중", "").strip()
            rev_share[name] = float(v)
    rev_share = normalize_shares(rev_share)
    
    # 미디어 비중
    perf_raw, viral_raw = {}, {}
    for c in cols["perf_cols"]:
        v = normalize_ratio(row.get(c, 0))
        if not np.isnan(v) and v > 0:
            perf_raw[str(c).replace("퍼포먼스마케팅_", "")] = float(v)
    
    for c in cols["viral_cols"]:
        v = normalize_ratio(row.get(c, 0))
        if not np.isnan(v) and v > 0:
            viral_raw[str(c).replace("바이럴마케팅_", "")] = float(v)
    
    perf_sum = sum(perf_raw.values())
    viral_sum = sum(viral_raw.values())
    total = perf_sum + viral_sum
    
    group_share = {"퍼포먼스": 1.0, "바이럴": 0.0} if total <= 0 else {
        "퍼포먼스": perf_sum / total,
        "바이럴": viral_sum / total
    }
    
    return {
        "rev_share": rev_share,
        "group_share": group_share,
        "perf_share": normalize_shares(perf_raw),
        "viral_share": normalize_shares(viral_raw)
    }

def get_scenario_kpi(row, cols):
    # KPI 자동 추출 (있으면 사용, 없으면 기본값)
    cpc = to_float(row.get("KPI_CPC", 300))
    cvr = to_float(row.get("KPI_CVR", 2)) / 100 if to_float(row.get("KPI_CVR", 2)) > 1 else to_float(row.get("KPI_CVR", 0.02))
    
    growth = normalize_ratio(row.get(cols["growth"], 0)) if cols["growth"] else 0
    ad_contrib = normalize_ratio(row.get(cols["ad_contrib"], 70)) if cols["ad_contrib"] else 0.7
    repurchase = normalize_ratio(row.get(cols["repurchase"], 20)) if cols["repurchase"] else 0.2
    
    return {
        "cpc": cpc,
        "cvr": cvr,
        "growth": growth,
        "ad_contrib": ad_contrib,
        "repurchase": repurchase
    }

def simulate_marketing(mode, budget, target_revenue, aov, cpc, cvr, ad_contrib, repurchase, role_params=None):
    # 기본 마케팅 성과 계산
    if mode == "예산 기반":
        ad_spend = budget
        clicks = ad_spend / cpc if cpc > 0 else 0
        orders_ad = clicks * cvr
        ad_revenue = orders_ad * aov
        total_revenue = ad_revenue / ad_contrib if ad_contrib > 0 else 0
    else:  # 매출 기반
        total_revenue = target_revenue
        ad_revenue = total_revenue * ad_contrib
        orders_ad = ad_revenue / aov if aov > 0 else 0
        clicks = orders_ad / cvr if cvr > 0 else 0
        ad_spend = clicks * cpc
    
    total_orders = total_revenue / aov if aov > 0 else 0
    repeat_revenue = total_revenue * repurchase
    first_revenue = total_revenue - repeat_revenue
    roas = total_revenue / ad_spend if ad_spend > 0 else 0
    
    result = {
        "revenue": total_revenue,
        "ad_spend": ad_spend,
        "ad_revenue": ad_revenue,
        "repeat_revenue": repeat_revenue,
        "first_revenue": first_revenue,
        "orders": total_orders,
        "roas": roas,
        "clicks": clicks
    }
    
    # 역할별 손익 계산
    if role_params:
        if role_params["role"] == "대행사":
            # 대행사 손익
            fee_revenue = ad_spend * (role_params["fee_rate"] / 100)
            payback_cost = ad_spend * (role_params["payback_rate"] / 100)
            viral_budget = ad_spend * role_params["viral_ratio"]
            viral_margin = viral_budget * (role_params["viral_margin_rate"] / 100)
            
            gross_profit = (fee_revenue - payback_cost) + viral_margin
            labor_cost = role_params["headcount"] * role_params["salary"]
            op_profit = gross_profit - labor_cost
            
            result.update({
                "fee_revenue": fee_revenue,
                "payback_cost": payback_cost,
                "viral_margin": viral_margin,
                "gross_profit": gross_profit,
                "labor_cost": labor_cost,
                "op_profit": op_profit,
                "total_billing": ad_spend + fee_revenue
            })
        
        else:  # 브랜드사
            cogs = total_revenue * (role_params["cost_rate"] / 100)
            logistics = total_orders * role_params["logistics_cost"]
            fixed_cost = role_params["fixed_cost"]
            
            op_profit = total_revenue - (ad_spend + cogs + logistics + fixed_cost)
            profit_margin = (op_profit / total_revenue * 100) if total_revenue > 0 else 0
            
            result.update({
                "cogs": cogs,
                "logistics": logistics,
                "fixed_cost": fixed_cost,
                "op_profit": op_profit,
                "profit_margin": profit_margin
            })
    
    return result

# =========================
# 차트 생성
# =========================
def create_revenue_treemap(rev_share, height=400):
    if not rev_share: return None
    
    df = pd.DataFrame(list(rev_share.items()), columns=["채널", "비중"])
    fig = px.treemap(df, path=["채널"], values="비중")
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{value:.1%}",
        textfont=dict(size=14, color="white")
    )
    fig.update_layout(height=height, title="매출 채널 구성")
    return fig

def create_waterfall_chart(result):
    if "fee_revenue" not in result: return None
    
    fig = go.Figure(go.Waterfall(
        name="Agency P&L",
        orientation="v",
        measure=["relative", "relative", "relative", "total", "relative", "total"],
        x=["광고비", "수수료수익", "페이백비용", "매출이익", "인건비", "영업이익"],
        y=[result["ad_spend"], result["fee_revenue"], -result["payback_cost"], 
           result["gross_profit"], -result["labor_cost"], 0],
        connector={"line": {"color": "rgba(63, 63, 63, 0.3)"}},
        texttemplate="%{y:,.0f}원"
    ))
    fig.update_layout(title="대행사 손익 구조", height=400)
    return fig

# =========================
# 사이드바 - 데이터 업로드 및 시나리오 선택
# =========================
st.sidebar.title("🎛️ IBR 시뮬레이터 Pro")

uploaded = st.sidebar.file_uploader("Backdata 업로드", type=["xlsx", "csv"])
if not uploaded:
    st.info("👈 좌측에서 backdata 파일을 업로드하세요.")
    st.stop()

df = load_backdata(uploaded)
if df.empty:
    st.error("파일을 읽을 수 없습니다.")
    st.stop()

cols = detect_columns(df)

# 필터링
st.sidebar.markdown("### 🔍 필터")
def get_unique_values(col):
    if col and col in df.columns:
        return ["전체"] + sorted([str(x) for x in df[col].dropna().unique()])
    return ["전체"]

stage_filter = st.sidebar.selectbox("단계", get_unique_values(cols["stage"]))
cat_filter = st.sidebar.selectbox("카테고리", get_unique_values(cols["cat"]))

# 필터 적용
filtered_df = df.copy()
if stage_filter != "전체" and cols["stage"]:
    filtered_df = filtered_df[filtered_df[cols["stage"]].astype(str) == stage_filter]
if cat_filter != "전체" and cols["cat"]:
    filtered_df = filtered_df[filtered_df[cols["cat"]].astype(str) == cat_filter]

# 시나리오 선택
scenario_options = filtered_df[cols["display"]].tolist()
selected_scenario = st.sidebar.selectbox("📌 시나리오 선택", scenario_options)

# 선택된 시나리오 데이터
scenario_row = filtered_df[filtered_df[cols["display"]] == selected_scenario].iloc[0]
shares = build_shares(scenario_row, cols)
kpi = get_scenario_kpi(scenario_row, cols)

st.sidebar.success(f"✅ {selected_scenario}")

# =========================
# 메인 탭
# =========================
tab_sim, tab_analysis, tab_ppt = st.tabs(["🎯 시뮬레이터", "📊 결과 분석", "📄 PPT 생성"])

# =========================
# 탭 1: 통합 시뮬레이터
# =========================
with tab_sim:
    st.markdown(f"""
    <div class="main-header">
        <h2>🎯 마케팅 시뮬레이터</h2>
        <p>시나리오: <strong>{selected_scenario}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # 시나리오 기본 정보
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("월 성장률", fmt_pct(kpi["growth"] * 100))
    col2.metric("광고기여율", fmt_pct(kpi["ad_contrib"] * 100))
    col3.metric("재구매율", fmt_pct(kpi["repurchase"] * 100))
    col4.metric("퍼포먼스 비중", fmt_pct(shares["group_share"]["퍼포먼스"] * 100))
    
    st.divider()
    
    # 시뮬레이션 설정
    col_left, col_right = st.columns([1, 1.2])
    
    with col_left:
        st.markdown("### ⚙️ 시뮬레이션 설정")
        
        # 역할 선택
        role = st.radio("분석 관점", ["대행사", "브랜드사"], horizontal=True)
        calc_mode = st.radio("계산 방식", ["예산 기반", "매출 기반"], horizontal=True)
        
        # 기본 KPI
        use_auto_kpi = st.toggle("시나리오 KPI 자동 사용", value=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            aov = st.number_input("객단가", value=50000, step=1000)
        with c2:
            cpc = st.number_input("CPC", value=kpi["cpc"] if use_auto_kpi else 300, step=10)
        with c3:
            cvr = st.number_input("CVR(%)", value=kpi["cvr"]*100 if use_auto_kpi else 2.0, step=0.1) / 100
        
        # 시나리오 변수
        c4, c5 = st.columns(2)
        with c4:
            ad_contrib = st.number_input("광고기여율(%)", value=kpi["ad_contrib"]*100, step=1) / 100
        with c5:
            repurchase = st.number_input("재구매율(%)", value=kpi["repurchase"]*100, step=1) / 100
        
        # 예산/매출 입력
        if calc_mode == "예산 기반":
            budget = st.number_input("월 광고 예산", value=50000000, step=1000000)
            target_revenue = 0
        else:
            target_revenue = st.number_input("월 목표 매출", value=200000000, step=10000000)
            budget = 0
        
        # 역할별 추가 설정
        role_params = {"role": role}
        
        if role == "대행사":
            st.markdown("#### 🏢 대행사 설정")
            r1, r2 = st.columns(2)
            with r1:
                role_params["fee_rate"] = st.number_input("수수료율(%)", value=15.0)
                role_params["payback_rate"] = st.number_input("페이백률(%)", value=5.0)
                role_params["viral_ratio"] = st.slider("바이럴 비중", 0.0, 1.0, 0.3)
            with r2:
                role_params["viral_margin_rate"] = st.number_input("바이럴 마진율(%)", value=20.0)
                role_params["headcount"] = st.number_input("투입 인원", value=2, step=1)
                role_params["salary"] = st.number_input("인당 월급", value=3500000, step=100000)
        else:
            st.markdown("#### 🏪 브랜드사 설정")
            r1, r2 = st.columns(2)
            with r1:
                role_params["cost_rate"] = st.number_input("원가율(%)", value=40.0)
                role_params["logistics_cost"] = st.number_input("건당 물류비", value=3000)
            with r2:
                role_params["fixed_cost"] = st.number_input("월 고정비", value=10000000, step=1000000)
    
    with col_right:
        # 시뮬레이션 실행
        result = simulate_marketing(
            calc_mode, budget, target_revenue, aov, cpc, cvr, 
            ad_contrib, repurchase, role_params
        )
        
        # 결과 저장 (다른 탭에서 사용)
        st.session_state.update({
            "sim_result": result,
            "scenario_name": selected_scenario,
            "shares": shares,
            "role": role
        })
        
        st.markdown("### 📊 시뮬레이션 결과")
        
        # 핵심 지표
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("총 매출", fmt_won_compact(result["revenue"]))
        m2.metric("광고비", fmt_won_compact(result["ad_spend"]))
        m3.metric("ROAS", f"{result['roas']:.1f}x")
        m4.metric("주문수", f"{result['orders']:,.0f}건")
        
        # 역할별 상세 지표
        if role == "대행사":
            st.markdown("#### 💼 대행사 손익")
            m5, m6, m7 = st.columns(3)
            m5.metric("총 청구액", fmt_won_compact(result["total_billing"]))
            m6.metric("총 마진", fmt_won_compact(result["gross_profit"]))
            m7.metric("영업이익", fmt_won_compact(result["op_profit"]))
            
            # Waterfall 차트
            fig_waterfall = create_waterfall_chart(result)
            if fig_waterfall:
                st.plotly_chart(fig_waterfall, use_container_width=True)
        
        else:
            st.markdown("#### 🏪 브랜드사 손익")
            m5, m6, m7 = st.columns(3)
            m5.metric("원가", fmt_won_compact(result["cogs"]))
            m6.metric("물류비", fmt_won_compact(result["logistics"]))
            m7.metric("영업이익", fmt_won_compact(result["op_profit"]))
            
            # 손익 차트
            fig_profit = px.bar(
                x=["매출", "광고비", "원가", "물류비", "고정비", "영업이익"],
                y=[result["revenue"], -result["ad_spend"], -result["cogs"], 
                   -result["logistics"], -result["fixed_cost"], result["op_profit"]],
                color=["매출", "광고비", "원가", "물류비", "고정비", "영업이익"],
                title="브랜드사 손익 분석"
            )
            st.plotly_chart(fig_profit, use_container_width=True)

# =========================
# 탭 2: 결과 분석
# =========================
with tab_analysis:
    st.markdown("## 📊 결과 분석")
    
    if "sim_result" not in st.session_state:
        st.warning("⚠️ 먼저 시뮬레이터 탭에서 분석을 실행하세요.")
    else:
        result = st.session_state["sim_result"]
        shares = st.session_state["shares"]
        
        # 매출 구조 분석
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💰 매출 채널 구성")
            fig_rev = create_revenue_treemap(shares["rev_share"])
            if fig_rev:
                st.plotly_chart(fig_rev, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 매출 구성")
            fig_composition = px.pie(
                values=[result["ad_revenue"], result["repeat_revenue"]],
                names=["광고기여 매출", "재구매 매출"],
                hole=0.4
            )
            fig_composition.update_layout(height=400)
            st.plotly_chart(fig_composition, use_container_width=True)
        
        # 채널별 상세 분석
        st.markdown("### 📋 채널별 예상 매출")
        
        channel_data = []
        for ch, share in sorted(shares["rev_share"].items(), key=lambda x: x[1], reverse=True):
            if share > 0:
                ch_revenue = result["revenue"] * share
                channel_data.append({
                    "채널": ch,
                    "비중": f"{share*100:.1f}%",
                    "예상 매출": fmt_won(ch_revenue),
                    "월 예상 주문": f"{(ch_revenue / 50000):,.0f}건"  # AOV 50,000 가정
                })
        
        if channel_data:
            df_channels = pd.DataFrame(channel_data)
            st.dataframe(df_channels, use_container_width=True, hide_index=True)

# =========================
# 탭 3: PPT 생성
# =========================
with tab_ppt:
    st.markdown("## 📄 PPT 생성")
    
    if not HAS_PPTX:
        st.error("❌ python-pptx 라이브러리가 필요합니다.")
        st.code("pip install python-pptx")
    elif "sim_result" not in st.session_state:
        st.warning("⚠️ 먼저 시뮬레이터 탭에서 분석을 실행하세요.")
    else:
        result = st.session_state["sim_result"]
        scenario_name = st.session_state["scenario_name"]
        role = st.session_state["role"]
        
        st.markdown(f"""
        <div class="success-box">
            <h4>✅ 시뮬레이션 완료</h4>
            <p><strong>시나리오:</strong> {scenario_name}</p>
            <p><strong>분석 관점:</strong> {role}</p>
            <p><strong>예상 매출:</strong> {fmt_won(result['revenue'])}</p>
            <p><strong>ROAS:</strong> {result['roas']:.1f}x</p>
        </div>
        """, unsafe_allow_html=True)
        
        # PPT 생성 옵션
        col1, col2 = st.columns([1, 1])
        
        with col1:
            ppt_template = st.file_uploader("PPT 템플릿 업로드 (선택사항)", type="pptx")
            
        with col2:
            if st.button("🚀 PPT 생성", type="primary", use_container_width=True):
                with st.spinner("PPT 생성 중..."):
                    try:
                        # PPT 생성
                        if ppt_template:
                            prs = Presentation(ppt_template)
                        else:
                            prs = Presentation()
                            
                            # 타이틀 슬라이드
                            slide = prs.slides.add_slide(prs.slide_layouts[0])
                            title = slide.shapes.title
                            title.text = "IBR 마케팅 시뮬레이션 결과"
                            
                            if len(slide.placeholders) > 1:
                                subtitle = slide.placeholders[1]
                                subtitle.text = f"""
시나리오: {scenario_name}
분석 관점: {role}
예상 매출: {fmt_won(result['revenue'])}
ROAS: {result['roas']:.1f}x
                                """.strip()
                            
                            # 결과 슬라이드
                            slide2 = prs.slides.add_slide(prs.slide_layouts[1])
                            title2 = slide2.shapes.title
                            title2.text = "시뮬레이션 핵심 결과"
                            
                            content = slide2.placeholders[1]
                            tf = content.text_frame
                            tf.text = f"총 매출: {fmt_won(result['revenue'])}"
                            
                            for line in [
                                f"광고비: {fmt_won(result['ad_spend'])}",
                                f"ROAS: {result['roas']:.1f}x",
                                f"광고기여 매출: {fmt_won(result['ad_revenue'])}",
                                f"재구매 매출: {fmt_won(result['repeat_revenue'])}",
                                f"영업이익: {fmt_won(result['op_profit'])}"
                            ]:
                                p = tf.add_paragraph()
                                p.text = line
                                p.level = 1
                        
                        # 저장
                        output = BytesIO()
                        prs.save(output)
                        output.seek(0)
                        
                        st.success("✅ PPT 생성 완료!")
                        
                        # 다운로드
                        st.download_button(
                            label="📥 PPT 다운로드",
                            data=output.getvalue(),
                            file_name=f"IBR_시뮬레이션_{scenario_name}_{role}.pptx",
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            use_container_width=True
                        )
                        
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ PPT 생성 오류: {str(e)}")
