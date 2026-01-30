import streamlit as st
import pandas as pd

st.set_page_config(page_title="마케팅 시뮬레이터", layout="wide")
st.title("📊 마케팅 시뮬레이터 – 백데이터 연결 확인")

CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRt3yFjt4OwY6Ym1-xCIJy75-6ccqAzpmGWfa7j7BscDWYL9bl2AmWEJtxo7SDvmQbysb5UEM-jOM2A/pub?output=csv"

@st.cache_data
def load_data():
    return pd.read_csv(CSV_URL)

df = load_data()

st.success("✅ Google Sheets CSV 로딩 성공")
st.dataframe(df, use_container_width=True)
st.write("컬럼 목록:", list(df.columns))
