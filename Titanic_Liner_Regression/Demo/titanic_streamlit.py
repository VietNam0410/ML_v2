import streamlit as st
import pandas as pd
import sys
sys.path.append("..")  # Đảm bảo import từ thư mục cha
from Titanic_Liner_Regression.Preprocessing import preprocess

st.title("🚢 Titanic Data Explorer")

# Tải file CSV
uploaded_file = st.file_uploader("Tải lên file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dữ liệu gốc")
    st.dataframe(df)

    # Biểu đồ phân phối
    st.subheader("📉 Phân phối giá trị 'Survived'")
    st.bar_chart(df["Survived"].value_counts())

    # Nút tiền xử lý dữ liệu
    if st.button("Tiền xử lý dữ liệu"):
        df = preprocess.preprocess_data(df)
        st.subheader("✅ Dữ liệu sau xử lý")
        st.dataframe(df)
