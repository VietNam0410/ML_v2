import streamlit as st
import pandas as pd
from Titanic_Liner_Regression.Preprocessing import preprocess
from Titanic_Liner_Regression.Models import train
import sys
import os

# Lấy đường dẫn thư mục gốc (Machine_Learning_v2)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Thêm đường dẫn Titanic_Liner_Regression vào sys.path
TITANIC_DIR = os.path.join(BASE_DIR, "Titanic_Liner_Regression")
sys.path.append(TITANIC_DIR)

# Import module từ Titanic_Liner_Regression
from Preprocessing import preprocess
from Models import train


# Tạo các tab trong giao diện Streamlit
tabs = ["Data", "Preprocessing", "Models", "Demo"]
selected_tab = st.sidebar.radio("Chọn tab:", tabs)

if selected_tab == "Data":
    st.title("📂 Data - Upload & Hiển thị dữ liệu")
    uploaded_file = st.file_uploader("Tải lên tệp CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        df.to_csv("../Data/titanic.csv", index=False)
        st.success("📁 Đã lưu dữ liệu vào thư mục Data!")

elif selected_tab == "Preprocessing":
    st.title("⚙️ Preprocessing - Xử lý dữ liệu")
    df = preprocess.load_data("../Data/titanic.csv")
    st.write("Dữ liệu trước xử lý:")
    st.dataframe(df.head())
    
    X_train, X_test, y_train, y_test = preprocess.preprocess_data(df)
    st.write("Dữ liệu sau khi tiền xử lý:")
    st.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    st.success("✅ Tiền xử lý hoàn tất!")

elif selected_tab == "Models":
    st.title("🤖 Models - Train & Evaluate")
    if st.button("Train Model"):
        train.train_and_log()
        st.success("✅ Model trained & logged with MLflow!")
    st.write("Kiểm tra MLflow UI bằng cách chạy `mlflow ui`")

elif selected_tab == "Demo":
    st.title("🎯 Demo - Dự đoán trên dữ liệu mới")
    st.write("🚀 Sẽ bổ sung sau...")

st.sidebar.info("💡 Chọn tab để thao tác với dữ liệu, mô hình và demo!")
