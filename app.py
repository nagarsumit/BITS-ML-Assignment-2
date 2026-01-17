import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Mobile Price Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER FUNCTIONS ---
def load_model(model_name):
    filename = f"model/{model_name.lower().replace(' ', '_')}.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    return None

def load_scaler():
    if os.path.exists("model/scaler.pkl"):
        with open("model/scaler.pkl", 'rb') as f:
            return pickle.load(f)
    return None

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📱 Navigation")
page = st.sidebar.radio("Go to:", ["📊 Dashboard", "ℹ️ Dataset Reference"])

st.sidebar.markdown("---")
st.sidebar.write("**Student:** Sumitkumar Nagar")
st.sidebar.write("**Roll No.:** 2025AA05220")
st.sidebar.write("**Assignment:** ML Assignment 2")

# --- PAGE 1: DASHBOARD (SETUP + PERFORMANCE) ---
if page == "📊 Dashboard":
    st.markdown("<h1 style='text-align: center;'>📱 Mobile Price Classification Dashboard</h1>", unsafe_allow_html=True)
    
    # 1. DATA SETUP SECTION
    st.subheader("🛠️ Step 1: Data & Model Configuration")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload Test Data (CSV)", type=["csv"])
        
        # Points to /datasets/test-data.csv
        DEFAULT_DATA_PATH = os.path.join("datasets", "test-data.csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Using uploaded file.")
        elif os.path.exists(DEFAULT_DATA_PATH):
            df = pd.read_csv(DEFAULT_DATA_PATH)
            st.info(f"ℹ️ Using default data from: `{DEFAULT_DATA_PATH}`")
        else:
            st.error(f"⚠️ Dataset not found at `{DEFAULT_DATA_PATH}`. Please upload a CSV to proceed.")
            df = None

    with col2:
        model_options = [
            "Logistic Regression", "Decision Tree", "K-Nearest Neighbor",
            "Naive Bayes", "Random Forest", "XGBoost"
        ]
        selected_model_name = st.selectbox("Choose Classifier", model_options)

    # 2. PERFORMANCE ANALYSIS SECTION
    if df is not None:
        st.divider()
        st.subheader(f"🚀 Performance: {selected_model_name}")
        
        target_col = "price_range"
        if target_col not in df.columns:
            target_col = st.selectbox("Select Target Column", df.columns)
            
        X_test = df.drop(columns=[target_col])
        y_test = df[target_col]
        
        model = load_model(selected_model_name)
        
        # Scaling
        if selected_model_name in ["Logistic Regression", "K-Nearest Neighbor"]:
            scaler = load_scaler()
            if scaler:
                try:
                    X_input = scaler.transform(X_test)
                except Exception as e:
                    st.warning(f"Scaling error: {e}. Using raw data.")
                    X_input = X_test
            else:
                X_input = X_test
        else:
            X_input = X_test

        if model:
            try:
                # Metrics Calculation
                y_pred = model.predict(X_input)
                try:
                    y_prob = model.predict_proba(X_input)
                    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
                except:
                    auc = 0.0

                # Top Row Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
                m2.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.3f}")
                m3.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.3f}")
                m4.metric("AUC", f"{auc:.3f}")

                # Visualizations
                c1, c2 = st.columns([1, 1.2])
                with c1:
                    st.markdown("**Confusion Matrix**")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlGnBu', ax=ax)
                    st.pyplot(fig)
                
                with c2:
                    st.markdown("**Classification Report**")
                    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
                    st.dataframe(report_df.style.background_gradient(cmap="YlGnBu", subset=['f1-score', 'precision', 'recall']))
                
                # Raw Data Preview (at the bottom)
                with st.expander("🔎 View Dataset Preview"):
                    st.dataframe(df.head(10))

            except Exception as e:
                st.error(f"Prediction Error: {e}")
        else:
            st.error(f"Model file for {selected_model_name} not found in `model/` folder.")

# --- PAGE 2: DATASET REFERENCE ---
elif page == "ℹ️ Dataset Reference":
    st.title("ℹ️ Dataset Information")
    
    st.markdown("""
    ### Mobile Price Classification
    This dataset contains features of various mobile phones and their corresponding price ranges.
    
    **Source:** [Kaggle - Mobile Price Classification](https://www.kaggle.com/datasets/sufyan145/mobile-price-classification)
    
    **Target Variable:**
    * `price_range`: This is the target variable with a value of:
        * 0 (low cost)
        * 1 (medium cost)
        * 2 (high cost)
        * 3 (very high cost)

    **Features Explanation:**
    - **battery_power:** Total energy a battery can store in one time measured in mAh
    - **blue:** Has bluetooth or not
    - **clock_speed:** speed at which microprocessor executes instructions
    - **dual_sim:** Has dual sim support or not
    - **fc:** Front Camera mega pixels
    - **four_g:** Has 4G or not
    - **int_memory:** Internal Memory in Gigabytes
    - **m_dep:** Mobile Depth in cm
    - **mobile_wt:** Weight of mobile phone
    - **n_cores:** Number of cores of processor
    - **pc:** Primary Camera mega pixels
    - **px_height:** Pixel Resolution Height
    - **px_width:** Pixel Resolution Width
    - **ram:** Random Access Memory in Mega Bytes
    - **sc_h:** Screen Height of mobile in cm
    - **sc_w:** Screen Width of mobile in cm
    - **talk_time:** longest time that a single battery charge will last
    - **three_g:** Has 3G or not
    - **touch_screen:** Has touch screen or not
    - **wifi:** Has wifi or not
    """)
    
    st.info("The models were trained to classify phones into these four price ranges based on the technical specifications above.")