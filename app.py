import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Credit Default Risk Studio", layout="wide")

st.title("💳 Credit Default Risk Prediction Studio")
st.markdown("Machine Learning Classification Evaluation on Consumer Credit Risk Data.")

# Model file paths
model_paths = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "K-Nearest Neighbors": "model/k_nearest_neighbors.pkl",
    "Gaussian Naive Bayes": "model/gaussian_naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl"
}

# --- Sidebar Controls ---
st.sidebar.header("Test Dataset")

# Option to use bundled dataset by default (like Samriddhi's app)
use_bundled = st.sidebar.checkbox("Use bundled test_data.csv", value=True)
uploaded_file = st.sidebar.file_uploader("Upload another CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox("Choose a classifier", list(model_paths.keys()))

# Determine which data source to read
df_test = None

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
elif use_bundled and os.path.exists("test_data.csv"):
    df_test = pd.read_csv("test_data.csv")

# --- Main Evaluation View ---
if df_test is not None:
    # Top Dataset Summary Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Test Records", f"{len(df_test):,}")
    m2.metric("Input Features", f"{df_test.shape[1] - 1}")
    m3.metric("ML Models", "5")
    
    # Identify target column
    target_col = None
    for col in df_test.columns:
        if "default" in col.lower() or "target" in col.lower():
            target_col = col
            break
            
    if target_col is None:
        st.error("Target column indicating credit default not found in dataset.")
        st.stop()
        
    y_test = df_test[target_col].astype(int)
    X_test = df_test.drop(columns=[target_col])
    
    default_rate = (y_test.sum() / len(y_test)) * 100
    m4.metric("Test Default Rate", f"{default_rate:.2f}%")
    
    st.markdown("---")
    st.subheader(f"Individual Model Evaluation: {selected_model_name}")
    
    # Load serialized model pipeline
    model_path = model_paths[selected_model_name]
    pipe = joblib.load(model_path)
    
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else y_pred
    
    # Compute the 6 primary metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Display the 6 metric cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("AUC Score", f"{auc:.4f}")
    c3.metric("Precision", f"{prec:.4f}")
    c4.metric("Recall", f"{rec:.4f}")
    c5.metric("F1 Score", f"{f1:.4f}")
    c6.metric("MCC Score", f"{mcc:.4f}")
    
    st.markdown("---")
    col_cm, col_cr = st.columns([1, 1])
    
    with col_cm:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["No Default (0)", "Default (1)"],
                    yticklabels=["No Default (0)", "Default (1)"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(fig)
        
    with col_cr:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.4f}"))
        
    st.markdown("---")
    with st.expander("Preview Test Dataset Rows"):
        st.dataframe(df_test.head(10))
else:
    st.info("👈 Please check 'Use bundled test_data.csv' or upload a CSV file from the sidebar.")
