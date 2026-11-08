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

# Sidebar controls
st.sidebar.header("1. Upload Test Dataset")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

st.sidebar.header("2. Model Selection")
selected_model_name = st.sidebar.selectbox("Choose a classifier", list(model_paths.keys()))

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Test Data Preview")
    st.dataframe(df_test.head(5))
    
    # Target column extraction
    target_col = None
    for col in df_test.columns:
        if "default" in col.lower() or "target" in col.lower():
            target_col = col
            break
            
    if target_col is None:
        st.error("Target column indicating credit default not found in CSV.")
        st.stop()
        
    y_test = df_test[target_col].astype(int)
    X_test = df_test.drop(columns=[target_col])
    
    # Load serialized model pipeline
    model_path = model_paths[selected_model_name]
    pipe = joblib.load(model_path)
    
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else y_pred
    
    # Compute Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    st.markdown("---")
    st.subheader(f"Evaluation Metrics: {selected_model_name}")
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
else:
    st.info("👈 Please upload `test_data.csv` from the sidebar to inspect metrics.")
