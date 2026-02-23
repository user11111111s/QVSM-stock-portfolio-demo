# app.py
# =========================================
# Quantum-Ready Stock Risk Prediction GUI
# Clean Ubuntu-style Dashboard
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics.pairwise import rbf_kernel

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Quantum Stock Risk System", layout="wide")

st.title("Quantum-Ready Stock Risk Prediction System")
st.subheader("Classical vs QSVM Comparison ")

# --------------------------
# Load Data
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/stock_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

df = load_data()

# --------------------------
# Feature Engineering
# --------------------------
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
df['Volatility'] = df['Log_Return'].rolling(window=10).std()
df['Volume_Scaled'] = (
    df['Volume'] - df['Volume'].rolling(window=20).mean()
) / df['Volume'].rolling(window=20).std()

df = df.dropna().reset_index(drop=True)

# Future Label
df['Future_Volatility'] = (
    df['Log_Return']
    .rolling(window=5)
    .std()
    .shift(-5)
)

df = df.dropna().reset_index(drop=True)

threshold = df['Future_Volatility'].median()
df['Risk_Label'] = (df['Future_Volatility'] > threshold).astype(int)

# --------------------------
# Train-Test Split
# --------------------------
X = df[['Log_Return', 'Volatility', 'Volume_Scaled']].values
y = df['Risk_Label'].values

split_index = int(len(X) * 0.75)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# Train Classical Models
# --------------------------
lr = LogisticRegression(max_iter=500)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_preds)

svm = SVC(kernel='rbf')
svm.fit(X_train_scaled, y_train)
svm_preds = svm.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, svm_preds)

# --------------------------
# QSVM Proxy (Stable)
# --------------------------
QSVM_MAX = 120
df_q = df.tail(QSVM_MAX).reset_index(drop=True)

X_q = df_q[['Log_Return', 'Volatility', 'Volume_Scaled']].values
y_q = df_q['Risk_Label'].values

split_q = int(len(X_q) * 0.75)
Xq_train, Xq_test = X_q[:split_q], X_q[split_q:]
yq_train, yq_test = y_q[:split_q], y_q[split_q:]

scaler_q = MinMaxScaler(feature_range=(-1, 1))
Xq_train_scaled = scaler_q.fit_transform(Xq_train)
Xq_test_scaled = scaler_q.transform(Xq_test)

K_train = rbf_kernel(Xq_train_scaled, Xq_train_scaled, gamma=1.0)
K_test = rbf_kernel(Xq_test_scaled, Xq_train_scaled, gamma=1.0)

qsvm = SVC(kernel='precomputed')
qsvm.fit(K_train, yq_train)
qsvm_preds = qsvm.predict(K_test)
qsvm_acc = accuracy_score(yq_test, qsvm_preds)

# --------------------------
# Accuracy Display
# --------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Logistic Regression", f"{lr_acc:.3f}")
col2.metric("SVM", f"{svm_acc:.3f}")
col3.metric("QSVM (Proxy Mode)", f"{qsvm_acc:.3f}")

# --------------------------
# Accuracy Graph
# --------------------------
st.subheader("Model Accuracy Comparison")

fig1, ax1 = plt.subplots()
models = ['LR', 'SVM', 'QSVM']
accuracies = [lr_acc, svm_acc, qsvm_acc]
ax1.bar(models, accuracies)
ax1.set_ylim(0,1)
ax1.set_ylabel("Accuracy")
st.pyplot(fig1)

# --------------------------
# Confusion Matrix Selector
# --------------------------
st.subheader("Confusion Matrix")

model_choice = st.selectbox(
    "Select Model",
    ["Logistic Regression", "SVM", "QSVM (Proxy)"]
)

if model_choice == "Logistic Regression":
    preds = lr_preds
    true = y_test
elif model_choice == "SVM":
    preds = svm_preds
    true = y_test
else:
    preds = qsvm_preds
    true = yq_test

cm = confusion_matrix(true, preds)

fig2, ax2 = plt.subplots()
ax2.imshow(cm)
ax2.set_title(f"Confusion Matrix - {model_choice}")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax2.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig2)

# --------------------------
# Classification Report
# --------------------------
st.subheader("Classification Report")

report = classification_report(true, preds, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# --------------------------
# Volatility Graph
# --------------------------
st.subheader("Rolling Volatility Over Time")

fig3, ax3 = plt.subplots(figsize=(10,4))
ax3.plot(df['Date'], df['Volatility'])
ax3.set_xlabel("Date")
ax3.set_ylabel("Volatility")
st.pyplot(fig3)

# --------------------------
# Explanation Section
# --------------------------
st.subheader("System Explanation")

st.write("""
• Log Returns convert price into stationary signal  
• Rolling Volatility measures market instability  
• Future Volatility used to create realistic risk labels  
• Chronological split prevents data leakage  
• QSVM uses kernel-based architecture inspired by quantum ML  
• Proxy mode ensures stable execution on all systems  
""")