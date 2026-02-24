# final_demo_qsvm_portfolio.py
# ===============================
# Quantum-Ready Stock Risk Model (Improved Balanced QSVM)
# ===============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_csv("../data/stock_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

print("Total rows loaded:", len(df))

# ---------------------------
# 2. Feature Engineering
# ---------------------------
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

df['Volatility'] = df['Log_Return'].rolling(window=10).std()

df['Volume_Scaled'] = (
    df['Volume'] - df['Volume'].rolling(window=20).mean()
) / df['Volume'].rolling(window=20).std()

df = df.dropna().reset_index(drop=True)
print("Rows after initial cleaning:", len(df))

# ---------------------------
# 3. Future Risk Label
# ---------------------------
df['Future_Volatility'] = (
    df['Log_Return']
    .rolling(window=5)
    .std()
    .shift(-5)
)

df = df.dropna().reset_index(drop=True)

future_threshold = df['Future_Volatility'].median()
df['Risk_Label'] = (df['Future_Volatility'] > future_threshold).astype(int)

print("Rows after future labeling:", len(df))

# ---------------------------
# 4. Define Features
# ---------------------------
features = ['Log_Return', 'Volatility', 'Volume_Scaled']
X_full = df[features].values
y_full = df['Risk_Label'].values

# ---------------------------
# 5. Chronological Split
# ---------------------------
split_index = int(len(X_full) * 0.75)
X_train = X_full[:split_index]
X_test  = X_full[split_index:]
y_train = y_full[:split_index]
y_test  = y_full[split_index:]

# ---------------------------
# 6. Scaling
# ---------------------------
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---------------------------
# 7. Classical Models (Balanced + Tuned)
# ---------------------------
print("\n--- Training Classical Models ---")

lr = LogisticRegression(max_iter=500)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)

# Tuned SVM
svc_rbf = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
svc_rbf.fit(X_train_scaled, y_train)
svm_preds = svc_rbf.predict(X_test_scaled)

lr_acc = accuracy_score(y_test, lr_preds)
svm_acc = accuracy_score(y_test, svm_preds)

print("\nLogistic Regression Accuracy:", lr_acc)
print("Balanced Tuned SVM Accuracy:", svm_acc)

# ---------------------------
# 8. QSVM (Improved Proxy)
# ---------------------------
print("\n--- Running QSVM (Balanced Proxy Mode) ---")

QSVM_MAX = 180   # Increased slightly (safe range)
df_qsvm_small = df.tail(QSVM_MAX).reset_index(drop=True)

X_q = df_qsvm_small[features].values
y_q = df_qsvm_small['Risk_Label'].values

split_q = int(len(X_q) * 0.75)
Xq_train = X_q[:split_q]
Xq_test  = X_q[split_q:]
yq_train = y_q[:split_q]
yq_test  = y_q[split_q:]

scaler_q = MinMaxScaler(feature_range=(-1, 1))
Xq_train_scaled = scaler_q.fit_transform(Xq_train)
Xq_test_scaled  = scaler_q.transform(Xq_test)

# Tuned gamma for better separation
K_train = rbf_kernel(Xq_train_scaled, Xq_train_scaled, gamma=0.5)
K_test  = rbf_kernel(Xq_test_scaled, Xq_train_scaled, gamma=0.5)

# Balanced QSVM
svc_q = SVC(kernel='precomputed', C=5, class_weight='balanced')
svc_q.fit(K_train, yq_train)

qsvm_preds = svc_q.predict(K_test)
qsvm_acc = accuracy_score(yq_test, qsvm_preds)

print("Balanced QSVM (Proxy) Accuracy:", qsvm_acc)

# ===============================
# VISUALIZATIONS
# ===============================

# Accuracy Comparison
plt.figure(figsize=(6,4))
models = ['Logistic Regression', 'Balanced SVM', 'Balanced QSVM']
accuracies = [lr_acc, svm_acc, qsvm_acc]
plt.bar(models, accuracies)
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()

# Confusion Matrix (QSVM now)
cm = confusion_matrix(yq_test, qsvm_preds)

plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix - Balanced QSVM")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

# Volatility Graph
plt.figure(figsize=(10,4))
plt.plot(df['Date'], df['Volatility'], label='Volatility')
plt.title("Rolling Volatility Over Time")
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.legend()
plt.show()

print("\nScript completed successfully with balanced QSVM improvements.")