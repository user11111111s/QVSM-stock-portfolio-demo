# final_demo_qsvm_portfolio.py
# ===============================
# Quantum-Ready Stock Risk Model (robust QSVM block)
# ===============================

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
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

VOL_WINDOW = 10
df['Volatility'] = df['Log_Return'].rolling(window=VOL_WINDOW).std()

VOL_Z_WINDOW = 20
df['Volume_Scaled'] = (
    df['Volume'] - df['Volume'].rolling(window=VOL_Z_WINDOW).mean()
) / df['Volume'].rolling(window=VOL_Z_WINDOW).std()

df = df.dropna().reset_index(drop=True)
print("Rows after initial cleaning:", len(df))

# ---------------------------
# 3. Future Risk Label (NO LEAKAGE)
# ---------------------------
FUTURE_WINDOW = 5
df['Future_Volatility'] = (
    df['Log_Return']
    .rolling(window=FUTURE_WINDOW)
    .std()
    .shift(-FUTURE_WINDOW)
)
df = df.dropna().reset_index(drop=True)

future_threshold = df['Future_Volatility'].median()
df['Risk_Label'] = (df['Future_Volatility'] > future_threshold).astype(int)
print("Rows after future labeling:", len(df))

# ---------------------------
# 4. Define Features
# ---------------------------
X_full = df[['Log_Return', 'Volatility', 'Volume_Scaled']].values
y_full = df['Risk_Label'].values

# ---------------------------
# 5. Chronological Split
# ---------------------------
split_index = int(len(X_full) * 0.75)
X_train = X_full[:split_index]; X_test = X_full[split_index:]
y_train = y_full[:split_index]; y_test = y_full[split_index:]

# ---------------------------
# 6. Scaling (Quantum-Ready)
# ---------------------------
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("Feature scaling complete.")
print("Scaled range:", X_train_scaled.min(), "to", X_train_scaled.max())

# ---------------------------
# 7. Classical Models
# ---------------------------
print("\n--- Training Classical Models ---")
lr = LogisticRegression(max_iter=500)
lr.fit(X_train_scaled, y_train)

svc_rbf = SVC(kernel='rbf')
svc_rbf.fit(X_train_scaled, y_train)

print("\nLogistic Regression Results:")
print(classification_report(y_test, lr.predict(X_test_scaled), digits=3))

print("\nClassical SVM Results:")
print(classification_report(y_test, svc_rbf.predict(X_test_scaled), digits=3))

print("LR Accuracy:", accuracy_score(y_test, lr.predict(X_test_scaled)))
print("SVM Accuracy:", accuracy_score(y_test, svc_rbf.predict(X_test_scaled)))

# ---------------------------
# 8. Real QSVM Implementation (robust & compatibility-safe)
# ---------------------------
print("\n--- Running QSVM (robust compatibility) ---")

# Prepare ultra-safe subset (chronological tail)
QSVM_MAX = 120
df_qsvm_small = df.tail(QSVM_MAX).reset_index(drop=True)

X_q = df_qsvm_small[['Log_Return', 'Volatility', 'Volume_Scaled']].values
y_q = df_qsvm_small['Risk_Label'].values

split_q = int(len(X_q) * 0.75)
Xq_train = X_q[:split_q]; Xq_test = X_q[split_q:]
yq_train = y_q[:split_q]; yq_test = y_q[split_q:]

# Scale for quantum/proxy
scaler_q = MinMaxScaler(feature_range=(-1, 1))
Xq_train_scaled = scaler_q.fit_transform(Xq_train)
Xq_test_scaled  = scaler_q.transform(Xq_test)

print("QSVM subset size:", len(X_q))

# We'll try multiple API options; track which one we used
qkernel = None
used_api = None

# Prepare the feature map (shallow)
try:
    from qiskit.circuit.library import ZZFeatureMap
    feature_map = ZZFeatureMap(feature_dimension=3, reps=1, entanglement='linear')
except Exception:
    feature_map = None

# 1) Try modern Sampler + FidelityQuantumKernel (qiskit 0.40+ / 1.x / 2.x environments)
if feature_map is not None:
    try:
        from qiskit.primitives import Sampler  # modern primitive
        from qiskit_machine_learning.kernels import FidelityQuantumKernel

        sampler = Sampler()
        qkernel = FidelityQuantumKernel(feature_map=feature_map, sampler=sampler)
        used_api = "fidelity_kernel_with_Sampler"
        print("Using API: FidelityQuantumKernel + Sampler")
    except Exception as e1:
        qkernel = None

# 2) Try qiskit_aer AerSampler (some installs expose AerSampler)
if qkernel is None and feature_map is not None:
    try:
        from qiskit_aer.primitives import AerSampler
        from qiskit_machine_learning.kernels import FidelityQuantumKernel

        sampler = AerSampler()
        qkernel = FidelityQuantumKernel(feature_map=feature_map, sampler=sampler)
        used_api = "fidelity_kernel_with_AerSampler"
        print("Using API: FidelityQuantumKernel + AerSampler (qiskit_aer)")
    except Exception:
        qkernel = None

# 3) Fallback: older QuantumKernel + QuantumInstance (if the environment supports it)
if qkernel is None and feature_map is not None:
    try:
        from qiskit_machine_learning.kernels import QuantumKernel
        # QuantumInstance exists only on some older-mid versions; try to import
        try:
            from qiskit.utils import QuantumInstance
            from qiskit_aer import AerSimulator
            backend = AerSimulator()
            qi = QuantumInstance(backend, shots=1024, seed_simulator=42, seed_transpiler=42)
            qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=qi)
            used_api = "quantumkernel_with_quantuminstance"
            print("Using API: QuantumKernel + QuantumInstance (fallback)")
        except Exception:
            qkernel = None
    except Exception:
        qkernel = None

# 4) If none of the quantum APIs are available — use a classical RBF kernel as a documented proxy
if qkernel is None:
    from sklearn.metrics.pairwise import rbf_kernel
    print("No compatible quantum-kernel API found. Using classical RBF kernel as a 'quantum-proxy' for demo.")
    used_api = "classical_rbf_proxy"

    # compute train & test classical kernel matrices (RBF)
    K_train = rbf_kernel(Xq_train_scaled, Xq_train_scaled, gamma=1.0)
    K_test  = rbf_kernel(Xq_test_scaled, Xq_train_scaled, gamma=1.0)

    svc_q = SVC(kernel='precomputed')
    svc_q.fit(K_train, yq_train)
    preds_q = svc_q.predict(K_test)

    print("\nQSVM (proxy) Results (API used: {}):".format(used_api))
    print(classification_report(yq_test, preds_q, digits=3))
    print("QSVM (proxy) Accuracy:", accuracy_score(yq_test, preds_q))

else:
    # Real quantum-kernel path
    try:
        print("Computing quantum kernel matrix (train)... (this may take a few seconds)")
        K_train = qkernel.evaluate(x_vec=Xq_train_scaled)

        svc_q = SVC(kernel='precomputed')
        svc_q.fit(K_train, yq_train)

        print("Computing quantum kernel matrix (test)...")
        K_test = qkernel.evaluate(x_vec=Xq_test_scaled, y_vec=Xq_train_scaled)

        preds_q = svc_q.predict(K_test)

        print("\nQSVM Results (API used: {}):".format(used_api))
        print(classification_report(yq_test, preds_q, digits=3))
        print("QSVM Accuracy:", accuracy_score(yq_test, preds_q))
    except Exception as e:
        print("QSVM attempt failed during kernel computation or SVM training.")
        print("Error:", e)
        print("As a safe fallback, you can run the script again after installing compatible qiskit packages or use the proxy path.")

print("\nScript completed (classical + QSVM/proxy).")