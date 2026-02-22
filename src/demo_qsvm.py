# demo_qsvm_portfolio.py

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. Create Synthetic Dataset
# ---------------------------
X, y = make_classification(
    n_samples=300,
    n_features=4,
    n_informative=3,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=1.2,
    random_state=42
)

df = pd.DataFrame(X, columns=['avg_return', 'volatility', 'volume', 'beta'])
df['label'] = y

print("Sample data (first 6 rows):")
print(df.head())

# ---------------------------
# 2. Train-Test Split & Scaling
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df[['avg_return', 'volatility', 'volume', 'beta']].values,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 3. Train Models
# ---------------------------
print("\n--- Training Models ---")

lr = LogisticRegression(max_iter=500)
lr.fit(X_train_scaled, y_train)

svc_rbf = SVC(kernel='rbf')
svc_rbf.fit(X_train_scaled, y_train)

# ---------------------------
# 4. Evaluation
# ---------------------------
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr.predict(X_test_scaled), digits=3))

print("\nClassical SVM (RBF) Classification Report:")
print(classification_report(y_test, svc_rbf.predict(X_test_scaled), digits=3))

lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
svm_acc = accuracy_score(y_test, svc_rbf.predict(X_test_scaled))

print(f"\nLogistic Regression Accuracy: {lr_acc:.3f}")
print(f"SVM (RBF) Accuracy: {svm_acc:.3f}")

# ---------------------------
# 5. Confusion Matrix Heatmap
# ---------------------------
cm = confusion_matrix(y_test, svc_rbf.predict(X_test_scaled))

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Classical SVM")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# ---------------------------
# 6. Accuracy Comparison Bar Chart
# ---------------------------
models = ['Logistic Regression', 'SVM (RBF)']
accuracies = [lr_acc, svm_acc]

plt.figure(figsize=(6, 4))
plt.bar(models, accuracies)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()

# ---------------------------
# 7. PCA Visualization
# ---------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(np.vstack([X_train_scaled, X_test_scaled]))
X_test_pca = X_pca[len(X_train_scaled):]

svc_preds = svc_rbf.predict(X_test_scaled)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_test, palette='tab10', s=60)
plt.title("Test Set - True Labels")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=svc_preds, palette='tab10', s=60)
plt.title("Classical SVM (RBF) - Predictions")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()
plt.show()

print("\nDemo complete.")
