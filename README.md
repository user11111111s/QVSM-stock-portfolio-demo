# Quantum Support Vector Machine for Stock Portfolio Classification

## Project Overview

This project demonstrates a machine learning approach to stock portfolio risk classification using classical Support Vector Machines (SVM) and compares it with Logistic Regression.

The objective is to simulate how machine learning models can classify stocks into:

* Low Risk (Class 0)
* High Risk (Class 1)

based on financial features such as average return, volatility, trading volume, and beta.

Although the original research paper focuses on Quantum Support Vector Machines (QSVM), this implementation demonstrates the classical SVM workflow clearly and includes optional quantum kernel support if Qiskit is installed.

This project is created for Machine Learning CIA-1 demonstration.

---

## Problem Statement

Portfolio management requires selecting financial assets while managing risk.
Machine learning can help classify stocks based on risk characteristics using historical data.

In this demo:

* Input: Financial features of stocks
* Output: Risk category (Low Risk or High Risk)
* Goal: Compare model performance and evaluate classification accuracy

---

## Dataset

This project uses a **synthetic dataset** generated using `sklearn.make_classification`.

Why synthetic data?

* It allows controlled experimentation
* It avoids dependency on external financial APIs
* It ensures reproducibility
* It simplifies demonstration for academic purposes

The dataset simulates:

* avg_return
* volatility
* volume
* beta

Total samples: 300
Test split: 25%

---

## Machine Learning Pipeline

The project follows a complete ML workflow:

1. Data Generation
2. Data Preprocessing (Normalization using MinMaxScaler)
3. Train-Test Split
4. Model Training

   * Logistic Regression
   * Support Vector Machine (RBF Kernel)
5. Model Evaluation

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * Confusion Matrix
6. Visualization

   * PCA projection plot
   * Confusion matrix heatmap
   * Accuracy comparison bar chart

---

## Models Used

### Logistic Regression

Used as a baseline classifier.
Works well for linearly separable data.

Accuracy achieved: ~73%

---

### Support Vector Machine (RBF Kernel)

Used to handle non-linear classification boundaries.
The RBF kernel maps input features into higher-dimensional space.

Accuracy achieved: ~86%

This demonstrates that the dataset has non-linear characteristics better captured by SVM.

---

## Visualization Explanation

### PCA Plot

PCA reduces 4-dimensional data into 2 principal components for visualization.

Left Plot:
True Labels (Actual Risk Category)

Right Plot:
Predicted Labels (SVM Output)

If both plots look similar, it indicates successful learning.

---

### Confusion Matrix Heatmap

The confusion matrix shows:

* Correct predictions (diagonal values)
* Incorrect predictions (off-diagonal values)

Example:

37 correct Low Risk
28 correct High Risk
1 false positive
9 false negatives

This confirms 86.7% accuracy.

---

### Accuracy Comparison Chart

Bar chart compares performance:

Logistic Regression → ~73%
SVM (RBF) → ~86%

This visually demonstrates model improvement.

---

## Optional Quantum Extension

If `qiskit` and `qiskit-machine-learning` are installed, the script can also evaluate a Quantum Kernel-based SVM (QSVM).

Note:
In this project, the quantum part is optional and uses a simulator backend.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Qiskit (optional)

---

## How to Run

1. Create virtual environment

python -m venv venv
venv\Scripts\activate

2. Install dependencies

pip install -r requirements.txt

3. Run script

python src/demo_qsvm.py

---

## Key Learning Outcomes

* Understanding supervised classification
* Comparing ML models
* Interpreting confusion matrices
* Visualizing high-dimensional data using PCA
* Evaluating model performance using accuracy and F1-score
* Understanding the role of kernel functions in SVM

---

## Future Improvements

* Replace synthetic dataset with real financial stock data
* Implement full Quantum Support Vector Machine comparison
* Add portfolio return optimization layer
* Integrate reinforcement learning for dynamic asset allocation


