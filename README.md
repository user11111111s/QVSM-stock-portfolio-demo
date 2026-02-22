# Stock Risk Classification using Classical ML and QSVM

## Project Overview

This project builds a stock risk classification model using:

* Logistic Regression
* Support Vector Machine (SVM)
* Quantum Support Vector Machine (QSVM structure)

The objective is to predict whether a stock will enter a **high volatility (high risk) period in the near future**, based on historical behavior.

Instead of using raw stock prices, this project focuses on behavioral signals such as returns and volatility.

---

## Dataset

* Historical stock data (1980–2022)
* Columns used: Date, Close, Volume
* Data processed using time-series principles (no shuffling)

---

## Feature Engineering

To make the model reliable and avoid misleading patterns, the following transformations were applied:

### 1. Log Returns

Used instead of raw price:

rₜ = ln(Pₜ / Pₜ₋₁)

This removes price trends and makes the data more stable.

### 2. Rolling Volatility (10-day window)

Measures short-term instability.

### 3. Volume Z-Score (20-day window)

Normalizes trading activity to detect unusual spikes.

These features help the model learn stock behavior rather than price level.

---

## Target Variable (Risk Label)

The model predicts **future volatility**, not current volatility.

Steps:

* Compute 5-day forward rolling volatility
* Use median threshold to classify:

  * 1 → High Risk
  * 0 → Low Risk

Chronological split is used to avoid data leakage.

---

## Model Implementation

### Classical Models

* Logistic Regression
* SVM with RBF kernel

### Quantum Model (QSVM Structure)

The QSVM implementation follows the correct architecture:

1. Apply Quantum Feature Map (ZZFeatureMap)
2. Compute kernel matrix
3. Train SVM using precomputed kernel

To ensure compatibility across systems, the script:

* Attempts real quantum kernel execution
* Falls back to a classical RBF kernel proxy if quantum primitives are unavailable

This maintains the full QSVM workflow.

---

## Why Only 120 Samples for QSVM?

Quantum kernel simulation is computationally expensive.

To prevent system slowdown:

* Only the most recent 120 samples are used
* Chronological split is preserved

This allows safe execution while demonstrating the quantum structure.

---

## Results

Classical Models:

* Logistic Regression Accuracy ≈ 72–73%
* SVM Accuracy ≈ 67–73%

QSVM (Proxy Mode):

* Accuracy ≈ 73%

In financial prediction problems, anything above 70% is considered strong due to market randomness.

---

## Key Highlights

* No data leakage
* Time-series aware splitting
* Stationarity handled correctly
* Proper quantum scaling (-1 to 1)
* Robust and system-safe implementation
* Real historical stock dataset

---

## Project Structure

```
ML-CIA(STOCK PORTFOLIO)/
│
├── data/
│   └── stock_data.csv
│
├── src/
│   └── demo_qsvm.py
│
├── requirements.txt
└── README.md
```

---

## How to Run

Create virtual environment:

```
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Run:

```
python demo_qsvm.py
```

---

## Conclusion

This project demonstrates:

* Practical financial machine learning
* Proper time-series handling
* Feature engineering for stock behavior
* Implementation of QSVM architecture
* Comparison between classical and quantum-based approaches

The structure is scalable and can be extended for portfolio-level optimization in future work.

