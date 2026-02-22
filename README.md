# Quantum-Ready Stock Risk Prediction (QSVM + Classical ML)

## Overview

This project builds a stock risk prediction model using:

- Logistic Regression (baseline)
- Classical SVM (RBF kernel)
- Quantum Support Vector Machine (QSVM) with safe fallback

The model predicts **future market risk (volatility)** using behavior-based financial features instead of raw stock prices.

---

## Why This Project Is Different

Instead of using raw stock prices (which are non-stationary), this project uses:

- Log Returns
- Rolling Volatility
- Normalized Volume (Z-score)

This ensures:
- No price-level bias
- No trend leakage
- Time-series correct splitting
- Proper future prediction setup

The model predicts whether the **next 5-day volatility** will be above or below median risk.

---

## Dataset

- Historical stock data (1980–2022)
- 10,000+ trading days
- Features engineered from raw OHLCV data

Data file location:
```
data/stock_data.csv
```

---

## Feature Engineering

1. **Log Return**
   r_t = ln(P_t / P_t-1)

2. **Rolling Volatility (10-day window)**

3. **Volume Z-score (20-day rolling normalization)**

4. **Future Volatility Label (5-day forward window)**  
   - Label = 1 → High future risk  
   - Label = 0 → Low future risk  

No data leakage is used.

---

## Model Pipeline

### 1. Chronological Train-Test Split
Time-series correct split (75% train, 25% test)

### 2. Feature Scaling
MinMaxScaler range (-1, 1)  
Quantum circuits require normalized input ranges.

### 3. Classical Models
- Logistic Regression
- SVM (RBF)

### 4. QSVM Block
- Attempts real quantum kernel using Qiskit
- If environment mismatch → safely falls back to classical RBF proxy
- Designed to never crash the script

---

## Visualizations

The script generates:

- Model Accuracy Comparison Bar Chart
- Confusion Matrix (SVM)
- Rolling Volatility Time-Series Plot

These graphs make the results easier to interpret.

---

## Example Results

Typical results (full dataset):

- Logistic Regression: ~72–73% accuracy
- Classical SVM: ~67–70% accuracy
- QSVM / Proxy: ~73% accuracy

Note:
Financial prediction is noisy and difficult.  
70% accuracy in future volatility classification is strong.

---

## Project Structure

```
ML-CIA(STOCK PORTFOLIO)/
│
├── data/
│   └── stock_data.csv
│
├── src/
│   └── final_demo_qsvm_portfolio.py
│
├── requirements.txt
└── README.md
```

---

## How to Run

1. Create virtual environment:

```
python -m venv venv
```

2. Activate it:

Windows:
```
venv\Scripts\activate
```

3. Install requirements:

```
pip install -r requirements.txt
```

4. Run the script:

```
python src/final_demo_qsvm_portfolio.py
```

---

## Optional: Enable Full Quantum Mode

If you want real QSVM execution:

```
pip install qiskit qiskit-machine-learning qiskit-aer
```

If Qiskit versions conflict, the script automatically switches to proxy mode.

---

## Key Highlights

✔ No data leakage  
✔ Time-series correct splitting  
✔ Stationary financial features  
✔ Quantum-ready scaling  
✔ Robust QSVM compatibility block  
✔ Clean visualizations  
✔ Safe fallback system  

---

## Conclusion

This project demonstrates:

- Proper financial feature engineering
- Classical vs Quantum kernel comparison
- Robust ML system design
- Practical constraints of quantum simulators

It combines financial modeling principles with Quantum Machine Learning experimentation in a stable, reproducible pipeline.

---

