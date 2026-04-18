# 📌 Credit Card Fraud Detection System

---

## 🚀 Overview

This project builds a Machine Learning system to detect fraudulent credit card transactions using real-world imbalanced data.

The goal is to **maximize fraud detection (recall)** while maintaining a reasonable number of false alerts.

---

## 📊 Problem Statement

Credit card fraud detection is a highly imbalanced classification problem:

* Legitimate transactions ≈ **99.7%**
* Fraud transactions ≈ **0.3%**

Traditional models fail by predicting everything as legitimate.

---

## 🎯 Objective

* Detect fraudulent transactions effectively
* Handle extreme class imbalance
* Optimize using **Recall & ROC-AUC**

---

## 🧠 Approach

### 🔹 1. Data Preprocessing

* Removed irrelevant columns
* Applied One-Hot Encoding
* Sampled dataset for faster training

### 🔹 2. Models Used

* **Baseline** → Random Forest (no imbalance handling)
* **Class Weight** → `class_weight="balanced"`
* **SMOTE** → Synthetic oversampling
* **XGBoost** → Boosting model

### 🔹 3. Handling Class Imbalance

* Class Weighting
* SMOTE

### 🔹 4. Evaluation Metrics

* Precision
* **Recall (Primary Focus)**
* F1-score
* ROC-AUC
* Confusion Matrix

---

## 🏆 Final Model

**Selected Model:**
Random Forest with Class Weight

```python
RandomForestClassifier(class_weight="balanced")
```

**Threshold:**

```python
threshold = 0.5
```

---

## 📈 Results

| Metric    | Value   |
| --------- | ------- |
| Accuracy  | 0.99    |
| Recall    | 0.60 🔥 |
| Precision | 0.38    |
| ROC-AUC   | 0.93    |

👉 Successfully detected **60% fraud cases**

---

## 📊 Feature Importance

* 💰 Transaction Amount
* 🌍 Location (Latitude, Longitude)
* 🏙 City Population
* 🛒 Transaction Category

---

## 📉 Threshold Tuning

* Lower threshold → higher recall but more false positives
* Best performance at **0.5**

---

## 🖥️ Deployment

Built using **Streamlit**

### Features:

* Upload CSV
* Real-time prediction
* Fraud probability output

---

## 🗂️ Project Structure

```bash
Credit_card_fraud_detection/
│
├── app/
│   └── app.py              # Streamlit app
│
├── notebooks/
│   └── EDA.ipynb          # Exploratory Data Analysis
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── main.py                # Run pipeline
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

```bash
git clone <https://github.com/TwinkleGhosh/CODSOFT_TASK1_Credit_card_fraud_dectection.git>
cd Credit_card_fraud_detection

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python main.py
streamlit run app/app.py
```

---

## 💡 Key Learnings

* Handling imbalanced datasets
* Importance of recall
* Precision vs recall trade-off
* Model deployment

---

## 🚀 Future Improvements

* Hyperparameter tuning
* Precision-Recall curves
* Cloud deployment
* API integration

---

## 👩‍💻 Author

**Twinkle Ghosh**

---

⭐ Star this repo if you found it useful!


---

