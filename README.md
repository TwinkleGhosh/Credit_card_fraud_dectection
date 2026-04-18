📌 Credit Card Fraud Detection System
🚀 Overview

This project builds a Machine Learning system to detect fraudulent credit card transactions using real-world imbalanced data.

The goal is to maximize fraud detection (recall) while maintaining a reasonable number of false alerts.

📊 Problem Statement

Credit card fraud detection is a highly imbalanced classification problem where:

Legitimate transactions ≈ 99.7%
Fraud transactions ≈ 0.3%

Traditional models fail by predicting everything as legitimate.

🎯 Objective
Detect fraudulent transactions effectively
Handle extreme class imbalance
Optimize model using evaluation metrics like Recall & ROC-AUC
🧠 Approach
1️⃣ Data Preprocessing
Removed irrelevant columns (ID, personal info, etc.)
Handled categorical variables using One-Hot Encoding
Sampled dataset for faster training
2️⃣ Models Used
Model	Description
Baseline	Random Forest without imbalance handling
Class Weight	Random Forest with class_weight="balanced"
SMOTE	Oversampling minority class using synthetic data
XGBoost	Advanced boosting model
3️⃣ Handling Class Imbalance
Applied Class Weighting
Applied SMOTE (Synthetic Oversampling)
4️⃣ Evaluation Metrics
Precision
Recall (Primary focus)
F1-score
ROC-AUC Score
Confusion Matrix
🏆 Final Model
✅ Selected Model:

Random Forest with Class Weight

RandomForestClassifier(class_weight="balanced")
✅ Final Threshold:
threshold = 0.5
📈 Final Results
Metric	Value
Accuracy	0.99
Fraud Recall	0.60 🔥
Precision	0.38
ROC-AUC	0.93

👉 Successfully detected 60% of fraud cases, a major improvement over baseline (0%).

📊 Feature Importance

Top features influencing fraud detection:

💰 Transaction Amount (amt)
🌍 Location (latitude, longitude)
🏙 City Population
🛒 Transaction Category

👉 Indicates fraud depends on transaction behavior + location patterns

📉 Threshold Tuning Insight
Lowering threshold increased recall but caused over-prediction of fraud
Optimal performance achieved at threshold = 0.5
🖥️ Deployment

Built an interactive web app using Streamlit:

Features:
Upload transaction dataset (CSV)
Predict fraud in real-time
Display fraud probability and predictions
🗂️ Project Structure
Credit_card_fraud_detection/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── model_baseline.pkl
│   ├── model_weighted.pkl
│   ├── model_smote.pkl
│   └── model_xgb.pkl
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── app/
│   └── app.py
│
├── main.py
└── requirements.txt
▶️ How to Run
1. Clone Repository
git clone <your-repo-link>
cd Credit_card_fraud_detection
2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Run Model
python main.py
5. Run Web App
streamlit run app/app.py
💡 Key Learnings
Handling imbalanced datasets
Importance of recall in fraud detection
Trade-off between precision vs recall
Real-world model evaluation
Deployment using Streamlit
🚀 Future Improvements
Hyperparameter tuning (GridSearchCV)
Precision-Recall Curve visualization
Deploy on cloud (Streamlit Cloud / AWS)
Real-time API integration
👩‍💻 Author

Twinkle Ghosh

⭐ If you found this useful, consider giving it a star!