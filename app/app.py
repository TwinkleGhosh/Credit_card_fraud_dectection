import streamlit as st
import joblib
import pandas as pd


# load model
model = joblib.load("models/model_weighted.pkl")


st.title("💳 Credit Card Fraud Detection")

st.write("This app predicts whether a transaction is Fraud or Legitimate")


# upload CSV option
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])


if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Preview of Data:")
    st.dataframe(data.head())

    if st.button("Predict Fraud"):

        # ensure same preprocessing as training
        data = data.drop(
            [
                "Unnamed: 0",
                "trans_date_trans_time",
                "cc_num",
                "first",
                "last",
                "street",
                "trans_num",
            ],
            axis=1,
            errors="ignore",
        )

        data = pd.get_dummies(data, drop_first=True)

        # align columns with training data
        model_features = model.feature_names_in_

        data = data.reindex(columns=model_features, fill_value=0)

        # predict probabilities
        probs = model.predict_proba(data)[:, 1]

        threshold = st.slider(
            "Select Fraud Detection Threshold (Lower = More Fraud Detection)",
            0.1,
            0.9,
            0.5,
            0.05,
        )

        if threshold < 0.3:
            st.warning("⚠️ Low threshold → High fraud detection but more false alarms")

        elif threshold > 0.7:
            st.warning("⚠️ High threshold → Fewer false alarms but may miss frauds")

        else:
            st.success("✅ Balanced threshold selected")

        predictions = (probs >= threshold).astype(int)

        data["Prediction"] = predictions
        data["Fraud Probability"] = probs

        st.write("Prediction Results:")
        st.dataframe(data.head())

        fraud_count = sum(predictions)
        legit_count = len(predictions) - fraud_count
        total_count = len(predictions)
        fraud_percent = (fraud_count / total_count) * 100

        # show count + percentage
        st.write(f"🚨 Fraud Transactions Detected: {fraud_count}")
        st.write(f"📈 Fraud Percentage: {fraud_percent:.2f}%")
        st.write(f"✅ Legit: {legit_count}")

        # graph 1: fraud vs legit count
        st.subheader("📊 Fraud vs Legit Transactions")

        chart_data = pd.DataFrame(
            {
                "Type": ["Legit", "Fraud"],
                "Count": [total_count - fraud_count, fraud_count],
            }
        )

        st.bar_chart(chart_data.set_index("Type"))

        # graph 2: probability distribution
        st.subheader("📉 Fraud Probability Distribution")

        st.line_chart(probs)

st.info(
    "🔍 Predictions are based on probability scores from the model. "
    "Transactions above the selected threshold are classified as fraud. "
    "Lower thresholds increase fraud detection (recall) but may increase false positives."
)
