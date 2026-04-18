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

        threshold = 0.1

        predictions = (probs >= threshold).astype(int)

        data["Prediction"] = predictions
        data["Fraud Probability"] = probs

        st.write("Prediction Results:")
        st.dataframe(data.head())

        fraud_count = sum(predictions)

        st.write(f"🚨 Fraud Transactions Detected: {fraud_count}")

else:

    st.info("Please upload a CSV file to begin.")
