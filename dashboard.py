import streamlit as st
import requests

st.title("🛒 E-commerce Customer Churn Dashboard")

st.write("Enter customer details to predict churn risk")

recency = st.number_input("Recency (days)", min_value=0)
frequency = st.number_input("Frequency (orders)", min_value=0)
monetary = st.number_input("Monetary Value", min_value=0.0)

if st.button("Predict Churn"):
    payload = {
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=payload)

    if response.status_code == 200:
        result = response.json()

        st.subheader("Prediction Result")

        if result["churn_prediction"] == 1:
            st.error(f"⚠ High Risk of Churn ({result['churn_probability']:.2f})")
        else:
            st.success(f"✅ Low Risk of Churn ({result['churn_probability']:.2f})")