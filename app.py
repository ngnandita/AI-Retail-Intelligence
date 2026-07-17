import streamlit as st
import pandas as pd
import joblib

# Load AI Model
model = joblib.load("models/xgboost_model.pkl")

st.set_page_config(
    page_title="AI Retail Intelligence",
    page_icon="🛒"
)

st.title("🛒 AI Retail Intelligence")
st.markdown("---")

st.info(
    "AI Powered Sales Prediction Dashboard"
)

st.write("Predict future sales using Machine Learning.")

quantity = st.number_input(
    "Quantity",
    min_value=1,
    value=5
)

discount = st.number_input(
    "Discount",
    min_value=0.0,
    max_value=1.0,
    value=0.10
)

profit = st.number_input(
    "Profit",
    value=120.0
)

if st.button("Predict Sales"):

    sample = pd.DataFrame({

        "Quantity":[quantity],

        "Discount":[discount],

        "Profit":[profit]

    })

    prediction = model.predict(sample)

    st.success(
        f"Predicted Sales : ₹ {prediction[0]:.2f}"
    )